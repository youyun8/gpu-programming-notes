// cuemu: a tiny CPU emulator for CUDA C++ kernels.
//
// Every CUDA thread runs as a fiber (user-space coroutine). Blocks execute one
// at a time; inside a block the fibers are scheduled cooperatively and only
// switch at synchronization points (__syncthreads, __syncwarp, warp shuffles,
// votes). This gives real barrier semantics, detects deadlocks, and makes it
// possible to verify kernels for correctness on a machine without a GPU.
//
// It is NOT a performance model and it does not emulate PTX, tensor cores,
// or memory-model subtleties beyond "barriers order memory".
#pragma once

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <math.h>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------
// Qualifiers
// ---------------------------------------------------------------------------
#define __global__
#define __device__
#define __host__
#define __noinline__ __attribute__((noinline))
#define __forceinline__ inline __attribute__((always_inline))
#define __restrict__ __restrict
#define __shared__ static
#define __constant__
#define __managed__
#define __launch_bounds__(...)
#define __align__(n) __attribute__((aligned(n)))
#define __grid_constant__
#define __CUDACC__ 1
#define __CUDA_ARCH__ 800
#define CUDART_VERSION 12000
#define CUDART_INF_F HUGE_VALF
#define CUDART_NAN_F NAN
#define CUDART_PI_F 3.141592654f

// ---------------------------------------------------------------------------
// Vector types
// ---------------------------------------------------------------------------
#define CUEMU_VEC2(T, N, A)                                                     \
    struct alignas(A) N##2 { T x, y; };                                         \
    inline N##2 make_##N##2(T x, T y) { return N##2{x, y}; }
#define CUEMU_VEC3(T, N)                                                        \
    struct N##3 { T x, y, z; };                                                 \
    inline N##3 make_##N##3(T x, T y, T z) { return N##3{x, y, z}; }
#define CUEMU_VEC4(T, N, A)                                                     \
    struct alignas(A) N##4 { T x, y, z, w; };                                   \
    inline N##4 make_##N##4(T x, T y, T z, T w) { return N##4{x, y, z, w}; }
#define CUEMU_VEC1(T, N)                                                        \
    struct N##1 { T x; };                                                       \
    inline N##1 make_##N##1(T x) { return N##1{x}; }

CUEMU_VEC1(float, float) CUEMU_VEC2(float, float, 8) CUEMU_VEC3(float, float) CUEMU_VEC4(float, float, 16)
CUEMU_VEC1(double, double) CUEMU_VEC2(double, double, 16) CUEMU_VEC3(double, double)
struct alignas(16) double4 { double x, y, z, w; };
inline double4 make_double4(double x, double y, double z, double w) { return double4{x, y, z, w}; }
CUEMU_VEC1(int, int) CUEMU_VEC2(int, int, 8) CUEMU_VEC3(int, int) CUEMU_VEC4(int, int, 16)
CUEMU_VEC1(unsigned int, uint) CUEMU_VEC2(unsigned int, uint, 8) CUEMU_VEC3(unsigned int, uint) CUEMU_VEC4(unsigned int, uint, 16)
CUEMU_VEC1(signed char, char) CUEMU_VEC2(signed char, char, 2) CUEMU_VEC3(signed char, char) CUEMU_VEC4(signed char, char, 4)
CUEMU_VEC1(unsigned char, uchar) CUEMU_VEC2(unsigned char, uchar, 2) CUEMU_VEC3(unsigned char, uchar) CUEMU_VEC4(unsigned char, uchar, 4)
CUEMU_VEC1(short, short) CUEMU_VEC2(short, short, 4) CUEMU_VEC3(short, short) CUEMU_VEC4(short, short, 8)
CUEMU_VEC1(unsigned short, ushort) CUEMU_VEC2(unsigned short, ushort, 4) CUEMU_VEC3(unsigned short, ushort) CUEMU_VEC4(unsigned short, ushort, 8)
CUEMU_VEC1(long long, longlong) CUEMU_VEC2(long long, longlong, 16)
CUEMU_VEC1(unsigned long long, ulonglong) CUEMU_VEC2(unsigned long long, ulonglong, 16)

struct dim3 {
    unsigned int x, y, z;
    constexpr dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1) : x(vx), y(vy), z(vz) {}
    constexpr dim3(uint3 v) : x(v.x), y(v.y), z(v.z) {}
    constexpr operator uint3() const { return uint3{x, y, z}; }
};

// ---------------------------------------------------------------------------
// Runtime state visible to kernels (swapped on every fiber switch)
// ---------------------------------------------------------------------------
inline uint3 threadIdx;
inline uint3 blockIdx;
inline dim3 blockDim;
inline dim3 gridDim;
constexpr int warpSize = 32;


// ----- x86-64 SysV context switch -------------------------------------------
#if !defined(__x86_64__)
#error "cuemu currently supports x86-64 only"
#endif
asm(R"(
    .text
    .weak cuemu_switch
    .type cuemu_switch,@function
cuemu_switch:
    pushq %rbp
    pushq %rbx
    pushq %r12
    pushq %r13
    pushq %r14
    pushq %r15
    subq $8, %rsp
    stmxcsr (%rsp)
    fnstcw 4(%rsp)
    movq %rsp, (%rdi)
    movq %rsi, %rsp
    ldmxcsr (%rsp)
    fldcw 4(%rsp)
    addq $8, %rsp
    popq %r15
    popq %r14
    popq %r13
    popq %r12
    popq %rbx
    popq %rbp
    ret
    .size cuemu_switch, .-cuemu_switch
    .weak cuemu_trampoline
    .type cuemu_trampoline,@function
cuemu_trampoline:
    andq $-16, %rsp
    call cuemuFiberMain@PLT
    ud2
    .size cuemu_trampoline, .-cuemu_trampoline
)");

namespace cuemu {

[[noreturn]] inline void fail(const char* msg) {
    std::fprintf(stderr, "cuemu error: %s\n", msg);
    std::fflush(stderr);
    std::abort();
}

// ----- fibers --------------------------------------------------------------
extern "C" void cuemu_switch(void** save_sp, void* load_sp);

enum class State { kRunnable, kWaitingBlock, kWaitingWarp, kDone };

// One cp.async copy (see __pipeline_memcpy_async below), deferred until waited for.
struct AsyncCopy {
    void* dst;
    const void* src;
    size_t size;
    size_t zfill;
    void perform() const {
        std::memcpy(dst, src, size - zfill);
        std::memset(static_cast<char*>(dst) + (size - zfill), 0, zfill);
    }
};

struct Fiber {
    void* sp = nullptr;
    char* stack = nullptr;
    uint3 tid{};
    int linear = 0;
    State state = State::kRunnable;
    std::vector<AsyncCopy> async_open;                  // issued, not yet committed
    std::vector<std::vector<AsyncCopy>> async_groups;  // committed groups, oldest first
};

struct WarpSlot {
    uint64_t value = 0;
    int arg = 0;
    int width = 32;
    int op = 0;
    unsigned mask = 0;
    uint64_t result = 0;
    bool arrived = false;
};

struct Block {
    std::vector<Fiber> fibers;
    int num_threads = 0;
    int live = 0;
    // __syncthreads
    int block_arrived = 0;
    int block_pred_sum = 0;
    int block_pred_result = 0;
    std::vector<int> block_pred_in;
    // warp operations
    std::vector<WarpSlot> slots;
    std::vector<int> warp_generation;
};

inline Block g_block;
inline int g_current = -1;
inline void* g_sched_sp = nullptr;
inline std::function<void()> g_body;
inline size_t g_stack_size = 256 * 1024;
inline std::vector<char*> g_stack_pool;
inline bool g_reverse_order = std::getenv("CUEMU_REVERSE") != nullptr;
inline std::vector<unsigned char> g_dyn_smem;

inline void yieldToScheduler() {
    Fiber& f = g_block.fibers[g_current];
    cuemu_switch(&f.sp, g_sched_sp);
}

extern "C" __attribute__((used, weak)) void cuemuFiberMain() {
    g_body();
    {
        // A thread that exits with copies in flight: they still land (as on hardware).
        Fiber& f = g_block.fibers[g_current];
        for (auto& group : f.async_groups)
            for (auto& c : group) c.perform();
        for (auto& c : f.async_open) c.perform();
        f.async_groups.clear();
        f.async_open.clear();
    }
    g_block.fibers[g_current].state = State::kDone;
    g_block.live--;
    yieldToScheduler();
    fail("resumed a finished fiber");
}

extern "C" void cuemu_trampoline();

inline void initFiber(Fiber& f) {
    // Stack layout expected by cuemu_switch: [mxcsr|fpucw] r15 r14 r13 r12 rbx rbp ret
    uintptr_t top = reinterpret_cast<uintptr_t>(f.stack + g_stack_size);
    top &= ~uintptr_t(15);
    auto* p = reinterpret_cast<uint64_t*>(top);
    *--p = 0;  // fake return address slot for the trampoline (keeps alignment)
    *--p = reinterpret_cast<uint64_t>(&cuemu_trampoline);
    for (int i = 0; i < 6; ++i) *--p = 0;  // rbp rbx r12 r13 r14 r15
    uint32_t mxcsr = 0x1F80;
    uint16_t fpucw = 0x037F;
    --p;
    std::memcpy(reinterpret_cast<char*>(p), &mxcsr, 4);
    std::memcpy(reinterpret_cast<char*>(p) + 4, &fpucw, 2);
    f.sp = p;
}

inline void checkWarpCompletion(int warp);

inline void deadlock() {
    std::fprintf(stderr, "cuemu error: deadlock in block (%u,%u,%u)\n", blockIdx.x, blockIdx.y, blockIdx.z);
    int shown = 0;
    for (auto& f : g_block.fibers) {
        if (f.state == State::kDone) continue;
        if (shown++ > 8) break;
        std::fprintf(stderr, "  thread (%u,%u,%u) waiting at %s\n", f.tid.x, f.tid.y, f.tid.z,
                     f.state == State::kWaitingBlock ? "__syncthreads" : "a warp-level primitive");
    }
    std::fprintf(stderr, "  hint: a barrier or *_sync call is not reached by every thread it names "
                         "(early return, divergent branch, or a full mask on a partial warp)\n");
    std::abort();
}

inline void runBlock(const dim3& bdim) {
    Block& b = g_block;
    const int n = static_cast<int>(bdim.x * bdim.y * bdim.z);
    b.num_threads = n;
    b.live = n;
    b.block_arrived = 0;
    b.fibers.assign(n, Fiber{});
    b.block_pred_in.assign(n, 0);
    const int num_warps = (n + 31) / 32;
    b.slots.assign(num_warps * 32, WarpSlot{});
    b.warp_generation.assign(num_warps, 0);
    while (static_cast<int>(g_stack_pool.size()) < n) {
        g_stack_pool.push_back(static_cast<char*>(std::malloc(g_stack_size)));
    }
    for (int i = 0; i < n; ++i) {
        Fiber& f = b.fibers[i];
        f.linear = i;
        f.tid = uint3{static_cast<unsigned>(i % bdim.x), static_cast<unsigned>((i / bdim.x) % bdim.y),
                      static_cast<unsigned>(i / (bdim.x * bdim.y))};
        f.stack = g_stack_pool[i];
        initFiber(f);
    }
    while (b.live > 0) {
        for (int k = 0; k < n; ++k) {
            const int i = g_reverse_order ? n - 1 - k : k;
            Fiber& f = b.fibers[i];
            if (f.state != State::kRunnable) continue;
            g_current = i;
            threadIdx = f.tid;
            cuemu_switch(&g_sched_sp, f.sp);
        }
        if (b.live == 0) break;
        // Barrier release: every live thread is waiting at __syncthreads.
        if (b.block_arrived > 0 && b.block_arrived == b.live) {
            b.block_arrived = 0;
            for (auto& f : b.fibers)
                if (f.state == State::kWaitingBlock) f.state = State::kRunnable;
            continue;
        }
        bool any_runnable = false;
        for (auto& f : b.fibers) any_runnable |= f.state == State::kRunnable;
        if (any_runnable) continue;
        // Nothing can run: report warp primitives waiting on exited lanes, else deadlock.
        for (int w = 0; w < num_warps; ++w) checkWarpCompletion(w);
        for (auto& f : b.fibers) any_runnable |= f.state == State::kRunnable;
        if (!any_runnable) deadlock();
    }
    g_current = -1;
}

// ----- __syncthreads -------------------------------------------------------
inline int blockBarrier(int pred, int mode) {
    Block& b = g_block;
    Fiber& f = b.fibers[g_current];
    b.block_pred_in[f.linear] = pred;
    b.block_arrived++;
    f.state = State::kWaitingBlock;
    yieldToScheduler();
    threadIdx = f.tid;
    if (mode == 0) return 0;
    int count = 0;
    for (int i = 0; i < b.num_threads; ++i) count += b.block_pred_in[i] != 0;
    // All threads read the same values: predicates are only rewritten at the
    // next barrier, which cannot complete before every thread got here.
    if (mode == 1) return count;
    if (mode == 2) return count == b.live ? 1 : 0;  // and
    return count > 0 ? 1 : 0;                        // or
}

// ----- warp operations -----------------------------------------------------
enum WarpOp { kSync = 0, kShfl, kShflUp, kShflDown, kShflXor, kBallot, kAny, kAll, kMatchAny, kReduceAdd,
              kReduceMin, kReduceMax, kReduceAnd, kReduceOr, kReduceXor };

inline void computeWarp(int warp) {
    Block& b = g_block;
    WarpSlot* s = &b.slots[warp * 32];
    for (int lane = 0; lane < 32; ++lane) {
        WarpSlot& me = s[lane];
        if (!me.arrived) continue;
        const int width = me.width;
        const int seg = lane & ~(width - 1);
        int src = lane;
        switch (me.op) {
            case kShfl: src = seg + (me.arg & (width - 1)); break;
            case kShflUp: src = lane - me.arg; if (src < seg) src = lane; break;
            case kShflDown: src = lane + me.arg; if (src >= seg + width) src = lane; break;
            case kShflXor: src = lane ^ me.arg; if (src >= seg + width || src < seg) src = lane; break;
            default: break;
        }
        switch (me.op) {
            case kShfl: case kShflUp: case kShflDown: case kShflXor:
                me.result = (src >= 0 && src < 32 && s[src].arrived) ? s[src].value : me.value;
                break;
            case kBallot: case kAny: case kAll: {
                unsigned bits = 0;
                for (int l = 0; l < 32; ++l)
                    if (s[l].arrived && ((me.mask >> l) & 1u) && s[l].value) bits |= 1u << l;
                unsigned participants = 0;
                for (int l = 0; l < 32; ++l)
                    if (s[l].arrived && ((me.mask >> l) & 1u)) participants |= 1u << l;
                if (me.op == kBallot) me.result = bits;
                else if (me.op == kAny) me.result = bits != 0;
                else me.result = bits == participants;
                break;
            }
            case kMatchAny: {
                unsigned bits = 0;
                for (int l = 0; l < 32; ++l)
                    if (s[l].arrived && ((me.mask >> l) & 1u) && s[l].value == me.value) bits |= 1u << l;
                me.result = bits;
                break;
            }
            case kReduceAdd: case kReduceMin: case kReduceMax: case kReduceAnd: case kReduceOr: case kReduceXor: {
                bool first = true;
                uint64_t acc = 0;
                for (int l = 0; l < 32; ++l) {
                    if (!s[l].arrived || !((me.mask >> l) & 1u)) continue;
                    const uint32_t v = static_cast<uint32_t>(s[l].value);
                    const bool is_signed = me.arg != 0;
                    if (first) { acc = v; first = false; continue; }
                    const uint32_t a = static_cast<uint32_t>(acc);
                    switch (me.op) {
                        case kReduceAdd: acc = a + v; break;
                        case kReduceMin: acc = is_signed ? static_cast<uint32_t>(std::min<int32_t>(a, v)) : std::min(a, v); break;
                        case kReduceMax: acc = is_signed ? static_cast<uint32_t>(std::max<int32_t>(a, v)) : std::max(a, v); break;
                        case kReduceAnd: acc = a & v; break;
                        case kReduceOr: acc = a | v; break;
                        default: acc = a ^ v; break;
                    }
                }
                me.result = acc;
                break;
            }
            default: me.result = 0; break;
        }
    }
    for (int lane = 0; lane < 32; ++lane) {
        if (!s[lane].arrived) continue;
        s[lane].arrived = false;
        Fiber& f = b.fibers[warp * 32 + lane];
        f.state = State::kRunnable;
    }
}

inline void checkWarpCompletion(int warp) {
    Block& b = g_block;
    WarpSlot* s = &b.slots[warp * 32];
    unsigned required = 0;
    bool any = false;
    for (int lane = 0; lane < 32; ++lane)
        if (s[lane].arrived) { required |= s[lane].mask; any = true; }
    if (!any) return;
    for (int lane = 0; lane < 32; ++lane) {
        if (!((required >> lane) & 1u)) continue;
        const int t = warp * 32 + lane;
        if (t >= b.num_threads || b.fibers[t].state == State::kDone) {
            std::fprintf(stderr, "cuemu error: warp-level primitive in block (%u,%u,%u) names lane %d of warp %d "
                                 "in its mask, but that thread %s\n", blockIdx.x, blockIdx.y, blockIdx.z, lane,
                         warp, t >= b.num_threads ? "does not exist (partial warp)" : "has already exited");
            std::abort();
        }
        if (!s[lane].arrived) return;  // still waiting for a participant
    }
    computeWarp(warp);
}

inline uint64_t warpOp(int op, unsigned mask, uint64_t value, int arg, int width) {
    Block& b = g_block;
    Fiber& f = b.fibers[g_current];
    const int warp = f.linear / 32;
    const int lane = f.linear % 32;
    if (!((mask >> lane) & 1u)) fail("calling lane is not part of the mask of a *_sync primitive");
    if (width <= 0 || width > 32 || (width & (width - 1)) != 0) fail("shuffle width must be a power of two <= 32");
    WarpSlot& me = b.slots[warp * 32 + lane];
    me.value = value;
    me.arg = arg;
    me.width = width;
    me.op = op;
    me.mask = mask;
    me.arrived = true;
    f.state = State::kWaitingWarp;
    checkWarpCompletion(warp);
    if (f.state == State::kWaitingWarp) yieldToScheduler();
    threadIdx = f.tid;
    return b.slots[warp * 32 + lane].result;
}

template <class T>
inline uint64_t toBits(T v) {
    static_assert(sizeof(T) <= 8, "shuffle of types larger than 8 bytes is not supported");
    uint64_t bits = 0;
    std::memcpy(&bits, &v, sizeof(T));
    return bits;
}
template <class T>
inline T fromBits(uint64_t bits) {
    T v;
    std::memcpy(&v, &bits, sizeof(T));
    return v;
}

// ----- launch --------------------------------------------------------------
struct Launcher {
    dim3 grid, block;
    size_t smem;
    Launcher(dim3 g, dim3 b, size_t s = 0, void* = nullptr) : grid(g), block(b), smem(s) {}

    template <class... Params, class... Args>
    void operator()(void (*kernel)(Params...), Args&&... args) {
        static_assert(sizeof...(Params) == sizeof...(Args), "kernel argument count mismatch");
        const size_t threads = static_cast<size_t>(block.x) * block.y * block.z;
        if (threads == 0 || threads > 1024) fail("invalid block size (must be 1..1024 threads)");
        if (block.z > 64) fail("blockDim.z must be <= 64");
        if (grid.x == 0 || grid.y == 0 || grid.z == 0) fail("invalid grid size (a dimension is 0)");
        if (grid.y > 65535 || grid.z > 65535) fail("gridDim.y and gridDim.z must be <= 65535");
        std::tuple<std::decay_t<Params>...> packed(static_cast<Params>(args)...);
        g_dyn_smem.assign(smem + 16, 0);
        blockDim = block;
        gridDim = grid;
        g_body = [&]() { std::apply(kernel, packed); };
        for (unsigned z = 0; z < grid.z; ++z)
            for (unsigned y = 0; y < grid.y; ++y)
                for (unsigned x = 0; x < grid.x; ++x) {
                    blockIdx = uint3{x, y, z};
                    runBlock(block);
                }
    }
};

template <class T>
inline T* dynamicSmem() { return reinterpret_cast<T*>(g_dyn_smem.data()); }

}  // namespace cuemu

// ---------------------------------------------------------------------------
// Synchronization and warp intrinsics
// ---------------------------------------------------------------------------
inline void __syncthreads() { cuemu::blockBarrier(0, 0); }
inline int __syncthreads_count(int pred) { return cuemu::blockBarrier(pred, 1); }
inline int __syncthreads_and(int pred) { return cuemu::blockBarrier(pred, 2); }
inline int __syncthreads_or(int pred) { return cuemu::blockBarrier(pred, 3); }
inline void __syncwarp(unsigned mask = 0xffffffffu) { cuemu::warpOp(cuemu::kSync, mask, 0, 0, 32); }
inline void __threadfence() {}
inline void __threadfence_block() {}
inline void __threadfence_system() {}
inline void __nanosleep(unsigned) {}
[[noreturn]] inline void __trap() { cuemu::fail("__trap() called"); }

inline unsigned __activemask() {
    const int warp = cuemu::g_block.fibers[cuemu::g_current].linear / 32;
    unsigned m = 0;
    for (int l = 0; l < 32; ++l) {
        const int t = warp * 32 + l;
        if (t < cuemu::g_block.num_threads && cuemu::g_block.fibers[t].state != cuemu::State::kDone) m |= 1u << l;
    }
    return m;
}

template <class T>
inline T __shfl_sync(unsigned mask, T var, int src_lane, int width = 32) {
    return cuemu::fromBits<T>(cuemu::warpOp(cuemu::kShfl, mask, cuemu::toBits(var), src_lane, width));
}
template <class T>
inline T __shfl_up_sync(unsigned mask, T var, unsigned delta, int width = 32) {
    return cuemu::fromBits<T>(cuemu::warpOp(cuemu::kShflUp, mask, cuemu::toBits(var), static_cast<int>(delta), width));
}
template <class T>
inline T __shfl_down_sync(unsigned mask, T var, unsigned delta, int width = 32) {
    return cuemu::fromBits<T>(cuemu::warpOp(cuemu::kShflDown, mask, cuemu::toBits(var), static_cast<int>(delta), width));
}
template <class T>
inline T __shfl_xor_sync(unsigned mask, T var, int lane_mask, int width = 32) {
    return cuemu::fromBits<T>(cuemu::warpOp(cuemu::kShflXor, mask, cuemu::toBits(var), lane_mask, width));
}
inline unsigned __ballot_sync(unsigned mask, int pred) {
    return static_cast<unsigned>(cuemu::warpOp(cuemu::kBallot, mask, pred != 0, 0, 32));
}
inline int __any_sync(unsigned mask, int pred) { return static_cast<int>(cuemu::warpOp(cuemu::kAny, mask, pred != 0, 0, 32)); }
inline int __all_sync(unsigned mask, int pred) { return static_cast<int>(cuemu::warpOp(cuemu::kAll, mask, pred != 0, 0, 32)); }
template <class T>
inline unsigned __match_any_sync(unsigned mask, T value) {
    return static_cast<unsigned>(cuemu::warpOp(cuemu::kMatchAny, mask, cuemu::toBits(value), 0, 32));
}
inline unsigned __reduce_add_sync(unsigned mask, unsigned v) { return static_cast<unsigned>(cuemu::warpOp(cuemu::kReduceAdd, mask, v, 0, 32)); }
inline int __reduce_add_sync(unsigned mask, int v) { return static_cast<int>(cuemu::warpOp(cuemu::kReduceAdd, mask, static_cast<uint32_t>(v), 1, 32)); }
inline unsigned __reduce_min_sync(unsigned mask, unsigned v) { return static_cast<unsigned>(cuemu::warpOp(cuemu::kReduceMin, mask, v, 0, 32)); }
inline int __reduce_min_sync(unsigned mask, int v) { return static_cast<int>(cuemu::warpOp(cuemu::kReduceMin, mask, static_cast<uint32_t>(v), 1, 32)); }
inline unsigned __reduce_max_sync(unsigned mask, unsigned v) { return static_cast<unsigned>(cuemu::warpOp(cuemu::kReduceMax, mask, v, 0, 32)); }
inline int __reduce_max_sync(unsigned mask, int v) { return static_cast<int>(cuemu::warpOp(cuemu::kReduceMax, mask, static_cast<uint32_t>(v), 1, 32)); }
inline unsigned __reduce_and_sync(unsigned mask, unsigned v) { return static_cast<unsigned>(cuemu::warpOp(cuemu::kReduceAnd, mask, v, 0, 32)); }
inline unsigned __reduce_or_sync(unsigned mask, unsigned v) { return static_cast<unsigned>(cuemu::warpOp(cuemu::kReduceOr, mask, v, 0, 32)); }
inline unsigned __reduce_xor_sync(unsigned mask, unsigned v) { return static_cast<unsigned>(cuemu::warpOp(cuemu::kReduceXor, mask, v, 0, 32)); }

// ---------------------------------------------------------------------------
// Atomics (fibers run on one OS thread, so plain read-modify-write is atomic)
// ---------------------------------------------------------------------------
#define CUEMU_ATOMIC_RMW(NAME, EXPR)                                            \
    template <class T, class U>                                                 \
    inline T NAME(T* address, U val_in) {                                       \
        const T val = static_cast<T>(val_in);                                   \
        T old = *address;                                                       \
        *address = (EXPR);                                                      \
        return old;                                                             \
    }
CUEMU_ATOMIC_RMW(atomicAdd, old + val)
CUEMU_ATOMIC_RMW(atomicSub, old - val)
CUEMU_ATOMIC_RMW(atomicExch, val)
CUEMU_ATOMIC_RMW(atomicMin, val < old ? val : old)
CUEMU_ATOMIC_RMW(atomicMax, val > old ? val : old)
CUEMU_ATOMIC_RMW(atomicAnd, old & val)
CUEMU_ATOMIC_RMW(atomicOr, old | val)
CUEMU_ATOMIC_RMW(atomicXor, old ^ val)
#define atomicAdd_block atomicAdd
#define atomicAdd_system atomicAdd
#define atomicMax_block atomicMax
#define atomicMin_block atomicMin
#define atomicExch_block atomicExch
template <class T, class U, class V>
inline T atomicCAS(T* address, U compare, V val) {
    T old = *address;
    if (old == static_cast<T>(compare)) *address = static_cast<T>(val);
    return old;
}
inline unsigned atomicInc(unsigned* address, unsigned val) {
    unsigned old = *address;
    *address = (old >= val) ? 0 : old + 1;
    return old;
}
inline unsigned atomicDec(unsigned* address, unsigned val) {
    unsigned old = *address;
    *address = (old == 0 || old > val) ? val : old - 1;
    return old;
}

// ---------------------------------------------------------------------------
// Math and bit intrinsics
// ---------------------------------------------------------------------------
template <class A, class B>
inline auto min(A a, B b) -> std::common_type_t<A, B> {
    using C = std::common_type_t<A, B>;
    return static_cast<C>(b) < static_cast<C>(a) ? static_cast<C>(b) : static_cast<C>(a);
}
template <class A, class B>
inline auto max(A a, B b) -> std::common_type_t<A, B> {
    using C = std::common_type_t<A, B>;
    return static_cast<C>(a) < static_cast<C>(b) ? static_cast<C>(b) : static_cast<C>(a);
}
inline float rsqrtf(float x) { return 1.0f / std::sqrt(x); }
inline double rsqrt(double x) { return 1.0 / std::sqrt(x); }
inline float rcbrtf(float x) { return 1.0f / std::cbrt(x); }
inline float __expf(float x) { return std::exp(x); }
inline float __exp10f(float x) { return std::pow(10.0f, x); }
inline float __logf(float x) { return std::log(x); }
inline float __log2f(float x) { return std::log2(x); }
inline float __log10f(float x) { return std::log10(x); }
inline float __powf(float x, float y) { return std::pow(x, y); }
inline float __sinf(float x) { return std::sin(x); }
inline float __cosf(float x) { return std::cos(x); }
inline float __tanf(float x) { return std::tan(x); }
inline void __sincosf(float x, float* s, float* c) { *s = std::sin(x); *c = std::cos(x); }
inline void sincosf(float x, float* s, float* c) { *s = std::sin(x); *c = std::cos(x); }
inline void sincos(double x, double* s, double* c) { *s = std::sin(x); *c = std::cos(x); }
inline void sincospif(float x, float* s, float* c) {
    *s = static_cast<float>(std::sin(M_PI * x));
    *c = static_cast<float>(std::cos(M_PI * x));
}
inline void sincospi(double x, double* s, double* c) { *s = std::sin(M_PI * x); *c = std::cos(M_PI * x); }
inline float sinpif(float x) { return static_cast<float>(std::sin(M_PI * x)); }
inline float cospif(float x) { return static_cast<float>(std::cos(M_PI * x)); }
inline float exp10f(float x) { return std::pow(10.0f, x); }
inline float __fdividef(float a, float b) { return a / b; }
inline float __frcp_rn(float a) { return 1.0f / a; }
inline float __fsqrt_rn(float a) { return std::sqrt(a); }
inline float __frsqrt_rn(float a) { return 1.0f / std::sqrt(a); }
inline float __fadd_rn(float a, float b) { return a + b; }
inline float __fsub_rn(float a, float b) { return a - b; }
inline float __fmul_rn(float a, float b) { return a * b; }
inline float __fdiv_rn(float a, float b) { return a / b; }
inline float __fmaf_rn(float a, float b, float c) { return std::fma(a, b, c); }
inline float __fadd_rz(float a, float b) { return a + b; }
inline float __fmul_rz(float a, float b) { return a * b; }
inline double __dadd_rn(double a, double b) { return a + b; }
inline double __dmul_rn(double a, double b) { return a * b; }
inline double __fma_rn(double a, double b, double c) { return std::fma(a, b, c); }
inline float __saturatef(float x) { return x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x); }
inline float normcdff(float x) { return 0.5f * std::erfc(-x * static_cast<float>(M_SQRT1_2)); }
inline double normcdf(double x) { return 0.5 * std::erfc(-x * M_SQRT1_2); }
inline float erfcinvf(float) { cuemu::fail("erfcinvf is not implemented in cuemu"); }
inline float __int_as_float(int v) { float f; std::memcpy(&f, &v, 4); return f; }
inline int __float_as_int(float v) { int i; std::memcpy(&i, &v, 4); return i; }
inline float __uint_as_float(unsigned v) { float f; std::memcpy(&f, &v, 4); return f; }
inline unsigned __float_as_uint(float v) { unsigned i; std::memcpy(&i, &v, 4); return i; }
inline double __longlong_as_double(long long v) { double d; std::memcpy(&d, &v, 8); return d; }
inline long long __double_as_longlong(double v) { long long i; std::memcpy(&i, &v, 8); return i; }
inline int __float2int_rn(float x) { return static_cast<int>(std::nearbyint(x)); }
inline int __float2int_rz(float x) { return static_cast<int>(x); }
inline int __float2int_rd(float x) { return static_cast<int>(std::floor(x)); }
inline int __float2int_ru(float x) { return static_cast<int>(std::ceil(x)); }
inline unsigned __float2uint_rn(float x) { return static_cast<unsigned>(std::nearbyint(x)); }
inline unsigned __float2uint_rz(float x) { return static_cast<unsigned>(x); }
inline long long __float2ll_rn(float x) { return static_cast<long long>(std::nearbyint(x)); }
inline long long __float2ll_rz(float x) { return static_cast<long long>(x); }
inline float __int2float_rn(int x) { return static_cast<float>(x); }
inline float __uint2float_rn(unsigned x) { return static_cast<float>(x); }
inline float __ll2float_rn(long long x) { return static_cast<float>(x); }
inline float __ull2float_rn(unsigned long long x) { return static_cast<float>(x); }
inline int __double2int_rn(double x) { return static_cast<int>(std::nearbyint(x)); }
inline float __double2float_rn(double x) { return static_cast<float>(x); }
inline int __popc(unsigned x) { return __builtin_popcount(x); }
inline int __popcll(unsigned long long x) { return __builtin_popcountll(x); }
inline int __clz(int x) { return x == 0 ? 32 : __builtin_clz(static_cast<unsigned>(x)); }
inline int __clzll(long long x) { return x == 0 ? 64 : __builtin_clzll(static_cast<unsigned long long>(x)); }
inline int __ffs(int x) { return __builtin_ffs(x); }
inline int __ffsll(long long x) { return __builtin_ffsll(x); }
inline unsigned __brev(unsigned x) {
    x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1);
    x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2);
    x = ((x >> 4) & 0x0F0F0F0Fu) | ((x & 0x0F0F0F0Fu) << 4);
    x = ((x >> 8) & 0x00FF00FFu) | ((x & 0x00FF00FFu) << 8);
    return (x >> 16) | (x << 16);
}
inline unsigned long long __brevll(unsigned long long x) {
    return (static_cast<unsigned long long>(__brev(static_cast<unsigned>(x))) << 32) | __brev(static_cast<unsigned>(x >> 32));
}
inline int __mul24(int a, int b) { return a * b; }
inline unsigned __umul24(unsigned a, unsigned b) { return a * b; }
inline int __mulhi(int a, int b) { return static_cast<int>((static_cast<long long>(a) * b) >> 32); }
inline unsigned __umulhi(unsigned a, unsigned b) { return static_cast<unsigned>((static_cast<unsigned long long>(a) * b) >> 32); }
inline unsigned long long __umul64hi(unsigned long long a, unsigned long long b) {
    return static_cast<unsigned long long>((static_cast<unsigned __int128>(a) * b) >> 64);
}
inline long long __mul64hi(long long a, long long b) { return static_cast<long long>((static_cast<__int128>(a) * b) >> 64); }
inline unsigned __funnelshift_l(unsigned lo, unsigned hi, unsigned shift) {
    shift &= 31;
    return shift ? (hi << shift) | (lo >> (32 - shift)) : hi;
}
inline unsigned __funnelshift_r(unsigned lo, unsigned hi, unsigned shift) {
    shift &= 31;
    return shift ? (lo >> shift) | (hi << (32 - shift)) : lo;
}
inline unsigned __byte_perm(unsigned x, unsigned y, unsigned s) {
    const uint64_t v = (static_cast<uint64_t>(y) << 32) | x;
    unsigned r = 0;
    for (int i = 0; i < 4; ++i) r |= ((v >> (8 * ((s >> (4 * i)) & 7))) & 0xFF) << (8 * i);
    return r;
}
inline int __dp4a(int a, int b, int c) {
    for (int i = 0; i < 4; ++i) c += static_cast<int8_t>(a >> (8 * i)) * static_cast<int8_t>(b >> (8 * i));
    return c;
}
inline long long clock64() { return 0; }
inline int clock() { return 0; }
template <class T> inline T __ldg(const T* p) { return *p; }
template <class T> inline T __ldcg(const T* p) { return *p; }
template <class T> inline T __ldca(const T* p) { return *p; }
template <class T> inline T __ldcs(const T* p) { return *p; }
template <class T> inline T __ldlu(const T* p) { return *p; }
template <class T> inline T __ldcv(const T* p) { return *p; }
template <class T> inline void __stcg(T* p, T v) { *p = v; }
template <class T> inline void __stcs(T* p, T v) { *p = v; }
template <class T> inline void __stwb(T* p, T v) { *p = v; }
template <class T> inline void __stwt(T* p, T v) { *p = v; }
#define __builtin_assume_aligned(p, ...) (p)

// ---------------------------------------------------------------------------
// half / bfloat16 (storage + float math, rounded to nearest even)
// ---------------------------------------------------------------------------
struct __half;
inline float __half2float(__half h);
inline __half __float2half(float f);
struct __half_raw { unsigned short x; };
struct __half {
    unsigned short __x;
    __half() = default;
    __half(__half_raw r) : __x(r.x) {}
    __half(float f);
    __half(double f) : __half(static_cast<float>(f)) {}
    __half(int v) : __half(static_cast<float>(v)) {}
    __half(unsigned v) : __half(static_cast<float>(v)) {}
    __half(long long v) : __half(static_cast<float>(v)) {}
    __half(unsigned long long v) : __half(static_cast<float>(v)) {}
    __half(short v) : __half(static_cast<float>(v)) {}
    __half(unsigned short v) : __half(static_cast<float>(v)) {}
    operator float() const;
    operator __half_raw() const { return __half_raw{__x}; }
    __half& operator+=(__half o);
    __half& operator-=(__half o);
    __half& operator*=(__half o);
    __half& operator/=(__half o);
};
using half = __half;
inline float __half2float(__half h) {
    _Float16 v;
    std::memcpy(&v, &h.__x, 2);
    return static_cast<float>(v);
}
inline __half __float2half(float f) {
    _Float16 v = static_cast<_Float16>(f);
    __half h;
    std::memcpy(&h.__x, &v, 2);
    return h;
}
inline __half::__half(float f) : __x(__float2half(f).__x) {}
inline __half::operator float() const { return __half2float(*this); }
inline __half operator+(__half a, __half b) { return __float2half(__half2float(a) + __half2float(b)); }
inline __half operator-(__half a, __half b) { return __float2half(__half2float(a) - __half2float(b)); }
inline __half operator*(__half a, __half b) { return __float2half(__half2float(a) * __half2float(b)); }
inline __half operator/(__half a, __half b) { return __float2half(__half2float(a) / __half2float(b)); }
inline __half operator-(__half a) { __half r = a; r.__x ^= 0x8000; return r; }
inline __half& __half::operator+=(__half o) { *this = *this + o; return *this; }
inline __half& __half::operator-=(__half o) { *this = *this - o; return *this; }
inline __half& __half::operator*=(__half o) { *this = *this * o; return *this; }
inline __half& __half::operator/=(__half o) { *this = *this / o; return *this; }
inline bool operator<(__half a, __half b) { return __half2float(a) < __half2float(b); }
inline bool operator>(__half a, __half b) { return __half2float(a) > __half2float(b); }
inline bool operator<=(__half a, __half b) { return __half2float(a) <= __half2float(b); }
inline bool operator>=(__half a, __half b) { return __half2float(a) >= __half2float(b); }
inline bool operator==(__half a, __half b) { return __half2float(a) == __half2float(b); }
inline bool operator!=(__half a, __half b) { return __half2float(a) != __half2float(b); }
inline __half __float2half_rn(float f) { return __float2half(f); }
inline __half __double2half(double d) { return __float2half(static_cast<float>(d)); }
inline __half __int2half_rn(int v) { return __float2half(static_cast<float>(v)); }
inline int __half2int_rn(__half h) { return static_cast<int>(std::nearbyint(__half2float(h))); }
inline int __half2int_rz(__half h) { return static_cast<int>(__half2float(h)); }
inline unsigned short __half_as_ushort(__half h) { return h.__x; }
inline short __half_as_short(__half h) { return static_cast<short>(h.__x); }
inline __half __ushort_as_half(unsigned short v) { __half h; h.__x = v; return h; }
inline __half __short_as_half(short v) { __half h; h.__x = static_cast<unsigned short>(v); return h; }
inline __half __hadd(__half a, __half b) { return a + b; }
inline __half __hsub(__half a, __half b) { return a - b; }
inline __half __hmul(__half a, __half b) { return a * b; }
inline __half __hdiv(__half a, __half b) { return a / b; }
inline __half __hneg(__half a) { return -a; }
inline __half __habs(__half a) { a.__x &= 0x7FFF; return a; }
inline __half __hfma(__half a, __half b, __half c) {
    return __float2half(std::fma(__half2float(a), __half2float(b), __half2float(c)));
}
inline __half __hmax(__half a, __half b) { return a < b ? b : a; }
inline __half __hmin(__half a, __half b) { return b < a ? b : a; }
inline bool __heq(__half a, __half b) { return a == b; }
inline bool __hlt(__half a, __half b) { return a < b; }
inline bool __hgt(__half a, __half b) { return a > b; }
inline bool __hle(__half a, __half b) { return a <= b; }
inline bool __hge(__half a, __half b) { return a >= b; }
inline bool __hisnan(__half a) { return std::isnan(__half2float(a)); }
inline __half hexp(__half a) { return __float2half(std::exp(__half2float(a))); }
inline __half hlog(__half a) { return __float2half(std::log(__half2float(a))); }
inline __half hsqrt(__half a) { return __float2half(std::sqrt(__half2float(a))); }
inline __half hrsqrt(__half a) { return __float2half(1.0f / std::sqrt(__half2float(a))); }
inline __half htanh(__half a) { return __float2half(std::tanh(__half2float(a))); }

struct alignas(4) __half2 {
    __half x, y;
};
using half2 = __half2;
inline __half2 __halves2half2(__half a, __half b) { return __half2{a, b}; }
inline __half2 make_half2(__half a, __half b) { return __half2{a, b}; }
inline __half2 __float2half2_rn(float f) { return __half2{__float2half(f), __float2half(f)}; }
inline __half2 __floats2half2_rn(float a, float b) { return __half2{__float2half(a), __float2half(b)}; }
inline __half2 __half2half2(__half a) { return __half2{a, a}; }
inline float2 __half22float2(__half2 h) { return float2{__half2float(h.x), __half2float(h.y)}; }
inline __half2 __float22half2_rn(float2 f) { return __floats2half2_rn(f.x, f.y); }
inline float __low2float(__half2 h) { return __half2float(h.x); }
inline float __high2float(__half2 h) { return __half2float(h.y); }
inline __half __low2half(__half2 h) { return h.x; }
inline __half __high2half(__half2 h) { return h.y; }
inline __half2 __hadd2(__half2 a, __half2 b) { return __half2{a.x + b.x, a.y + b.y}; }
inline __half2 __hsub2(__half2 a, __half2 b) { return __half2{a.x - b.x, a.y - b.y}; }
inline __half2 __hmul2(__half2 a, __half2 b) { return __half2{a.x * b.x, a.y * b.y}; }
inline __half2 __hfma2(__half2 a, __half2 b, __half2 c) { return __half2{__hfma(a.x, b.x, c.x), __hfma(a.y, b.y, c.y)}; }
inline __half2 operator+(__half2 a, __half2 b) { return __hadd2(a, b); }
inline __half2 operator*(__half2 a, __half2 b) { return __hmul2(a, b); }

struct __nv_bfloat16 {
    unsigned short __x;
    __nv_bfloat16() = default;
    __nv_bfloat16(float f) {
        unsigned u;
        std::memcpy(&u, &f, 4);
        if ((u & 0x7FFFFFFFu) > 0x7F800000u) { __x = static_cast<unsigned short>((u >> 16) | 0x40); return; }
        u += 0x7FFFu + ((u >> 16) & 1u);
        __x = static_cast<unsigned short>(u >> 16);
    }
    operator float() const {
        unsigned u = static_cast<unsigned>(__x) << 16;
        float f;
        std::memcpy(&f, &u, 4);
        return f;
    }
};
using nv_bfloat16 = __nv_bfloat16;
struct alignas(4) __nv_bfloat162 { __nv_bfloat16 x, y; };
using nv_bfloat162 = __nv_bfloat162;
inline float __bfloat162float(__nv_bfloat16 b) { return static_cast<float>(b); }
inline __nv_bfloat16 __float2bfloat16(float f) { return __nv_bfloat16(f); }
inline __nv_bfloat16 __float2bfloat16_rn(float f) { return __nv_bfloat16(f); }

// ---------------------------------------------------------------------------
// Runtime API subset
// ---------------------------------------------------------------------------
typedef int cudaError_t;
typedef void* cudaStream_t;
typedef void* cudaEvent_t;
enum cudaMemcpyKind { cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice, cudaMemcpyDefault };
enum cudaFuncAttribute { cudaFuncAttributeMaxDynamicSharedMemorySize, cudaFuncAttributePreferredSharedMemoryCarveout };
constexpr cudaError_t cudaSuccess = 0;
struct cudaDeviceProp {
    char name[256] = "cuemu";
    int multiProcessorCount = 108;
    size_t sharedMemPerBlock = 48 * 1024;
    size_t sharedMemPerBlockOptin = 163 * 1024;
    size_t sharedMemPerMultiprocessor = 164 * 1024;
    int maxThreadsPerBlock = 1024;
    int maxThreadsPerMultiProcessor = 2048;
    int warpSize = 32;
    int regsPerBlock = 65536;
    int major = 8;
    int minor = 0;
    size_t totalGlobalMem = size_t(40) << 30;
    size_t l2CacheSize = 40 << 20;
    int maxGridSize[3] = {2147483647, 65535, 65535};
    int maxThreadsDim[3] = {1024, 1024, 64};
};
enum cudaDeviceAttr { cudaDevAttrMultiProcessorCount, cudaDevAttrMaxSharedMemoryPerBlockOptin, cudaDevAttrMaxThreadsPerBlock };

inline cudaError_t cudaMalloc(void** p, size_t n) {
    *p = std::aligned_alloc(256, ((n + 255) / 256) * 256 + 256);
    return cudaSuccess;
}
template <class T> inline cudaError_t cudaMalloc(T** p, size_t n) { return cudaMalloc(reinterpret_cast<void**>(p), n); }
template <class T> inline cudaError_t cudaMallocAsync(T** p, size_t n, cudaStream_t = nullptr) { return cudaMalloc(reinterpret_cast<void**>(p), n); }
template <class T> inline cudaError_t cudaMallocManaged(T** p, size_t n, unsigned = 0) { return cudaMalloc(reinterpret_cast<void**>(p), n); }
inline cudaError_t cudaFree(void* p) { std::free(p); return cudaSuccess; }
inline cudaError_t cudaFreeAsync(void* p, cudaStream_t = nullptr) { std::free(p); return cudaSuccess; }
inline cudaError_t cudaMemcpy(void* d, const void* s, size_t n, cudaMemcpyKind) { std::memmove(d, s, n); return cudaSuccess; }
inline cudaError_t cudaMemcpyAsync(void* d, const void* s, size_t n, cudaMemcpyKind, cudaStream_t = nullptr) { std::memmove(d, s, n); return cudaSuccess; }
inline cudaError_t cudaMemset(void* d, int v, size_t n) { std::memset(d, v, n); return cudaSuccess; }
inline cudaError_t cudaMemsetAsync(void* d, int v, size_t n, cudaStream_t = nullptr) { std::memset(d, v, n); return cudaSuccess; }
template <class T>
inline cudaError_t cudaMemcpyToSymbol(T& symbol, const void* src, size_t n, size_t offset = 0, cudaMemcpyKind = cudaMemcpyHostToDevice) {
    std::memcpy(reinterpret_cast<char*>(&symbol) + offset, src, n);
    return cudaSuccess;
}
template <class T>
inline cudaError_t cudaMemcpyFromSymbol(void* dst, const T& symbol, size_t n, size_t offset = 0, cudaMemcpyKind = cudaMemcpyDeviceToHost) {
    std::memcpy(dst, reinterpret_cast<const char*>(&symbol) + offset, n);
    return cudaSuccess;
}
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaStreamCreate(cudaStream_t* s) { *s = nullptr; return cudaSuccess; }
inline cudaError_t cudaStreamDestroy(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaEventCreate(cudaEvent_t* e) { *e = nullptr; return cudaSuccess; }
inline cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t = nullptr) { return cudaSuccess; }
inline cudaError_t cudaEventSynchronize(cudaEvent_t) { return cudaSuccess; }
inline cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t, cudaEvent_t) { *ms = 0.0f; return cudaSuccess; }
inline cudaError_t cudaEventDestroy(cudaEvent_t) { return cudaSuccess; }
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline cudaError_t cudaPeekAtLastError() { return cudaSuccess; }
inline const char* cudaGetErrorString(cudaError_t) { return "no error"; }
inline cudaError_t cudaGetDevice(int* d) { *d = 0; return cudaSuccess; }
inline cudaError_t cudaSetDevice(int) { return cudaSuccess; }
inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* p, int) { *p = cudaDeviceProp{}; return cudaSuccess; }
inline cudaError_t cudaDeviceGetAttribute(int* v, cudaDeviceAttr a, int) {
    *v = a == cudaDevAttrMultiProcessorCount ? 108 : (a == cudaDevAttrMaxThreadsPerBlock ? 1024 : 163 * 1024);
    return cudaSuccess;
}
template <class F> inline cudaError_t cudaFuncSetAttribute(F, cudaFuncAttribute, int) { return cudaSuccess; }
template <class F>
inline cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* n, F, int block_size, size_t) {
    *n = std::max(1, 2048 / std::max(1, block_size));
    return cudaSuccess;
}

// ---------------------------------------------------------------------------
// nvcuda::wmma (tensor core API) — functional emulation.
// Every thread holds the whole tile in its fragment and computes the full
// product, so results are correct for code that treats fragments as opaque
// (load / mma / store / elementwise ops over x[0..num_elements)). Code that
// depends on the hardware's element-to-lane mapping is not supported.
// ---------------------------------------------------------------------------
namespace nvcuda {
namespace wmma {
struct matrix_a {};
struct matrix_b {};
struct accumulator {};
struct row_major {};
struct col_major {};
enum layout_t { mem_row_major, mem_col_major };
namespace precision {
struct tf32 {};
}

template <class Use, int M, int N, int K> struct FragDims;
template <int M, int N, int K> struct FragDims<matrix_a, M, N, K> { static constexpr int kRows = M, kCols = K; };
template <int M, int N, int K> struct FragDims<matrix_b, M, N, K> { static constexpr int kRows = K, kCols = N; };
template <int M, int N, int K> struct FragDims<accumulator, M, N, K> { static constexpr int kRows = M, kCols = N; };

template <class T> struct FragStorage { using type = T; };
template <> struct FragStorage<precision::tf32> { using type = float; };

template <class Use, int M, int N, int K, class T, class Layout = void>
struct fragment {
    using Dims = FragDims<Use, M, N, K>;
    using element_type = typename FragStorage<T>::type;
    static constexpr int kRows = Dims::kRows;
    static constexpr int kCols = Dims::kCols;
    static constexpr int num_elements = kRows * kCols;
    element_type x[num_elements];  // logical row-major tile
};

template <class Use, int M, int N, int K, class T, class L, class V>
inline void fill_fragment(fragment<Use, M, N, K, T, L>& f, const V& v) {
    for (int i = 0; i < f.num_elements; ++i) f.x[i] = static_cast<typename fragment<Use, M, N, K, T, L>::element_type>(v);
}

template <class P>
inline void checkWmmaPointer(const P* p, unsigned ldm) {
    if (reinterpret_cast<uintptr_t>(p) % 32 != 0) cuemu::fail("wmma load/store pointer must be 256-bit (32-byte) aligned");
    if ((ldm * sizeof(P)) % 16 != 0) cuemu::fail("wmma ldm must be a multiple of 16 bytes");
}

template <class Use, int M, int N, int K, class T, class L, class P>
inline void load_matrix_sync(fragment<Use, M, N, K, T, L>& f, const P* p, unsigned ldm) {
    using F = fragment<Use, M, N, K, T, L>;
    checkWmmaPointer(p, ldm);
    constexpr bool kColMajor = std::is_same_v<L, col_major>;
    for (int r = 0; r < F::kRows; ++r)
        for (int c = 0; c < F::kCols; ++c)
            f.x[r * F::kCols + c] = static_cast<typename F::element_type>(kColMajor ? p[c * ldm + r] : p[r * ldm + c]);
}

template <int M, int N, int K, class T, class P>
inline void load_matrix_sync(fragment<accumulator, M, N, K, T, void>& f, const P* p, unsigned ldm, layout_t layout) {
    using F = fragment<accumulator, M, N, K, T, void>;
    checkWmmaPointer(p, ldm);
    for (int r = 0; r < F::kRows; ++r)
        for (int c = 0; c < F::kCols; ++c)
            f.x[r * F::kCols + c] = layout == mem_col_major ? p[c * ldm + r] : p[r * ldm + c];
}

template <int M, int N, int K, class T, class P>
inline void store_matrix_sync(P* p, const fragment<accumulator, M, N, K, T, void>& f, unsigned ldm, layout_t layout) {
    using F = fragment<accumulator, M, N, K, T, void>;
    checkWmmaPointer(p, ldm);
    for (int r = 0; r < F::kRows; ++r)
        for (int c = 0; c < F::kCols; ++c) {
            P& dst = layout == mem_col_major ? p[c * ldm + r] : p[r * ldm + c];
            dst = static_cast<P>(f.x[r * F::kCols + c]);
        }
}

template <class Acc> struct MmaCompute { using type = float; };
template <> struct MmaCompute<int> { using type = int; };

template <int M, int N, int K, class TA, class LA, class TB, class LB, class TC, class TD>
inline void mma_sync(fragment<accumulator, M, N, K, TD>& d, const fragment<matrix_a, M, N, K, TA, LA>& a,
                     const fragment<matrix_b, M, N, K, TB, LB>& b, const fragment<accumulator, M, N, K, TC>& c,
                     bool = false) {
    using Compute = typename MmaCompute<TD>::type;
    Compute out[M * N];
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            Compute acc = static_cast<Compute>(c.x[i * N + j]);
            for (int k = 0; k < K; ++k)
                acc += static_cast<Compute>(a.x[i * K + k]) * static_cast<Compute>(b.x[k * N + j]);
            out[i * N + j] = acc;
        }
    for (int i = 0; i < M * N; ++i) d.x[i] = static_cast<TD>(out[i]);
}

inline float __float_to_tf32(float x) {
    unsigned u = __float_as_uint(x);
    u = (u + 0x1000u) & 0xFFFFE000u;  // round to 10 mantissa bits
    return __uint_as_float(u);
}
}  // namespace wmma
}  // namespace nvcuda


// ---------------------------------------------------------------------------
// Asynchronous copies: the <cuda_pipeline.h> primitives that compile to cp.async
// (sm_80+). Copies are *deferred* until __pipeline_wait_prior() covers their
// group, so a kernel that reads a stage before waiting for it sees stale shared
// memory here as it would on a GPU.
// ---------------------------------------------------------------------------
inline void __pipeline_memcpy_async(void* dst, const void* src, size_t size_and_align, size_t zfill = 0) {
    if (size_and_align != 4 && size_and_align != 8 && size_and_align != 16)
        cuemu::fail("__pipeline_memcpy_async: size must be 4, 8 or 16 bytes");
    if (zfill > size_and_align) cuemu::fail("__pipeline_memcpy_async: zfill larger than the copy");
    if (reinterpret_cast<uintptr_t>(dst) % size_and_align != 0 ||
        (zfill < size_and_align && reinterpret_cast<uintptr_t>(src) % size_and_align != 0))
        cuemu::fail("__pipeline_memcpy_async: source and destination must be aligned to the copy size");
    cuemu::g_block.fibers[cuemu::g_current].async_open.push_back({dst, src, size_and_align, zfill});
}
inline void __pipeline_commit() {
    auto& f = cuemu::g_block.fibers[cuemu::g_current];
    f.async_groups.push_back(std::move(f.async_open));
    f.async_open.clear();
}
inline void __pipeline_wait_prior(size_t prior) {
    auto& f = cuemu::g_block.fibers[cuemu::g_current];
    while (f.async_groups.size() > prior) {
        for (auto& c : f.async_groups.front()) c.perform();
        f.async_groups.erase(f.async_groups.begin());
    }
}

// ---------------------------------------------------------------------------
// Warp-wide matrix instructions written as inline PTX in real kernels
// (ldmatrix, mma.sync). cuemu cannot run PTX, so kernels call these instead
// under `#ifdef __CUEMU__`. They implement the documented PTX fragment layouts,
// so a kernel that indexes its fragments wrongly fails here too.
// ---------------------------------------------------------------------------
#define __CUEMU__ 1
namespace cuemu {
inline const void* g_lane_ptr[1024];
inline uint32_t g_lane_regs[1024][10];
inline float halfBits(uint32_t bits16) { return __half2float(__ushort_as_half(static_cast<unsigned short>(bits16))); }
}  // namespace cuemu

// ldmatrix.sync.aligned.m8n8.x{1,2,4}[.trans].shared.b16: lanes 8i..8i+7 supply the row
// addresses of 8x8 matrix i (16 bytes per row); register i of lane l receives row l/4,
// columns 2(l%4) and 2(l%4)+1 of matrix i (of its transpose with .trans).
inline void cuemuLdmatrix(uint32_t* regs, int num, bool trans, const void* row_ptr) {
    const int me = cuemu::g_block.fibers[cuemu::g_current].linear;
    const int base = me & ~31, lane = me & 31;
    if (reinterpret_cast<uintptr_t>(row_ptr) % 16 != 0) cuemu::fail("ldmatrix: row address must be 16-byte aligned");
    cuemu::g_lane_ptr[me] = row_ptr;
    __syncwarp();
    for (int i = 0; i < num; ++i) {
        uint16_t v[2];
        for (int h = 0; h < 2; ++h) {
            const int row = trans ? 2 * (lane % 4) + h : lane / 4;
            const int col = trans ? lane / 4 : 2 * (lane % 4) + h;
            v[h] = static_cast<const uint16_t*>(cuemu::g_lane_ptr[base + 8 * i + row])[col];
        }
        regs[i] = v[0] | (static_cast<uint32_t>(v[1]) << 16);
    }
    __syncwarp();
}

// mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 d, a, b, c. With g = lane / 4 and
// t = lane % 4 (pairs of halves packed low element first):
//   a0 = A[g][2t..2t+1]   a1 = A[g+8][2t..]   a2 = A[g][2t+8..]   a3 = A[g+8][2t+8..]
//   b0 = B[2t..2t+1][g]   b1 = B[2t+8..2t+9][g]
//   c0, c1 = C[g][2t], C[g][2t+1]             c2, c3 = C[g+8][2t], C[g+8][2t+1]
inline void cuemuMmaM16N8K16(float* d, const uint32_t* a, const uint32_t* b, const float* c) {
    const int me = cuemu::g_block.fibers[cuemu::g_current].linear;
    const int base = me & ~31, lane = me & 31;
    for (int i = 0; i < 4; ++i) cuemu::g_lane_regs[me][i] = a[i];
    for (int i = 0; i < 2; ++i) cuemu::g_lane_regs[me][4 + i] = b[i];
    __syncwarp();
    float am[16][16], bm[16][8];
    for (int l = 0; l < 32; ++l) {
        const uint32_t* r = cuemu::g_lane_regs[base + l];
        const int g = l / 4, t = l % 4;
        for (int h = 0; h < 2; ++h) {
            am[g][2 * t + h] = cuemu::halfBits(r[0] >> (16 * h));
            am[g + 8][2 * t + h] = cuemu::halfBits(r[1] >> (16 * h));
            am[g][2 * t + 8 + h] = cuemu::halfBits(r[2] >> (16 * h));
            am[g + 8][2 * t + 8 + h] = cuemu::halfBits(r[3] >> (16 * h));
            bm[2 * t + h][g] = cuemu::halfBits(r[4] >> (16 * h));
            bm[2 * t + 8 + h][g] = cuemu::halfBits(r[5] >> (16 * h));
        }
    }
    const int g = lane / 4, t = lane % 4;
    float out[4];
    for (int i = 0; i < 4; ++i) {
        const int row = g + 8 * (i / 2), col = 2 * t + (i % 2);
        float acc = c[i];
        for (int k = 0; k < 16; ++k) acc += am[row][k] * bm[k][col];
        out[i] = acc;
    }
    __syncwarp();
    for (int i = 0; i < 4; ++i) d[i] = out[i];
}
