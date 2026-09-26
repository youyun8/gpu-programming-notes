// Chapter 12: convolution and stencils.
//
//   conv1dTiled      1-D convolution: filter in __constant__ memory, input tile + halo in shared memory
//   conv2dNaive      2-D convolution, every tap read from global memory (through the caches)
//   conv2dTiled      2-D convolution with a (16 + 2R)^2 shared-memory input tile
//   stencil3dNaive   one Jacobi step of the 7-point 3-D stencil, 7 global loads per point
//   stencil3d25D     the same with 2.5-D blocking: march along z, keep z-1 / z / z+1 in registers,
//                    share the xy neighbours of the current plane through shared memory
//
// All convolutions are "same" size with zero padding: out has the shape of in, and taps that fall
// outside the input read 0. The stencil keeps boundary cells unchanged (Dirichlet boundary).
//
// Build: nvcc -O3 -arch=sm_80 -std=c++17 12-convolution-stencil.cu -o convolution_stencil
#include "check.cuh"

constexpr int kMaxRadius1d = 64;
constexpr int kMaxRadius2d = 4;
__constant__ float c_filter1d[2 * kMaxRadius1d + 1];
__constant__ float c_filter2d[(2 * kMaxRadius2d + 1) * (2 * kMaxRadius2d + 1)];

// ---- 1-D ------------------------------------------------------------------------------------
constexpr int kThreads1d = 256;

// Block b computes out[b*256 .. b*256+255]. It needs in[b*256 - R .. b*256+255 + R]: the tile
// plus a halo of R elements on each side, loaded once into shared memory and read 2R+1 times.
__global__ void conv1dTiled(const float* in, float* out, int n, int radius) {
    __shared__ float tile[kThreads1d + 2 * kMaxRadius1d];
    const int base = blockIdx.x * kThreads1d;
    for (int i = threadIdx.x; i < kThreads1d + 2 * radius; i += blockDim.x) {
        const int g = base - radius + i;
        tile[i] = (g >= 0 && g < n) ? in[g] : 0.0f;          // zero padding outside the input
    }
    __syncthreads();
    const int o = base + threadIdx.x;
    if (o >= n) return;                                      // no barrier below: safe
    float acc = 0.0f;
    for (int k = -radius; k <= radius; ++k)                  // all lanes read the same c_filter1d[k]: broadcast
        acc = fmaf(c_filter1d[k + radius], tile[threadIdx.x + radius + k], acc);
    out[o] = acc;
}

// ---- 2-D ------------------------------------------------------------------------------------
constexpr int kTile2d = 16;   // 16 x 16 outputs per block, one per thread

__global__ void conv2dNaive(const float* in, float* out, int height, int width, int radius) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const int side = 2 * radius + 1;
    float acc = 0.0f;
    for (int dy = -radius; dy <= radius; ++dy)
        for (int dx = -radius; dx <= radius; ++dx) {
            const int yy = y + dy, xx = x + dx;
            if (yy >= 0 && yy < height && xx >= 0 && xx < width)
                acc = fmaf(c_filter2d[(dy + radius) * side + dx + radius], in[yy * width + xx], acc);
        }
    out[y * width + x] = acc;
}

__global__ void conv2dTiled(const float* in, float* out, int height, int width, int radius) {
    constexpr int kSide = kTile2d + 2 * kMaxRadius2d;
    __shared__ float tile[kSide][kSide + 1];                 // +1: column reads of the halo are conflict-free
    const int x0 = blockIdx.x * kTile2d - radius;            // input coordinates of tile[0][0]
    const int y0 = blockIdx.y * kTile2d - radius;
    const int side = kTile2d + 2 * radius;
    // Load (16 + 2R)^2 values with 256 threads: a strided loop over the tile.
    for (int i = threadIdx.y * kTile2d + threadIdx.x; i < side * side; i += kTile2d * kTile2d) {
        const int ty = i / side, tx = i % side;
        const int yy = y0 + ty, xx = x0 + tx;
        tile[ty][tx] = (yy >= 0 && yy < height && xx >= 0 && xx < width) ? in[yy * width + xx] : 0.0f;
    }
    __syncthreads();
    const int x = blockIdx.x * kTile2d + threadIdx.x;
    const int y = blockIdx.y * kTile2d + threadIdx.y;
    if (x >= width || y >= height) return;
    const int fside = 2 * radius + 1;
    float acc = 0.0f;
    for (int dy = 0; dy < fside; ++dy)
        for (int dx = 0; dx < fside; ++dx)
            acc = fmaf(c_filter2d[dy * fside + dx], tile[threadIdx.y + dy][threadIdx.x + dx], acc);
    out[y * width + x] = acc;
}

// ---- 3-D 7-point stencil ----------------------------------------------------------------------
// out = c0 * in + c1 * (sum of the 6 face neighbours) for interior points; boundary points copied.
constexpr int kBx = 32, kBy = 8;

__device__ __forceinline__ size_t at(int x, int y, int z, int nx, int ny) {
    return (static_cast<size_t>(z) * ny + y) * nx + x;
}

__global__ void stencil3dNaive(const float* in, float* out, int nx, int ny, int nz, float c0, float c1) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z;
    if (x >= nx || y >= ny) return;
    const size_t i = at(x, y, z, nx, ny);
    if (x == 0 || y == 0 || z == 0 || x == nx - 1 || y == ny - 1 || z == nz - 1) {
        out[i] = in[i];
        return;
    }
    const size_t plane = static_cast<size_t>(nx) * ny;
    out[i] = c0 * in[i] + c1 * (in[i - 1] + in[i + 1] + in[i - nx] + in[i + nx] + in[i - plane] + in[i + plane]);
}

// A block owns a 32 x 8 column of the domain and walks it from z = 0 to nz - 1.
// Per point: 1 global load (the plane above) instead of 7; x/y neighbours come from shared memory.
__global__ void stencil3d25D(const float* in, float* out, int nx, int ny, int nz, float c0, float c1) {
    __shared__ float plane[kBy + 2][kBx + 2];                // current plane with a 1-cell halo
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int x = blockIdx.x * kBx + tx;
    const int y = blockIdx.y * kBy + ty;
    const bool inside = x < nx && y < ny;
    const bool boundary_xy = x == 0 || y == 0 || x == nx - 1 || y == ny - 1;
    auto load = [&](int xx, int yy, int zz) {
        return (xx >= 0 && xx < nx && yy >= 0 && yy < ny && zz >= 0 && zz < nz) ? in[at(xx, yy, zz, nx, ny)] : 0.0f;
    };
    float below = load(x, y, 0);                             // registers: planes z-1, z, z+1 of my column
    float cur = load(x, y, 1);
    if (inside) out[at(x, y, 0, nx, ny)] = below;            // plane z = 0 is boundary
    for (int z = 1; z < nz - 1; ++z) {
        const float above = load(x, y, z + 1);
        __syncthreads();                                     // everyone is done reading the previous plane
        plane[ty + 1][tx + 1] = cur;
        if (tx == 0) plane[ty + 1][0] = load(x - 1, y, z);   // halo columns and rows of plane z
        if (tx == kBx - 1) plane[ty + 1][kBx + 1] = load(x + 1, y, z);
        if (ty == 0) plane[0][tx + 1] = load(x, y - 1, z);
        if (ty == kBy - 1) plane[kBy + 1][tx + 1] = load(x, y + 1, z);
        __syncthreads();
        if (inside) {
            const float neighbours = plane[ty + 1][tx] + plane[ty + 1][tx + 2] + plane[ty][tx + 1] +
                                     plane[ty + 2][tx + 1] + below + above;
            out[at(x, y, z, nx, ny)] = boundary_xy ? cur : c0 * cur + c1 * neighbours;
        }
        below = cur;
        cur = above;
    }
    if (inside && nz > 1) out[at(x, y, nz - 1, nx, ny)] = cur;   // plane z = nz - 1 is boundary
}

// ---- Checks -----------------------------------------------------------------------------------
std::vector<float> makeFilter(int taps, uint32_t seed) { return ex::randomVector(taps, seed, -0.5f, 0.5f); }

void checkConv1d(int n, int radius) {
    const std::vector<float> x = ex::randomVector(n, 1), f = makeFilter(2 * radius + 1, 2);
    CUDA_CHECK(cudaMemcpyToSymbol(c_filter1d, f.data(), f.size() * sizeof(float)));
    ex::DeviceArray<float> d_in(x), d_out(n);
    conv1dTiled<<<ex::ceilDiv(n, kThreads1d), kThreads1d>>>(d_in.ptr, d_out.ptr, n, radius);
    std::vector<double> ref(n, 0.0);
    for (int o = 0; o < n; ++o)
        for (int k = -radius; k <= radius; ++k)
            if (o + k >= 0 && o + k < n) ref[o] += static_cast<double>(f[k + radius]) * x[o + k];
    char name[64];
    std::snprintf(name, sizeof(name), "conv1dTiled n=%d R=%d", n, radius);
    ex::checkClose(name, d_out.download(), ref, 1e-4, 1e-5);
}

void checkConv2d(int height, int width, int radius) {
    const std::vector<float> x = ex::randomVector(static_cast<size_t>(height) * width, 3);
    const int side = 2 * radius + 1;
    const std::vector<float> f = makeFilter(side * side, 4);
    CUDA_CHECK(cudaMemcpyToSymbol(c_filter2d, f.data(), f.size() * sizeof(float)));
    ex::DeviceArray<float> d_in(x), d_out(x.size());
    std::vector<double> ref(x.size(), 0.0);
    for (int y = 0; y < height; ++y)
        for (int xx = 0; xx < width; ++xx)
            for (int dy = -radius; dy <= radius; ++dy)
                for (int dx = -radius; dx <= radius; ++dx)
                    if (y + dy >= 0 && y + dy < height && xx + dx >= 0 && xx + dx < width)
                        ref[y * width + xx] += static_cast<double>(f[(dy + radius) * side + dx + radius]) *
                                               x[(y + dy) * width + xx + dx];
    const dim3 block(kTile2d, kTile2d), grid(ex::ceilDiv(width, kTile2d), ex::ceilDiv(height, kTile2d));
    char name[64];
    conv2dNaive<<<grid, block>>>(d_in.ptr, d_out.ptr, height, width, radius);
    std::snprintf(name, sizeof(name), "conv2dNaive %dx%d R=%d", height, width, radius);
    ex::checkClose(name, d_out.download(), ref, 1e-4, 1e-5);
    conv2dTiled<<<grid, block>>>(d_in.ptr, d_out.ptr, height, width, radius);
    std::snprintf(name, sizeof(name), "conv2dTiled %dx%d R=%d", height, width, radius);
    ex::checkClose(name, d_out.download(), ref, 1e-4, 1e-5);
}

void checkStencil(int nx, int ny, int nz) {
    const float c0 = 0.4f, c1 = 0.1f;
    const std::vector<float> x = ex::randomVector(static_cast<size_t>(nx) * ny * nz, 5);
    ex::DeviceArray<float> d_in(x), d_out(x.size());
    std::vector<double> ref(x.size());
    for (int z = 0; z < nz; ++z)
        for (int y = 0; y < ny; ++y)
            for (int xx = 0; xx < nx; ++xx) {
                const size_t i = (static_cast<size_t>(z) * ny + y) * nx + xx;
                if (xx == 0 || y == 0 || z == 0 || xx == nx - 1 || y == ny - 1 || z == nz - 1) {
                    ref[i] = x[i];
                } else {
                    const size_t p = static_cast<size_t>(nx) * ny;
                    ref[i] = c0 * x[i] + c1 * (static_cast<double>(x[i - 1]) + x[i + 1] + x[i - nx] + x[i + nx] +
                                               x[i - p] + x[i + p]);
                }
            }
    char name[64];
    stencil3dNaive<<<dim3(ex::ceilDiv(nx, kBx), ex::ceilDiv(ny, kBy), nz), dim3(kBx, kBy)>>>(d_in.ptr, d_out.ptr, nx,
                                                                                          ny, nz, c0, c1);
    std::snprintf(name, sizeof(name), "stencil3dNaive %dx%dx%d", nx, ny, nz);
    ex::checkClose(name, d_out.download(), ref, 1e-5, 1e-6);
    CUDA_CHECK(cudaMemset(d_out.ptr, 0, x.size() * sizeof(float)));
    stencil3d25D<<<dim3(ex::ceilDiv(nx, kBx), ex::ceilDiv(ny, kBy)), dim3(kBx, kBy)>>>(d_in.ptr, d_out.ptr, nx, ny,
                                                                                    nz, c0, c1);
    std::snprintf(name, sizeof(name), "stencil3d25D %dx%dx%d", nx, ny, nz);
    ex::checkClose(name, d_out.download(), ref, 1e-5, 1e-6);
}

void bench() {
    {
        const int h = 4096, w = 4096, radius = 3;
        const std::vector<float> f = makeFilter(49, 6);
        CUDA_CHECK(cudaMemcpyToSymbol(c_filter2d, f.data(), f.size() * sizeof(float)));
        ex::DeviceArray<float> d_in(ex::randomVector(static_cast<size_t>(h) * w, 7)), d_out(static_cast<size_t>(h) * w);
        const dim3 block(kTile2d, kTile2d), grid(w / kTile2d, h / kTile2d);
        const double bytes = 8.0 * h * w;
        ex::reportBandwidth("conv2dNaive 4096^2, 7x7", ex::timeMs([&] {
            conv2dNaive<<<grid, block>>>(d_in.ptr, d_out.ptr, h, w, radius);
        }), bytes);
        ex::reportBandwidth("conv2dTiled 4096^2, 7x7", ex::timeMs([&] {
            conv2dTiled<<<grid, block>>>(d_in.ptr, d_out.ptr, h, w, radius);
        }), bytes);
    }
    {
        const int nx = 512, ny = 512, nz = 256;
        const size_t count = static_cast<size_t>(nx) * ny * nz;
        ex::DeviceArray<float> d_in(ex::randomVector(count, 8)), d_out(count);
        const double bytes = 8.0 * count;
        ex::reportBandwidth("stencil3dNaive 512x512x256", ex::timeMs([&] {
            stencil3dNaive<<<dim3(nx / kBx, ny / kBy, nz), dim3(kBx, kBy)>>>(d_in.ptr, d_out.ptr, nx, ny, nz, 0.4f, 0.1f);
        }), bytes);
        ex::reportBandwidth("stencil3d25D 512x512x256", ex::timeMs([&] {
            stencil3d25D<<<dim3(nx / kBx, ny / kBy), dim3(kBx, kBy)>>>(d_in.ptr, d_out.ptr, nx, ny, nz, 0.4f, 0.1f);
        }), bytes);
    }
}

int main(int argc, char** argv) {
    checkConv1d(1, 3);
    checkConv1d(1000, 3);
    checkConv1d(777, 17);
    checkConv1d(300, 64);
    checkConv2d(1, 1, 1);
    checkConv2d(37, 53, 1);
    checkConv2d(40, 33, 3);
    checkConv2d(17, 70, 4);
    checkStencil(3, 3, 3);
    checkStencil(40, 13, 9);
    checkStencil(70, 17, 5);
    if (ex::wantBench(argc, argv)) bench();
    return ex::finish("12-convolution-stencil");
}
