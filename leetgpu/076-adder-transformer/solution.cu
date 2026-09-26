// Adder Transformer Inference (LeetGPU)
// https://leetgpu.com/challenges/adder-transformer-inference
//
// Greedy decoding (11 steps) of a 10-parameter, 1-layer, 2-dim transformer.
// Only the last position's logits are needed per step, and with a single
// layer every position's K/V depends on that position's token alone. So each
// step costs O(seq_len) and one thread can decode one sequence end to end,
// keeping the (<= 41-token) sequence in registers/local memory.
// Constants and float operation order follow the reference model.
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

constexpr int kVocab = 10;
constexpr int kPromptLen = 31;
constexpr int kOutputDigits = 11;
constexpr int kMaxLen = kPromptLen + kOutputDigits;
constexpr float kRmsEps = 1e-6f;
constexpr float kEmbedConst = 1000.0f;

// RoPE frequency and attention scale, derived on the host in double precision.
struct ModelConsts {
    float omega;
    float attn_scale;
};

// RMSNorm of the 2-D hidden state without a weight.
__device__ __forceinline__ void unitRmsNorm(float& x0, float& x1) {
    const float r = rsqrtf((x0 * x0 + x1 * x1) * 0.5f + kRmsEps);
    x0 *= r;
    x1 *= r;
}

__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }

__global__ void adderDecode(const int* prompts, float* output, const float* w, int batch, ModelConsts mc) {
    // One thread runs the whole greedy decode of one batch element: the model has a 2-D
    // hidden state, so everything fits in registers and local arrays.
    const int bi = blockIdx.x * blockDim.x + threadIdx.x;
    if (bi >= batch) return;

    // Token embeddings built from the parameters: e(d) = (w0 - w1 d^2, -d).
    float emb0[kVocab];
    float emb1[kVocab];
    for (int d = 0; d < kVocab; ++d) {
        const float df = static_cast<float>(d);
        emb0[d] = w[0] - w[1] * df * df;
        emb1[d] = -df;
    }
    // Remaining weights: query projection, value, MLP gates, carry weight and final norm.
    const float q_w0 = w[2], q_w1 = w[3], v_w = w[4];
    const float a_gate = w[5], c_gate = w[6], carry_w = w[7];
    const float norm_w0 = w[8], norm_w1 = w[9];

    // Per-position rotated key (k1 is 0 before RoPE) and value, cached across steps.
    float k_rot0[kMaxLen];
    float k_rot1[kMaxLen];
    float val[kMaxLen];
    int len = 0;
    auto appendToken = [&](int tok) {
        float h0 = emb0[tok], h1 = emb1[tok];
        unitRmsNorm(h0, h1);
        float k0 = h0, k1 = 0.0f;
        unitRmsNorm(k0, k1);
        const float angle = static_cast<float>(len) * mc.omega;
        const float c = cosf(angle), s = sinf(angle);
        k_rot0[len] = k0 * c - k1 * s;
        k_rot1[len] = k0 * s + k1 * c;
        val[len] = h1 * v_w;
        ++len;
    };
    // Prefill: cache the rotated keys and values of the 31 prompt tokens.
    for (int t = 0; t < kPromptLen; ++t) appendToken(prompts[static_cast<size_t>(bi) * kPromptLen + t]);

    int last_tok = prompts[static_cast<size_t>(bi) * kPromptLen + kPromptLen - 1];
    // Greedy decode of 11 output digits.
    for (int step = 0; step < kOutputDigits; ++step) {
        const int pos = len - 1;
        // Query of the last position.
        float h0 = emb0[last_tok], h1 = emb1[last_tok];
        float n0 = h0, n1 = h1;
        unitRmsNorm(n0, n1);
        float q0 = n0 * q_w0, q1 = n0 * q_w1;
        unitRmsNorm(q0, q1);
        const float angle = static_cast<float>(pos) * mc.omega;
        const float c = cosf(angle), s = sinf(angle);
        const float qr0 = q0 * c - q1 * s;
        const float qr1 = q0 * s + q1 * c;

        // Causal softmax attention over positions 0..pos (value dim 0 only).
        float mx = -FLT_MAX;
        for (int j = 0; j < len; ++j) mx = fmaxf(mx, (qr0 * k_rot0[j] + qr1 * k_rot1[j]) * mc.attn_scale);
        float denom = 0.0f, numer = 0.0f;
        for (int j = 0; j < len; ++j) {
            const float p = expf((qr0 * k_rot0[j] + qr1 * k_rot1[j]) * mc.attn_scale - mx);
            denom += p;
            numer += p * val[j];
        }
        const float attn = numer / denom;

        // Residual (O projection writes into dim 1), MLP, final norm, tied logits.
        h1 += attn;
        float m0 = h0, m1 = h1;
        unitRmsNorm(m0, m1);
        const float g0 = m0 * a_gate + m1 * c_gate;
        const float g1 = m0 * (a_gate - c_gate / kEmbedConst) + m1 * c_gate;
        h1 += carry_w * (silu(g1) * m0 - silu(g0) * m0);
        const float rms = sqrtf((h0 * h0 + h1 * h1) * 0.5f + kRmsEps);
        const float f0 = h0 / rms * norm_w0;
        const float f1 = h1 / rms * norm_w1;

        // Tied output embedding: logits = final hidden state . e(d); write them and take the arg-max.
        float* out = output + (static_cast<size_t>(bi) * kOutputDigits + step) * kVocab;
        int best = 0;
        float best_logit = -FLT_MAX;
        for (int d = 0; d < kVocab; ++d) {
            const float logit = f0 * emb0[d] + f1 * emb1[d];
            out[d] = logit;
            if (logit > best_logit) {
                best_logit = logit;
                best = d;
            }
        }
        // Feed the chosen digit back as the next input token.
        last_tok = best;
        if (len < kMaxLen) appendToken(best);
    }
}

// prompts, output, weights are device pointers
extern "C" void solve(const int* prompts, float* output, const float* weights, int batch_size) {
    // Model constants: RoPE frequency 2 pi / 19 and the attention scale implied by the construction.
    const double pi = 3.14159265358979323846;
    const double omega = 2.0 * pi / 19.0;
    const double amplitude = std::log(10.0) / (std::cos(omega * 0.3) - std::cos(omega * 0.7));
    const double qk_norm_sq = amplitude / std::sqrt(2.0);
    const ModelConsts mc{static_cast<float>(omega), static_cast<float>(std::pow(2.0, -0.5) * qk_norm_sq)};
    adderDecode<<<(batch_size + 127) / 128, 128>>>(prompts, output, weights, batch_size, mc);
    cudaDeviceSynchronize();
}
