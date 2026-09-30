// tests/q4_0_expert_check.cu - the Q4_0 grouped-expert GPU path (Fmt<2>, via iq_mmvq, the same row_dot the grouped
// kernels use) against (a) Strata's independent dense Q4_0 kernel (native_mmvq, pinned to llama.cpp's CUDA MMVQ) on
// the same Q8_1 bytes, (b) a CPU replay of llama.cpp's vec_dot_q4_0_q8_1 formula, and (c) the float reference, next
// to the centred formula (sum (q-8) * q8) so the formula's own rounding is visible.
//   q4_0_expert_check <model.gguf> [layer]   (layer's expert 7 gate rows, 16 random activations)
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace k = strata::kernels;

static float h2f(uint16_t h) { __half x; std::memcpy(&x, &h, 2); return __half2float(x); }

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: q4_0_expert_check <model.gguf> [layer]\n"); return 2; }
    const int layer = argc > 2 ? std::atoi(argv[2]) : 20;
    const int H = 2560, FF = 640, E = 7, NC = 8;
    strata::GgufFile g(argv[1]);
    const strata::TensorInfo* t = nullptr;
    for (const auto& ti : g.tensors())
        if (ti.name == "blk." + std::to_string(layer) + ".ffn_gate_exps.weight") t = &ti;
    if (!t || t->type != 2) { std::fprintf(stderr, "layer %d gate is not Q4_0\n", layer); return 1; }
    const size_t row = k::iq_row_bytes(2, H), mat = row * FF;
    const uint8_t* w = g.tensor_data(*t) + (size_t) E * mat;

    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) NC * H);
    for (auto& v : x) v = nd(rng);

    void *dw, *dx, *dq, *dy1, *dy2;
    cudaMalloc(&dw, mat); cudaMalloc(&dx, x.size() * 4); cudaMalloc(&dq, (size_t) NC * H / 32 * 36);
    cudaMalloc(&dy1, (size_t) NC * FF * 4); cudaMalloc(&dy2, (size_t) NC * FF * 4);
    cudaStream_t s; cudaStreamCreate(&s);
    cudaMemcpy(dw, w, mat, cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
    k::quantize_q8_1_rows((const float*) dx, NC, H, dq, s);                 // the grouped path's quantizer
    k::iq_mmvq(2, dw, dq, (float*) dy1, H, FF, NC, s);                        // Fmt<2> row_dot
    for (int c = 0; c < NC; ++c)                                              // Strata's dense Q4_0 MMVQ, one column each
        k::native_mmvq(2, dw, (const uint8_t*) dq + (size_t) c * H / 32 * 36, (float*) dy2 + (size_t) c * FF, H, FF, 1, s);
    cudaStreamSynchronize(s);
    if (cudaGetLastError() != cudaSuccess) { std::fprintf(stderr, "cuda error\n"); return 1; }
    std::vector<float> y1((size_t) NC * FF), y2((size_t) NC * FF);
    std::vector<uint8_t> q((size_t) NC * H / 32 * 36);
    cudaMemcpy(y1.data(), dy1, y1.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(y2.data(), dy2, y2.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(q.data(), dq, q.size(), cudaMemcpyDeviceToHost);

    // CPU replays on the same Q8_1 bytes (block: half d, half sum, int8 qs[32])
    double e_new_vs_dense = 0, n_dense = 0, e_new_vs_repl = 0, n_repl = 0, e_new_vs_cent = 0, n_cent = 0;
    double e_repl_ref = 0, e_cent_ref = 0, n_ref = 0;
    for (int c = 0; c < NC; ++c)
        for (int r = 0; r < FF; ++r) {
            const uint8_t* wr = w + (size_t) r * row;
            double repl = 0, cent = 0, ref = 0;
            for (int b = 0; b < H / 32; ++b) {
                const uint8_t* wb = wr + b * 18;
                const uint8_t* qb = q.data() + ((size_t) c * H / 32 + b) * 36;
                uint16_t dh, d8h, s8h; std::memcpy(&dh, wb, 2); std::memcpy(&d8h, qb, 2); std::memcpy(&s8h, qb + 2, 2);
                const float d4 = h2f(dh), d8 = h2f(d8h), s8 = h2f(s8h);
                const int8_t* q8 = (const int8_t*) (qb + 4);
                long sumi = 0, sumc = 0;
                for (int j = 0; j < 16; ++j) {
                    const int lo = wb[2 + j] & 0xF, hi = wb[2 + j] >> 4;
                    sumi += lo * q8[j] + hi * q8[j + 16];
                    sumc += (lo - 8) * q8[j] + (hi - 8) * q8[j + 16];
                    ref += (double) d4 * ((lo - 8) * (double) x[(size_t) c * H + b * 32 + j] +
                                          (hi - 8) * (double) x[(size_t) c * H + b * 32 + j + 16]);
                }
                repl += (double) d4 * ((double) sumi * d8 - 8.0 * s8);          // llama.cpp vec_dot_q4_0_q8_1
                cent += (double) d4 * d8 * (double) sumc;
            }
            const double a = y1[(size_t) c * FF + r], bd = y2[(size_t) c * FF + r];
            e_new_vs_dense += (a - bd) * (a - bd); n_dense += bd * bd;
            e_new_vs_repl += (a - repl) * (a - repl); n_repl += repl * repl;
            e_new_vs_cent += (a - cent) * (a - cent); n_cent += cent * cent;
            e_repl_ref += (repl - ref) * (repl - ref); e_cent_ref += (cent - ref) * (cent - ref); n_ref += ref * ref;
        }
    std::printf("layer %d expert %d gate (%d rows x %d cols x %d tokens)\n", layer, E, FF, H, NC);
    std::printf("  new Fmt<2> vs Strata dense Q4_0 MMVQ : rel %.2e\n", std::sqrt(e_new_vs_dense / n_dense));
    std::printf("  new Fmt<2> vs llama.cpp formula (CPU): rel %.2e\n", std::sqrt(e_new_vs_repl / n_repl));
    std::printf("  new Fmt<2> vs centred formula (CPU)  : rel %.2e\n", std::sqrt(e_new_vs_cent / n_cent));
    std::printf("  llama.cpp formula vs float reference : rel %.2e\n", std::sqrt(e_repl_ref / n_ref));
    std::printf("  centred formula   vs float reference : rel %.2e\n", std::sqrt(e_cent_ref / n_ref));

    // the dequantizers: Q4_0 (the prompt path's fallback) on this matrix, Q8_0 on token_embd rows (the native
    // embedding table), both against an exact CPU dequant; exact equality expected (one fp32 multiply each)
    {
        std::vector<float> got((size_t) FF * H);
        float* dd; cudaMalloc(&dd, got.size() * 4);
        k::iq_dequant_f32(2, dw, (int64_t) FF * H, dd, s);
        cudaStreamSynchronize(s);
        cudaMemcpy(got.data(), dd, got.size() * 4, cudaMemcpyDeviceToHost);
        long bad = 0;
        for (size_t i = 0; i < got.size(); ++i) {
            const size_t r = i / H, c = i % H, b = c / 32, j = c % 32;
            const uint8_t* wb = w + r * row + b * 18;
            uint16_t dh; std::memcpy(&dh, wb, 2);
            const int qv = j < 16 ? (wb[2 + j] & 0xF) : (wb[2 + j - 16] >> 4);
            if (got[i] != h2f(dh) * (float) (qv - 8)) ++bad;
        }
        std::printf("  Q4_0 dequant: %ld of %zu values differ\n", bad, got.size());
        cudaFree(dd);
        const strata::TensorInfo* te = nullptr;
        for (const auto& ti : g.tensors()) if (ti.name == "token_embd.weight") te = &ti;
        if (te && te->type == 8) {
            const size_t erow = k::iq_row_bytes(8, H);
            const int toks[4] = {0, 1000, 151643, 248000};
            std::vector<float> ge((size_t) 4 * H);
            void* dt; int32_t* dtok; float* de;
            cudaMalloc(&dt, erow * 248320); cudaMalloc(&dtok, 16); cudaMalloc(&de, ge.size() * 4);
            cudaMemcpy(dt, g.tensor_data(*te), erow * 248320, cudaMemcpyHostToDevice);
            cudaMemcpy(dtok, toks, 16, cudaMemcpyHostToDevice);
            k::iq_embed_rows(8, dt, erow, dtok, 4, H, de, s);
            cudaStreamSynchronize(s);
            cudaMemcpy(ge.data(), de, ge.size() * 4, cudaMemcpyDeviceToHost);
            long badq8 = 0;
            for (int t2 = 0; t2 < 4; ++t2)
                for (int c = 0; c < H; ++c) {
                    const uint8_t* eb = g.tensor_data(*te) + (size_t) toks[t2] * erow + (c / 32) * 34;
                    uint16_t dh; std::memcpy(&dh, eb, 2);
                    if (ge[(size_t) t2 * H + c] != h2f(dh) * (float) (int8_t) eb[2 + c % 32]) ++badq8;
                }
            std::printf("  Q8_0 token_embd rows: %ld of %d values differ\n", badq8, 4 * H);
        }
    }
    return 0;
}
