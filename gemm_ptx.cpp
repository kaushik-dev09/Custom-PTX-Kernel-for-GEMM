// gemm_ptx.cu
// Target: sm_80 (A100) or sm_86 (RTX 3090)
// Compile: nvcc -O3 -arch=sm_86 -lineinfo gemm_ptx.cu -o gemm_ptx

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

#define BM 64
#define BN 64
#define BK 16
#define WM 32   // warp tile rows
#define WN 32   // warp tile cols
#define TM 8    // thread tile rows (per thread)
#define TN 8    // thread tile cols (per thread)

// ─────────────────────────────────────────────────
//  PTX HELPER MACROS
// ─────────────────────────────────────────────────

// Convert generic ptr → shared memory address
__device__ __forceinline__
uint32_t cvta_shared(const void* smem_ptr) {
    uint32_t addr;
    asm volatile(
        "cvta.to.shared.u32 %0, %1;"
        : "=r"(addr)
        : "r"(__cvta_generic_to_shared(smem_ptr))
    );
    return addr;
}

// Load from global memory → register (f32)
__device__ __forceinline__
float ld_global_f32(const float* ptr) {
    float val;
    asm volatile(
        "ld.global.f32 %0, [%1];"
        : "=f"(val)
        : "l"(ptr)
    );
    return val;
}

// Load from global memory with cache streaming hint (bypass L1)
__device__ __forceinline__
float ld_global_cs_f32(const float* ptr) {
    float val;
    asm volatile(
        "ld.global.cs.f32 %0, [%1];"   // .cs = cache-streaming
        : "=f"(val)
        : "l"(ptr)
    );
    return val;
}

// Store register → shared memory (f32)
__device__ __forceinline__
void st_shared_f32(uint32_t smem_addr, float val) {
    asm volatile(
        "st.shared.f32 [%0], %1;"
        :
        : "r"(smem_addr), "f"(val)
    );
}

// Load from shared memory → register (f32)
__device__ __forceinline__
float ld_shared_f32(uint32_t smem_addr) {
    float val;
    asm volatile(
        "ld.shared.f32 %0, [%1];"
        : "=f"(val)
        : "r"(smem_addr)
    );
    return val;
}

// FMA: d = a * b + c  (round-to-nearest)
__device__ __forceinline__
float fma_rn(float a, float b, float c) {
    float d;
    asm volatile(
        "fma.rn.f32 %0, %1, %2, %3;"
        : "=f"(d)
        : "f"(a), "f"(b), "f"(c)
    );
    return d;
}

// Memory fence (shared memory scope — cheaper than full membar)
__device__ __forceinline__
void membar_cta() {
    asm volatile("membar.cta;");
}

// ─────────────────────────────────────────────────
//  SMEM LAYOUT  (avoid bank conflicts)
//  A tile: [BK][BM] = [16][64] — col-major for A^T access
//  B tile: [BK][BN] = [16][64] — row-major
// ─────────────────────────────────────────────────
__device__ __forceinline__
uint32_t smem_a_addr(uint32_t smem_base_a, int k, int m) {
    // Padded: stride = BM+4 to avoid bank conflicts
    return smem_base_a + (k * (BM + 4) + m) * sizeof(float);
}

__device__ __forceinline__
uint32_t smem_b_addr(uint32_t smem_base_b, int k, int n) {
    return smem_base_b + (k * (BN + 4) + n) * sizeof(float);
}

// ─────────────────────────────────────────────────
//  MAIN KERNEL
// ─────────────────────────────────────────────────
__global__ __launch_bounds__(256, 2)
void gemm_ptx_kernel(
    const float* __restrict__ A,   // [M×K] row-major
    const float* __restrict__ B,   // [K×N] row-major
    float*       __restrict__ C,   // [M×N] row-major
    int M, int N, int K,
    float alpha, float beta
) {
    // ── Block indices ──────────────────────────────
    const int brow = blockIdx.y;   // which BM-tile row
    const int bcol = blockIdx.x;   // which BN-tile col

    // ── Thread indices ─────────────────────────────
    const int tid    = threadIdx.x;        // 0..255
    const int warp   = tid >> 5;           // 0..7
    const int lane   = tid & 31;           // 0..31

    // 8 warps arranged as 2×4 warp grid within block
    const int warp_row = warp >> 2;        // 0..1  (WM=32 rows each)
    const int warp_col = warp & 3;         // 0..3  (WN=16 cols each)

    // Thread position within warp tile (4×8 threads → 8×8 elements each)
    const int tr = lane >> 3;              // 0..3
    const int tc = lane & 7;              // 0..7

    // ── Shared Memory Allocation ───────────────────
    // Double-buffered: 2 × (A_tile + B_tile)
    //   A: [BK][BM+4] = 16×68 floats = 4352 bytes per buffer
    //   B: [BK][BN+4] = 16×68 floats = 4352 bytes per buffer
    __shared__ float smem[2][BK * (BM + 4) + BK * (BN + 4)];

    uint32_t smem_a[2], smem_b[2];
    // Get base PTX shared addresses for each double-buffer slot
    asm volatile(
        "cvta.to.shared.u32 %0, %1;"
        : "=r"(smem_a[0]) : "r"(__cvta_generic_to_shared(&smem[0][0]))
    );
    asm volatile(
        "cvta.to.shared.u32 %0, %1;"
        : "=r"(smem_b[0]) : "r"(__cvta_generic_to_shared(&smem[0][BK*(BM+4)]))
    );
    asm volatile(
        "cvta.to.shared.u32 %0, %1;"
        : "=r"(smem_a[1]) : "r"(__cvta_generic_to_shared(&smem[1][0]))
    );
    asm volatile(
        "cvta.to.shared.u32 %0, %1;"
        : "=r"(smem_b[1]) : "r"(__cvta_generic_to_shared(&smem[1][BK*(BM+4)]))
    );

    // ── Register Tile (8×8 accumulators per thread) ─
    // Explicitly declared in PTX .reg space via inline asm
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++)
            asm volatile("mov.f32 %0, 0f00000000;" : "=f"(acc[i][j]));  // 0.0f

    // ── Global Memory Pointers ─────────────────────
    // Each thread loads a column of A-tile and a row of B-tile
    // Stride across K-loop: 4 threads per row, each loads 1 element
    const int load_a_row = brow * BM + (tid >> 2);        // row in A
    const int load_a_col = (tid & 3) * 4;                 // col offset in K (4 per thread = vec4)
    const int load_b_row = (tid >> 4) * 1;                // row in K per tile
    const int load_b_col = bcol * BN + (tid & 15) * 4;   // col in B

    const float* A_ptr = A + load_a_row * K + load_a_col;
    const float* B_ptr = B + load_b_row * N + load_b_col;

    // ── K-loop with Double Buffering ───────────────
    int buf = 0;
    int num_tiles = (K + BK - 1) / BK;

    // Load first tile eagerly
    // [For brevity, simplified load — production code vectorizes with ld.global.v4.f32]
    #pragma unroll
    for (int i = 0; i < BK; i++) {
        // Thread loads 1 element of A into smem_a[buf]
        int a_m = tid % BM;
        int a_k = i;
        float a_val = ld_global_cs_f32(A + (brow * BM + a_m) * K + a_k);
        uint32_t a_addr = smem_a[buf] + (a_k * (BM+4) + a_m) * sizeof(float);
        st_shared_f32(a_addr, a_val);

        // Thread loads 1 element of B
        int b_k = i;
        int b_n = tid % BN;
        float b_val = ld_global_cs_f32(B + b_k * N + (bcol * BN + b_n));
        uint32_t b_addr = smem_b[buf] + (b_k * (BN+4) + b_n) * sizeof(float);
        st_shared_f32(b_addr, b_val);
    }
    asm volatile("bar.sync 0;");   // __syncthreads() in PTX

    // ── Main K-Tile Loop ───────────────────────────
    for (int tile = 0; tile < num_tiles; tile++) {
        int next_buf = 1 - buf;

        // ─ Prefetch next tile into next_buf (async-style)
        if (tile + 1 < num_tiles) {
            #pragma unroll
            for (int i = 0; i < BK; i++) {
                int a_m = tid % BM;
                int a_k = (tile + 1) * BK + i;
                float a_val = (a_k < K) ? ld_global_cs_f32(A + (brow*BM+a_m)*K + a_k) : 0.f;
                uint32_t a_addr = smem_a[next_buf] + (i*(BM+4)+a_m)*sizeof(float);
                st_shared_f32(a_addr, a_val);

                int b_k = (tile + 1) * BK + i;
                int b_n = tid % BN;
                float b_val = (b_k < K) ? ld_global_cs_f32(B + b_k*N + (bcol*BN+b_n)) : 0.f;
                uint32_t b_addr = smem_b[next_buf] + (i*(BN+4)+b_n)*sizeof(float);
                st_shared_f32(b_addr, b_val);
            }
        }

        // ─ Compute over current tile in smem (BK steps)
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            // Load register fragments for A (TM=8 rows)
            float a_frag[TM];
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                int m = warp_row * WM + tr * TM + i;  // actual row within BM
                uint32_t a_addr = smem_a[buf] + (k*(BM+4)+m)*sizeof(float);
                a_frag[i] = ld_shared_f32(a_addr);
            }

            // Load register fragments for B (TN=8 cols)
            float b_frag[TN];
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                int n = warp_col * WN + tc * TN + j;  // actual col within BN
                uint32_t b_addr = smem_b[buf] + (k*(BN+4)+n)*sizeof(float);
                b_frag[j] = ld_shared_f32(b_addr);
            }

            // Outer product: 8×8 FMAs using PTX fma.rn.f32
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    acc[i][j] = fma_rn(a_frag[i], b_frag[j], acc[i][j]);
        }

        asm volatile("bar.sync 0;");
        buf = next_buf;
    }

    // ── Write Output with alpha/beta scaling ───────
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int row = brow * BM + warp_row * WM + tr * TM + i;
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int col = bcol * BN + warp_col * WN + tc * TN + j;
            if (row < M && col < N) {
                float c_old;
                asm volatile(
                    "ld.global.f32 %0, [%1];"
                    : "=f"(c_old)
                    : "l"(C + row * N + col)
                );
                float c_new;
                asm volatile(
                    "fma.rn.f32 %0, %1, %2, %3;"   // c_new = alpha*acc + beta*c_old
                    : "=f"(c_new)
                    : "f"(alpha), "f"(acc[i][j]), "f"(c_old)
                );
                asm volatile(
                    "st.global.f32 [%0], %1;"
                    :
                    : "l"(C + row * N + col), "f"(c_new)
                );
            }
        }
    }
}
