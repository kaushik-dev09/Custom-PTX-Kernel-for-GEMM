Global Memory (GMEM)
       │  ld.global.f32 (coalesced 128-byte transactions)
       ▼
Shared Memory (SMEM)    ← 64×16 tile of A, 16×64 tile of B
       │  ld.shared.f32
       ▼
Register File           ← 8×8 thread-local accumulator tile
       │  fma.rn.f32 / mma.sync
       ▼
Global Memory           ← st.global.f32
