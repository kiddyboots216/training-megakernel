# Third-party software

Training Megakernel builds on the following packages. Their licenses govern
them independently of this project's MIT license.

- NVIDIA CUTLASS DSL 4.6.0 is distributed under the NVIDIA Software License
  Agreement. As required for applications that incorporate its Python source:
  “This software contains source code provided by NVIDIA Corporation.”
- FlashAttention 4 is distributed under the BSD 3-Clause License.
- Quack kernels are distributed under the Apache License 2.0.

The program source includes modified copies of functions from FlashAttention 4
and Quack. Each file names the functions it adapts and says what changed.

- From FlashAttention 4 (commit 890f238), under the BSD 3-Clause License in
  [LICENSES/FlashAttention-BSD-3-Clause.txt](LICENSES/FlashAttention-BSD-3-Clause.txt):
  `fa4_forward_kernel.py`, `fa4_forward_call.py`, `fa4_backward_kernel.py`,
  `fa4_backward_call.py`, `fa4_postprocess.py` and one tile-scheduler method in
  `attention.py`, in `kernel/program/`.
- From quack-kernels 0.6.0, under the Apache License 2.0 in
  [LICENSES/Quack-Apache-2.0.txt](LICENSES/Quack-Apache-2.0.txt):
  `quack_gemm_bodies.py`, `quack_rmsnorm_bodies.py`, and the GEMM member, epilogue
  and tile-scheduler methods adapted in `gemm_members.py`, `projection_members.py`,
  `attention_projections.py`, `mlp_projections.py` and `tile_schedulers.py`, in
  `kernel/program/`.

Installations of the packages include their complete license texts. See their
upstream repositories for source and notices:
[CUTLASS](https://github.com/NVIDIA/cutlass),
[FlashAttention](https://github.com/Dao-AILab/flash-attention), and
[Quack](https://github.com/Dao-AILab/quack).
