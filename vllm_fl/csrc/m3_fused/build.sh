#!/bin/bash
# Build the M3 fused-op native extension for the PPU (CUDA-compatible) stack.
# Arch mirrors deep_gemm's device probe: ppu001 -> compute_80a/sm_80a,
# ppu1v5 -> compute_89/sm_89. A wrong arch compiles fine but the PPU runtime
# silently drops every kernel launch (hggcErrorInvalidValue / no-op).
set -e
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT="${ROOT}/vllm_fl/_m3_fused_C.so"
CUDA_HOME="${CUDA_HOME:-/usr/local/PPU_SDK/CUDA_SDK}"

NVCC="$CUDA_HOME/bin/nvcc"
if [ ! -x "$NVCC" ]; then
  NVCC="$(command -v nvcc)"
fi
echo "nvcc: $NVCC ($($NVCC --version | tail -1))"

INCLUDES=$(python - <<'PY'
import sysconfig
import torch.utils.cpp_extension  # noqa: F401 (populate attribute)
incs = [sysconfig.get_paths()['include']]
for p in torch.utils.cpp_extension.include_paths():
    incs.append(p)
print(' '.join(f'-I{p}' for p in incs))
PY
)

LIBDIR=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")

# Match deep_gemm's is_ppu1v5_device() heuristic.
ARCH=$(python - <<'PY'
import torch
name = torch.cuda.get_device_name(0).lower()
ppu15 = any(k in name for k in ('ppu1v5', 'ppu-1v5', 'zw80', 'ppu0015'))
print('compute_89,sm_89' if ppu15 else 'compute_80a,sm_80a')
PY
)
echo "arch: $ARCH"

"$NVCC" -shared -Xcompiler -fPIC -O3 -std=c++17 \
  -ppu-simt-branch=false -ppu-patch-fence-ppu=false -wno-loop-miss-transform \
  -ppu-cg-to-kp1=true -ppu-fix-uninit=true \
  -gencode arch=$ARCH \
  --expt-relaxed-constexpr --expt-extended-lambda \
  "--ptxas-options=--register-usage-level=10" \
  "--diag-suppress=39,174,177,940" \
  "-D_GLIBCXX_USE_CXX11_ABI=1" \
  -DTORCH_EXTENSION_NAME=_m3_fused_C \
  -DTORCH_API_INCLUDE_EXTENSION_H \
  $INCLUDES \
  "${ROOT}/vllm_fl/csrc/m3_fused/m3_fused_kernel.cu" \
  -L"$LIBDIR" -ltorch -ltorch_cpu -ltorch_python -lc10 -ltorch_cuda -lc10_cuda \
  -o "$OUT"

echo "built: $OUT"
