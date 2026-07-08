# TRELLIS.2 Environment Fixes

Environment: conda env `metrologyIR`, Python 3.10, PyTorch 2.6.0+cu124, RTX 3090 Ti (SM 8.6)

---

## 1. Downgrade `transformers` to 5.3.0

**Error:** `AttributeError: 'DINOv3ViTModel' object has no attribute 'layer'`

**Cause:** In `transformers` 5.4.0, `DINOv3ViTModel` was refactored — transformer layers moved from `model.layer` into a nested `DINOv3ViTEncoder` at `model.model.layer`. TRELLIS.2 was written against the pre-5.4 API.

**Fix:**
```bash
pip install "transformers==5.3.0"
```

---

## 2. Build `flash-attn` from source

**Error:** `ImportError: undefined symbol: _ZN3c105ErrorC2ENS_14SourceLocationENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE`

**Cause:** All prebuilt `flash-attn` wheels on PyPI and GitHub have a C++ ABI mismatch with the pip-installed `torch 2.6.0+cu124`. Without `FLASH_ATTENTION_FORCE_BUILD=TRUE`, `flash-attn`'s `setup.py` silently downloads one of these broken prebuilt wheels instead of compiling.

**Fix:** Force a true source build against the local torch installation:
```bash
conda activate metrologyIR
FLASH_ATTENTION_FORCE_BUILD=TRUE TORCH_CUDA_ARCH_LIST="8.6" pip install flash-attn --no-build-isolation
```

> Takes ~20 min. `TORCH_CUDA_ARCH_LIST="8.6"` limits compilation to RTX 3090 Ti only to avoid unnecessary kernel variants.

---

## 3. Clean up accidental CUDA 13 packages

**Error:** `RuntimeError: cuDNN error: CUDNN_STATUS_NOT_INITIALIZED`

**Cause:** During `flash-attn` installation, pip accidentally pulled in `torch 2.11.0` as a dependency (from the prebuilt wheel metadata), which dragged in CUDA 13.x packages. After restoring `torch 2.6.0`, these CUDA 13 packages remained and conflicted with CUDA 12.4.

**Fix:** Remove the CUDA 13 packages and restore the corrupted CUDA 12 ones:
```bash
pip uninstall -y \
  cuda-bindings cuda-pathfinder cuda-toolkit \
  nvidia-cublas nvidia-cuda-cupti nvidia-cuda-nvrtc nvidia-cuda-runtime \
  nvidia-cudnn-cu13 nvidia-cufft nvidia-cufile nvidia-curand \
  nvidia-cusolver nvidia-cusparse nvidia-cusparselt-cu13 \
  nvidia-nccl-cu13 nvidia-nvjitlink nvidia-nvshmem-cu13 nvidia-nvtx

pip install --force-reinstall nvidia-cudnn-cu12==9.1.0.70 nvidia-nccl-cu12
```

---

## Attention backend (optional)

The attention backend is configurable via environment variable. Useful if `flash-attn` is unavailable:

```bash
export ATTN_BACKEND=sdpa   # use PyTorch native attention (no extra deps)
# options: flash_attn | sdpa | xformers | naive
```

To persist across sessions:
```bash
conda env config vars set ATTN_BACKEND=sdpa -n metrologyIR
```

---

## Verification

```bash
conda activate metrologyIR
python -c "
import torch, flash_attn
print('torch:', torch.__version__)
print('cuda:', torch.cuda.is_available())
print('cudnn:', torch.backends.cudnn.version())
print('flash_attn:', flash_attn.__version__)
"
```

Expected output:
```
torch: 2.6.0+cu124
cuda: True
cudnn: 90100
flash_attn: 2.8.3
```
