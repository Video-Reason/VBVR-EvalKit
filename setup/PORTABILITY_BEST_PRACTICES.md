# Installation Portability Best Practices

This document outlines strategies to ensure VMEvalKit setup scripts work reliably across different machines, filesystems, and environments.

## Executive Summary

**Primary Changes Made:**
1. ✅ Added `--no-cache-dir` to all `flash-attn` installations
2. ✅ Added `--no-cache-dir` to all `git+` pip installations
3. ✅ Documented best practices for future setup scripts

## Core Portability Issues and Solutions

### 1. Cross-Device Link Errors (Errno 18)

**Problem:**
- Occurs when pip cache and temp directories are on different filesystems
- Common in HPC clusters, Docker containers, and systems with complex mount configurations
- Particularly affects large packages like flash-attention, PyTorch

**Solution:**
```bash
# Use --no-cache-dir for large packages and git installs
pip install -q --no-cache-dir flash-attn==2.7.0.post2 --no-build-isolation
pip install -q --no-cache-dir git+https://github.com/user/repo.git@version
pip install -q --no-cache-dir torch==2.4.0+cu121 --index-url https://download.pytorch.org/whl/cu121
```

**When to use:**
- ✅ All flash-attention installations
- ✅ All PyTorch/torchvision installations (large wheels)
- ✅ All git-based pip installs
- ✅ Build-from-source packages (--no-build-isolation)
- ⚠️  Optional for small standard packages (adds minor overhead)

### 2. Dependency Version Pinning

**Problem:**
- Unpinned versions cause breakage when upstream releases incompatible updates
- Different machines may install different versions at different times

**Solution:**
```bash
# ✅ GOOD: Exact versions
pip install -q diffusers==0.31.0
pip install -q "peft>=0.12.0,<0.13.0"  # Range with upper bound

# ❌ BAD: Unpinned or unbounded
pip install -q diffusers
pip install -q "peft>=0.12.0"
```

**Per workspace rules: Always use exact versions in setup scripts**

### 3. PyTorch CUDA Compatibility

**Problem:**
- pip may install CPU-only wheels if not explicitly specified
- Different CUDA versions across machines

**Solution:**
```bash
# Always specify CUDA version and verify
pip install -q --no-cache-dir \
    torch==2.4.0+cu121 \
    torchvision==0.19.0+cu121 \
    torchaudio==2.4.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Verify compiled ops are present
python -c "import torchvision; assert hasattr(torchvision.ops, 'nms')"
```

### 4. Virtual Environment Isolation

**Problem:**
- Package conflicts between different models
- System site-packages interfering

**Solution:**
```bash
# Always use model-specific venvs (already implemented in share.sh)
create_model_venv "$MODEL"
activate_model_venv "$MODEL"
# ... install packages ...
deactivate
```

### 5. HuggingFace Hub Downloads

**Problem:**
- Network interruptions during large downloads
- Symlink issues on some filesystems

**Solution:**
```bash
# Always use --resume-download and --local-dir-use-symlinks False
huggingface-cli download tencent/HunyuanVideo-I2V \
    --local-dir "${TARGET_DIR}" \
    --local-dir-use-symlinks False \
    --resume-download
```

### 6. Build Dependencies

**Problem:**
- Missing build tools on fresh systems
- flash-attention requires specific build environment

**Solution:**
```bash
# Install build dependencies BEFORE attempting builds
pip install -q packaging ninja psutil wheel setuptools

# Then install packages requiring compilation
pip install -q --no-cache-dir flash-attn==2.7.0.post2 --no-build-isolation
```

## Setup Script Checklist

When creating new model setup scripts, verify:

- [ ] Uses `set -euo pipefail` at the top
- [ ] Sources `share.sh` for common functions
- [ ] Creates and activates model-specific venv
- [ ] All pip installs use **exact versions** (e.g., `==X.Y.Z`)
- [ ] Large packages use `--no-cache-dir`
- [ ] Git-based installs use `--no-cache-dir`
- [ ] PyTorch installs specify CUDA version
- [ ] PyTorch installs are verified (torchvision ops check)
- [ ] HuggingFace downloads use `--resume-download` and `--local-dir-use-symlinks False`
- [ ] Deactivates venv at the end
- [ ] Prints success message

## Testing Portability

To test setup scripts on different environments:

```bash
# Test on fresh environment
docker run --rm -it --gpus all nvidia/cuda:12.1.0-devel-ubuntu22.04 bash

# Test with limited disk space
# Test with /tmp on different filesystem than /home
# Test with read-only pip cache
export PIP_CACHE_DIR=/read-only-path  # Should still work with --no-cache-dir
```

## Performance Considerations

**--no-cache-dir Impact:**
- First install: No performance difference (nothing to cache)
- Subsequent installs: Downloads packages again (~5-30s extra per package)
- Trade-off: Reliability > Speed for setup scripts
- Production deploys: Cache is rarely useful anyway (one-time setup)

**When to skip --no-cache-dir:**
- Small, stable packages (<10MB)
- Pure Python packages (no compilation)
- Local development iteration (can use pip cache locally)

## Common Pitfalls to Avoid

### ❌ Don't Do This:
```bash
# Unpinned versions
pip install torch torchvision

# No CUDA specification
pip install torch==2.4.0

# Missing --no-cache-dir on large packages
pip install flash-attn==2.7.0.post2 --no-build-isolation

# Using pip cache in production
pip install -r requirements.txt  # Cache issues possible

# Assuming /tmp is on same filesystem
TMPDIR=/tmp pip install ...  # May fail
```

### ✅ Do This Instead:
```bash
# Exact versions with CUDA
pip install -q --no-cache-dir \
    torch==2.4.0+cu121 \
    torchvision==0.19.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Safe flash-attention install
pip install -q --no-cache-dir flash-attn==2.7.0.post2 --no-build-isolation

# Safe git install
pip install -q --no-cache-dir git+https://github.com/user/repo.git@v1.2.3

# Portable requirements
pip install -q -r requirements.txt --no-cache-dir  # If needed
```

## References

- [pip caching documentation](https://pip.pypa.io/en/stable/topics/caching/)
- [PyTorch installation guide](https://pytorch.org/get-started/locally/)
- [HuggingFace Hub CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)
- [flash-attention installation](https://github.com/Dao-AILab/flash-attention)

## Affected Files (Current Implementation)

```
✅ setup/models/morphic-frames-to-video/setup.sh
   - Added --no-cache-dir to flash-attn==2.7.0.post2

✅ setup/models/hunyuan-video-i2v/setup.sh
   - Added --no-cache-dir to flash-attention@v2.6.3
   - Added --no-cache-dir to CLIP git install
   - Already using --no-cache-dir for PyTorch (good!)
```

## Future Improvements

1. **Parallel installs:** Consider using `pip install package1 package2 ...` for unrelated deps
2. **Retry logic:** Add retry mechanism for network-dependent operations
3. **Offline mode:** Document how to create offline install packages
4. **CI/CD testing:** Add tests that run installs in Docker containers with different filesystem configs
5. **Cache strategy:** For development, document how to safely use pip cache locally

---

**Last Updated:** 2025-12-29
**Maintainer:** VMEvalKit Team

