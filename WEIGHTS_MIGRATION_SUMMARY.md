# Model Weights Centralization - Migration Summary

## What Changed

All model weights in VMEvalKit have been reorganized into a centralized `weights/` directory at the project root. This provides better organization, cleaner git tracking, and easier weight management.

## New Directory Structure

```
VMEvalKit/
├── weights/                          # ✨ NEW: Centralized weights directory
│   ├── dynamicrafter/
│   │   ├── dynamicrafter_256_v1/
│   │   ├── dynamicrafter_512_v1/
│   │   └── dynamicrafter_1024_v1/
│   ├── videocrafter/
│   │   └── base_512_v2/
│   ├── wan/
│   │   └── Wan2.2-I2V-A14B/
│   ├── morphic/
│   ├── hunyuan/
│   ├── ltx-video/
│   └── svd/
├── submodules/                       # Model source code only (no weights)
├── envs/                             # Virtual environments
└── ...
```

## Files Modified

### Core Configuration
- ✅ `setup/lib/common.sh` - Added `WEIGHTS_DIR` variable and updated checkpoint paths
- ✅ `env.template` - Updated default weight paths for Morphic/Wan models

### Model Inference Code
- ✅ `vmevalkit/models/dynamicrafter_inference.py` - Updated checkpoint paths
- ✅ `vmevalkit/models/videocrafter_inference.py` - Updated checkpoint paths
- ✅ `vmevalkit/models/morphic_inference.py` - Updated default weight paths

### Git Configuration
- ✅ `.gitignore` - Added `weights/` directory and weight file patterns

### Documentation
- ✅ `setup/README.md` - Updated with new weights structure
- ✅ `setup/SETUP.md` - Updated cleanup commands
- ✅ `README.md` - Added reference to weights documentation
- ✅ `docs/WEIGHTS_STRUCTURE.md` - NEW: Comprehensive weights documentation
- ✅ `weights/README.md` - NEW: Quick reference in weights directory

### Migration Tools
- ✅ `setup/migrate_weights.sh` - NEW: Script to migrate existing weights

## What You Need to Do

### 1. Migrate Existing Weights (If You Have Any)

If you already have model weights downloaded in the old locations:

```bash
cd /home/hokindeng/VMEvalKit
./setup/migrate_weights.sh
```

This will automatically move:
- `Wan2.2-I2V-A14B/` → `weights/wan/Wan2.2-I2V-A14B/`
- `morphic-frames-lora-weights/` → `weights/morphic/`
- `submodules/DynamiCrafter/checkpoints/` → `weights/dynamicrafter/`
- `submodules/VideoCrafter/checkpoints/` → `weights/videocrafter/`

### 2. Update Your .env File (If You Have Custom Paths)

If you have a `.env` file with custom weight paths, update them:

```env
# Old paths (update these)
MORPHIC_WAN2_CKPT_DIR=./Wan2.2-I2V-A14B
MORPHIC_LORA_WEIGHTS_PATH=./morphic-frames-lora-weights/lora_interpolation_high_noise_final.safetensors

# New paths (use these)
MORPHIC_WAN2_CKPT_DIR=./weights/wan/Wan2.2-I2V-A14B
MORPHIC_LORA_WEIGHTS_PATH=./weights/morphic/lora_interpolation_high_noise_final.safetensors
```

### 3. Clean Up Git Status

The models showing up in `git status` will now be ignored:

```bash
# These are now in .gitignore:
# - weights/
# - Wan2.2-I2V-A14B/
# - morphic-frames-lora-weights/
# - *.pth, *.safetensors, *.ckpt, *.bin, *.pt
```

After migration, run:
```bash
git status
```

You should see that model weight directories are no longer listed as untracked files.

### 4. Download New Weights (If Starting Fresh)

If you don't have weights yet, just run the setup:

```bash
./setup/RUN_SETUP.sh
```

All weights will be downloaded to the correct `weights/` directory automatically.

## Benefits of This Change

✅ **Better Organization**: All weights in one place, separated from code  
✅ **Cleaner Git**: Weights properly ignored, no accidental commits  
✅ **Easier Management**: Simple to backup, migrate, or clean up weights  
✅ **Consistent Structure**: All models follow the same pattern  
✅ **Better Documentation**: Clear docs on where everything lives  

## Backwards Compatibility

The code still checks for weights in the new location first, but the old environment variable names are preserved for compatibility. If you have scripts that set custom paths, they will continue to work.

## Need Help?

- **Full Documentation**: [docs/WEIGHTS_STRUCTURE.md](docs/WEIGHTS_STRUCTURE.md)
- **Setup Guide**: [setup/README.md](setup/README.md)
- **Migration Issues**: Run `./setup/migrate_weights.sh` and check the output

## Questions?

If you encounter any issues with the migration:

1. Check that paths in your `.env` file are updated
2. Run `./setup/3a_validate_opensource.sh` to validate setup
3. Check logs in `logs/` directory for any errors
4. See [docs/WEIGHTS_STRUCTURE.md](docs/WEIGHTS_STRUCTURE.md) for troubleshooting

---

**Summary**: All model weights now live in `weights/` directory. Run `./setup/migrate_weights.sh` if you have existing weights, or `./setup/RUN_SETUP.sh` for fresh setup.

