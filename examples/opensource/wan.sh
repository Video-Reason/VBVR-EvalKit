# WAN 2.1 FLF2V 14B 720P - First-Last Frame to Video generation
uv run examples/generate_videos.py --model wan --task chess maze

# WAN 2.1 FLF2V 14B 720P - First-Last Frame to Video generation (full name)
uv run examples/generate_videos.py --model wan-2.1-flf2v-720p --task chess maze

# WAN 2.2 I2V A14B - Image to Video generation with 14B parameters
uv run examples/generate_videos.py --model wan-2.2-i2v-a14b --task chess maze

# WAN 2.1 I2V 14B 480P - Image to Video generation at 480p resolution
uv run examples/generate_videos.py --model wan-2.1-i2v-480p --task chess maze

# WAN 2.1 I2V 14B 720P - Image to Video generation at 720p resolution
uv run examples/generate_videos.py --model wan-2.1-i2v-720p --task chess maze

# WAN 2.2 TI2V 5B - Text + Image to Video generation with 5B parameters
uv run examples/generate_videos.py --model wan-2.2-ti2v-5b --task chess maze

# WAN 2.1 VACE 14B - Video generation with 14B parameters
uv run examples/generate_videos.py --model wan-2.1-vace-14b --task chess maze

# WAN 2.1 VACE 1.3B - Lightweight video generation with 1.3B parameters
uv run examples/generate_videos.py --model wan-2.1-vace-1.3b --task chess maze