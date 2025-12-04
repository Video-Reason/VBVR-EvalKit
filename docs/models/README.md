
## 🎬 Supported Models

VMEvalKit provides unified access to **40 video generation models** across **11 provider families**:

For commercial APIs, we support Luma Dream Machine, Google Veo, Google Veo 3.1, WaveSpeed WAN 2.1, WaveSpeed WAN 2.2, Runway ML, OpenAI Sora.
For open-source models, we support HunyuanVideo, VideoCrafter, DynamiCrafter, Stable Video Diffusion, Morphic, LTX-Video.

### Commercial APIs (32 models)

| Provider | Models | Key Features | API Required |
|----------|---------|-------------|--------------|
| **Luma Dream Machine** | 2 | `luma-ray-2`, `luma-ray-flash-2` | `LUMA_API_KEY` |
| **Google Veo** | 3 | `veo-2.0-generate`, `veo-3.0-generate`, `veo-3.0-fast-generate` | GCP credentials |
| **Google Veo 3.1** | 4 | Native 1080p, audio generation (via WaveSpeed) | `WAVESPEED_API_KEY` |
| **WaveSpeed WAN 2.1** | 8 | 480p/720p variants with LoRA and ultra-fast options | `WAVESPEED_API_KEY` |
| **WaveSpeed WAN 2.2** | 10 | Enhanced 5B models, improved quality | `WAVESPEED_API_KEY` |
| **Runway ML** | 3 | Gen-3A Turbo, Gen-4 Turbo, Gen-4 Aleph | `RUNWAY_API_SECRET` |
| **OpenAI Sora** | 2 | Sora-2, Sora-2-Pro (4s/8s/12s durations) | `OPENAI_API_KEY` |

### Open-Source Models

| Provider | Models | Key Features | Hardware Requirements |
|----------|---------|-------------|----------------------|
| **HunyuanVideo** | 1 | High-quality 720p I2V | GPU with 24GB+ VRAM |
| **VideoCrafter** | 1 | Text-guided video synthesis | GPU with 16GB+ VRAM |
| **DynamiCrafter** | 3 | 256p/512p/1024p, image animation | GPU with 12-24GB VRAM |
| **Stable Video Diffusion** | 1 | Video generation | GPU with 16GB+ VRAM |
| **Morphic** | 1 | Video generation | GPU with 16GB+ VRAM |
| **LTX-Video** | 1 | Video generation | GPU with 16GB+ VRAM |





