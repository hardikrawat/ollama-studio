# Local Intelligence Workstation (LIW)

A high-density, professional local AI engineering workstation and multi-modality client interface designed for self-hosted artificial intelligence.

**Local Intelligence Workstation** coordinates multiple independent local inference engines into a single, cohesive technical operating environment—supporting autoregressive text generation, vision-language models, diffusion image generation, speech-to-text audio transcription, and neural text-to-speech synthesis without cloud dependencies or subscription APIs.

---

## Architecture: Multi-Backend Orchestration

Local Intelligence Workstation operates as an intelligent web gateway (`server.py`) serving a zero-build client application that communicates concurrently with local daemons:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   Local Intelligence Workstation                      │
│            (Single-Page Engineering Canvas · HTML5 / CSS3 / ES6)       │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │ HTTP / REST / Streams
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      server.py · API Gateway                           │
│     - Multi-Backend Status & Latency Monitoring                        │
│     - Model Hub Scrapers (Ollama · Hugging Face · CivitAI)             │
│     - Reverse Proxy with Split GGUF Shard Verification                 │
└───────┬──────────────┬──────────────┬──────────────┬───────────────────┘
        │              │              │              │
        ▼              ▼              ▼              ▼
┌──────────────┐┌──────────────┐┌──────────────┐┌───────────────────────┐
│    Ollama    ││   SD WebUI   ││   Whisper    ││       Piper TTS       │
│ (Port 11434) ││ (Port 7860)  ││ (Port 8178)  ││      (Port 5000)      │
│  Text & VLM  ││ Diffusion AI ││ Speech-to-Txt││     Text-to-Speech    │
└──────────────┘└──────────────┘└──────────────┘└───────────────────────┘
```

### Supported Modalities & Inference Engines

| Modality | Default Engine | Port | Capabilities |
|:---|:---|:---|:---|
| **💬 Text LLM & Vision** | **Ollama** (`llama.cpp`) | `11434` | Autoregressive reasoning, thinking scratchpad, image attachments |
| **🎨 Diffusion Images** | **SD WebUI** / **ComfyUI** | `7860` / `8188` | Text-to-Image, aspect ratio presets, CFG, samplers, seeds |
| **🎙️ Speech-to-Text** | **whisper.cpp** / **faster-whisper** | `8178` | Live microphone capture, audio upload, timestamped transcripts |
| **🔊 Text-to-Speech** | **Piper TTS** / Web Speech API | `5000` | Real-time neural voice synthesis, inline assistant voice playback |
| **🌐 Model Hub** | **Multi-Source Gateway** | Gateway | Ollama Registry, HuggingFace Hub (GGUF), CivitAI (Checkpoints) |

---

## Quick Start

### 1. Launch the Workstation

Run the launcher script corresponding to your operating system:

```bash
# macOS / Linux
./start.sh

# macOS Finder One-Click
Double-click start.command

# Windows
start.bat

# Direct Python Execution
python3 server.py 3300
```

Open your browser to `http://localhost:3300`.

### 2. Preflight Check & Setup Guide

Open `http://localhost:3300/setup` (or click **SETUP ↗** in the bottom telemetry bar) to run automated connectivity checks and access step-by-step installation instructions for every backend across macOS, Linux, Windows, and Docker.

### 3. Optional: Turnkey Docker Deployment

To run all inference backends containerized:

```bash
docker compose up -d
```

---

## Workstation Studios

### 💬 Chat & Reasoning Studio
- **High-Density Technical Canvas**: Monospace document ledger with hairline rules, sharp 0px corners, and archival parchment theme.
- **Thinking / Chain-of-Thought Scratchpad**: Isolates `<think>` blocks emitted by reasoning architectures (DeepSeek R1, Qwen 2.5) with a live generation timer, token speed counter, and copy controls.
- **Vision & Multimodal Attachments**: Paste or attach images, documents, and code files directly into the prompt composer.
- **`/imagine <prompt>` Slash Command**: Type `/imagine cinematic view of mountains` directly in the chat composer to synthesize diffusion images and view them inline with download and lightbox controls.
- **🎙️ Composer Voice Dictation**: Click the microphone button in the composer bar to dictate speech directly into the prompt box via Whisper ASR.
- **🔊 Voice Playback (`Listen`)**: Click `Read Aloud` on any assistant response to stream speech via Piper TTS or browser Web Speech.

### 🎨 Image Generation Studio
- **Diffusion Inspector**: Configure checkpoint models, synthesis prompts, negative prompts, sampling methods (Euler a, DPM++ 2M Karras, DDIM), sampling steps, and CFG scale.
- **Aspect Ratio Presets**: One-click switching between `1:1 (512x512)`, `1:1 HD (1024x1024)`, `3:4 Portrait (768x1024)`, and `16:9 Landscape (1024x576)`.
- **Live Progress & Sampler Feedback**: Real-time progress percentage bar and step ETA indicators.
- **Stage Canvas**: High-resolution image canvas with pan, zoom lightbox modal, PNG download, and base64 copying.
- **Filmstrip History**: Local history gallery of recent generations; click any thumbnail to review parameters or inject into chat.

### 🎙️ Audio Lab
- **Microphone Capture**: Browser `MediaRecorder` audio recording with live countdown timer and hairline VU level meter.
- **Audio File Upload**: Drag-and-drop or browse `.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`, or `.webm` files.
- **Audio Scrubber Player**: Integrated waveform scrubber for playback and review.
- **Whisper ASR Transcription**: Single-click transcription with multi-language detection, formatted transcript editing, clipboard copy, and `.txt` export.
- **Neural TTS Synthesizer**: Text input card to synthesize neural speech using Piper voices or system browser voices with speed adjustment.

### 📱 Mobile & Responsive Experience
- **Adaptive Command Grid**: Balanced 2x2 mobile header grid maintaining strict 28px button heights, sharp monochrome borders, and readable typography.
- **Flush Action Composer**: Single-row mobile prompt bar with fixed 36px flush alignment across the prompt input, microphone dictation, attachment picker, and send button.
- **Segmented Toolbars**: Horizontally scrollable single-row tab bars for Model Hub categories and Settings modals, preventing vertical wrapping on small screens.
- **Non-Collapsing Card Layout**: Isolated workspace panes and Model Hub cards preventing layout shifting, element stacking, or touch target collisions on mobile viewports.

---

## Universal Model Hub

Access the Model Hub via the **Hub ↓** button in the command header:

1. **Ollama Registry**:
   - Live scraping and search across all models on `ollama.com`.
   - Real-time tag resolution and one-click pulling into Ollama.
2. **Hugging Face Hub**:
   - Search tens of thousands of community GGUF models across all official pipeline categories (`text-generation`, `image-text-to-text`, `text-to-image`, `text-to-speech`, `automatic-speech-recognition`, etc.).
   - **Modality-Aware Action Architecture**: Distinguishes autoregressive LLMs/VLMs from text-to-image diffusion models (such as FLUX.1, SDXL, and SD 1.5/3).
     - **Autoregressive Models (LLMs / VLMs)**: Provide 1-click `↓ Pull to Ollama`, dynamic quantization tag selectors, and custom Ollama Modelfile generation.
     - **Diffusion Models (FLUX / Stable Diffusion)**: Clearly labeled with `⚡ ComfyUI / Forge (Not Ollama)` notices. Provides `⬇ ComfyUI / Forge GGUF` direct browser download links, copyable direct URLs, and copyable `wget` terminal commands targeting `ComfyUI/models/diffusion_models/` or `models/unet/`. Intercepts invalid pull attempts with technical guidance.
   - **Split GGUF Shard Detection**: Automatically flags multi-part shards (`-00001-of-00002.gguf`) with warning badges and prevents failed pulls by recommending single-file variants.
   - High-speed mirror selector (`huggingface.co` vs `hf-mirror.com`) and token authentication.
   - Modelfile Builder: Generate custom Ollama Modelfiles with one click for supported LLMs.
3. **CivitAI Diffusion Hub**:
   - Search Stable Diffusion checkpoints, LoRAs, and VAEs.
   - View base model tags (`Pony`, `SDXL 1.0`, `SD 1.5`, `FLUX.1 D`), download statistics, and direct safetensors download links.

---

## Configuration & Settings

Open **Settings (⚙)** to manage hardware parameters and inference backends:

- **Backends & Engines**: View live connectivity and ping latency for Ollama, SD WebUI, ComfyUI, Whisper ASR, and Piper TTS. Save and persist custom URLs to `backends.json`.
- **Hardware & Context**: Adjust context window size (`num_ctx`) from 2,048 up to 262,144 tokens (256K). Configure temperature and sampling parameters.
- **System Persona**: Choose academic, software engineering, concise, or Socratic system prompt directives.
- **Voice & Speech**: Enable auto-speech playback for assistant messages, choose preferred TTS voices, and set speech rate.

---

## Privacy & Security

Local Intelligence Workstation operates **entirely on your local machine**:
- No external telemetry, tracking, or cloud phone-home scripts.
- Sessions and configurations persist strictly in your browser's local storage and `backends.json`.
- Fully functional in air-gapped and offline environments.
