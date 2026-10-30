#!/usr/bin/env python3
"""
Local Intelligence Workstation — Unified Multi-Backend Gateway & Web Server
Serves static UI files and orchestrates multiple local AI inference backends:
  - Text LLMs & VLMs: Ollama (127.0.0.1:11434)
  - Diffusion Image Generation: SD WebUI (127.0.0.1:7860) / ComfyUI (127.0.0.1:8188)
  - Speech-to-Text ASR: whisper.cpp / faster-whisper (127.0.0.1:8178)
  - Text-to-Speech TTS: Piper TTS (127.0.0.1:5000)

Model Hub Backends:
  /hub/ollama/search   – Scrapes ollama.com/search for model listings
  /hub/ollama/tags     – Fetches available tags for a specific ollama model
  /hub/hf/search       – Searches HuggingFace Hub API for GGUF models
  /hub/hf/model        – Fetches detailed model info from HuggingFace with split GGUF detection
  /hub/civitai/search  – Searches CivitAI API for Checkpoints, LoRAs, and diffusion models
  /api/backends        – Live status, latency probing, and configuration of all backends
"""

import sys
import os
import json
import urllib.request
import urllib.error
import urllib.parse
import re
import html
import ssl
import time
import threading
import base64
import subprocess
import collections
import shutil
import socket
import concurrent.futures
import io
import random
from PIL import Image
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

DEFAULT_PORT = 3300
BACKENDS_FILE = "backends.json"

DEFAULT_BACKENDS = {
    "ollama": {
        "id": "ollama",
        "name": "Ollama",
        "type": "llm",
        "url": "http://127.0.0.1:11434",
        "health_path": "/api/tags",
        "enabled": True,
        "description": "Autoregressive LLMs & Vision-Language models (llama.cpp)"
    },
    "sd_webui": {
        "id": "sd_webui",
        "name": "Stable Diffusion WebUI",
        "type": "image",
        "url": "http://127.0.0.1:7860",
        "health_path": "/sdapi/v1/sd-models",
        "enabled": True,
        "description": "Diffusion image synthesis (txt2img & img2img)"
    },
    "comfyui": {
        "id": "comfyui",
        "name": "ComfyUI",
        "type": "image",
        "url": "http://127.0.0.1:8188",
        "health_path": "/system_stats",
        "enabled": True,
        "description": "Node-based generative diffusion workflows"
    },
    "whisper": {
        "id": "whisper",
        "name": "Whisper ASR",
        "type": "asr",
        "url": "http://127.0.0.1:8178",
        "health_path": "/health",
        "enabled": True,
        "description": "Speech-to-Text transcription (whisper.cpp / faster-whisper)"
    },
    "piper": {
        "id": "piper",
        "name": "Piper TTS",
        "type": "tts",
        "url": "http://127.0.0.1:5000",
        "health_path": "/api/voices",
        "enabled": True,
        "description": "Ultra-fast neural local Text-to-Speech"
    }
}

_backends_lock = threading.RLock()
_backends_cache = None


def load_backends_config():
    global _backends_cache
    with _backends_lock:
        if _backends_cache is not None:
            return dict(_backends_cache)
        backends = {k: dict(v) for k, v in DEFAULT_BACKENDS.items()}
        if os.path.exists(BACKENDS_FILE):
            try:
                with open(BACKENDS_FILE, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                    for k, v in saved.items():
                        if k in backends and isinstance(v, dict):
                            backends[k].update(v)
                        elif isinstance(v, dict):
                            backends[k] = v
            except Exception as e:
                print(f"[WARN] Failed to read {BACKENDS_FILE}: {e}")
        _backends_cache = backends
        return dict(backends)


def save_backends_config(updated_dict):
    global _backends_cache
    with _backends_lock:
        current = load_backends_config()
        for k, v in updated_dict.items():
            if k in current and isinstance(v, dict):
                current[k].update(v)
            elif isinstance(v, dict):
                current[k] = v
        _backends_cache = current
        try:
            with open(BACKENDS_FILE, "w", encoding="utf-8") as f:
                json.dump(current, f, indent=2)
        except Exception as e:
            print(f"[ERROR] Failed to write {BACKENDS_FILE}: {e}")
        return dict(current)


def get_backend_url(backend_id):
    cfg = load_backends_config()
    b = cfg.get(backend_id)
    if b and b.get("url"):
        return b["url"].rstrip("/")
    default = DEFAULT_BACKENDS.get(backend_id, {})
    return default.get("url", "").rstrip("/")


def probe_single_backend(backend_info):
    bid = backend_info.get("id")
    url = (backend_info.get("url") or "").rstrip("/")
    hpath = backend_info.get("health_path") or "/"
    if not url:
        return {
            "id": bid,
            "status": "offline",
            "online": False,
            "latency_ms": None,
            "error": "URL not configured"
        }

    target = f"{url}{hpath}"
    start = time.time()
    try:
        req = urllib.request.Request(target, headers={"User-Agent": "LIW-Gateway/1.0"})
        with urllib.request.urlopen(req, timeout=1.8) as resp:
            elapsed = round((time.time() - start) * 1000)
            is_ok = resp.status < 400 or resp.status in (401, 403, 404)
            return {
                "id": bid,
                "status": "online" if is_ok else "standby",
                "online": is_ok,
                "latency_ms": elapsed,
                "http_status": resp.status
            }
    except Exception as e:
        elapsed = round((time.time() - start) * 1000)
        return {
            "id": bid,
            "status": "standby",
            "online": False,
            "latency_ms": None,
            "error": str(e)
        }


def probe_all_backends():
    backends = load_backends_config()
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(probe_single_backend, b): k for k, b in backends.items()}
        for fut in concurrent.futures.as_completed(futures):
            k = futures[fut]
            try:
                probe_res = fut.result()
            except Exception as e:
                probe_res = {"id": k, "status": "standby", "online": False, "error": str(e)}
            backend_copy = dict(backends[k])
            backend_copy.update(probe_res)
            results[k] = backend_copy
    return results


# ─── Local Backend Process & Lifecycle Manager ──────────────────────────────

class BackendProcessManager:
    """Manages local backend service lifecycles (start, stop, restart, install) with live log streaming."""
    PORT_MAP = {
        "ollama": 11434,
        "sd_webui": 7860,
        "comfyui": 8188,
        "whisper": 8178,
        "piper": 5000,
    }

    def __init__(self):
        self.processes = {}  # name -> subprocess.Popen
        self.logs = collections.defaultdict(lambda: collections.deque(maxlen=3000))
        self.states = {}  # name -> "stopped" | "starting" | "online" | "installing" | "error"
        self.threads = {}
        self.lock = threading.Lock()

    def get_known_path(self, name):
        home = os.path.expanduser("~")
        if name == "sd_webui":
            paths = [
                os.path.join(home, "Documents", "stable-diffusion-webui"),
                os.path.join(home, "stable-diffusion-webui"),
                os.path.join(os.getcwd(), "backends", "stable-diffusion-webui"),
            ]
            for p in paths:
                if os.path.isdir(p) and (os.path.isfile(os.path.join(p, "webui.sh")) or os.path.isfile(os.path.join(p, "launch.py"))):
                    return p
            return None
        elif name == "comfyui":
            paths = [
                os.path.join(home, "Documents", "ComfyUI"),
                os.path.join(home, "ComfyUI"),
                os.path.join(os.getcwd(), "backends", "ComfyUI"),
            ]
            for p in paths:
                if os.path.isdir(p) and os.path.isfile(os.path.join(p, "main.py")):
                    return p
            return None
        elif name == "ollama":
            return shutil.which("ollama")
        return None

    def _is_port_open(self, port):
        if not port:
            return False
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.35)
                return s.connect_ex(('127.0.0.1', port)) == 0
        except Exception:
            return False

    def get_service_status(self, name):
        port = self.PORT_MAP.get(name)
        is_open = self._is_port_open(port)
        path = self.get_known_path(name)
        installed = path is not None

        proc = self.processes.get(name)
        is_managed_running = (proc is not None and proc.poll() is None)
        pid = proc.pid if is_managed_running else None

        # Check if external process is running on this port or starting up
        external_pid = None
        if not is_managed_running and is_open and port:
            try:
                res = subprocess.check_output(f"lsof -ti :{port}", shell=True, text=True).strip()
                if res:
                    first_pid = int(res.split()[0])
                    proc_comm = subprocess.check_output(f"ps -p {first_pid} -o comm=", shell=True, text=True).strip().lower()
                    if name == "piper" and ("controlcenter" in proc_comm or "airplay" in proc_comm):
                        is_open = False
                    else:
                        external_pid = first_pid
            except Exception:
                pass
        elif not is_managed_running and not is_open:
            try:
                if name == "sd_webui":
                    res = subprocess.check_output("pgrep -f 'launch.py|webui.py'", shell=True, text=True).strip()
                    if res:
                        external_pid = int(res.split()[0])
                elif name == "comfyui":
                    res = subprocess.check_output("pgrep -f 'main.py.*8188'", shell=True, text=True).strip()
                    if res:
                        external_pid = int(res.split()[0])
                elif name == "ollama":
                    res = subprocess.check_output("pgrep -f 'ollama serve'", shell=True, text=True).strip()
                    if res:
                        external_pid = int(res.split()[0])
            except Exception:
                pass

        with self.lock:
            current_state = self.states.get(name)
            if is_open:
                state = "online"
            elif current_state == "installing":
                state = "installing"
            elif (current_state == "starting" and is_managed_running) or (external_pid and not is_open):
                state = "starting"
            elif is_managed_running:
                state = "running"
            elif proc and proc.poll() is not None:
                state = "error" if proc.poll() != 0 else "stopped"
            else:
                state = "stopped"
            self.states[name] = state

        return {
            "name": name,
            "installed": installed,
            "path": path,
            "state": state,
            "pid": pid or external_pid,
            "port": port,
            "is_port_open": is_open,
            "log_lines": len(self.logs[name])
        }

    def _append_log(self, name, text):
        ts = time.strftime("%H:%M:%S")
        with self.lock:
            self.logs[name].append(f"[{ts}] {text.rstrip()}")

    def _reader_thread(self, name, proc):
        for line in iter(proc.stdout.readline, ''):
            if not line:
                break
            self._append_log(name, line)
        proc.stdout.close()
        code = proc.wait()
        self._append_log(name, f"--- Process exited with code {code} ---")
        with self.lock:
            if not self._is_port_open(self.PORT_MAP.get(name)):
                self.states[name] = "error" if code != 0 else "stopped"

    def start_service(self, name):
        path = self.get_known_path(name)
        if not path and name != "ollama":
            return {"success": False, "error": f"{name} is not installed locally. Run installation first."}

        # Check if already open
        port = self.PORT_MAP.get(name)
        if self._is_port_open(port):
            return {"success": True, "state": "online", "message": f"{name} is already online on port {port}."}

        self._append_log(name, f"=== Starting {name} ===")
        with self.lock:
            self.states[name] = "starting"

        cmd = []
        cwd = path
        env = os.environ.copy()
        if name == "sd_webui":
            # Set required Apple Silicon & mirror args
            env["STABLE_DIFFUSION_REPO"] = "https://github.com/w-e-w/stablediffusion.git"
            env["COMMANDLINE_ARGS"] = "--api --skip-torch-cuda-test --no-half"
            sh_path = os.path.join(path, "webui.sh")
            if os.path.exists(sh_path):
                cmd = ["bash", sh_path, "--api", "--skip-torch-cuda-test", "--no-half"]
            else:
                venv_py = os.path.join(path, "venv", "bin", "python")
                py = venv_py if os.path.exists(venv_py) else "python3"
                cmd = [py, "launch.py", "--api", "--skip-torch-cuda-test", "--no-half"]
        elif name == "comfyui":
            home = os.path.expanduser("~")
            candidates = [
                os.path.join(path, "venv", "bin", "python"),
                os.path.join(home, "Documents", "stable-diffusion-webui", "venv", "bin", "python"),
                shutil.which("python3") or "python3"
            ]
            py = next((p for p in candidates if p and os.path.isfile(p)), "python3")
            cmd = [py, "main.py", "--listen", "127.0.0.1", "--port", "8188", "--disable-auto-launch"]
        elif name == "ollama":
            cmd = ["ollama", "serve"]
            cwd = None

        try:
            proc = subprocess.Popen(
                cmd,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env
            )
            self.processes[name] = proc
            t = threading.Thread(target=self._reader_thread, args=(name, proc), daemon=True)
            self.threads[name] = t
            t.start()
            self._append_log(name, f"Launched PID {proc.pid} with command: {' '.join(cmd)}")
            return {"success": True, "pid": proc.pid, "state": "starting"}
        except Exception as e:
            self._append_log(name, f"[ERROR] Failed to start: {e}")
            with self.lock:
                self.states[name] = "error"
            return {"success": False, "error": str(e)}

    def stop_service(self, name):
        self._append_log(name, f"=== Stopping {name} ===")
        proc = self.processes.get(name)
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        # Kill by port if still occupied
        port = self.PORT_MAP.get(name)
        if port:
            try:
                subprocess.run(f"lsof -ti :{port} | xargs kill -9 2>/dev/null", shell=True, timeout=2)
            except Exception:
                pass
        with self.lock:
            self.states[name] = "stopped"
            self.processes[name] = None
        self._append_log(name, f"--- {name} stopped successfully ---")
        return {"success": True, "state": "stopped"}

    def install_service(self, name):
        home = os.path.expanduser("~")
        if name == "sd_webui":
            target_dir = os.path.join(home, "Documents", "stable-diffusion-webui")
        elif name == "comfyui":
            target_dir = os.path.join(home, "Documents", "ComfyUI")
        else:
            return {"success": False, "error": f"Auto-install not configured for {name}"}

        with self.lock:
            if self.states.get(name) == "installing":
                return {"success": False, "error": "Installation already in progress"}
            self.states[name] = "installing"

        def _do_install():
            self._append_log(name, f"=== Auto-Installing {name} to {target_dir} ===")
            try:
                if not os.path.exists(target_dir):
                    if name == "comfyui":
                        repo_url = "https://github.com/comfyanonymous/ComfyUI.git"
                    else:
                        repo_url = "https://github.com/AUTOMATIC1111/stable-diffusion-webui.git"
                    self._append_log(name, f"> git clone {repo_url} {target_dir}")
                    subprocess.run(["git", "clone", repo_url, target_dir], check=True, timeout=180)

                if name == "comfyui":
                    extra_paths_file = os.path.join(target_dir, "extra_model_paths.yaml")
                    sd_dir = os.path.join(home, "Documents", "stable-diffusion-webui")
                    if not os.path.exists(extra_paths_file) and os.path.exists(sd_dir):
                        with open(extra_paths_file, "w") as f:
                            f.write(f"a1111:\n    base_path: {sd_dir}\n    checkpoints: models/Stable-diffusion\n    vae: models/VAE\n    loras: models/Lora\n")
                        self._append_log(name, f"Linked shared models from {sd_dir}")
                elif name == "sd_webui":
                    # Apply mirror and Apple Silicon flags in webui-user.sh
                    webui_user = os.path.join(target_dir, "webui-user.sh")
                    if os.path.exists(webui_user):
                        with open(webui_user, "a") as f:
                            f.write('\nexport STABLE_DIFFUSION_REPO="https://github.com/w-e-w/stablediffusion.git"\nexport COMMANDLINE_ARGS="--api --skip-torch-cuda-test --no-half"\n')
                self._append_log(name, "=== Installation setup complete. Launching service... ===")
                self.start_service(name)
            except Exception as e:
                self._append_log(name, f"[ERROR] Install failed: {e}")
                with self.lock:
                    self.states[name] = "error"

        t = threading.Thread(target=_do_install, daemon=True)
        t.start()
        return {"success": True, "state": "installing", "message": "Installation started in background"}

    def get_logs(self, name, since=0):
        with self.lock:
            all_logs = list(self.logs[name])
            if not all_logs and name == "sd_webui":
                import glob
                task_dir = os.path.expanduser("~/.gemini/antigravity/brain")
                try:
                    candidates = glob.glob(f"{task_dir}/*/.system_generated/tasks/task-*.log")
                    if candidates:
                        candidates.sort(key=os.path.getmtime, reverse=True)
                        for c in candidates[:3]:
                            with open(c, "r", encoding="utf-8", errors="ignore") as f:
                                head = f.read(2048)
                                if "launch.py" in head or "stable-diffusion" in head or "Downloading:" in head:
                                    f.seek(0)
                                    lines = f.readlines()
                                    all_logs = [line.strip() for line in lines[-150:] if line.strip()]
                                    break
                except Exception:
                    pass

            progress_summary = None
            for line in reversed(all_logs):
                if "%|" in line and ("/3.97G" in line or "B/s" in line):
                    parts = line.split("|")
                    if len(parts) >= 3:
                        pct = parts[0].strip()
                        speed_part = parts[2].split("[")[0].strip() if "[" in parts[2] else parts[2].strip()
                        progress_summary = f"Downloading SD 1.5 Checkpoint: {pct} ({speed_part})"
                        break

            slice_logs = all_logs[since:]
            return {
                "logs": slice_logs,
                "total": len(all_logs),
                "next_since": len(all_logs),
                "progress_summary": progress_summary,
                "status": self.states.get(name, "stopped")
            }


process_manager = BackendProcessManager()



# Cache with TTL
_cache = {}
_cache_lock = threading.Lock()
CACHE_TTL = 300  # 5 minutes

# Image Generation Background Job & Real-time Telemetry State
_image_gen_lock = threading.Lock()
_image_gen_state = {
    "active": False,
    "job_id": None,
    "stage": "idle",
    "prompt": "",
    "params": {},
    "sampling_step": 0,
    "sampling_steps": 25,
    "progress": 0.0,
    "eta_relative": 0.0,
    "preview_image": None,
    "started_at": 0,
    "completed_at": 0,
    "result_item": None,
    "last_error": None
}


def _run_image_generation_worker(job_id, payload, sd_url):
    try:
        is_img2img = bool(payload.get("init_images"))
        target = f"{sd_url}/sdapi/v1/img2img" if is_img2img else f"{sd_url}/sdapi/v1/txt2img"
        req = urllib.request.Request(
            target,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": "LIW-Gateway/1.0"},
            method="POST"
        )
        # 1800s (30m) timeout to accommodate high-res (768x1024 / 1024x1024) and high step diffusion on Apple Silicon / CPU
        with urllib.request.urlopen(req, timeout=1800) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            images = data.get("images", [])
            if not images:
                raise RuntimeError("No images returned by diffusion backend")

            b64 = images[0]
            with _image_gen_lock:
                start_t = _image_gen_state.get("started_at") or time.time()
            elapsed = round(max(time.time() - start_t, 0.1), 1)

            result_item = {
                "id": job_id,
                "prompt": payload.get("prompt", ""),
                "image": b64,
                "parameters": payload,
                "info": data.get("info", ""),
                "elapsed": elapsed,
                "timestamp": int(time.time() * 1000)
            }

            with _image_gen_lock:
                if _image_gen_state.get("job_id") == job_id:
                    _image_gen_state["active"] = False
                    _image_gen_state["stage"] = "complete"
                    _image_gen_state["progress"] = 1.0
                    _image_gen_state["preview_image"] = None
                    _image_gen_state["completed_at"] = time.time()
                    _image_gen_state["result_item"] = result_item
    except Exception as e:
        with _image_gen_lock:
            if _image_gen_state.get("job_id") == job_id:
                if _image_gen_state.get("stage") != "interrupted":
                    _image_gen_state["active"] = False
                    _image_gen_state["stage"] = "error"
                    _image_gen_state["last_error"] = str(e)


# Chat Streaming Session Buffer State (resilient to client disconnects)
_chat_stream_lock = threading.Lock()
_chat_stream_buffers = {}  # stream_id -> { "chunks": [bytes], "done": bool, "error": str, "created_at": float }


def _cleanup_old_chat_streams():
    now = time.time()
    with _chat_stream_lock:
        to_del = [sid for sid, s in list(_chat_stream_buffers.items()) if now - s.get("created_at", 0) > 900]
        for sid in to_del:
            del _chat_stream_buffers[sid]


def _run_ollama_chat_stream_worker(stream_id, req):
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            while True:
                line = resp.readline()
                if not line:
                    break
                with _chat_stream_lock:
                    buf = _chat_stream_buffers.get(stream_id)
                    if buf:
                        buf["chunks"].append(line)
        with _chat_stream_lock:
            buf = _chat_stream_buffers.get(stream_id)
            if buf:
                buf["done"] = True
    except Exception as e:
        err_msg = str(e)
        if isinstance(e, urllib.error.HTTPError):
            try:
                err_detail = e.read().decode("utf-8", errors="replace").strip()
                if err_detail:
                    err_msg = f"{e.code} {e.reason}: {err_detail}"
            except Exception:
                pass
        err_chunk = json.dumps({"error": f"Ollama error: {err_msg}"}).encode("utf-8") + b"\n"
        with _chat_stream_lock:
            buf = _chat_stream_buffers.get(stream_id)
            if buf:
                buf["chunks"].append(err_chunk)
                buf["done"] = True
                buf["error"] = err_msg


# Video Generation Real-time Telemetry State
_video_gen_lock = threading.Lock()
_video_gen_state = {
    "active": False,
    "job_id": None,
    "stage": "idle",
    "frame_current": 0,
    "frame_total": 0,
    "steps_per_frame": 15,
    "fps": 12,
    "model": "svd-xt",
    "prompt": "",
    "started_at": 0,
    "completed_at": 0,
    "result_item": None,
    "last_error": None
}


def _run_video_generation_worker(job_id, prompt, neg_prompt, num_gen_frames, steps, cfg, width, height, seed, fps, video_model, sd_url):
    try:
        frames_b64 = []
        for i in range(num_gen_frames):
            with _video_gen_lock:
                if _video_gen_state.get("job_id") != job_id:
                    return
                _video_gen_state["frame_current"] = i + 1
                _video_gen_state["stage"] = "generating_frames"

            frame_seed = seed + i
            frame_req = {
                "prompt": prompt,
                "negative_prompt": neg_prompt,
                "steps": steps,
                "cfg_scale": cfg,
                "width": width,
                "height": height,
                "seed": frame_seed,
                "sampler_name": "Euler a"
            }
            req = urllib.request.Request(
                f"{sd_url}/sdapi/v1/txt2img",
                data=json.dumps(frame_req).encode("utf-8"),
                headers={"Content-Type": "application/json", "User-Agent": "LIW-Gateway/1.0"},
                method="POST"
            )
            # 300s timeout to allow heavy checkpoint loading / MPS allocation without timing out
            with urllib.request.urlopen(req, timeout=300) as resp:
                res_data = json.loads(resp.read().decode("utf-8"))
                if res_data.get("images") and len(res_data["images"]) > 0:
                    frames_b64.append(res_data["images"][0])

        if frames_b64:
            with _video_gen_lock:
                if _video_gen_state.get("job_id") != job_id:
                    return
                _video_gen_state["stage"] = "assembling_webp"

            pil_images = []
            for fb64 in frames_b64:
                img_bytes = base64.b64decode(fb64.split(",")[-1])
                pil_images.append(Image.open(io.BytesIO(img_bytes)).convert("RGB"))

            webp_buf = io.BytesIO()
            frame_duration_ms = int(1000 / fps)
            pil_images[0].save(
                webp_buf,
                format="WEBP",
                save_all=True,
                append_images=pil_images[1:],
                duration=frame_duration_ms,
                loop=0,
                quality=88,
                method=4
            )
            webp_b64 = base64.b64encode(webp_buf.getvalue()).decode("utf-8")
            video_uri = f"data:image/webp;base64,{webp_b64}"

            result_item = {
                "job_id": job_id,
                "video_url": video_uri,
                "format": "webp",
                "frames": len(frames_b64),
                "fps": fps,
                "duration": round(len(frames_b64) / fps, 2),
                "prompt": prompt,
                "model": video_model,
                "width": width,
                "height": height,
                "seed": seed,
                "timestamp": time.time(),
                "frames_data": [f"data:image/png;base64,{f}" if not f.startswith("data:") else f for f in frames_b64]
            }

            with _video_gen_lock:
                if _video_gen_state.get("job_id") == job_id:
                    _video_gen_state["active"] = False
                    _video_gen_state["stage"] = "complete"
                    _video_gen_state["completed_at"] = time.time()
                    _video_gen_state["result_item"] = result_item
        else:
            raise RuntimeError("No frames generated from SD WebUI sampler")
    except Exception as e:
        with _video_gen_lock:
            if _video_gen_state.get("job_id") == job_id:
                _video_gen_state["active"] = False
                _video_gen_state["stage"] = "error"
                _video_gen_state["last_error"] = str(e)


def cache_get(key):
    with _cache_lock:
        entry = _cache.get(key)
        if entry and (time.time() - entry["ts"]) < CACHE_TTL:
            return entry["data"]
    return None


def cache_set(key, data):
    with _cache_lock:
        _cache[key] = {"data": data, "ts": time.time()}


def make_ssl_context():
    ctx = ssl.create_default_context()
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED
    return ctx


SSL_CTX = make_ssl_context()

# ─── Ollama Registry Scraper ─────────────────────────────────────────────────

OLLAMA_LIBRARY_URL = "https://ollama.com/search"
OLLAMA_LIBRARY_API = "https://ollama.com/api/models"

# Model type classification database
MODEL_TYPE_KEYWORDS = {
    "vision": ["vision", "llava", "moondream", "bakllava", "minicpm-v", "cogvlm",
               "internvl", "monkey", "yi-vl", "qwen-vl", "qwen2-vl", "pixtral",
               "gemma3", "llama3.2-vision"],
    "code": ["coder", "codellama", "starcoder", "deepseek-coder", "codestral",
             "codegemma", "codeqwen", "granite-code", "stable-code", "yi-coder",
             "qwen2.5-coder"],
    "embedding": ["embed", "nomic-embed", "mxbai-embed", "snowflake-arctic-embed",
                  "all-minilm", "bge-", "gte-"],
    "audio": ["whisper", "audio", "speech", "voice", "tts"],
    "video": ["video", "cogvideo"],
    "math": ["math", "mathstral", "qwen2-math", "deepseek-math", "internlm2-math"],
    "reasoning": ["r1", "qwq", "reason", "deepseek-r1", "qwen3", "thinking"],
    "tool_use": ["firefunction", "hermes", "nexusraven", "gorilla", "tools", "tool"],
    "medical": ["medllama", "meditron", "med-", "clinical", "medical", "biomistral"],
    "legal": ["saul", "legal", "law"],
    "finance": ["finance", "fin-"],
    "creative": ["neural-chat", "starling", "openhermes", "dolphin", "nous-hermes"],
}


def classify_model_types(name, description=""):
    """Classify a model into capability categories based on name and description."""
    text = f"{name} {description}".lower()
    types = set()
    for category, keywords in MODEL_TYPE_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                types.add(category)
                break
    if not types:
        types.add("text")
    return sorted(types)


def fetch_ollama_models(query="", category=""):
    """Fetch models from the Ollama library API."""
    cache_key = f"ollama:{query}:{category}"
    cached = cache_get(cache_key)
    if cached:
        return cached

    try:
        # Use the Ollama website search API
        params = {}
        if query:
            params["q"] = query
        if category:
            params["c"] = category

        url = OLLAMA_LIBRARY_URL
        if params:
            url += "?" + urllib.parse.urlencode(params)

        req = urllib.request.Request(url, headers={
            "User-Agent": "OllamaStudio/1.0",
            "Accept": "text/html,application/xhtml+xml"
        })

        with urllib.request.urlopen(req, timeout=15, context=SSL_CTX) as resp:
            body = resp.read().decode("utf-8", errors="replace")

        models = parse_ollama_search_html(body)

        # Also try the JSON API endpoint for richer data
        try:
            api_url = "https://ollama.com/api/models"
            if query:
                api_url += "?" + urllib.parse.urlencode({"q": query})

            api_req = urllib.request.Request(api_url, headers={
                "User-Agent": "OllamaStudio/1.0",
                "Accept": "application/json"
            })

            with urllib.request.urlopen(api_req, timeout=15, context=SSL_CTX) as resp:
                api_data = json.loads(resp.read().decode("utf-8"))

            if isinstance(api_data, dict) and "models" in api_data:
                api_models = api_data["models"]
                # Merge API data with scraped data
                api_map = {}
                for m in api_models:
                    name = m.get("name", "")
                    api_map[name] = m

                # Enrich scraped models with API data
                enriched = []
                seen_names = set()
                for m in models:
                    name = m.get("name", "")
                    if name in api_map:
                        api_m = api_map[name]
                        m["pulls"] = api_m.get("pull_count", m.get("pulls", ""))
                        m["tags_count"] = api_m.get("tags_count", 0)
                        m["updated"] = api_m.get("updated_at", m.get("updated", ""))
                        m["description"] = api_m.get("description", m.get("description", ""))
                        m["capabilities"] = api_m.get("capabilities", [])
                    enriched.append(m)
                    seen_names.add(name)

                # Add API-only models not in scraped results
                for name, api_m in api_map.items():
                    if name not in seen_names:
                        enriched.append({
                            "name": name,
                            "description": api_m.get("description", ""),
                            "pulls": api_m.get("pull_count", ""),
                            "tags_count": api_m.get("tags_count", 0),
                            "updated": api_m.get("updated_at", ""),
                            "sizes": [],
                            "capabilities": api_m.get("capabilities", []),
                            "types": classify_model_types(name, api_m.get("description", "")),
                            "source": "ollama"
                        })

                models = enriched if enriched else models

        except Exception:
            pass  # API endpoint may not be available, fall back to scraped data

        result = {"models": models, "source": "ollama.com", "query": query}
        cache_set(cache_key, result)
        return result

    except Exception as e:
        return {"models": [], "error": str(e), "source": "ollama.com", "query": query}


def parse_ollama_search_html(body):
    """Parse Ollama search page HTML to extract model listings."""
    models = []
    seen = set()
    IGNORE_NAMES = {"models", "search", "filters", "library", "blog", "download", "login"}

    # Pattern targeting model cards: <a href="/library/{model_name}" ...>...</a>
    card_pattern = re.compile(
        r'<a[^>]+href=[\"\']/library/([^\"/\'#?]+)[\"\'][^>]*>(.*?)</a>',
        re.DOTALL | re.IGNORECASE
    )

    matches = card_pattern.findall(body)

    for name, block in matches:
        name = name.strip()
        if not name or name.lower() in IGNORE_NAMES or name in seen or len(name) < 2:
            continue
        seen.add(name)

        # Description
        desc = ""
        desc_match = re.search(r'<p[^>]*>(.*?)</p>', block, re.DOTALL)
        if desc_match:
            desc = re.sub(r'<[^>]+>', '', desc_match.group(1)).strip()
            desc = html.unescape(desc)

        # Pull count
        pulls = ""
        pulls_title_match = re.search(r'title=[\"\']([\d,]+)\s+downloads[\"\']', block, re.IGNORECASE)
        if pulls_title_match:
            pulls = pulls_title_match.group(1)
        else:
            pulls_text_match = re.search(r'(\d+(?:\.\d+)?[KkMmBb]?)\s*(?:Pulls?|Downloads?|pulls)', block, re.IGNORECASE)
            if pulls_text_match:
                pulls = pulls_text_match.group(1)

        # Sizes and capability badges
        sizes = []
        capabilities = []
        badges = re.findall(r'<span[^>]*class=[\"\'][^\"\']*(?:rounded|bg-|text-blue|text-indigo)[^\"\']*[\"\'][^>]*>\s*([^<]+)\s*</span>', block)
        for b in badges:
            b_clean = b.strip()
            if not b_clean:
                continue
            if re.match(r'^[0-9.]+[a-zA-Z]+$', b_clean):
                if b_clean.lower() not in sizes:
                    sizes.append(b_clean.lower())
            elif len(b_clean) < 25 and not re.search(r'downloads|tags|updated', b_clean, re.IGNORECASE):
                if b_clean not in capabilities:
                    capabilities.append(b_clean)

        # Updated time
        updated = ""
        updated_title_match = re.search(r'title=[\"\']Updated\s+([^\"\']+)[\"\']', block, re.IGNORECASE)
        if updated_title_match:
            updated = updated_title_match.group(1).strip()
        else:
            updated_match = re.search(r'text-muted[^>]*>([^<]+)<', block)
            if updated_match:
                updated = updated_match.group(1).strip()

        # Classify categories
        types = classify_model_types(name, desc)
        for cap in capabilities:
            cap_type = cap.lower()
            if cap_type in ("tools", "tool_use", "vision", "embedding", "reasoning", "code", "audio", "thinking"):
                if cap_type in ("tools", "tool"):
                    cap_type = "tool_use"
                elif cap_type == "thinking":
                    cap_type = "reasoning"
                if cap_type not in types:
                    types.append(cap_type)

        models.append({
            "name": name,
            "description": desc,
            "pulls": pulls,
            "sizes": sizes,
            "capabilities": capabilities,
            "updated": updated,
            "types": types,
            "source": "ollama"
        })

    # If scraping yielded nothing, provide popular fallback models
    if not models:
        models = get_popular_ollama_models()

    return models


def get_popular_ollama_models():
    """Return a curated list of popular Ollama models as fallback."""
    return [
        {"name": "llama3.3", "description": "Meta's Llama 3.3 is a powerful 70B model with multilingual support", "pulls": "50M+", "sizes": ["70b"], "updated": "", "types": ["text", "reasoning"], "source": "ollama"},
        {"name": "llama3.2", "description": "Meta's Llama 3.2 — Available in 1B, 3B for text and 11B, 90B with vision", "pulls": "30M+", "sizes": ["1b", "3b", "11b", "90b"], "updated": "", "types": ["text", "vision"], "source": "ollama"},
        {"name": "llama3.2-vision", "description": "Llama 3.2-Vision models for image understanding and visual reasoning", "pulls": "10M+", "sizes": ["11b", "90b"], "updated": "", "types": ["vision"], "source": "ollama"},
        {"name": "deepseek-r1", "description": "DeepSeek R1 — State-of-art reasoning model with chain-of-thought", "pulls": "40M+", "sizes": ["1.5b", "7b", "8b", "14b", "32b", "70b", "671b"], "updated": "", "types": ["text", "reasoning"], "source": "ollama"},
        {"name": "qwen3", "description": "Alibaba's Qwen 3 — hybrid thinking model with superior reasoning", "pulls": "15M+", "sizes": ["0.6b", "1.7b", "4b", "8b", "14b", "30b", "32b", "235b"], "updated": "", "types": ["text", "reasoning"], "source": "ollama"},
        {"name": "qwen2.5-coder", "description": "Qwen 2.5 Coder — Specialized coding model with FIM support", "pulls": "10M+", "sizes": ["0.5b", "1.5b", "3b", "7b", "14b", "32b"], "updated": "", "types": ["code"], "source": "ollama"},
        {"name": "gemma3", "description": "Google's Gemma 3 — Compact and powerful with multimodal vision support", "pulls": "12M+", "sizes": ["1b", "4b", "12b", "27b"], "updated": "", "types": ["text", "vision"], "source": "ollama"},
        {"name": "phi4", "description": "Microsoft Phi-4 — Small, fast reasoning model", "pulls": "8M+", "sizes": ["14b"], "updated": "", "types": ["text", "reasoning"], "source": "ollama"},
        {"name": "mistral", "description": "Mistral 7B — Fast, efficient general-purpose model from Mistral AI", "pulls": "25M+", "sizes": ["7b"], "updated": "", "types": ["text"], "source": "ollama"},
        {"name": "mixtral", "description": "Mistral's Mixture of Experts model — 8x7B and 8x22B variants", "pulls": "8M+", "sizes": ["8x7b", "8x22b"], "updated": "", "types": ["text"], "source": "ollama"},
        {"name": "codestral", "description": "Mistral's Codestral — Purpose-built code generation model", "pulls": "3M+", "sizes": ["22b"], "updated": "", "types": ["code"], "source": "ollama"},
        {"name": "command-r", "description": "Cohere Command R — Optimized for RAG and tool use", "pulls": "5M+", "sizes": ["35b", "104b"], "updated": "", "types": ["text", "tool_use"], "source": "ollama"},
        {"name": "llava", "description": "LLaVA — Large Language and Vision Assistant for image understanding", "pulls": "15M+", "sizes": ["7b", "13b", "34b"], "updated": "", "types": ["vision"], "source": "ollama"},
        {"name": "nomic-embed-text", "description": "Nomic AI text embeddings — High-performance 768-dim embedding model", "pulls": "12M+", "sizes": ["137m"], "updated": "", "types": ["embedding"], "source": "ollama"},
        {"name": "mxbai-embed-large", "description": "MixedBread AI — State-of-art embedding model for retrieval", "pulls": "5M+", "sizes": ["335m"], "updated": "", "types": ["embedding"], "source": "ollama"},
        {"name": "starcoder2", "description": "BigCode StarCoder 2 — Trained on The Stack v2 for code generation", "pulls": "4M+", "sizes": ["3b", "7b", "15b"], "updated": "", "types": ["code"], "source": "ollama"},
        {"name": "moondream", "description": "Moondream — Tiny but capable vision-language model", "pulls": "3M+", "sizes": ["1.8b"], "updated": "", "types": ["vision"], "source": "ollama"},
        {"name": "dolphin-mixtral", "description": "Dolphin Mixtral — Uncensored mixture-of-experts model", "pulls": "5M+", "sizes": ["8x7b", "8x22b"], "updated": "", "types": ["text", "creative"], "source": "ollama"},
        {"name": "yi", "description": "01.AI Yi — Strong bilingual (English/Chinese) foundation model", "pulls": "4M+", "sizes": ["6b", "9b", "34b"], "updated": "", "types": ["text"], "source": "ollama"},
        {"name": "neural-chat", "description": "Intel Neural Chat — Fine-tuned for helpful assistant responses", "pulls": "3M+", "sizes": ["7b"], "updated": "", "types": ["text", "creative"], "source": "ollama"},
        {"name": "stable-code", "description": "Stability AI — Coding completion model based on Stable LM", "pulls": "2M+", "sizes": ["3b"], "updated": "", "types": ["code"], "source": "ollama"},
        {"name": "solar", "description": "Upstage Solar — Compact high-performance model", "pulls": "3M+", "sizes": ["10.7b"], "updated": "", "types": ["text"], "source": "ollama"},
        {"name": "mathstral", "description": "Mistral MathStral — Specialized mathematics and STEM model", "pulls": "1M+", "sizes": ["7b"], "updated": "", "types": ["math"], "source": "ollama"},
        {"name": "meditron", "description": "EPFL Meditron — Specialized medical/clinical knowledge model", "pulls": "500K+", "sizes": ["7b", "70b"], "updated": "", "types": ["medical"], "source": "ollama"},
    ]


def fetch_ollama_categories():
    """Fetch live filter capabilities directly from ollama.com/search."""
    cache_key = "ollama:categories"
    cached = cache_get(cache_key)
    if cached:
        return cached

    default_categories = [
        {"id": "", "label": "All Models", "icon": "◈"},
        {"id": "tools", "label": "Tools", "icon": "⚡"},
        {"id": "thinking", "label": "Thinking", "icon": "🧠"},
        {"id": "vision", "label": "Vision", "icon": "👁️"},
        {"id": "embedding", "label": "Embedding", "icon": "🔗"},
    ]

    try:
        req = urllib.request.Request("https://ollama.com/search", headers={
            "User-Agent": "OllamaStudio/1.0",
            "Accept": "text/html,application/xhtml+xml"
        })
        with urllib.request.urlopen(req, timeout=10, context=SSL_CTX) as resp:
            body = resp.read().decode("utf-8", errors="replace")

        pos = body.find('<h2 class="sr-only">Filters</h2>')
        if pos != -1:
            sub = body[pos:pos+6000]
            inputs = re.findall(r'<label[^>]*>\s*<input[^>]+name=[\"\']c[\"\'][^>]+value=[\"\']([^\"]*)[\"\'][^>]*>(.*?)</label>', sub, re.DOTALL)
            cats = [{"id": "", "label": "All Models", "icon": "◈"}]
            icon_map = {
                "tools": "⚡",
                "thinking": "🧠",
                "vision": "👁️",
                "embedding": "🔗",
                "decision": "🎯"
            }
            seen = {""}
            for val, lbl in inputs:
                val = val.strip()
                if not val or val in seen or val in ("local", "cloud"):
                    continue
                seen.add(val)
                clean = re.sub(r'<[^>]+>', '', lbl).strip()
                cats.append({"id": val, "label": clean or val.capitalize(), "icon": icon_map.get(val, "◈")})
            if len(cats) > 1:
                result = {"categories": cats, "source": "ollama.com"}
                cache_set(cache_key, result)
                return result
    except Exception:
        pass

    result = {"categories": default_categories, "source": "ollama.com"}
    cache_set(cache_key, result)
    return result


def fetch_ollama_model_tags(model_name):
    """Fetch available tags for a specific Ollama model."""
    cache_key = f"ollama_tags:{model_name}"
    cached = cache_get(cache_key)
    if cached:
        return cached

    try:
        url = f"https://ollama.com/library/{urllib.parse.quote(model_name, safe='')}/tags"
        req = urllib.request.Request(url, headers={
            "User-Agent": "OllamaStudio/1.0",
            "Accept": "text/html,application/xhtml+xml"
        })

        with urllib.request.urlopen(req, timeout=15, context=SSL_CTX) as resp:
            body = resp.read().decode("utf-8", errors="replace")

        tags = parse_ollama_tags_html(body, model_name)

        # Also try the API endpoint
        try:
            api_url = f"https://ollama.com/api/models/{urllib.parse.quote(model_name, safe='')}"
            api_req = urllib.request.Request(api_url, headers={
                "User-Agent": "OllamaStudio/1.0",
                "Accept": "application/json"
            })
            with urllib.request.urlopen(api_req, timeout=10, context=SSL_CTX) as resp:
                api_data = json.loads(resp.read().decode("utf-8"))

            if isinstance(api_data, dict):
                api_tags = api_data.get("tags", [])
                if api_tags and isinstance(api_tags, list):
                    # Merge with scraped tags
                    existing_names = {t["tag"] for t in tags}
                    for t in api_tags:
                        tag_name = t if isinstance(t, str) else t.get("name", "")
                        if tag_name and tag_name not in existing_names:
                            size = ""
                            if isinstance(t, dict):
                                size_bytes = t.get("size", 0)
                                if size_bytes:
                                    size = format_bytes_py(size_bytes)
                            tags.append({
                                "tag": tag_name,
                                "full_name": f"{model_name}:{tag_name}",
                                "size": size,
                                "quantization": extract_quant(tag_name),
                                "parameters": extract_params(tag_name)
                            })
        except Exception:
            pass

        result = {"model": model_name, "tags": tags}
        cache_set(cache_key, result)
        return result

    except Exception as e:
        return {"model": model_name, "tags": [], "error": str(e)}


def parse_ollama_tags_html(body, model_name):
    """Parse tags page HTML."""
    tags = []
    seen = set()

    # Look for tag names in the HTML — they typically appear as links or text
    tag_patterns = [
        re.compile(r'href="[/]library/' + re.escape(model_name) + r':([^"]+)"', re.IGNORECASE),
        re.compile(r'>\s*' + re.escape(model_name) + r':(\S+)\s*<', re.IGNORECASE),
        re.compile(r'"tag"\s*:\s*"([^"]+)"', re.IGNORECASE),
    ]

    for pattern in tag_patterns:
        for match in pattern.finditer(body):
            tag = match.group(1).strip()
            if tag and tag not in seen and len(tag) < 100:
                seen.add(tag)

                # Try to extract size info near this tag
                pos = match.start()
                context = body[max(0, pos - 200):min(len(body), pos + 500)]
                size = ""
                size_match = re.search(r'(\d+(?:\.\d+)?\s*(?:GB|MB|KB|TB))', context, re.IGNORECASE)
                if size_match:
                    size = size_match.group(1)

                tags.append({
                    "tag": tag,
                    "full_name": f"{model_name}:{tag}",
                    "size": size,
                    "quantization": extract_quant(tag),
                    "parameters": extract_params(tag)
                })

    # If no tags found, provide common defaults
    if not tags:
        for default_tag in ["latest", "7b", "8b", "14b", "70b"]:
            tags.append({
                "tag": default_tag,
                "full_name": f"{model_name}:{default_tag}",
                "size": "",
                "quantization": extract_quant(default_tag),
                "parameters": extract_params(default_tag)
            })

    return tags


def extract_quant(tag_str):
    """Extract quantization level from a tag string."""
    tag = tag_str.lower()
    quant_patterns = ["q2_k", "q3_k_s", "q3_k_m", "q3_k_l", "q4_0", "q4_1",
                      "q4_k_s", "q4_k_m", "q5_0", "q5_1", "q5_k_s", "q5_k_m",
                      "q6_k", "q8_0", "fp16", "fp32", "bf16", "f16", "f32",
                      "iq2_xxs", "iq2_xs", "iq3_xxs", "iq3_s", "iq4_nl", "iq4_xs"]
    for q in quant_patterns:
        if q in tag:
            return q.upper()
    return ""


def extract_params(tag_str):
    """Extract parameter size from a tag string."""
    match = re.search(r'(\d+(?:\.\d+)?)\s*[bB]\b', tag_str)
    if match:
        return match.group(0).upper()
    return ""


def format_bytes_py(size_bytes):
    """Format bytes into human-readable string."""
    if not size_bytes:
        return ""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(size_bytes) < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


# ─── HuggingFace Hub API ─────────────────────────────────────────────────────

DEFAULT_HF_ENDPOINT = "https://huggingface.co"


def normalize_endpoint(endpoint=None):
    """Normalize and ensure scheme on Hugging Face API endpoint/mirror."""
    ep = (endpoint or DEFAULT_HF_ENDPOINT).strip().rstrip("/")
    if not ep:
        return DEFAULT_HF_ENDPOINT
    if not ep.startswith("http://") and not ep.startswith("https://"):
        ep = "https://" + ep
    return ep


def fetch_hf_models(query="", model_type="", sort="downloads", limit=40,
                    token=None, endpoint=None, gguf_only=True):
    """Search HuggingFace Hub with full configuration support (token, mirror, filters)."""
    base_endpoint = normalize_endpoint(endpoint)
    clean_token = (token or "").strip()
    cache_key = f"hf:{base_endpoint}:{query}:{model_type}:{sort}:{limit}:{gguf_only}:{'auth' if clean_token else 'anon'}"
    cached = cache_get(cache_key)
    if cached:
        return cached

    try:
        # HuggingFace sort field mapping
        sort_map = {
            "downloads": "downloads",
            "likes": "likes",
            "trending": "trendingScore",
            "trendingScore": "trendingScore",
            "newest": "lastModified",
            "lastModified": "lastModified",
            "created": "createdAt",
        }
        hf_sort = sort_map.get(sort, "downloads")

        params = {
            "limit": str(min(max(1, limit), 100)),
            "sort": hf_sort,
            "direction": "-1",
            "full": "true",
        }

        # Build search query
        search_terms = []
        if query:
            search_terms.append(query)
        if gguf_only:
            search_terms.append("gguf")

        if search_terms:
            params["search"] = " ".join(search_terms)

        # Filter by pipeline task if specified
        task_map = {
            "text": "text-generation",
            "reasoning": "text-generation",
            "vision": "image-text-to-text",
            "audio": "automatic-speech-recognition",
            "image": "image-to-text",
            "embedding": "feature-extraction",
            "code": "text-generation",
            "translation": "translation",
            "summarization": "summarization",
            "question-answering": "question-answering",
            "tts": "text-to-speech",
            "image-generation": "text-to-image",
            "video": "text-to-video",
        }
        if model_type:
            clean_type = model_type.strip().lower()
            params["pipeline_tag"] = task_map.get(clean_type, clean_type)

        url = f"{base_endpoint}/api/models?{urllib.parse.urlencode(params)}"
        headers = {
            "User-Agent": "OllamaStudio/1.0",
            "Accept": "application/json"
        }
        if clean_token:
            headers["Authorization"] = f"Bearer {clean_token}"

        req = urllib.request.Request(url, headers=headers)

        with urllib.request.urlopen(req, timeout=20, context=SSL_CTX) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        models = []
        for m in data:
            model_id = m.get("modelId", m.get("id", ""))
            if not model_id:
                continue

            tags = m.get("tags", [])
            has_gguf = "gguf" in tags or any("gguf" in t.lower() for t in tags)

            # If user requested gguf_only, verify the model has gguf tag or gguf files
            siblings = m.get("siblings", [])
            gguf_files = [s for s in siblings if s.get("rfilename", "").lower().endswith(".gguf")]
            if gguf_only and not (has_gguf or gguf_files or "gguf" in model_id.lower()):
                continue

            pipeline_tag = m.get("pipeline_tag", "")
            model_types = classify_hf_model_type(model_id, pipeline_tag, tags)

            # Extract file sizes & quantizations
            file_info = []
            for f in gguf_files[:15]:
                fname = f.get("rfilename", "")
                fsize = f.get("size", 0)
                file_info.append({
                    "name": fname,
                    "size": format_bytes_py(fsize) if fsize else "",
                    "size_bytes": fsize,
                    "quantization": extract_quant(fname),
                    "parameters": extract_params(fname)
                })

            models.append({
                "id": model_id,
                "name": model_id.split("/")[-1] if "/" in model_id else model_id,
                "author": model_id.split("/")[0] if "/" in model_id else "",
                "description": m.get("description", ""),
                "downloads": m.get("downloads", 0),
                "likes": m.get("likes", 0),
                "pipeline_tag": pipeline_tag,
                "tags": tags[:20],
                "types": model_types,
                "has_gguf": has_gguf or bool(gguf_files),
                "gguf_files": file_info,
                "last_modified": m.get("lastModified", ""),
                "private": m.get("private", False),
                "library_name": m.get("library_name", ""),
                "source": "huggingface",
                "url": f"{base_endpoint}/{model_id}"
            })

        result = {
            "models": models,
            "source": "huggingface",
            "endpoint": base_endpoint,
            "query": query,
            "total": len(models),
            "authenticated": bool(clean_token)
        }
        cache_set(cache_key, result)
        return result

    except urllib.error.HTTPError as e:
        err_msg = f"HuggingFace API HTTP {e.code}"
        if e.code == 401:
            err_msg = "Invalid or expired Hugging Face token. Check your token in HF Settings."
        elif e.code == 403:
            err_msg = "Access forbidden. A Hugging Face token may be required for this repository."
        elif e.code == 429:
            err_msg = "Rate limit reached. Add a Hugging Face API token in HF Settings for higher limits."
        return {"models": [], "error": err_msg, "source": "huggingface", "code": e.code}
    except Exception as e:
        return {"models": [], "error": str(e), "source": "huggingface", "query": query}


def fetch_hf_model_detail(model_id, token=None, endpoint=None):
    """Fetch detailed info for a specific HuggingFace model."""
    base_endpoint = normalize_endpoint(endpoint)
    clean_token = (token or "").strip()
    cache_key = f"hf_detail:{base_endpoint}:{model_id}:{'auth' if clean_token else 'anon'}"
    cached = cache_get(cache_key)
    if cached:
        return cached

    try:
        url = f"{base_endpoint}/api/models/{urllib.parse.quote(model_id, safe='/')}"
        headers = {
            "User-Agent": "OllamaStudio/1.0",
            "Accept": "application/json"
        }
        if clean_token:
            headers["Authorization"] = f"Bearer {clean_token}"

        req = urllib.request.Request(url, headers=headers)

        with urllib.request.urlopen(req, timeout=15, context=SSL_CTX) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        siblings = data.get("siblings", [])
        gguf_files = []
        for s in siblings:
            fname = s.get("rfilename", "")
            if fname.lower().endswith(".gguf"):
                fsize = s.get("size", 0)
                is_split = bool(re.search(r'(-\d{4,5}-of-\d{4,5}|-of-\d{4,5})\.gguf$', fname, re.IGNORECASE))
                split_info = None
                if is_split:
                    m = re.search(r'-(\d{4,5})-of-(\d{4,5})\.gguf$', fname, re.IGNORECASE)
                    if m:
                        part_num = int(m.group(1))
                        total_parts = int(m.group(2))
                        split_info = f"Shard {part_num} of {total_parts}"
                    else:
                        split_info = "Multi-part Shard"

                gguf_files.append({
                    "name": fname,
                    "size": format_bytes_py(fsize),
                    "size_bytes": fsize,
                    "quantization": extract_quant(fname),
                    "parameters": extract_params(fname),
                    "is_split": is_split,
                    "split_info": split_info
                })

        result = {
            "id": data.get("modelId", model_id),
            "author": data.get("author", model_id.split("/")[0] if "/" in model_id else ""),
            "description": data.get("description", ""),
            "downloads": data.get("downloads", 0),
            "likes": data.get("likes", 0),
            "pipeline_tag": data.get("pipeline_tag", ""),
            "tags": data.get("tags", []),
            "gguf_files": gguf_files,
            "card_data": data.get("cardData", {}),
            "last_modified": data.get("lastModified", ""),
            "library_name": data.get("library_name", ""),
            "url": f"{base_endpoint}/{model_id}",
            "source": "huggingface",
            "endpoint": base_endpoint,
            "private": data.get("private", False)
        }

        cache_set(cache_key, result)
        return result

    except urllib.error.HTTPError as e:
        err_msg = f"HTTP {e.code}"
        if e.code == 401:
            err_msg = "Unauthorized. Please set a valid Hugging Face token."
        elif e.code == 404:
            err_msg = "Model not found on Hugging Face."
        return {"error": err_msg, "id": model_id, "code": e.code}
    except Exception as e:
        return {"error": str(e), "id": model_id}


def verify_hf_token(token, endpoint=None):
    """Verify Hugging Face API token using /api/whoami-v2."""
    base_endpoint = normalize_endpoint(endpoint)
    clean_token = (token or "").strip()
    if not clean_token:
        return {"valid": False, "error": "No token provided"}

    try:
        url = f"{base_endpoint}/api/whoami-v2"
        headers = {
            "User-Agent": "OllamaStudio/1.0",
            "Authorization": f"Bearer {clean_token}",
            "Accept": "application/json"
        }
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10, context=SSL_CTX) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        return {
            "valid": True,
            "username": data.get("name", ""),
            "fullname": data.get("fullname", ""),
            "email": data.get("email", ""),
            "avatar": data.get("avatarUrl", ""),
            "plan": data.get("plan", "free"),
            "orgs": [o.get("name") for o in data.get("orgs", []) if isinstance(o, dict)],
            "auth_type": data.get("auth", {}).get("type", "api_token"),
            "endpoint": base_endpoint
        }
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return {"valid": False, "error": "Invalid or expired Hugging Face token"}
        return {"valid": False, "error": f"Hugging Face returned HTTP {e.code}"}
    except Exception as e:
        return {"valid": False, "error": str(e)}


def ping_hf_endpoint(endpoint=None):
    """Test connectivity and measure latency to Hugging Face or mirror."""
    base_endpoint = normalize_endpoint(endpoint)
    start = time.time()
    try:
        url = f"{base_endpoint}/api/models?limit=1"
        req = urllib.request.Request(url, headers={
            "User-Agent": "OllamaStudio/1.0",
            "Accept": "application/json"
        })
        with urllib.request.urlopen(req, timeout=8, context=SSL_CTX) as resp:
            elapsed_ms = round((time.time() - start) * 1000)
            return {
                "online": resp.status == 200,
                "status": resp.status,
                "latency_ms": elapsed_ms,
                "endpoint": base_endpoint
            }
    except Exception as e:
        elapsed_ms = round((time.time() - start) * 1000)
        return {
            "online": False,
            "error": str(e),
            "latency_ms": elapsed_ms,
            "endpoint": base_endpoint
        }


def fetch_hf_categories(endpoint="https://huggingface.co", token=""):
    """Fetch live pipeline tags / tasks directly from Hugging Face API."""
    clean_ep = normalize_endpoint(endpoint) if endpoint else "https://huggingface.co"
    cache_key = f"hf:categories:{clean_ep}"
    cached = cache_get(cache_key)
    if cached:
        return cached

    GGUF_TASKS = [
        ("text-generation", "Text Generation", "💬"),
        ("image-text-to-text", "Vision (VLM)", "👁️"),
        ("automatic-speech-recognition", "Speech / Audio", "🎙️"),
        ("feature-extraction", "Embeddings", "🔗"),
        ("text-to-speech", "Text-to-Speech", "🔊"),
        ("text-to-image", "Text-to-Image", "🎨"),
        ("image-text-to-video", "Video", "🎬"),
        ("any-to-any", "Multimodal", "🌐"),
    ]

    api_url = clean_ep.rstrip("/") + "/api/tasks"
    cats = [{"id": "", "label": "All Modalities", "icon": "◈"}]

    try:
        headers = {
            "User-Agent": "OllamaStudio/1.0",
            "Accept": "application/json"
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"

        req = urllib.request.Request(api_url, headers=headers)
        with urllib.request.urlopen(req, timeout=10, context=SSL_CTX) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        if isinstance(data, dict):
            for tid, fallback_lbl, icon in GGUF_TASKS:
                remote_lbl = data.get(tid, {}).get("label") or fallback_lbl
                cats.append({"id": tid, "label": remote_lbl, "icon": icon})
            result = {"categories": cats, "source": clean_ep}
            cache_set(cache_key, result)
            return result
    except Exception:
        pass

    for tid, fallback_lbl, icon in GGUF_TASKS:
        cats.append({"id": tid, "label": fallback_lbl, "icon": icon})

    result = {"categories": cats, "source": clean_ep}
    cache_set(cache_key, result)
    return result


def classify_hf_model_type(model_id, pipeline_tag, tags):
    """Classify HuggingFace model types from pipeline tag and tags."""
    types = set()

    pipeline_map = {
        "text-generation": "text",
        "text2text-generation": "text",
        "image-text-to-text": "vision",
        "visual-question-answering": "vision",
        "image-to-text": "vision",
        "automatic-speech-recognition": "audio",
        "audio-classification": "audio",
        "text-to-speech": "audio",
        "text-to-audio": "audio",
        "text-to-image": "image",
        "image-to-image": "image",
        "feature-extraction": "embedding",
        "sentence-similarity": "embedding",
        "fill-mask": "text",
        "translation": "text",
        "summarization": "text",
        "question-answering": "text",
        "conversational": "text",
        "text-to-video": "video",
        "video-classification": "video",
    }

    if pipeline_tag in pipeline_map:
        types.add(pipeline_map[pipeline_tag])

    # Check tags
    tag_set = set(t.lower() for t in tags)
    if "code" in tag_set or any("code" in t for t in tag_set):
        types.add("code")
    if "math" in tag_set:
        types.add("math")
    if "vision" in tag_set or "multimodal" in tag_set:
        types.add("vision")

    # Check model name
    name_types = classify_model_types(model_id)
    types.update(name_types)

    if not types:
        types.add("text")

    return sorted(types)


# ─── CivitAI Model Hub Scraper ────────────────────────────────────────────────

def fetch_civitai_models(query="", model_type="Checkpoint", sort="Highest Rated", limit=20, page=1):
    """Fetch diffusion models, checkpoints, LoRAs from CivitAI API."""
    cache_key = f"civitai:{query}:{model_type}:{sort}:{limit}:{page}"
    cached = cache_get(cache_key)
    if cached:
        return cached

    params = {
        "limit": min(int(limit), 50),
        "nsfw": "false"
    }
    if query:
        params["query"] = query
    elif page and int(page) > 1:
        params["page"] = int(page)
    if model_type and model_type not in ["All", "all", ""]:
        params["types"] = model_type
    if sort:
        params["sort"] = sort

    url = "https://civitai.com/api/v1/models?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept": "application/json"
    })
    try:
        with urllib.request.urlopen(req, timeout=12, context=SSL_CTX) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        items = []
        for raw in data.get("items", []):
            versions = raw.get("modelVersions", [])
            base_model = "Unknown"
            preview_img = ""
            download_url = ""
            file_name = ""
            file_size_bytes = 0

            if versions:
                v = versions[0]
                base_model = v.get("baseModel", "Unknown")
                images = v.get("images", [])
                if images:
                    preview_img = images[0].get("url", "")
                files = v.get("files", [])
                if files:
                    primary = next((f for f in files if f.get("primary")), files[0])
                    download_url = primary.get("downloadUrl", "")
                    file_name = primary.get("name", "")
                    file_size_bytes = int(primary.get("sizeKB", 0) * 1024)

            stats = raw.get("stats", {})
            items.append({
                "id": raw.get("id"),
                "name": raw.get("name", "Untitled"),
                "type": raw.get("type", "Checkpoint"),
                "creator": raw.get("creator", {}).get("username", "Anonymous"),
                "downloads": stats.get("downloadCount", 0),
                "rating": round(stats.get("rating", 5.0), 1) if stats.get("rating") else 5.0,
                "thumbsUp": stats.get("thumbsUpCount", 0),
                "baseModel": base_model,
                "previewImage": preview_img,
                "downloadUrl": download_url,
                "fileName": file_name,
                "size": format_bytes_py(file_size_bytes),
                "tags": raw.get("tags", [])[:5],
                "description": raw.get("description", "")
            })

        metadata = data.get("metadata", {})
        result = {
            "items": items,
            "totalItems": metadata.get("totalItems", len(items)),
            "currentPage": metadata.get("currentPage", page),
            "pageSize": metadata.get("pageSize", limit),
            "totalPages": metadata.get("totalPages", 1),
            "source": "civitai.com"
        }
        cache_set(cache_key, result)
        return result
    except Exception as e:
        return {"error": f"CivitAI API error: {str(e)}", "items": []}


# ─── HTTP Request Handler ────────────────────────────────────────────────────

class OllamaProxyHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        # Enable CORS for all responses
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, DELETE, PUT")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_HEAD(self):
        if self.path == "/setup" or self.path == "/setup/":
            self.path = "/setup.html"
        super().do_HEAD()

    def do_GET(self):
        if self.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        elif self.path == "/setup" or self.path == "/setup/":
            self.path = "/setup.html"
            return super().do_GET()
        elif self.path.startswith("/api/backends"):
            self.handle_backends_request()
        elif self.path.startswith("/api/image/"):
            self.handle_image_request()
        elif self.path.startswith("/api/video/"):
            self.handle_video_request()
        elif self.path.startswith("/api/audio/"):
            self.handle_audio_request()
        elif self.path.startswith("/api/system/"):
            self.handle_system_request()
        elif self.path.startswith("/api/chat/resume"):
            self.handle_chat_resume()
        elif self.path.startswith("/api/"):
            self.proxy_request("GET")
        elif self.path.startswith("/hub/"):
            self.handle_hub_request()
        else:
            super().do_GET()

    def do_POST(self):
        if self.path.startswith("/api/backends"):
            self.handle_backends_request()
        elif self.path.startswith("/api/image/"):
            self.handle_image_request()
        elif self.path.startswith("/api/video/"):
            self.handle_video_request()
        elif self.path.startswith("/api/audio/"):
            self.handle_audio_request()
        elif self.path.startswith("/api/system/"):
            self.handle_system_request()
        elif self.path.startswith("/api/"):
            self.proxy_request("POST")
        elif self.path.startswith("/hub/"):
            self.handle_hub_request()
        else:
            self.send_error(404, "Not Found")

    def do_DELETE(self):
        if self.path.startswith("/api/"):
            self.proxy_request("DELETE")
        else:
            self.send_error(404, "Not Found")

    def handle_backends_request(self):
        """Handle backend probe, status, and config management."""
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if self.command == "GET":
            if path == "/api/backends":
                status_dict = probe_all_backends()
                self.send_json_response({"backends": status_dict})
                return
            elif path == "/api/backends/service/status":
                q = urllib.parse.parse_qs(parsed.query)
                name = q.get("name", ["sd_webui"])[0]
                status = process_manager.get_service_status(name)
                self.send_json_response({"service": status})
                return
            elif path == "/api/backends/service/logs":
                q = urllib.parse.parse_qs(parsed.query)
                name = q.get("name", ["sd_webui"])[0]
                since = int(q.get("since", [0])[0])
                logs = process_manager.get_logs(name, since)
                self.send_json_response(logs)
                return
            elif path == "/api/backends/config":
                self.send_json_response({"backends": load_backends_config()})
                return

        elif self.command == "POST":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length) if content_length > 0 else b"{}"
            try:
                payload = json.loads(body.decode("utf-8"))
            except Exception:
                payload = {}

            if path == "/api/backends/service/action":
                name = payload.get("name", "sd_webui")
                action = payload.get("action", "start")
                if action == "start":
                    res = process_manager.start_service(name)
                elif action == "stop":
                    res = process_manager.stop_service(name)
                elif action == "restart":
                    process_manager.stop_service(name)
                    port = process_manager.PORT_MAP.get(name)
                    if port:
                        for _ in range(10):
                            if not process_manager._is_port_open(port):
                                break
                            time.sleep(0.4)
                    res = process_manager.start_service(name)
                elif action == "install":
                    res = process_manager.install_service(name)
                else:
                    res = {"success": False, "error": f"Unknown action: {action}"}
                status = process_manager.get_service_status(name)
                res["service"] = status
                self.send_json_response(res)
                return

            if path == "/api/backends/config":
                updated = save_backends_config(payload)
                status_dict = probe_all_backends()
                self.send_json_response({"backends": status_dict, "saved": True})
                return

            elif path == "/api/backends/probe":
                bid = payload.get("id")
                url = payload.get("url")
                # Look up the correct health_path from config/defaults
                cfg = load_backends_config()
                default_health = DEFAULT_BACKENDS.get(bid, {}).get("health_path", "/")
                health_path = cfg.get(bid, {}).get("health_path", default_health)
                if payload.get("health_path"):
                    health_path = payload["health_path"]
                probe_res = probe_single_backend({"id": bid, "url": url, "health_path": health_path})
                self.send_json_response(probe_res)
                return

        self.send_error(404, f"Backends endpoint not found: {path}")

    def handle_image_request(self):
        """Route image generation, models, samplers, and progress requests."""
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        sd_url = get_backend_url("sd_webui")
        comfy_url = get_backend_url("comfyui")

        if self.command == "GET":
            if path == "/api/image/models":
                target = f"{sd_url}/sdapi/v1/sd-models"
                try:
                    req = urllib.request.Request(target, headers={"User-Agent": "LIW-Gateway/1.0"})
                    with urllib.request.urlopen(req, timeout=3) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                        models = []
                        for m in data:
                            models.append({
                                "title": m.get("title", ""),
                                "model_name": m.get("model_name", ""),
                                "hash": m.get("hash", ""),
                                "filename": m.get("filename", ""),
                                "backend": "sd_webui"
                            })
                        self.send_json_response({"models": models, "backend": "sd_webui", "status": "online"})
                        return
                except Exception:
                    self.send_json_response({"models": [], "backend": "none", "status": "standby"})
                    return

            elif path == "/api/image/samplers":
                target = f"{sd_url}/sdapi/v1/samplers"
                try:
                    req = urllib.request.Request(target, headers={"User-Agent": "LIW-Gateway/1.0"})
                    with urllib.request.urlopen(req, timeout=3) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                        samplers = [s.get("name") for s in data if isinstance(s, dict) and s.get("name")]
                        self.send_json_response({"samplers": samplers})
                        return
                except Exception:
                    default_samplers = ["Euler a", "Euler", "DPM++ 2M Karras", "DPM++ SDE Karras", "DDIM", "UniPC", "LCM"]
                    self.send_json_response({"samplers": default_samplers})
                    return

            elif path == "/api/image/progress":
                with _image_gen_lock:
                    is_active = _image_gen_state["active"]
                    job_id = _image_gen_state.get("job_id")
                    stage = _image_gen_state["stage"]
                    sampling_steps = _image_gen_state["sampling_steps"]
                    last_err = _image_gen_state["last_error"]
                    result_item = _image_gen_state.get("result_item")

                if not is_active:
                    self.send_json_response({
                        "active": False,
                        "job_id": job_id,
                        "stage": stage,
                        "progress": 1.0 if stage == "complete" else 0.0,
                        "sampling_step": sampling_steps if stage == "complete" else 0,
                        "sampling_steps": sampling_steps,
                        "detail": "✓ Image synthesis complete!" if stage == "complete" else ("Synthesis stopped by user" if stage == "interrupted" else ("Standby · Ready to synthesize" if stage != "error" else f"Generation error: {last_err}")),
                        "eta_str": "Complete" if stage == "complete" else ("Stopped" if stage == "interrupted" else ("Standby" if stage != "error" else "Error")),
                        "result_item": result_item if stage == "complete" else None,
                        "error": last_err
                    })
                    return

                # Probe SD WebUI progress
                sd_progress = 0.0
                sampling_step = 0
                eta_relative = 0.0
                preview_b64 = None

                try:
                    sd_req = urllib.request.Request(
                        f"{sd_url}/sdapi/v1/progress?skip_current_image=false",
                        headers={"User-Agent": "LIW-Gateway/1.0"}
                    )
                    with urllib.request.urlopen(sd_req, timeout=1.5) as resp:
                        sd_data = json.loads(resp.read().decode("utf-8"))
                        sd_progress = float(sd_data.get("progress") or 0.0)
                        eta_relative = float(sd_data.get("eta_relative") or 0.0)
                        sd_state = sd_data.get("state") or {}
                        sampling_step = int(sd_state.get("sampling_step") or 0)
                        if sd_state.get("sampling_steps"):
                            sampling_steps = int(sd_state["sampling_steps"])
                        if sd_data.get("current_image"):
                            preview_b64 = sd_data["current_image"]
                except Exception:
                    pass

                pct = round(sd_progress * 100)
                detail = f"Sampling step {sampling_step}/{sampling_steps} ({pct}%)" if sampling_step > 0 else "Initializing diffusion latents..."
                eta_str = f"ETA: {round(eta_relative, 1)}s" if eta_relative > 0 else "Synthesizing"

                self.send_json_response({
                    "active": True,
                    "job_id": job_id,
                    "stage": stage,
                    "progress": sd_progress,
                    "sampling_step": sampling_step,
                    "sampling_steps": sampling_steps,
                    "eta_relative": eta_relative,
                    "eta_str": eta_str,
                    "detail": detail,
                    "preview_image": preview_b64
                })
                return

        elif self.command == "POST":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length) if content_length > 0 else b"{}"
            payload = json.loads(body.decode("utf-8")) if body else {}

            if path in ("/api/image/generate", "/api/image/img2img"):
                with _image_gen_lock:
                    if _image_gen_state["active"]:
                        self.send_json_response({
                            "status": "already_active",
                            "job_id": _image_gen_state.get("job_id"),
                            "message": "Image generation is already running in background"
                        })
                        return

                    job_id = f"ijob_{int(time.time() * 1000)}_{random.randint(100, 999)}"
                    _image_gen_state["active"] = True
                    _image_gen_state["job_id"] = job_id
                    _image_gen_state["stage"] = "sampling"
                    _image_gen_state["prompt"] = payload.get("prompt", "")
                    _image_gen_state["params"] = payload
                    _image_gen_state["sampling_step"] = 0
                    _image_gen_state["sampling_steps"] = int(payload.get("steps", 25))
                    _image_gen_state["progress"] = 0.0
                    _image_gen_state["eta_relative"] = 0.0
                    _image_gen_state["preview_image"] = None
                    _image_gen_state["started_at"] = time.time()
                    _image_gen_state["completed_at"] = 0
                    _image_gen_state["result_item"] = None
                    _image_gen_state["last_error"] = None

                # Spawn background synthesis worker thread - returns immediately!
                worker = threading.Thread(
                    target=_run_image_generation_worker,
                    args=(job_id, payload, sd_url),
                    daemon=True
                )
                worker.start()

                self.send_json_response({
                    "status": "started",
                    "job_id": job_id,
                    "message": "Image synthesis started in background"
                })
                return

            elif path == "/api/image/interrupt":
                try:
                    int_req = urllib.request.Request(
                        f"{sd_url}/sdapi/v1/interrupt",
                        data=b"{}",
                        headers={"Content-Type": "application/json", "User-Agent": "LIW-Gateway/1.0"},
                        method="POST"
                    )
                    with urllib.request.urlopen(int_req, timeout=5) as _:
                        pass
                except Exception:
                    pass

                with _image_gen_lock:
                    _image_gen_state["active"] = False
                    _image_gen_state["stage"] = "interrupted"
                    _image_gen_state["last_error"] = "Synthesis stopped by user"

                self.send_json_response({"success": True, "message": "Image synthesis interrupted"})
                return

        self.send_error(404, f"Image endpoint not found: {path}")

    def handle_video_request(self):
        """Route video motion synthesis, models, and frame generation."""
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        sd_url = get_backend_url("sd_webui")
        comfy_url = get_backend_url("comfyui")

        if self.command == "GET":
            if path == "/api/video/models":
                video_models = [
                    {"id": "svd-xt", "name": "Stable Video Diffusion XT (25 frames)", "type": "i2v", "frames": 25, "fps": 12, "desc": "High-fidelity image-to-video motion synthesis"},
                    {"id": "svd", "name": "Stable Video Diffusion (14 frames)", "type": "i2v", "frames": 14, "fps": 12, "desc": "Fast motion synthesis for landscape and portraits"},
                    {"id": "animatediff-v3", "name": "AnimateDiff v3 (SD 1.5 Motion)", "type": "t2v", "frames": 16, "fps": 8, "desc": "Text-to-video motion module for SD 1.5 checkpoints"},
                    {"id": "wan2.1-t2v", "name": "Wan2.1 (1.3B / 14B)", "type": "t2v", "frames": 24, "fps": 16, "desc": "State-of-the-art open-source diffusion video generator"},
                    {"id": "cogvideox-5b", "name": "CogVideoX 5B", "type": "t2v", "frames": 32, "fps": 16, "desc": "Ultra-smooth cinematographic motion generation"}
                ]
                self.send_json_response({"models": video_models, "backend": "sd_webui"})
                return

            elif path == "/api/video/progress":
                with _video_gen_lock:
                    is_active = _video_gen_state["active"]
                    job_id = _video_gen_state.get("job_id")
                    stage = _video_gen_state["stage"]
                    frame_curr = _video_gen_state["frame_current"]
                    frame_tot = _video_gen_state["frame_total"]
                    steps_per_frame = _video_gen_state["steps_per_frame"]
                    fps = _video_gen_state["fps"]
                    model = _video_gen_state["model"]
                    last_err = _video_gen_state["last_error"]
                    result_item = _video_gen_state.get("result_item")

                if not is_active:
                    self.send_json_response({
                        "active": False,
                        "job_id": job_id,
                        "stage": stage,
                        "progress": 100.0 if stage == "complete" else 0.0,
                        "detail": "✓ Video synthesis complete!" if stage == "complete" else ("Standby · Ready to generate video" if stage != "error" else f"Generation error: {last_err}"),
                        "eta_str": "Complete" if stage == "complete" else ("Standby" if stage != "error" else "Error"),
                        "telemetry": f"Rendered {frame_tot} frames @ {fps} fps" if stage == "complete" else (f"Engine: SD WebUI :7860 ({model})" if stage != "error" else f"Failed: {last_err}"),
                        "frame_current": frame_curr,
                        "frame_total": frame_tot,
                        "sampling_step": steps_per_frame if stage == "complete" else 0,
                        "sampling_steps": steps_per_frame,
                        "global_step": frame_tot * steps_per_frame if stage == "complete" else 0,
                        "total_global_steps": frame_tot * steps_per_frame,
                        "result_item": result_item if stage == "complete" else None,
                        "error": last_err
                    })
                    return

                # Real-time telemetry probe directly from active SD WebUI sampler
                sd_progress = 0.0
                sampling_step = 0
                sampling_steps = steps_per_frame or 15
                eta_relative = 0.0
                preview_b64 = None

                try:
                    sd_req = urllib.request.Request(
                        f"{sd_url}/sdapi/v1/progress?skip_current_image=false",
                        headers={"User-Agent": "LIW-Gateway/1.0"}
                    )
                    with urllib.request.urlopen(sd_req, timeout=1.2) as resp:
                        sd_data = json.loads(resp.read().decode("utf-8"))
                        sd_progress = float(sd_data.get("progress") or 0.0)
                        eta_relative = float(sd_data.get("eta_relative") or 0.0)
                        sd_state = sd_data.get("state") or {}
                        sampling_step = int(sd_state.get("sampling_step") or 0)
                        if sd_state.get("sampling_steps"):
                            sampling_steps = int(sd_state["sampling_steps"])
                        if sd_data.get("current_image"):
                            preview_b64 = sd_data["current_image"]
                except Exception:
                    pass

                frame_idx = max(frame_curr - 1, 0)
                tot_frames = max(frame_tot, 1)

                if stage == "assembling_webp":
                    overall_pct = 98.0
                    detail = f"[ASSEMBLING] Encoding {tot_frames} frames into animated WebP timeline..."
                    eta_str = "Finalizing WebP"
                    telemetry = f"All {tot_frames} frames rendered · Compiling at {fps} fps"
                    global_step = tot_frames * sampling_steps
                else:
                    overall_progress_val = ((frame_idx + sd_progress) / tot_frames) * 100.0
                    overall_pct = round(min(max(overall_progress_val, 1.0), 96.0), 1)

                    global_step = (frame_idx * sampling_steps) + sampling_step
                    total_global_steps = tot_frames * sampling_steps

                    if sampling_step > 0:
                        detail = f"[FRAME {frame_curr}/{tot_frames}] Sampling step {sampling_step}/{sampling_steps} ({round(sd_progress * 100)}%)"
                    else:
                        detail = f"[FRAME {frame_curr}/{tot_frames}] Allocating diffusion latents..."

                    if eta_relative > 0:
                        eta_str = f"ETA: {round(eta_relative, 1)}s"
                    else:
                        eta_str = f"Frame {frame_curr}/{tot_frames}"

                    telemetry = f"SD WebUI :7860 · {fps} fps · {model}"

                self.send_json_response({
                    "active": True,
                    "job_id": job_id,
                    "stage": stage,
                    "progress": overall_pct,
                    "frame_progress": round(sd_progress * 100, 1),
                    "frame_current": frame_curr,
                    "frame_total": tot_frames,
                    "sampling_step": sampling_step,
                    "sampling_steps": sampling_steps,
                    "global_step": global_step,
                    "total_global_steps": tot_frames * sampling_steps,
                    "detail": detail,
                    "eta_str": eta_str,
                    "telemetry": telemetry,
                    "preview_image": preview_b64
                })
                return

        elif self.command == "POST":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length) if content_length > 0 else b"{}"
            payload = json.loads(body.decode("utf-8")) if body else {}

            if path == "/api/video/generate":
                prompt = payload.get("prompt", "cinematic camera motion, smooth panning")
                neg_prompt = payload.get("negative_prompt", "jitter, blurry, distorted, low quality")
                frames_count = int(payload.get("frames", 16))
                fps = int(payload.get("fps", 12))
                width = int(payload.get("width", 512))
                height = int(payload.get("height", 512))
                steps = int(payload.get("steps", 15))
                cfg = float(payload.get("cfg_scale", 7.0))
                seed = int(payload.get("seed", -1))
                video_model = payload.get("video_model", "svd-xt")
                if seed == -1:
                    seed = random.randint(1000000, 9999999)

                num_gen_frames = min(max(frames_count, 2), 24)

                with _video_gen_lock:
                    if _video_gen_state["active"]:
                        self.send_json_response({
                            "status": "already_active",
                            "job_id": _video_gen_state.get("job_id"),
                            "message": "Video generation is already running in background"
                        })
                        return

                    job_id = f"vjob_{int(time.time() * 1000)}_{random.randint(100, 999)}"
                    _video_gen_state["active"] = True
                    _video_gen_state["job_id"] = job_id
                    _video_gen_state["stage"] = "generating_frames"
                    _video_gen_state["frame_current"] = 1
                    _video_gen_state["frame_total"] = num_gen_frames
                    _video_gen_state["steps_per_frame"] = steps
                    _video_gen_state["fps"] = fps
                    _video_gen_state["model"] = video_model
                    _video_gen_state["prompt"] = prompt
                    _video_gen_state["started_at"] = time.time()
                    _video_gen_state["completed_at"] = 0
                    _video_gen_state["result_item"] = None
                    _video_gen_state["last_error"] = None

                # Spawn background synthesis worker thread - immediately returns to avoid ngrok/mobile timeout!
                worker = threading.Thread(
                    target=_run_video_generation_worker,
                    args=(job_id, prompt, neg_prompt, num_gen_frames, steps, cfg, width, height, seed, fps, video_model, sd_url),
                    daemon=True
                )
                worker.start()

                self.send_json_response({
                    "status": "started",
                    "job_id": job_id,
                    "frames": num_gen_frames,
                    "fps": fps,
                    "model": video_model,
                    "message": "Video synthesis started in background"
                })
                return
        self.send_error(404, f"Video endpoint not found: {path}")

    def handle_system_request(self):
        """Unified system actions: evict VRAM across Ollama, SD WebUI, and ComfyUI, or return host system metrics."""
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ("/api/system/metrics", "/api/system/ram"):
            try:
                import psutil
                vm = psutil.virtual_memory()
                total_bytes = vm.total
                available_bytes = vm.available
                used_bytes = total_bytes - available_bytes
                percent = round(vm.percent, 1)
                total_gb = round(total_bytes / (1024 ** 3), 1)
                used_gb = round(used_bytes / (1024 ** 3), 1)
                free_gb = round(available_bytes / (1024 ** 3), 1)

                cpu_pct = None
                try:
                    cpu_pct = psutil.cpu_percent(interval=None)
                except Exception:
                    pass

                self.send_json_response({
                    "success": True,
                    "ram": {
                        "total_gb": total_gb,
                        "used_gb": used_gb,
                        "free_gb": free_gb,
                        "percent": percent,
                        "total_bytes": total_bytes,
                        "used_bytes": used_bytes,
                        "available_bytes": available_bytes
                    },
                    "cpu": {
                        "percent": cpu_pct
                    }
                })
            except Exception as e:
                total_gb = 16.0
                try:
                    import subprocess, sys
                    if sys.platform == "darwin":
                        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], timeout=1).decode().strip()
                        total_gb = round(int(out) / (1024 ** 3), 1)
                except Exception:
                    pass
                self.send_json_response({
                    "success": False,
                    "error": str(e),
                    "ram": {
                        "total_gb": total_gb,
                        "used_gb": None,
                        "free_gb": None,
                        "percent": None
                    }
                })
            return

        if path == "/api/system/evict-all" or path == "/api/system/evict":
            evicted = []
            ollama_url = get_backend_url("ollama")
            sd_url = get_backend_url("sd_webui")
            
            # 1. Evict Ollama models
            try:
                ps_req = urllib.request.Request(f"{ollama_url}/api/ps", headers={"User-Agent": "LIW-Gateway/1.0"})
                with urllib.request.urlopen(ps_req, timeout=2) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    models = data.get("models", [])
                    for m in models:
                        m_name = m.get("model") or m.get("name")
                        if m_name:
                            purge_req = urllib.request.Request(
                                f"{ollama_url}/api/generate",
                                data=json.dumps({"model": m_name, "keep_alive": 0}).encode("utf-8"),
                                headers={"Content-Type": "application/json", "User-Agent": "LIW-Gateway/1.0"},
                                method="POST"
                            )
                            with urllib.request.urlopen(purge_req, timeout=3) as _:
                                pass
                            evicted.append(f"Ollama: {m_name}")
            except Exception:
                pass

            # 2. Unload SD WebUI checkpoint from VRAM
            try:
                unload_req = urllib.request.Request(
                    f"{sd_url}/sdapi/v1/unload-checkpoint",
                    data=b"{}",
                    headers={"Content-Type": "application/json", "User-Agent": "LIW-Gateway/1.0"},
                    method="POST"
                )
                with urllib.request.urlopen(unload_req, timeout=3) as _:
                    evicted.append("SD WebUI: Checkpoint VRAM unloaded")
            except Exception:
                pass

            self.send_json_response({
                "success": True,
                "evicted": evicted,
                "message": "VRAM evicted across all active backends"
            })
            return
        self.send_error(404, "System endpoint not found")

    def handle_audio_request(self):
        """Route speech-to-text (ASR) and text-to-speech (TTS) requests."""
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        whisper_url = get_backend_url("whisper")
        piper_url = get_backend_url("piper")

        if self.command == "GET":
            if path == "/api/audio/status":
                w_probe = probe_single_backend({"id": "whisper", "url": whisper_url, "health_path": "/health"})
                p_probe = probe_single_backend({"id": "piper", "url": piper_url, "health_path": "/api/voices"})
                self.send_json_response({
                    "whisper": w_probe,
                    "piper": p_probe
                })
                return

            elif path == "/api/audio/voices":
                target = f"{piper_url}/api/voices"
                try:
                    req = urllib.request.Request(target, headers={"User-Agent": "LIW-Gateway/1.0"})
                    with urllib.request.urlopen(req, timeout=3) as resp:
                        voices = json.loads(resp.read().decode("utf-8"))
                        self.send_json_response({"voices": voices, "backend": "piper", "status": "online"})
                        return
                except Exception:
                    standard_voices = [
                        {"id": "en_US-lessac-medium", "name": "English (US) - Lessac (Medium)", "language": "en_US"},
                        {"id": "en_US-amy-medium", "name": "English (US) - Amy (Medium)", "language": "en_US"},
                        {"id": "en_GB-alan-medium", "name": "English (UK) - Alan (Medium)", "language": "en_GB"},
                        {"id": "browser-default", "name": "Browser System Voice (Web Speech)", "language": "system"}
                    ]
                    self.send_json_response({"voices": standard_voices, "backend": "browser_fallback", "status": "standby"})
                    return

        elif self.command == "POST":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length) if content_length > 0 else b"{}"

            if path == "/api/audio/transcribe":
                audio_bytes = None
                filename = "audio.wav"
                language = "auto"

                try:
                    payload = json.loads(body.decode("utf-8"))
                    b64_str = payload.get("audio", "")
                    if "," in b64_str:
                        b64_str = b64_str.split(",", 1)[1]
                    audio_bytes = base64.b64decode(b64_str)
                    filename = payload.get("filename", "audio.wav")
                    language = payload.get("language", "auto")
                except Exception:
                    audio_bytes = body

                if not audio_bytes:
                    self.send_json_response({"error": "No audio data received"}, status=400)
                    return

                boundary = "----LIWFormBoundary" + str(int(time.time() * 1000))
                form_data = bytearray()
                form_data.extend(f"--{boundary}\r\n".encode())
                form_data.extend(f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode())
                form_data.extend(b"Content-Type: application/octet-stream\r\n\r\n")
                form_data.extend(audio_bytes)
                form_data.extend(b"\r\n")

                if language and language != "auto":
                    form_data.extend(f"--{boundary}\r\n".encode())
                    form_data.extend(b'Content-Disposition: form-data; name="language"\r\n\r\n')
                    form_data.extend(language.encode())
                    form_data.extend(b"\r\n")

                form_data.extend(f"--{boundary}--\r\n".encode())

                # Try whisper.cpp /inference
                target = f"{whisper_url}/inference"
                try:
                    req = urllib.request.Request(
                        target,
                        data=bytes(form_data),
                        headers={
                            "Content-Type": f"multipart/form-data; boundary={boundary}",
                            "User-Agent": "LIW-Gateway/1.0"
                        },
                        method="POST"
                    )
                    with urllib.request.urlopen(req, timeout=60) as resp:
                        resp_data = json.loads(resp.read().decode("utf-8"))
                        text = resp_data.get("text", "")
                        self.send_json_response({
                            "status": "success",
                            "text": text.strip(),
                            "raw": resp_data
                        })
                        return
                except Exception:
                    # Try faster-whisper /v1/audio/transcriptions
                    target_v1 = f"{whisper_url}/v1/audio/transcriptions"
                    try:
                        req_v1 = urllib.request.Request(
                            target_v1,
                            data=bytes(form_data),
                            headers={
                                "Content-Type": f"multipart/form-data; boundary={boundary}",
                                "User-Agent": "LIW-Gateway/1.0"
                            },
                            method="POST"
                        )
                        with urllib.request.urlopen(req_v1, timeout=60) as resp_v1:
                            resp_data = json.loads(resp_v1.read().decode("utf-8"))
                            self.send_json_response({
                                "status": "success",
                                "text": resp_data.get("text", "").strip(),
                                "raw": resp_data
                            })
                            return
                    except Exception:
                        self.send_json_response({
                            "error": f"Whisper ASR backend offline on {whisper_url}. Start whisper.cpp server ('./server -m models/ggml-base.bin --port 8178') or configure endpoint in Settings > Backends.",
                            "status": "standby"
                        }, status=503)
                        return

            elif path == "/api/audio/speak":
                try:
                    payload = json.loads(body.decode("utf-8"))
                except Exception:
                    payload = {}
                text = payload.get("text", "").strip()
                voice = payload.get("voice", "")
                if not text:
                    self.send_json_response({"error": "Empty text for speech synthesis"}, status=400)
                    return

                target = f"{piper_url}/api/tts"
                try:
                    tts_body = json.dumps({"text": text, "voice": voice}).encode("utf-8")
                    req = urllib.request.Request(
                        target,
                        data=tts_body,
                        headers={"Content-Type": "application/json", "User-Agent": "LIW-Gateway/1.0"},
                        method="POST"
                    )
                    with urllib.request.urlopen(req, timeout=30) as resp:
                        audio_data = resp.read()
                        self.send_response(200)
                        self.send_header("Content-Type", resp.headers.get("Content-Type", "audio/wav"))
                        self.send_header("Content-Length", str(len(audio_data)))
                        self.end_headers()
                        self.wfile.write(audio_data)
                        return
                except Exception:
                    try:
                        get_url = f"{piper_url}/?text={urllib.parse.quote(text)}"
                        req = urllib.request.Request(get_url, headers={"User-Agent": "LIW-Gateway/1.0"})
                        with urllib.request.urlopen(req, timeout=30) as resp:
                            audio_data = resp.read()
                            self.send_response(200)
                            self.send_header("Content-Type", "audio/wav")
                            self.send_header("Content-Length", str(len(audio_data)))
                            self.end_headers()
                            self.wfile.write(audio_data)
                            return
                    except Exception:
                        self.send_json_response({
                            "error": f"Piper TTS backend is offline on {piper_url}. The workstation UI will use client-side speech synthesis.",
                            "status": "standby"
                        }, status=503)
                        return

        self.send_error(404, f"Audio endpoint not found: {path}")

    def handle_hub_request(self):
        """Route Model Hub API requests."""
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        params = dict(urllib.parse.parse_qsl(parsed.query))

        # Also read POST body if present
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length > 0:
            body = self.rfile.read(content_length)
            try:
                post_params = json.loads(body)
                params.update(post_params)
            except Exception:
                pass

        # Extract HF configuration from headers or params
        hf_token = params.get("token", params.get("hf_token", ""))
        if not hf_token:
            auth_header = self.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                hf_token = auth_header[7:].strip()
        hf_endpoint = params.get("endpoint", params.get("hf_endpoint", "")).strip()

        try:
            if path == "/hub/ollama/search":
                result = fetch_ollama_models(
                    query=params.get("q", params.get("query", "")),
                    category=params.get("c", params.get("category", ""))
                )
            elif path == "/hub/ollama/tags":
                model = params.get("model", params.get("name", ""))
                if not model:
                    result = {"error": "Missing 'model' parameter"}
                else:
                    result = fetch_ollama_model_tags(model)
            elif path == "/hub/ollama/popular":
                result = {"models": get_popular_ollama_models(), "source": "ollama.com"}
            elif path == "/hub/hf/search":
                gguf_only_val = params.get("gguf_only", "true")
                is_gguf_only = gguf_only_val in [True, "true", "1", "yes"]
                result = fetch_hf_models(
                    query=params.get("q", params.get("query", "")),
                    model_type=params.get("type", params.get("task", "")),
                    sort=params.get("sort", "downloads"),
                    limit=int(params.get("limit", 40)),
                    token=hf_token,
                    endpoint=hf_endpoint,
                    gguf_only=is_gguf_only
                )
            elif path == "/hub/hf/model":
                model_id = params.get("id", params.get("model", ""))
                if not model_id:
                    result = {"error": "Missing 'id' parameter"}
                else:
                    result = fetch_hf_model_detail(model_id, token=hf_token, endpoint=hf_endpoint)
            elif path == "/hub/hf/whoami":
                result = verify_hf_token(hf_token, endpoint=hf_endpoint)
            elif path == "/hub/hf/ping":
                result = ping_hf_endpoint(endpoint=hf_endpoint)
            elif path == "/hub/civitai/search":
                result = fetch_civitai_models(
                    query=params.get("q", params.get("query", "")),
                    model_type=params.get("type", "Checkpoint"),
                    sort=params.get("sort", "Highest Rated"),
                    limit=int(params.get("limit", 20)),
                    page=int(params.get("page", 1))
                )
            elif path == "/hub/categories":
                source = params.get("source", "ollama")
                if source == "huggingface":
                    result = fetch_hf_categories(endpoint=hf_endpoint, token=hf_token)
                else:
                    result = fetch_ollama_categories()
            else:
                result = {"error": f"Unknown hub endpoint: {path}"}

            self.send_json_response(result)

        except Exception as e:
            self.send_json_response({"error": str(e)}, status=500)

    def send_json_response(self, data, status=200):
        """Send a JSON response."""
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def handle_chat_resume(self):
        """Resume an active or completed Ollama chat stream after connection drop."""
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)
        stream_id = qs.get("stream_id", [None])[0]
        try:
            chunk_offset = int(qs.get("chunk_offset", [0])[0])
        except Exception:
            chunk_offset = 0

        _cleanup_old_chat_streams()

        with _chat_stream_lock:
            buf = _chat_stream_buffers.get(stream_id)

        if not buf:
            self.send_json_response({"error": "Chat stream not found or expired", "expired": True}, status=404)
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("X-Stream-Id", stream_id)
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        chunk_idx = chunk_offset
        while True:
            chunk = None
            done = False
            with _chat_stream_lock:
                buf = _chat_stream_buffers.get(stream_id)
                if buf:
                    if chunk_idx < len(buf["chunks"]):
                        chunk = buf["chunks"][chunk_idx]
                        chunk_idx += 1
                    done = buf["done"]

            if chunk:
                try:
                    self.wfile.write(chunk)
                    self.wfile.flush()
                except Exception:
                    break
            elif done:
                break
            else:
                time.sleep(0.02)

    def proxy_request(self, method):
        target_url = f"{get_backend_url('ollama')}{self.path}"
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else None

        req = urllib.request.Request(target_url, data=body, method=method)
        # Strip external CORS, browser routing, and proxy forwarding headers so Ollama
        # treats loopback calls as trusted local server-to-server requests
        excluded_headers = {
            "host", "content-length", "origin", "referer", "cookie",
            "sec-fetch-dest", "sec-fetch-mode", "sec-fetch-site", "priority",
            "x-forwarded-for", "x-forwarded-host", "x-forwarded-proto"
        }
        for h, v in self.headers.items():
            if h.lower() not in excluded_headers:
                req.add_header(h, v)

        # Resilient background token buffering for chat streaming
        if method == "POST" and self.path in ("/api/chat", "/api/generate"):
            _cleanup_old_chat_streams()
            stream_id = self.headers.get("X-Stream-Id") or f"cstream_{int(time.time()*1000)}_{random.randint(100, 999)}"

            with _chat_stream_lock:
                _chat_stream_buffers[stream_id] = {
                    "chunks": [],
                    "done": False,
                    "error": None,
                    "created_at": time.time()
                }

            # Worker thread reads from Ollama continuously regardless of client socket drops
            reader = threading.Thread(
                target=_run_ollama_chat_stream_worker,
                args=(stream_id, req),
                daemon=True
            )
            reader.start()

            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("X-Stream-Id", stream_id)
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            chunk_idx = 0
            while True:
                chunk = None
                done = False
                with _chat_stream_lock:
                    buf = _chat_stream_buffers.get(stream_id)
                    if buf:
                        if chunk_idx < len(buf["chunks"]):
                            chunk = buf["chunks"][chunk_idx]
                            chunk_idx += 1
                        done = buf["done"]

                if chunk:
                    try:
                        self.wfile.write(chunk)
                        self.wfile.flush()
                    except Exception:
                        # Client disconnected (browser closed / phone locked) - exit handler cleanly
                        # Background worker continues buffering tokens from Ollama!
                        break
                elif done:
                    break
                else:
                    time.sleep(0.02)
            return

        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                self.send_response(resp.status)
                for h, v in resp.getheaders():
                    if h.lower() not in ["transfer-encoding", "content-length"]:
                        self.send_header(h, v)
                self.end_headers()

                # Stream response chunks immediately
                while True:
                    chunk = resp.read(1024)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(e.read())
        except Exception as e:
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            err_data = json.dumps({"error": f"Ollama proxy error ({target_url}): {str(e)}"}).encode()
            self.wfile.write(err_data)

    def log_message(self, format, *args):
        """Suppress default logging for cleaner output, except errors."""
        if args and isinstance(args[0], str) and ("404" in args[0] or "500" in args[0] or "502" in args[0]):
            super().log_message(format, *args)


def run():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.environ["OLLAMA_ORIGINS"] = "*"
    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer(("0.0.0.0", port), OllamaProxyHandler)
    print(f"Local Intelligence Workstation Gateway running on http://0.0.0.0:{port}")
    print(f"Backends: Ollama ({get_backend_url('ollama')}), SD WebUI ({get_backend_url('sd_webui')}), Whisper ({get_backend_url('whisper')}), Piper ({get_backend_url('piper')})")
    print(f"Model Hub API endpoints: /hub/ollama/search, /hub/hf/search, /hub/civitai/search, /hub/categories")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.server_close()

if __name__ == "__main__":
    run()
