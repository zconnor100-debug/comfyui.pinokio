#!/usr/bin/env python3
"""
AI Video Generator - Prompt-based pipeline: SDXL text-to-image -> SVD image-to-video
"""
import os, json, uuid, time, random, threading, subprocess, tempfile
from pathlib import Path
import requests
from flask import Flask, request, jsonify, send_from_directory, render_template_string

app = Flask(__name__)
COMFYUI_URL = "http://127.0.0.1:8188"
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
jobs = {}

HTML = open(BASE_DIR / "prompt_ui.html").read()

def queue_comfyui(workflow):
    client_id = str(uuid.uuid4())
    r = requests.post(f"{COMFYUI_URL}/prompt", json={"prompt": workflow, "client_id": client_id}, timeout=30)
    r.raise_for_status()
    return r.json()["prompt_id"]

def wait_for(prompt_id, timeout=900):
    start = time.time()
    while time.time() - start < timeout:
        r = requests.get(f"{COMFYUI_URL}/history/{prompt_id}", timeout=10)
        data = r.json()
        if prompt_id in data and data[prompt_id].get("status", {}).get("completed"):
            return data[prompt_id]
        time.sleep(2)
    raise TimeoutError("ComfyUI timed out after %ds" % timeout)

def get_images(history):
    imgs = []
    for node_output in history.get("outputs", {}).values():
        imgs.extend(node_output.get("images", []))
    return imgs

def download_img(img_info):
    r = requests.get(f"{COMFYUI_URL}/view",
        params={"filename": img_info["filename"], "subfolder": img_info.get("subfolder",""), "type": img_info.get("type","output")},
        timeout=60)
    r.raise_for_status()
    return r.content

def upload_img(data, filename):
    r = requests.post(f"{COMFYUI_URL}/upload/image", files={"image": (filename, data, "image/png")}, timeout=30)
    r.raise_for_status()
    return r.json()["name"]

def txt2img_workflow(prompt, negative, model, seed, steps):
    if seed < 0: seed = random.randint(0, 999999999)
    turbo = "turbo" in model.lower()
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": model}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1",1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["1",1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 576, "batch_size": 1}},
        "5": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": min(steps,4) if turbo else steps,
            "cfg": 1.0 if turbo else 7.0,
            "sampler_name": "euler_ancestral" if turbo else "dpmpp_2m",
            "scheduler": "karras", "denoise": 1.0,
            "model": ["1",0], "positive": ["2",0], "negative": ["3",0], "latent_image": ["4",0]
        }},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5",0], "vae": ["1",2]}},
        "7": {"class_type": "SaveImage", "inputs": {"filename_prefix": "pvgen_img", "images": ["6",0]}}
    }

def img2vid_workflow(uploaded, model, motion, fps, seed):
    if seed < 0: seed = random.randint(0, 999999999)
    frames = 25 if "xt" in model.lower() else 14
    return {
        "1": {"class_type": "ImageOnlyCheckpointLoader", "inputs": {"ckpt_name": model}},
        "2": {"class_type": "LoadImage", "inputs": {"image": uploaded, "upload": "image"}},
        "3": {"class_type": "SVD_img2vid_Conditioning", "inputs": {
            "width": 1024, "height": 576, "video_frames": frames,
            "motion_bucket_id": motion, "fps": fps, "augmentation_level": 0.0,
            "clip_vision": ["1",1], "init_image": ["2",0], "vae": ["1",2]
        }},
        "4": {"class_type": "VideoLinearCFGGuidance", "inputs": {"min_cfg": 1.0, "model": ["1",0]}},
        "5": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": 20, "cfg": 2.5,
            "sampler_name": "euler", "scheduler": "karras", "denoise": 1.0,
            "model": ["4",0], "positive": ["3",0], "negative": ["3",1], "latent_image": ["3",2]
        }},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5",0], "vae": ["1",2]}},
        "7": {"class_type": "SaveImage", "inputs": {"filename_prefix": "pvgen_frame", "images": ["6",0]}}
    }

def frames_to_mp4(frame_paths, out_path, fps):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for p in sorted(frame_paths):
            f.write(f"file {repr(str(p))}
")
            f.write(f"duration {1/fps}
")
        lst = f.name
    try:
        result = subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst,
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "23",
            str(out_path)
        ], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr[-500:])
    finally:
        os.unlink(lst)

def upd(job_id, **kw):
    jobs[job_id].update(kw)

def pipeline(job_id, p):
    try:
        upd(job_id, stage=0, progress=5, status="running", message="Generating image from prompt...")
        wf = txt2img_workflow(p["prompt"], p["negative"], p["img_model"], p["seed"], p["steps"])
        pid = queue_comfyui(wf)
        upd(job_id, progress=10, message="Image queued — waiting for ComfyUI...")
        hist = wait_for(pid)
        imgs = get_images(hist)
        if not imgs: raise RuntimeError("No image output from SDXL")
        img_data = download_img(imgs[0])
        img_file = f"pvgen_{job_id}.png"
        (OUTPUT_DIR / img_file).write_bytes(img_data)
        upd(job_id, progress=35, stage=1, image_url=f"/output/{img_file}",
            message="Image ready — animating with SVD (2-5 min)...")

        uploaded = upload_img(img_data, img_file)
        wf2 = img2vid_workflow(uploaded, p["vid_model"], p["motion"], p["fps"], p["seed"])
        pid2 = queue_comfyui(wf2)
        upd(job_id, progress=40, message="SVD animation queued — this takes a few minutes...")
        hist2 = wait_for(pid2, timeout=1200)
        frames = get_images(hist2)
        if not frames: raise RuntimeError("No frames output from SVD")
        upd(job_id, progress=75, message=f"Got {len(frames)} frames — downloading...")

        frame_paths = []
        for i, fi in enumerate(frames):
            fp = OUTPUT_DIR / f"pvgen_{job_id}_f{i:04d}.png"
            fp.write_bytes(download_img(fi))
            frame_paths.append(fp)
            upd(job_id, progress=75+int(10*i/len(frames)), message=f"Downloading frame {i+1}/{len(frames)}...")

        upd(job_id, stage=2, progress=88, message="Encoding video with ffmpeg...")
        vid_file = f"pvgen_{job_id}.mp4"
        frames_to_mp4(frame_paths, OUTPUT_DIR / vid_file, p["fps"])
        for fp in frame_paths:
            try: fp.unlink()
            except: pass
        upd(job_id, status="done", stage=3, progress=100,
            message="Video ready!", video_url=f"/output/{vid_file}")
    except Exception as e:
        upd(job_id, status="error", error=str(e), message=f"Error: {e}")

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/models")
def models():
    try:
        r1 = requests.get(f"{COMFYUI_URL}/object_info/CheckpointLoaderSimple", timeout=8)
        r2 = requests.get(f"{COMFYUI_URL}/object_info/ImageOnlyCheckpointLoader", timeout=8)
        img_models = r1.json().get("CheckpointLoaderSimple",{}).get("input",{}).get("required",{}).get("ckpt_name",[[[]]]) [0]
        vid_models = r2.json().get("ImageOnlyCheckpointLoader",{}).get("input",{}).get("required",{}).get("ckpt_name",[[[]]]) [0]
        return jsonify({"img": img_models, "vid": vid_models})
    except Exception as e:
        return jsonify({"img": [], "vid": [], "error": str(e)})

@app.route("/generate", methods=["POST"])
def generate():
    try:
        requests.get(f"{COMFYUI_URL}/system_stats", timeout=5).raise_for_status()
    except Exception:
        return jsonify({"error": "ComfyUI is not running at %s. Start it first from the launcher." % COMFYUI_URL}), 503
    data = request.json
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {"status": "queued", "progress": 0, "stage": 0, "message": "Starting..."}
    threading.Thread(target=pipeline, args=(job_id, data), daemon=True).start()
    return jsonify({"job_id": job_id})

@app.route("/status/<job_id>")
def status(job_id):
    return jsonify(jobs.get(job_id, {"status": "not_found"}))

@app.route("/output/<path:filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"
  AI Video Generator -> http://127.0.0.1:{port}")
    print(f"  ComfyUI expected at -> {COMFYUI_URL}
")
    app.run(host="0.0.0.0", port=port, debug=False)
