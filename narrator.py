#!/usr/bin/env python3
"""
AI Video Narrator
Two modes: manual script entry OR Ollama AI script generation
Pipeline: script → edge-tts narration → SDXL images → SVD animation → ffmpeg assembly
No paid API required.
"""

import os, json, uuid, time, asyncio, threading, subprocess, tempfile, re
from pathlib import Path
import requests
from flask import Flask, request, jsonify, send_from_directory, render_template_string, Response, stream_with_context

app = Flask(__name__)
COMFYUI_URL = "http://127.0.0.1:8188"
OLLAMA_URL  = "http://localhost:11434"
BASE_DIR    = Path(__file__).parent
OUTPUT_DIR  = BASE_DIR / "narrator_output"
OUTPUT_DIR.mkdir(exist_ok=True)

_lock = threading.Lock()
jobs  = {}

# ─── Job helpers ─────────────────────────────────────────────────────────────

def j_upd(jid, **kw):
    with _lock:
        jobs[jid].update(kw)

def j_scene(jid, idx, **kw):
    with _lock:
        jobs[jid]["scenes"][idx].update(kw)

# ─── ComfyUI helpers ─────────────────────────────────────────────────────────

def comfy_queue(wf):
    r = requests.post(f"{COMFYUI_URL}/prompt",
                      json={"prompt": wf, "client_id": str(uuid.uuid4())}, timeout=30)
    r.raise_for_status()
    return r.json()["prompt_id"]

def comfy_wait(pid, timeout=1200):
    t0 = time.time()
    while time.time() - t0 < timeout:
        d = requests.get(f"{COMFYUI_URL}/history/{pid}", timeout=10).json()
        if pid in d and d[pid].get("status", {}).get("completed"):
            return d[pid]
        time.sleep(2)
    raise TimeoutError(f"ComfyUI timed out after {timeout}s")

def comfy_images(hist):
    out = []
    for v in hist.get("outputs", {}).values():
        out.extend(v.get("images", []))
    return out

def comfy_dl(info):
    r = requests.get(f"{COMFYUI_URL}/view",
        params={"filename": info["filename"],
                "subfolder": info.get("subfolder", ""),
                "type": info.get("type", "output")}, timeout=60)
    r.raise_for_status()
    return r.content

def comfy_upload(data, name):
    r = requests.post(f"{COMFYUI_URL}/upload/image",
        files={"image": (name, data, "image/png")}, timeout=30)
    r.raise_for_status()
    return r.json()["name"]

def svd_available():
    try:
        r = requests.get(f"{COMFYUI_URL}/object_info/ImageOnlyCheckpointLoader", timeout=5)
        m = r.json().get("ImageOnlyCheckpointLoader", {}) \
              .get("input", {}).get("required", {}).get("ckpt_name", [[]])[0]
        return len(m) > 0
    except:
        return False

# ─── ComfyUI workflows ────────────────────────────────────────────────────────

import random as _rng

NEGATIVE = "blurry, low quality, distorted, watermark, text, cartoon, deformed, bad anatomy, ugly"

STYLE_SUFFIX = {
    "documentary": "photorealistic, documentary photography, cinematic lighting, 8k, detailed",
    "cinematic":   "cinematic film still, anamorphic lens, dramatic color grade, movie quality",
    "educational": "clean professional illustration, bright lighting, clear subject, detailed",
    "storytelling":"atmospheric, painterly, mood lighting, storybook illustration style",
    "news":        "photojournalistic, high contrast, editorial photography, professional",
}

def wf_txt2img(prompt, model, seed=-1, steps=20):
    if seed < 0: seed = _rng.randint(0, 999999999)
    turbo = "turbo" in model.lower()
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": model}},
        "2": {"class_type": "CLIPTextEncode",          "inputs": {"text": prompt, "clip": ["1",1]}},
        "3": {"class_type": "CLIPTextEncode",          "inputs": {"text": NEGATIVE, "clip": ["1",1]}},
        "4": {"class_type": "EmptyLatentImage",        "inputs": {"width": 1024, "height": 576, "batch_size": 1}},
        "5": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": min(steps,4) if turbo else steps,
            "cfg": 1.0 if turbo else 7.0,
            "sampler_name": "euler_ancestral" if turbo else "dpmpp_2m",
            "scheduler": "karras", "denoise": 1.0,
            "model":["1",0], "positive":["2",0], "negative":["3",0], "latent_image":["4",0]}},
        "6": {"class_type": "VAEDecode",  "inputs": {"samples":["5",0], "vae":["1",2]}},
        "7": {"class_type": "SaveImage",  "inputs": {"filename_prefix":"nrt_img", "images":["6",0]}},
    }

def wf_img2vid(uploaded, model, motion=127, fps=6, seed=-1):
    if seed < 0: seed = _rng.randint(0, 999999999)
    frames = 25 if "xt" in model.lower() else 14
    return {
        "1": {"class_type": "ImageOnlyCheckpointLoader", "inputs": {"ckpt_name": model}},
        "2": {"class_type": "LoadImage",                 "inputs": {"image": uploaded, "upload":"image"}},
        "3": {"class_type": "SVD_img2vid_Conditioning",  "inputs": {
            "width":1024, "height":576, "video_frames":frames,
            "motion_bucket_id":motion, "fps":fps, "augmentation_level":0.0,
            "clip_vision":["1",1], "init_image":["2",0], "vae":["1",2]}},
        "4": {"class_type": "VideoLinearCFGGuidance",    "inputs": {"min_cfg":1.0, "model":["1",0]}},
        "5": {"class_type": "KSampler", "inputs": {
            "seed":seed, "steps":20, "cfg":2.5,
            "sampler_name":"euler", "scheduler":"karras", "denoise":1.0,
            "model":["4",0], "positive":["3",0], "negative":["3",1], "latent_image":["3",2]}},
        "6": {"class_type": "VAEDecode",  "inputs": {"samples":["5",0], "vae":["1",2]}},
        "7": {"class_type": "SaveImage",  "inputs": {"filename_prefix":"nrt_frame", "images":["6",0]}},
    }

# ─── ffmpeg helpers ───────────────────────────────────────────────────────────

W, H, FPS_OUT, AR, CH = 1024, 576, 25, "44100", "2"

def ff(*args, check=True):
    r = subprocess.run(["ffmpeg", "-y"] + list(args), capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{r.stderr[-500:]}")
    return r

def probe_duration(path):
    r = subprocess.run(["ffprobe","-v","error","-show_entries","format=duration",
        "-of","default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True)
    try: return float(r.stdout.strip())
    except: return 6.0

def frames_to_clip(frame_paths, out, fps=6):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for p in sorted(frame_paths):
            f.write(f"file {repr(str(p))}\n")
            f.write(f"duration {1/fps:.6f}\n")
        lst = f.name
    try:
        ff("-f","concat","-safe","0","-i",lst,
           "-vf",f"scale={W}:{H},fps={FPS_OUT},format=yuv420p",
           "-c:v","libx264","-preset","fast","-crf","23", str(out))
    finally:
        os.unlink(lst)

def image_to_clip(img, duration, out):
    ff("-loop","1","-i",str(img),
       "-t",str(duration),
       "-vf",f"scale={W}:{H},fps={FPS_OUT},format=yuv420p",
       "-c:v","libx264","-preset","fast","-crf","23","-tune","stillimage", str(out))

def mux_scene(clip, audio, duration, out):
    """Loop video clip to match audio duration, mux together, normalize codec."""
    if audio and Path(audio).exists():
        ff("-stream_loop","-1","-i",str(clip),
           "-i",str(audio),
           "-t",str(duration+0.4),
           "-map","0:v","-map","1:a",
           "-c:v","libx264","-pix_fmt","yuv420p","-r",str(FPS_OUT),"-preset","fast","-crf","23",
           "-c:a","aac","-ar",AR,"-ac",CH,
           "-shortest", str(out))
    else:
        ff("-stream_loop","-1","-i",str(clip),
           "-f","lavfi","-i",f"anullsrc=r={AR}:cl=stereo",
           "-t",str(duration+0.4),
           "-map","0:v","-map","1:a",
           "-c:v","libx264","-pix_fmt","yuv420p","-r",str(FPS_OUT),"-preset","fast","-crf","23",
           "-c:a","aac","-ar",AR,"-ac",CH,
           "-shortest", str(out))

def make_title_card(title, n_scenes, out, dur=3.5):
    img_p = Path(str(out).replace(".mp4",".png"))
    made  = False
    try:
        from PIL import Image, ImageDraw, ImageFont
        img  = Image.new("RGB", (W, H), (8, 8, 15))
        draw = ImageDraw.Draw(img)
        font_t = font_s = None
        for fp in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                   "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                   "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"]:
            if Path(fp).exists():
                try:
                    font_t = ImageFont.truetype(fp, 60)
                    font_s = ImageFont.truetype(fp.replace("-Bold","").replace("Bold",""), 28)
                except: pass
                break
        if not font_t:
            font_t = font_s = ImageFont.load_default()
        bb = draw.textbbox((0,0), title, font=font_t)
        tw, th = bb[2]-bb[0], bb[3]-bb[1]
        draw.text(((W-tw)//2, H//2-th//2-28), title, font=font_t, fill=(168,85,247))
        sub = f"{n_scenes}-scene AI generated video"
        sb  = draw.textbbox((0,0), sub, font=font_s)
        sw  = sb[2]-sb[0]
        draw.text(((W-sw)//2, H//2+th//2+18), sub, font=font_s, fill=(100,100,140))
        img.save(str(img_p))
        made = True
    except ImportError:
        pass

    if made:
        ff("-loop","1","-i",str(img_p),
           "-f","lavfi","-i",f"anullsrc=r={AR}:cl=stereo",
           "-t",str(dur),
           "-vf",f"scale={W}:{H},fps={FPS_OUT},format=yuv420p",
           "-c:v","libx264","-preset","fast","-crf","20","-tune","stillimage",
           "-c:a","aac","-ar",AR,"-ac",CH,"-shortest", str(out))
        try: img_p.unlink()
        except: pass
    else:
        ff("-f","lavfi","-i",f"color=c=0x08080f:s={W}x{H}:d={dur}",
           "-f","lavfi","-i",f"anullsrc=r={AR}:cl=stereo",
           "-t",str(dur),
           "-c:v","libx264","-pix_fmt","yuv420p","-r",str(FPS_OUT),"-preset","fast",
           "-c:a","aac","-ar",AR,"-ac",CH,"-shortest", str(out))

def concat_all(parts, out):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for p in parts:
            f.write(f"file {repr(str(p))}\n")
        lst = f.name
    try:
        try:
            ff("-f","concat","-safe","0","-i",lst,"-c","copy", str(out))
        except RuntimeError:
            ff("-f","concat","-safe","0","-i",lst,
               "-c:v","libx264","-c:a","aac","-pix_fmt","yuv420p","-preset","fast", str(out))
    finally:
        os.unlink(lst)

# ─── TTS ──────────────────────────────────────────────────────────────────────

def tts(text, voice, out_path):
    async def _go():
        import edge_tts
        await edge_tts.Communicate(text, voice).save(str(out_path))
    asyncio.run(_go())
    return probe_duration(out_path)

# ─── Ollama script generation ─────────────────────────────────────────────────

OLLAMA_PROMPT = """\
You are a video script writer. Write a {n}-scene video script about: {idea}
Style: {style}

Respond ONLY with a valid JSON object. No explanation, no markdown fences.

{{
  "title": "Short compelling title (max 60 chars)",
  "scenes": [
    {{
      "id": 1,
      "title": "Scene title (3-6 words)",
      "image_prompt": "Detailed description for AI image generation. Describe subject, environment, lighting, composition. 20-40 words. No text or watermarks.",
      "narration": "2-3 sentences of natural spoken narration. 15-40 words."
    }}
  ]
}}

Include exactly {n} scenes."""

def extract_json(text):
    text = text.strip()
    try: return json.loads(text)
    except: pass
    m = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if m:
        try: return json.loads(m.group(1).strip())
        except: pass
    m = re.search(r'\{[\s\S]*\}', text)
    if m:
        try: return json.loads(m.group())
        except: pass
    raise ValueError("Could not parse JSON from model response")

def ollama_script(idea, n, style, model):
    prompt = OLLAMA_PROMPT.format(n=n, idea=idea, style=style)
    r = requests.post(f"{OLLAMA_URL}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=180)
    r.raise_for_status()
    raw = r.json()["response"]
    return extract_json(raw)

def ollama_models():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return [m["name"] for m in r.json().get("models", [])]
    except:
        return []

# ─── Main pipeline ────────────────────────────────────────────────────────────

def pipeline(jid, params, script):
    job_dir = OUTPUT_DIR / jid
    job_dir.mkdir(parents=True, exist_ok=True)
    try:
        scenes = script["scenes"]
        title  = script.get("title", "AI Video")
        style  = params.get("style", "cinematic")
        voice  = params.get("voice", "en-US-JennyNeural")

        scene_states = [{"id": s["id"], "title": s["title"],
                         "narration": s.get("narration",""), "status":"pending",
                         "thumb_url": None} for s in scenes]
        j_upd(jid, title=title, scenes=scene_states, progress=5,
              message=f'Script loaded: "{title}"')

        # Phase 1: Generate all TTS audio concurrently
        j_upd(jid, stage="tts", progress=8, message="Generating narration audio...")
        audio_dur = [6.0] * len(scenes)

        def do_tts(i, scene):
            j_scene(jid, i, status="audio")
            ap = job_dir / f"s{i+1:02d}_audio.mp3"
            try:
                d = tts(scene.get("narration",""), voice, ap)
                audio_dur[i] = max(d, 2.0)
            except Exception as e:
                print(f"TTS scene {i+1} failed: {e}")

        threads = [threading.Thread(target=do_tts, args=(i,s)) for i,s in enumerate(scenes)]
        for t in threads: t.start()
        for t in threads: t.join()
        j_upd(jid, progress=18, message="Audio ready. Checking video models...")

        # Phase 2: Check SVD + build title card
        use_static = not svd_available()
        if use_static:
            j_upd(jid, message="No SVD model found — using static images (download SVD for animation)")
        j_upd(jid, stage="title", progress=20, message="Creating title card...")
        title_vid = job_dir / "title_card.mp4"
        make_title_card(title, len(scenes), title_vid)
        segments = [title_vid]

        # Phase 3: Process each scene through ComfyUI (sequential)
        img_model = params.get("img_model", "sd_xl_base_1.0.safetensors")
        vid_model = params.get("vid_model", "svd_xt.safetensors")
        motion    = int(params.get("motion", 127))
        fps       = int(params.get("fps", 6))
        steps     = int(params.get("steps", 20))

        for i, scene in enumerate(scenes):
            base_p = 22 + int(68 * i / len(scenes))
            j_upd(jid, stage="scene", current_scene=i, progress=base_p,
                  message=f"Scene {i+1}/{len(scenes)}: generating image...")
            j_scene(jid, i, status="image")

            style_sfx  = STYLE_SUFFIX.get(style, "")
            full_prompt = f"{scene.get('image_prompt','')}, {style_sfx}"
            wf = wf_txt2img(full_prompt, img_model, steps=steps)
            hist = comfy_wait(comfy_queue(wf))
            imgs = comfy_images(hist)
            if not imgs:
                raise RuntimeError(f"No image output for scene {i+1}")
            img_data = comfy_dl(imgs[0])
            img_path = job_dir / f"s{i+1:02d}_img.png"
            img_path.write_bytes(img_data)
            j_scene(jid, i, status="video" if not use_static else "assembling",
                    thumb_url=f"/narrator_output/{jid}/{img_path.name}")

            dur  = audio_dur[i]
            clip = job_dir / f"s{i+1:02d}_clip.mp4"

            if not use_static:
                j_upd(jid, progress=base_p+16,
                      message=f"Scene {i+1}/{len(scenes)}: animating with SVD...")
                uploaded = comfy_upload(img_data, img_path.name)
                hist2 = comfy_wait(comfy_queue(wf_img2vid(uploaded, vid_model, motion, fps)))
                frames = comfy_images(hist2)
                fps_paths = []
                for fi, finfo in enumerate(frames):
                    fp = job_dir / f"s{i+1:02d}_f{fi:04d}.png"
                    fp.write_bytes(comfy_dl(finfo))
                    fps_paths.append(fp)
                frames_to_clip(fps_paths, clip, fps)
                for fp in fps_paths:
                    try: fp.unlink()
                    except: pass
            else:
                image_to_clip(img_path, dur, clip)

            j_upd(jid, progress=base_p+28,
                  message=f"Scene {i+1}/{len(scenes)}: adding narration...")
            seg  = job_dir / f"s{i+1:02d}_seg.mp4"
            ap   = job_dir / f"s{i+1:02d}_audio.mp3"
            mux_scene(clip, ap if ap.exists() else None, dur, seg)
            try: clip.unlink()
            except: pass
            segments.append(seg)
            j_scene(jid, i, status="done")

        # Phase 4: Concatenate
        j_upd(jid, stage="concat", progress=92, message="Assembling final video...")
        final = job_dir / "final.mp4"
        concat_all(segments, final)
        for s in segments:
            try: s.unlink()
            except: pass

        j_upd(jid, status="done", progress=100, stage="done",
              message=f'"{title}" is ready!',
              video_url=f"/narrator_output/{jid}/final.mp4")

    except Exception as e:
        import traceback; traceback.print_exc()
        j_upd(jid, status="error", error=str(e), message=f"Error: {e}")

# ─── Flask routes ─────────────────────────────────────────────────────────────

HTML = open(BASE_DIR / "narrator_ui.html").read()

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/ollama/models")
def get_ollama_models():
    return jsonify({"models": ollama_models(), "online": len(ollama_models()) > 0})

@app.route("/ollama/generate", methods=["POST"])
def generate_script_route():
    d = request.json
    try:
        script = ollama_script(d["idea"], int(d.get("n_scenes",5)),
                               d.get("style","cinematic"), d.get("model","llama3"))
        return jsonify({"ok": True, "script": script})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/comfyui/models")
def comfyui_models():
    try:
        r1 = requests.get(f"{COMFYUI_URL}/object_info/CheckpointLoaderSimple", timeout=8)
        r2 = requests.get(f"{COMFYUI_URL}/object_info/ImageOnlyCheckpointLoader", timeout=8)
        img = r1.json().get("CheckpointLoaderSimple",{}).get("input",{}).get("required",{}).get("ckpt_name",[[]])[0]
        vid = r2.json().get("ImageOnlyCheckpointLoader",{}).get("input",{}).get("required",{}).get("ckpt_name",[[]])[0]
        return jsonify({"img": img, "vid": vid})
    except Exception as e:
        return jsonify({"img":[],"vid":[],"error":str(e)})

@app.route("/generate", methods=["POST"])
def generate():
    try:
        requests.get(f"{COMFYUI_URL}/system_stats", timeout=5).raise_for_status()
    except:
        return jsonify({"error": f"ComfyUI is not running at {COMFYUI_URL}. Start it first."}), 503

    data   = request.json
    script = data.get("script")
    params = data.get("params", {})
    if not script or not script.get("scenes"):
        return jsonify({"error": "No script provided. Build or generate a script first."}), 400

    jid = str(uuid.uuid4())[:8]
    with _lock:
        jobs[jid] = {"status":"starting","progress":0,"stage":"init",
                     "message":"Starting...","scenes":[],"title":"","video_url":None}
    threading.Thread(target=pipeline, args=(jid, params, script), daemon=True).start()
    return jsonify({"job_id": jid})

@app.route("/stream/<jid>")
def stream(jid):
    @stream_with_context
    def events():
        while True:
            job = jobs.get(jid)
            if not job:
                yield f"data: {json.dumps({'status':'not_found'})}\n\n"
                break
            yield f"data: {json.dumps(job)}\n\n"
            if job["status"] in ("done","error"):
                break
            time.sleep(1.5)
    return Response(events(), mimetype="text/event-stream",
                    headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

@app.route("/narrator_output/<jid>/<path:fn>")
def serve_out(jid, fn):
    return send_from_directory(OUTPUT_DIR / jid, fn)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7861))
    print(f"\n  AI Video Narrator  ->  http://127.0.0.1:{port}")
    print(f"  ComfyUI            ->  {COMFYUI_URL}")
    print(f"  Ollama (optional)  ->  {OLLAMA_URL}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
