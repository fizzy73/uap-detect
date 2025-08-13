# multi_feed.py — v0.4.5
# + Heartbeat (FPS/throughput) per stream
# + detections.csv logger
# + mask-overlap suppression (IoU) for HUD/UI false hits
# + read_frame + robust reconnects (from v0.4.4)
# Python 3.13, PyTorch >= 2.6

import os, csv, time, threading, yaml, subprocess, socket
import cv2, imageio, torch, numpy as np
from datetime import datetime
from ultralytics import YOLO

# ========= Global defaults =========
SAVE_FOLDER   = os.path.abspath("output")
STREAMS_FILE  = "streams.yaml"
SHOW_WINDOW   = False
GLOBAL_STORAGE_GB_CAP = 2.0
BASE_W, BASE_H = 1280, 720

DETECTIONS_CSV = os.path.join(SAVE_FOLDER, "detections.csv")
HEARTBEAT_EVERY_SEC = 10.0  # print stream stats this often

# ========= PyTorch >= 2.6 safe-load allowlist =========
from torch import nn
from torch.serialization import add_safe_globals
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.conv import Conv
add_safe_globals([
    nn.modules.conv.Conv2d,
    nn.modules.batchnorm.BatchNorm2d,
    nn.modules.activation.SiLU,
    nn.modules.container.Sequential,
    nn.modules.linear.Linear,
    DetectionModel, Conv,
])

# ========= Safe YOLO loader =========
def safe_load_yolo():
    weights_url  = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
    weights_path = "yolov8n.pt"
    if not os.path.exists(weights_path):
        print("🔽 Downloading YOLOv8n weights...")
        import urllib.request
        urllib.request.urlretrieve(weights_url, weights_path)
    return YOLO(weights_path)

# ========= YouTube via Streamlink (local HTTP proxy) =========
_port_lock = threading.Lock()
_ports_in_use = set()

def is_youtube(url: str) -> bool:
    u = url.lower()
    return ("youtube.com" in u) or ("youtu.be" in u)

def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.25)
        return s.connect_ex(("127.0.0.1", port)) == 0

def find_free_port(preferred=None) -> int:
    if preferred and not port_in_use(preferred):
        return preferred
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

def start_streamlink_proxy(youtube_watch_url: str, fixed_port=None, timeout_s: int = 25) -> str | None:
    with _port_lock:
        port = find_free_port(fixed_port)
        while port in _ports_in_use or port_in_use(port):
            port = find_free_port()
        _ports_in_use.add(port)
    try:
        cmd = [
            "streamlink", "--player-external-http", f"--player-external-http-port={port}",
            "--retry-open","5","--retry-streams","5","--loglevel","warning",
            youtube_watch_url, "best",
        ]
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print("[streamlink] Not installed. pip install streamlink")
        return None
    except Exception as e:
        print(f"[streamlink] Error: {e}")
        return None
    waited = 0.0
    while waited < timeout_s:
        if port_in_use(port):
            print(f"[streamlink] proxy on 127.0.0.1:{port}")
            return f"http://127.0.0.1:{port}/"
        time.sleep(0.25); waited += 0.25
    print("[streamlink] Failed to start local HTTP server (timeout).")
    return None

# ---------- yt-dlp fallback ----------
def resolve_with_ytdlp(youtube_watch_url: str) -> str | None:
    try:
        import yt_dlp
        ydl_opts = {
            "quiet": True, "nocheckcertificate": True, "format": "best",
            "skip_download": True, "noplaylist": True, "geo_bypass": True,
            "live_from_start": False,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_watch_url, download=False)
            url = info.get("url")
            if url:
                print("[yt-dlp] ✅ resolved direct media URL.")
                return url
            else:
                print("[yt-dlp] ❌ no 'url' in result.")
    except Exception as e:
        print(f"[yt-dlp] failed: {e}")
    return None

def _open_with_imageio(url: str):
    ffmpeg_params = [
        "-protocol_whitelist","file,http,https,tcp,tls,crypto",
        "-user_agent","Mozilla/5.0",
        "-reconnect","1","-reconnect_streamed","1","-reconnect_on_network_error","1",
        "-rw_timeout","15000000",
    ]
    return imageio.get_reader(url, "ffmpeg", ffmpeg_params=ffmpeg_params)

# ========= robust open_stream with optional force_ytdlp =========
def open_stream(url: str, fixed_port=None, force_ytdlp: bool = False):
    def try_opencv(u):
        cap = cv2.VideoCapture(u, cv2.CAP_FFMPEG)
        return ("opencv", cap) if cap is not None and cap.isOpened() else (None, None)
    def try_imageio(u):
        try:
            rdr = _open_with_imageio(u)
            # probe a frame to verify
            try:
                _ = rdr.get_next_data()
            except Exception:
                pass
            return ("imageio", rdr)
        except Exception as e:
            print(f"[open] imageio failed: {e}")
            return (None, None)

    if is_youtube(url):
        if force_ytdlp:
            print("[open] ⛳ force_ytdlp=True — resolving direct URL via yt-dlp…")
            direct = resolve_with_ytdlp(url)
            if direct:
                backend, src = try_opencv(direct)
                if backend: 
                    print("[open] OpenCV connected to direct media URL."); return src, backend
                backend, src = try_imageio(direct)
                if backend: return src, backend
            print("[open] force_ytdlp failed to provide a playable URL.")
        else:
            print("[input] YouTube URL detected — using Streamlink proxy.")
            local = start_streamlink_proxy(url, fixed_port, timeout_s=25)
            if local:
                backend, src = try_opencv(local)
                if backend: 
                    print("[open] OpenCV connected to proxy."); return src, backend
                backend, src = try_imageio(local)
                if backend: return src, backend
            print("[open] Falling back to yt-dlp direct URL…")
            direct = resolve_with_ytdlp(url)
            if direct:
                backend, src = try_opencv(direct)
                if backend: 
                    print("[open] OpenCV connected to direct media URL."); return src, backend
                backend, src = try_imageio(direct)
                if backend: return src, backend

        print("[open] Final fallback: try raw watch URL with OpenCV/imageio…")
        backend, src = try_opencv(url)
        if backend: return src, backend
        backend, src = try_imageio(url)
        if backend: return src, backend
        raise RuntimeError("No backend could open the YouTube source.")

    backend, src = try_opencv(url)
    if backend: 
        print("[open] OpenCV connected to direct URL."); return src, backend
    print("[open] OpenCV failed, trying imageio…")
    backend, src = try_imageio(url)
    if backend: return src, backend
    raise RuntimeError("No backend could open the source URL.")

# ========= Frame readers =========
def read_frame(source, backend):
    if backend == "opencv":
        ok, frame = source.read()
        return frame if ok else None
    elif backend == "imageio":
        try:
            frame = source.get_next_data()  # RGB
            if frame is None: return None
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        except Exception:
            return None
    return None

def close_source(source, backend):
    try:
        if backend == "opencv" and hasattr(source, "release"): source.release()
        elif backend == "imageio" and hasattr(source, "close"): source.close()
    except Exception: pass

# ========= Utilities =========
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def folder_size_bytes(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try: total += os.path.getsize(os.path.join(root, f))
            except Exception: pass
    return total

def enforce_global_cap(base_folder: str, cap_gb: float):
    cap_bytes = int(cap_gb * (1024**3))
    total = folder_size_bytes(base_folder)
    if total <= cap_bytes: return
    all_jpgs = []
    for root, _, files in os.walk(base_folder):
        for f in files:
            if f.lower().endswith(".jpg"):
                p = os.path.join(root, f)
                try: all_jpgs.append((os.path.getmtime(p), p))
                except Exception: pass
    all_jpgs.sort(key=lambda x: x[0])
    for _, p in all_jpgs:
        try: os.remove(p)
        except Exception: pass
        if folder_size_bytes(base_folder) <= cap_bytes:
            print(f"[storage] Global cap OK (<= {cap_gb} GB).")
            break

def _scale_masks(masks, frame_w, frame_h):
    sx = frame_w / float(BASE_W); sy = frame_h / float(BASE_H)
    scaled = []
    for x, y, w, h in masks:
        X = int(x * sx); Y = int(y * sy); W = int(w * sx); H = int(h * sy)
        X = max(0, min(X, frame_w-1)); Y = max(0, min(Y, frame_h-1))
        if X + W > frame_w: W = frame_w - X
        if Y + H > frame_h: H = frame_h - Y
        if W > 0 and H > 0: scaled.append((X, Y, W, H))
    return scaled

def apply_masks(frame, masks):
    for x, y, w, h in masks:
        frame[y:y+h, x:x+w] = 0
    return frame

def detect_motion(prev, curr, threshold=25, min_pixels=5000):
    delta = cv2.absdiff(prev, curr)
    gray  = cv2.cvtColor(delta, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    return cv2.countNonZero(dilated) > int(min_pixels)

# ---- Box vs mask suppression (IoU) ----
def rect_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0: return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return inter / union

def inflate_rect(x, y, w, h, pad, W, H):
    x1 = max(0, x - pad); y1 = max(0, y - pad)
    x2 = min(W, x + w + pad); y2 = min(H, y + h + pad)
    return (x1, y1, x2, y2)

# ========= Streams config =========
def load_streams_config():
    with open(STREAMS_FILE, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    streams = {}
    for name, cfg in raw.items():
        if isinstance(cfg, str): url, cfg = cfg, {}
        else: url = cfg.get("url", "")
        if not url: continue
        streams[name] = {
            "url": url,
            "masks": cfg.get("masks", []),
            "excluded_classes": cfg.get("excluded_classes", []),
            "whitelist_classes": cfg.get("whitelist_classes", []),
            "confidence": float(cfg.get("confidence", 0.5)),
            "min_box_area": int(cfg.get("min_box_area", 0)),
            "frame_skip": int(cfg.get("frame_skip", 2)),
            "motion_gate": bool(cfg.get("motion_gate", True)),
            "motion_pixels": int(cfg.get("motion_pixels", 5000)),
            "save_boxed_frames": bool(cfg.get("save_boxed_frames", True)),
            "save_raw_frames": bool(cfg.get("save_raw_frames", False)),
            "delete_older_minutes": int(cfg.get("delete_older_minutes", 60)),
            "max_saved_frames": int(cfg.get("max_saved_frames", 300)),
            "proxy_port": cfg.get("proxy_port", None),
            "force_ytdlp": bool(cfg.get("force_ytdlp", False)),
        }
    return streams

def cleanup_stream_folder(folder_path, max_age_minutes=60, max_files=300):
    now = time.time()
    try:
        files = [os.path.join(folder_path, f)
                 for f in os.listdir(folder_path)
                 if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(".jpg")]
    except FileNotFoundError:
        return
    for p in files:
        try:
            age_min = (now - os.path.getmtime(p)) / 60.0
            if age_min > max_age_minutes: os.remove(p)
        except Exception: pass
    try:
        files = [os.path.join(folder_path, f)
                 for f in os.listdir(folder_path)
                 if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(".jpg")]
    except FileNotFoundError:
        return
    files.sort(key=lambda p: os.path.getmtime(p))
    while len(files) > max_files:
        p = files.pop(0)
        try: os.remove(p)
        except Exception: pass

# ========= CSV logger =========
_csv_lock = threading.Lock()
def _csv_init_if_needed():
    ensure_dir(SAVE_FOLDER)
    if not os.path.exists(DETECTIONS_CSV):
        with open(DETECTIONS_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["ts_utc","stream","backend","class","conf","x1","y1","x2","y2","w","h","file"])
def log_detection(ts_utc, stream, backend, cls, conf, x1,y1,x2,y2, w,h, fname):
    with _csv_lock:
        with open(DETECTIONS_CSV, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([ts_utc, stream, backend, cls, f"{conf:.4f}",
                                    x1,y1,x2,y2, w,h, fname])

# ========= Per-stream worker =========
def process_stream(name, cfg, model):
    url = cfg["url"]
    save_dir = os.path.join(SAVE_FOLDER, name)
    ensure_dir(save_dir)
    _csv_init_if_needed()

    print(f"🚀 Start: {name}")
    print(f"📡 URL: {url}")
    print("-" * 60)

    frame_idx = 0
    last_maint = last_hb = time.monotonic()
    maint_every = 20.0
    frames_since_hb = saved_since_hb = 0

    while True:
        source = backend = None
        try:
            source, backend = open_stream(url, cfg.get("proxy_port"), cfg.get("force_ytdlp", False))
            prev_frame = None
            scaled_masks = None
            mask_rects = []

            while True:
                frame = read_frame(source, backend)
                if frame is None:
                    raise ValueError("No frame received")

                h, w = frame.shape[:2]
                if scaled_masks is None and cfg["masks"]:
                    scaled_masks = _scale_masks(cfg["masks"], w, h)
                    mask_rects = [inflate_rect(x, y, mw, mh, 6, w, h) for (x,y,mw,mh) in scaled_masks]

                frames_since_hb += 1
                frame_idx += 1
                if cfg["frame_skip"] > 1 and (frame_idx % cfg["frame_skip"] != 0):
                    # even skipped frames count toward FPS
                    pass
                else:
                    # Motion gate
                    if cfg["motion_gate"]:
                        if prev_frame is None:
                            prev_frame = frame.copy()
                            continue
                        if not detect_motion(prev_frame, frame, min_pixels=cfg["motion_pixels"]):
                            prev_frame = frame.copy()
                            continue
                        prev_frame = frame.copy()

                    # Masks
                    masked = frame if not scaled_masks else apply_masks(frame.copy(), scaled_masks)

                    # Inference
                    results = model(masked, verbose=False)[0]
                    dets = []
                    try:
                        for box in results.boxes:
                            cls_id = int(box.cls[0]); conf = float(box.conf[0])
                            cls_name = model.names[cls_id]
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            wbox = max(0, x2 - x1); hbox = max(0, y2 - y1)
                            area = wbox * hbox
                            if conf < cfg["confidence"]: continue
                            if cfg["min_box_area"] and area < cfg["min_box_area"]: continue
                            if cfg["whitelist_classes"]:
                                if cls_name.lower() not in [c.lower() for c in cfg["whitelist_classes"]]:
                                    continue
                            elif cfg["excluded_classes"] and cls_name:
                                if cls_name.lower() in [c.lower() for c in cfg["excluded_classes"]]:
                                    continue
                            # Suppress anything overlapping masked areas too much
                            if mask_rects and any(rect_iou((x1,y1,x2,y2), m) > 0.30 for m in mask_rects):
                                continue
                            dets.append(((x1, y1, x2, y2), cls_name, conf))
                    except Exception as e:
                        print(f"[{name}] parse error: {e}")
                        continue

                    if dets and (cfg["save_boxed_frames"] or cfg["save_raw_frames"]):
                        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                        raw_path  = os.path.join(save_dir, f"{name}_{ts}_raw.jpg")
                        boxed_path= os.path.join(save_dir, f"{name}_{ts}_boxed.jpg")

                        if cfg["save_raw_frames"]:
                            cv2.imwrite(raw_path, frame)

                        if cfg["save_boxed_frames"]:
                            vis = frame.copy()
                            for (x1, y1, x2, y2), cls, conf in dets:
                                cv2.rectangle(vis, (x1, y1), (x2, y2), (0,255,0), 2)
                                cv2.putText(vis, f"{cls} {conf:.2f}", (x1, y1-8),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                            cv2.imwrite(boxed_path, vis)
                            print(f"[{name}] saved: {os.path.basename(boxed_path)}")
                            saved_since_hb += 1
                            # Log every detection (one row per box)
                            for (x1, y1, x2, y2), cls, conf in dets:
                                log_detection(datetime.utcnow().isoformat(), name, backend, cls, conf,
                                              x1,y1,x2,y2, w,h, os.path.basename(boxed_path))

                # Maintenance
                now = time.monotonic()
                if now - last_maint >= maint_every:
                    cleanup_stream_folder(save_dir, cfg["delete_older_minutes"], cfg["max_saved_frames"])
                    enforce_global_cap(SAVE_FOLDER, GLOBAL_STORAGE_GB_CAP)
                    last_maint = now

                # Heartbeat
                if now - last_hb >= HEARTBEAT_EVERY_SEC:
                    fps = frames_since_hb / (now - last_hb)
                    print(f"[{name}] ⏱️ {fps:.1f} fps | backend={backend} | "
                          f"frames={frames_since_hb} | saves={saved_since_hb}")
                    frames_since_hb = 0; saved_since_hb = 0; last_hb = now

                if SHOW_WINDOW:
                    cv2.imshow(name, frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        raise KeyboardInterrupt

        except Exception as e:
            print(f"[{name}] stream error: {e}")
            print("🔁 reconnecting in 5s…")
            time.sleep(5)
        finally:
            if source is not None:
                close_source(source, backend)

# ========= Main =========
def main():
    os.makedirs(SAVE_FOLDER, exist_ok=True)
    model = safe_load_yolo()
    cfgs  = load_streams_config()
    threads = []
    for name, cfg in cfgs.items():
        t = threading.Thread(target=process_stream, args=(name, cfg, model), daemon=False)
        t.start(); threads.append(t)
    for t in threads: t.join()

if __name__ == "__main__":
    main()
