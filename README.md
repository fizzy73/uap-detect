                     _.-'~~~~`-._
                  .-~            ~-.
                 / .----.  .----.  \
                | /      \/      \  |
                ||  .--.  || .--. | |
                || (====) ||(====)||    U A P   D E T E C T
                | \ '--' /  \ '--' /     (v0.4.5)
                 \ '-..-' /\ '-..-'/
               .--`-.__.-'  `-.__.-'--.
              /__\__________________/__\

*Multi-stream, YOLOv8-powered UAP anomaly detection pipeline*

---
## ✨ Features
- Per-stream **masks** (auto‑scaled from 1280×720) + **mask-overlap suppression (IoU)**
- **Motion gate** + class filters + min box area
- **Streamlink** HTTP proxy for YouTube + **yt‑dlp** fallback (+ optional `imageio[ffmpeg]`)
- **Heartbeat** (FPS + throughput) every 10s
- `output/detections.csv` with `ts, stream, class, conf, box, size, file`
- Global storage cap + per‑stream cleanup

## 🚀 Quickstart (Windows / PowerShell)
```pwsh
git clone https://github.com/KINGLERMAN/uap-detect
cd uap-detect
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python multi_feed.py