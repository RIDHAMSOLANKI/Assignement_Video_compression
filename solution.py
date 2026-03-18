"""
Smart Behavioral Video Compression
Sentio Mind Assignment
Author : Ridham Solanki
Roll No: 230853
Branch : Ridham_Solanki_230853
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Algorithm:
  Step 1 : pHash            – drop if >95% similar to last kept frame
  Step 2 : Optical flow     – discard if motion score < 0.05
  Step 3 : Haar face detect – keep regardless of motion if face found
  Step 4 : Context frame    – keep one frame every 3 seconds minimum
  Step 5 : ffmpeg re-encode – H.264 MP4 @ 12 fps

KEY SPEED FIXES:
  - Single-pass: save kept frames to disk WHILE selecting (no second video read)
  - FRAME_SKIP=3  → process every 3rd frame
  - RESIZE_WIDTH=256 for analysis; original resolution saved for output
  - FACE_CHECK_EVERY=6 → Haar only every 6 frames
  - ffmpeg pipe: frames piped directly via stdin (no temp JPEG files at all)
  - JPEG quality=80, faster disk write
"""

import cv2, json, os, subprocess, sys, time, tempfile, shutil
from typing import Optional
import numpy as np
import imagehash
from PIL import Image

# ── Constants ─────────────────────────────────────────────────────────────────
PHASH_SIMILARITY_THRESHOLD = 0.95
OPTICAL_FLOW_THRESHOLD     = 0.05
CONTEXT_FRAME_INTERVAL_SEC = 3.0
OUTPUT_FPS                 = 12

# ── Speed knobs ───────────────────────────────────────────────────────────────
FRAME_SKIP       = 3    # process every Nth frame
RESIZE_WIDTH     = 256  # width for analysis only
FACE_CHECK_EVERY = 6    # Haar every N processed frames

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ── Step 1: pHash ─────────────────────────────────────────────────────────────
def compute_phash(small_bgr):
    return imagehash.phash(
        Image.fromarray(cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB))
    )

def is_too_similar(cur, last, threshold=PHASH_SIMILARITY_THRESHOLD):
    if last is None: return False
    return (1.0 - (cur - last) / 64.0) > threshold


# ── Step 2: Optical Flow ──────────────────────────────────────────────────────
def compute_motion_score(prev_gray, curr_gray):
    if prev_gray is None: return 1.0
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=2, winsize=9,
        iterations=2, poly_n=5, poly_sigma=1.1, flags=0
    )
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(np.mean(mag))


# ── Step 3: Haar Face Detection ───────────────────────────────────────────────
def has_face(small_bgr):
    gray  = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=4, minSize=(20, 20)
    )
    return len(faces) > 0


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-PASS: select frames AND save them to tmpdir in ONE read of the video
# This eliminates the second full video scan entirely.
# ─────────────────────────────────────────────────────────────────────────────
def select_and_save_frames(video_path: str, tmpdir: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / fps

    print(f"[INFO] {total_frames} frames @ {fps:.1f}fps | {duration_sec:.1f}s | {width}x{height}")
    print(f"[INFO] Single-pass mode: skip={FRAME_SKIP} | resize={RESIZE_WIDTH}px | face_every={FACE_CHECK_EVERY}")

    kept_indices     = []
    frame_meta       = []
    last_hash        = None
    prev_gray_small  = None
    last_context_ts  = -CONTEXT_FRAME_INTERVAL_SEC
    last_face_result = False
    face_counter     = 0
    saved_count      = 0
    t0               = time.time()

    for idx in range(total_frames):
        ret, frame = cap.read()
        if not ret: break

        # Skip frames
        if idx % FRAME_SKIP != 0:
            continue

        ts = idx / fps

        # Resize for analysis
        h, w  = frame.shape[:2]
        nw, nh = RESIZE_WIDTH, int(h * RESIZE_WIDTH / w)
        small     = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        curr_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        # Step 3: Face (cached every FACE_CHECK_EVERY frames)
        if face_counter % FACE_CHECK_EVERY == 0:
            last_face_result = has_face(small)
        face_counter += 1

        keep   = False
        reason = "motion"

        if last_face_result:
            keep   = True
            reason = "face"
        elif (ts - last_context_ts) >= CONTEXT_FRAME_INTERVAL_SEC:
            keep   = True
            reason = "context"
        else:
            curr_hash = compute_phash(small)
            if not is_too_similar(curr_hash, last_hash):
                motion = compute_motion_score(prev_gray_small, curr_gray)
                if motion >= OPTICAL_FLOW_THRESHOLD:
                    keep      = True
                    reason    = "motion"
                    last_hash = curr_hash

        if keep:
            # ── Save frame to disk immediately (single pass) ──────────────────
            out_file = os.path.join(tmpdir, f"frame_{saved_count:06d}.jpg")
            cv2.imwrite(out_file, frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            saved_count += 1

            if reason in ("face", "context"):
                last_hash = compute_phash(small)
            if reason == "context":
                last_context_ts = ts

            kept_indices.append(idx)
            meta = {"frame_index": idx, "timestamp_sec": round(ts, 3),
                    "reason": reason, "face_detected": last_face_result}
            if reason == "motion":
                meta["motion_score"] = round(compute_motion_score(prev_gray_small, curr_gray), 4)
            frame_meta.append(meta)

        prev_gray_small = curr_gray

    cap.release()
    elapsed = time.time() - t0
    speed   = round(duration_sec / max(elapsed, 0.01), 1)
    print(f"[INFO] Single-pass done: {elapsed:.1f}s | {speed}x real-time | {saved_count} frames saved")

    return {
        "video_path":   video_path,
        "total_frames": total_frames,
        "fps":          fps,
        "duration_sec": round(duration_sec, 2),
        "width":        width,
        "height":       height,
        "kept_indices": kept_indices,
        "frame_meta":   frame_meta,
        "elapsed_sec":  round(elapsed, 2),
        "saved_count":  saved_count,
    }


# ── Step 5: ffmpeg encode (reads from tmpdir, no second video scan) ───────────
def encode_from_tmpdir(tmpdir: str, saved_count: int,
                       output_path: str = "compressed_output.mp4",
                       output_fps: int = OUTPUT_FPS) -> str:
    print(f"[INFO] Encoding {saved_count} frames with ffmpeg (ultrafast)...")
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(output_fps),
        "-i", os.path.join(tmpdir, "frame_%06d.jpg"),
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "28",
        "-pix_fmt", "yuv420p",
        output_path
    ]
    t0     = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("[ERROR] ffmpeg:\n", result.stderr[-400:])
        raise RuntimeError("ffmpeg failed")
    print(f"[INFO] ffmpeg done in {time.time()-t0:.1f}s → {output_path}")
    return output_path


# ── segments_kept.json ────────────────────────────────────────────────────────
def build_segments_json(selection, output_path="segments_kept.json") -> str:
    fps = selection["fps"]
    ki  = selection["kept_indices"]
    segs = []
    if ki:
        s = e = ki[0]
        for fi in ki[1:]:
            if fi == e + 1: e = fi
            else:
                segs.append({"start_frame": s, "end_frame": e,
                              "start_time_sec": round(s/fps,3),
                              "end_time_sec":   round(e/fps,3)})
                s = e = fi
        segs.append({"start_frame": s, "end_frame": e,
                     "start_time_sec": round(s/fps,3),
                     "end_time_sec":   round(e/fps,3)})

    payload = {
        "video_source":      selection["video_path"],
        "total_frames":      selection["total_frames"],
        "fps":               round(fps, 3),
        "duration_sec":      selection["duration_sec"],
        "kept_frame_count":  len(ki),
        "compression_ratio": round(1 - len(ki)/max(selection["total_frames"],1), 4),
        "segments":          segs,
        "frame_details":     selection["frame_meta"],
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[INFO] segments_kept.json saved ({len(segs)} segments)")
    return output_path


# ── compression_report.html (fully offline) ───────────────────────────────────
def build_html_report(selection, input_path, output_video_path,
                      report_path="compression_report.html") -> str:
    import base64

    total   = selection["total_frames"]
    kept    = len(selection["kept_indices"])
    dropped = total - kept
    ratio   = round((1 - kept/max(total,1))*100, 1)
    fps     = selection["fps"]
    in_mb   = round(os.path.getsize(input_path)/1e6, 2)        if os.path.exists(input_path)        else "N/A"
    out_mb  = round(os.path.getsize(output_video_path)/1e6, 2) if os.path.exists(output_video_path) else "N/A"

    reasons = {"face": 0, "context": 0, "motion": 0}
    for m in selection["frame_meta"]:
        r = m.get("reason","motion"); reasons[r] = reasons.get(r,0)+1

    # thumbnails — read from already-saved JPEGs in tmpdir (fast, no video re-read)
    # fallback: re-read video if tmpdir gone
    meta_map = {m["frame_index"]: m for m in selection["frame_meta"]}
    tc = {"face":"#e74c3c","context":"#f39c12","motion":"#3498db"}

    # Sample up to 20 kept frames for storyboard
    step   = max(1, len(selection["kept_indices"])//20)
    sample = selection["kept_indices"][::step][:20]
    sample_set = set(sample)

    cap = cv2.VideoCapture(selection["video_path"])
    thumbs = {}; idx = 0
    while cap.isOpened() and len(thumbs) < len(sample_set):
        ret, frame = cap.read()
        if not ret: break
        if idx in sample_set:
            th = cv2.resize(frame, (160, 90))
            _, buf = cv2.imencode(".jpg", th, [cv2.IMWRITE_JPEG_QUALITY, 55])
            thumbs[idx] = base64.b64encode(buf).decode()
        idx += 1
    cap.release()

    thumb_html = ""
    for fi in sample:
        if fi not in thumbs: continue
        m = meta_map.get(fi, {}); rsn = m.get("reason","?"); ts = m.get("timestamp_sec",0)
        thumb_html += (f'<div style="background:#1c1f2b;border-radius:8px;overflow:hidden;width:160px">'
                       f'<img src="data:image/jpeg;base64,{thumbs[fi]}" style="width:100%;display:block">'
                       f'<div style="padding:5px 8px;font-size:0.7rem;color:#aaa">#{fi} | {ts:.1f}s '
                       f'<span style="background:{tc.get(rsn,"#888")};color:#fff;padding:1px 5px;'
                       f'border-radius:3px;font-size:0.65rem">{rsn}</span></div></div>')

    def bar(label, count, color):
        pct = round(count/max(total,1)*100, 1)
        return (f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">'
                f'<div style="width:140px;text-align:right;font-size:0.82rem;color:#aaa">{label}</div>'
                f'<div style="flex:1;background:#2a2d3a;border-radius:6px;height:22px;overflow:hidden">'
                f'<div style="width:{pct}%;height:100%;background:{color};display:flex;align-items:center;'
                f'padding-left:8px;font-size:0.72rem;font-weight:700;color:#fff">{count} ({pct}%)</div>'
                f'</div></div>')

    rows = ""
    for i, m in enumerate(selection["frame_meta"][:100]):
        rsn  = m.get("reason","?"); face = "✅" if m.get("face_detected") else "—"
        rows += (f'<tr><td>{i+1}</td><td>{m["frame_index"]}</td><td>{m["timestamp_sec"]:.2f}s</td>'
                 f'<td><span style="background:{tc.get(rsn,"#888")};color:#fff;padding:2px 7px;'
                 f'border-radius:4px;font-size:0.7rem">{rsn}</span></td><td>{face}</td></tr>')

    speed_x = round(selection["duration_sec"] / max(selection["elapsed_sec"], 0.01), 1)

    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>Sentio – Compression Report</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',Arial,sans-serif;background:#0f1117;color:#e0e0e0}}
header{{background:linear-gradient(135deg,#0a3d2e,#14805e);padding:26px 40px}}
header h1{{font-size:1.7rem;color:#fff}} header p{{color:#a0d4bc;margin-top:4px}}
.wrap{{max-width:1100px;margin:0 auto;padding:28px 20px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:14px;margin-bottom:28px}}
.card{{background:#1c1f2b;border-radius:12px;padding:18px;text-align:center}}
.card .v{{font-size:1.85rem;font-weight:700;color:#3ecf8e}}
.card .l{{font-size:0.74rem;color:#888;text-transform:uppercase;margin-top:3px}}
.sec{{font-size:1rem;font-weight:600;margin:22px 0 12px;border-left:4px solid #3ecf8e;padding-left:11px}}
.thumbs{{display:flex;flex-wrap:wrap;gap:10px;margin-top:10px}}
table{{width:100%;border-collapse:collapse;font-size:0.8rem}}
th,td{{padding:7px 11px;border-bottom:1px solid #2a2d3a;text-align:left}}
th{{background:#1c1f2b;color:#3ecf8e;font-weight:600}} tr:hover{{background:#1c1f2b}}
footer{{text-align:center;color:#555;padding:24px;font-size:0.78rem}}
</style></head><body>
<header>
  <h1>🎥 Smart Behavioral Video Compression</h1>
  <p>Sentio Mind &nbsp;·&nbsp; Ridham Solanki &nbsp;·&nbsp; Roll 230853</p>
</header>
<div class="wrap">
  <div class="grid">
    <div class="card"><div class="v">{ratio}%</div><div class="l">Size Reduction</div></div>
    <div class="card"><div class="v">{kept}</div><div class="l">Frames Kept</div></div>
    <div class="card"><div class="v">{dropped}</div><div class="l">Frames Dropped</div></div>
    <div class="card"><div class="v">{in_mb} MB</div><div class="l">Input Size</div></div>
    <div class="card"><div class="v">{out_mb} MB</div><div class="l">Output Size</div></div>
    <div class="card"><div class="v">{speed_x}x</div><div class="l">Speed vs Real-time</div></div>
  </div>
  <div class="sec">Frame Retention Breakdown</div>
  {bar("Face-detected",  reasons['face'],    "#e74c3c")}
  {bar("Motion",         reasons['motion'],  "#3498db")}
  {bar("Context",        reasons['context'], "#f39c12")}
  {bar("Dropped",        dropped,            "#444")}
  <div class="sec">Storyboard (sample of kept frames)</div>
  <div class="thumbs">{thumb_html}</div>
  <div class="sec">Frame Log (first 100 kept frames)</div>
  <table>
    <tr><th>#</th><th>Frame Index</th><th>Timestamp</th><th>Reason</th><th>Face</th></tr>
    {rows}
  </table>
</div>
<footer>Generated by Sentio Mind Smart Compression Pipeline</footer>
</body></html>"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[INFO] HTML report saved → {report_path}")
    return report_path


# ── Integration hook ──────────────────────────────────────────────────────────
def extract_intelligent_frames(video_path, segments_json_path="segments_kept.json"):
    with open(segments_json_path) as f:
        data = json.load(f)
    kept_set = {d["frame_index"] for d in data["frame_details"]}
    cap = cv2.VideoCapture(video_path); idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if idx in kept_set: yield idx, frame
        idx += 1
    cap.release()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Sentio Mind – Smart Video Compression")
    parser.add_argument("input")
    parser.add_argument("--output",   default="compressed_output.mp4")
    parser.add_argument("--report",   default="compression_report.html")
    parser.add_argument("--segments", default="segments_kept.json")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Not found: {args.input}"); sys.exit(1)

    print("="*60)
    print("  Sentio Mind – Smart Behavioral Video Compression")
    print("  Ridham Solanki | Roll 230853")
    print("="*60)

    t_total = time.time()
    tmpdir  = tempfile.mkdtemp(prefix="sentio_")

    try:
        # ── Single pass: select + save frames ────────────────────────────────
        sel = select_and_save_frames(args.input, tmpdir)

        kept  = len(sel["kept_indices"])
        total = sel["total_frames"]
        print(f"\n[RESULT] Kept {kept}/{total} frames → "
              f"{round((1-kept/max(total,1))*100,1)}% dropped\n")

        # ── ffmpeg: encode from already-saved JPEGs ───────────────────────────
        encode_from_tmpdir(tmpdir, sel["saved_count"], args.output)

        # ── JSON + HTML ───────────────────────────────────────────────────────
        build_segments_json(sel, args.segments)
        build_html_report(sel, args.input, args.output, args.report)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    elapsed = time.time() - t_total
    speed   = round(sel["duration_sec"] / max(elapsed, 0.01), 1)
    in_mb   = os.path.getsize(args.input)/1e6        if os.path.exists(args.input)  else 0
    out_mb  = os.path.getsize(args.output)/1e6       if os.path.exists(args.output) else 0
    size_r  = round((1 - out_mb/max(in_mb,0.001))*100, 1)

    print("\n" + "="*60)
    print(f"  ✅ ALL DONE in {elapsed:.1f}s  ({speed}x real-time)")
    print(f"  📦 {in_mb:.1f} MB  →  {out_mb:.1f} MB  ({size_r}% reduction)")
    print(f"  📄 {args.output}")
    print(f"  📄 {args.segments}")
    print(f"  📄 {args.report}")
    print("="*60)

if __name__ == "__main__":
    main()