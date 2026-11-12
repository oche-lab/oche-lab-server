"""OpenCV drawing for Oche Lab coach review: skeleton overlays, key-frame JPEGs, optional overlay MP4.

Needs cv2 + numpy only (no mediapipe), so it can be exercised on synthetic frames.
"""
import shutil
import subprocess
import textwrap
from pathlib import Path

import cv2
import numpy as np

FRAME_MAX_SIDE = 720          # key-frame JPEGs are downscaled to this
JPEG_QUALITY = 80
OVERLAY_MAX_SIDE = 640        # overlay video resolution cap
OVERLAY_MAX_SECONDS = 60      # only the first N seconds are rendered
OVERLAY_CODECS = ('avc1', 'H264')  # iOS needs H.264. Fallback: write mp4v, then transcode with ffmpeg if installed
FFMPEG_TIMEOUT_S = 90

# BGR colours
ARM = (94, 197, 34)        # green: shoulder-elbow-wrist
HEAD = (11, 158, 245)      # amber: head-shoulder line
SHOULDERS = (250, 165, 96)  # light blue: shoulder line
WHITE = (255, 255, 255)
PANEL = (32, 18, 11)


def _ascii(text):
    return (text or '').replace('°', ' deg').replace('–', '-').replace('’', "'").encode('ascii', 'ignore').decode()


def _px(p, w, h):
    return None if p is None else (int(round(p[0] * w)), int(round(p[1] * h)))


def _panel(img, y0, y1, alpha=0.65):
    y0, y1 = max(0, y0), min(img.shape[0], y1)
    roi = img[y0:y1]
    img[y0:y1] = cv2.addWeighted(roi, 1 - alpha, np.full_like(roi, PANEL), alpha, 0)


def draw_overlay(frame, points, elbow_angle=None, label=None, caption=None, time_s=None):
    """Draw tracked arm, head/shoulder lines, elbow angle, header label and caption. Returns a new image."""
    img = frame.copy()
    h, w = img.shape[:2]
    s = max(0.6, min(w, h) / 540)  # scale strokes/text with resolution
    th = max(2, int(3 * s))
    P = {k: _px(points.get(k), w, h) for k in ('head', 'shoulder', 'elbow', 'wrist', 'other_shoulder')}
    if P['shoulder'] and P['other_shoulder']:
        cv2.line(img, P['other_shoulder'], P['shoulder'], SHOULDERS, th, cv2.LINE_AA)
    if P['head'] and P['shoulder']:
        cv2.line(img, P['head'], P['shoulder'], HEAD, th, cv2.LINE_AA)
    for a, b in (('shoulder', 'elbow'), ('elbow', 'wrist')):
        if P[a] and P[b]:
            cv2.line(img, P[a], P[b], ARM, th + 2, cv2.LINE_AA)
    for k, c in (('head', HEAD), ('shoulder', ARM), ('elbow', ARM), ('wrist', ARM), ('other_shoulder', SHOULDERS)):
        if P[k]:
            cv2.circle(img, P[k], th + 4, c, -1, cv2.LINE_AA)
            cv2.circle(img, P[k], th + 4, WHITE, 1, cv2.LINE_AA)
    if P['elbow'] and elbow_angle is not None:
        txt = f'{elbow_angle:.0f} deg'
        org = (P['elbow'][0] + int(12 * s), P['elbow'][1] - int(12 * s))
        cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s, (0, 0, 0), th + 2, cv2.LINE_AA)
        cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s, WHITE, max(1, th - 1), cv2.LINE_AA)
    if label or time_s is not None:
        bar = int(44 * s)
        _panel(img, 0, bar)
        if label:
            cv2.putText(img, _ascii(label), (int(12 * s), int(30 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s, WHITE, max(1, th - 1), cv2.LINE_AA)
        if time_s is not None:
            t = f'{time_s:.2f}s'
            (tw, _), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s, 1)
            cv2.putText(img, t, (w - tw - int(12 * s), int(30 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s, (200, 220, 230), 1, cv2.LINE_AA)
    if caption:
        chars = max(20, int(w / (11 * s)))
        lines = textwrap.wrap(_ascii(caption), chars)[:3]
        lh = int(26 * s)
        top = h - lh * len(lines) - int(16 * s)
        _panel(img, top, h, 0.72)
        for k, line in enumerate(lines):
            cv2.putText(img, line, (int(12 * s), top + int(8 * s) + lh * (k + 1) - int(6 * s)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.62 * s, WHITE, max(1, th - 2), cv2.LINE_AA)
    return img


def _fit(img, max_side):
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    k = max_side / max(h, w)
    return cv2.resize(img, (int(w * k) // 2 * 2, int(h * k) // 2 * 2), interpolation=cv2.INTER_AREA)


def render_key_frames(video_path, requests, out_dir):
    """requests: [{'frame_idx', 'name', 'points', 'elbow_angle_deg', 'label', 'caption', 'time_s'}].
    Reads the video once, writes JPEGs. Returns the set of names written."""
    out_dir = Path(out_dir)
    wanted = {}
    for r in requests:
        wanted.setdefault(int(r['frame_idx']), []).append(r)
    written, idx = set(), 0
    if not wanted:
        return written
    last = max(wanted)
    cap = cv2.VideoCapture(str(video_path))
    try:
        while idx <= last:
            ok, frame = cap.read()
            if not ok:
                break
            for r in wanted.get(idx, []):
                img = draw_overlay(_fit(frame, FRAME_MAX_SIDE), r['points'], r.get('elbow_angle_deg'),
                                   r.get('label'), r.get('caption'), r.get('time_s'))
                if cv2.imwrite(str(out_dir / r['name']), img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
                    written.add(r['name'])
            idx += 1
    finally:
        cap.release()
    return written


def render_overlay_video(video_path, rows, windows, out_path):
    """Annotated MP4 of the clip (first OVERLAY_MAX_SECONDS). rows need 'frame_idx', 'time_s', 'points',
    'elbow_angle_deg'; windows: [(throw_no, start_s, end_s)]. Returns out_path or None if no H.264 encoder."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    ok, frame = cap.read()
    if not ok:
        cap.release()
        return None
    first = _fit(frame, OVERLAY_MAX_SIDE)
    size = (first.shape[1], first.shape[0])
    out_path = Path(out_path)
    target, writer, transcode = out_path, None, False
    for codec in OVERLAY_CODECS:
        wr = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*codec), fps, size)
        if wr.isOpened():
            writer = wr
            break
        wr.release()
    if writer is None and shutil.which('ffmpeg'):
        target, transcode = out_path.with_suffix('.raw.mp4'), True
        writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
        if not writer.isOpened():
            writer = None
    if writer is None:
        cap.release()
        out_path.unlink(missing_ok=True)
        return None
    sampled = [r['frame_idx'] for r in rows]
    j, idx, limit = 0, 0, int(fps * OVERLAY_MAX_SECONDS)
    try:
        while ok and idx < limit:
            while j + 1 < len(rows) and sampled[j + 1] <= idx:
                j += 1
            r = rows[j]
            t = idx / fps
            active = next((n for n, a, b in windows if a <= t <= b), None)
            img = draw_overlay(_fit(frame, OVERLAY_MAX_SIDE), r['points'], r['elbow_angle_deg'],
                               f'Throw {active}' if active else 'Oche Lab', None, t)
            if img.shape[1] != size[0] or img.shape[0] != size[1]:
                img = cv2.resize(img, size)
            writer.write(img)
            ok, frame = cap.read()
            idx += 1
    finally:
        writer.release()
        cap.release()
    if transcode:
        try:
            subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(target), '-c:v', 'libx264', '-preset', 'veryfast',
                            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-an', str(out_path)],
                           check=True, timeout=FFMPEG_TIMEOUT_S, capture_output=True)
        except (subprocess.SubprocessError, OSError):
            out_path.unlink(missing_ok=True)
        finally:
            target.unlink(missing_ok=True)
    return out_path if out_path.exists() and out_path.stat().st_size > 0 else None
