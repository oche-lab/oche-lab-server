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
ARM = (61, 31, 255)        # neon red (BGR #FF1F3D): shoulder-elbow-wrist
HEAD = (11, 158, 245)      # amber: head-shoulder line
SHOULDERS = (215, 210, 210)  # off-white: shoulder line
WHITE = (255, 255, 255)
PANEL = (14, 12, 12)       # matte black HUD panels


# ---- Upright frames ------------------------------------------------------------------------------------
# iPhones store portrait video as landscape pixels plus a rotation tag. OpenCV's pip builds don't apply the
# tag by default, so pose detection, key frames and the overlay film all came out sideways. Every reader
# goes through open_video()/upright(): auto-rotation is switched off and the tag is applied explicitly, so the
# result is the same on any OpenCV build. The overlay MP4 is written from upright pixels with no rotation tag.
_ROTATE = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def _ffprobe_rotation(path):
    """Display rotation from the stream side data (ffprobe convention: + = counter-clockwise), or 0."""
    if not shutil.which('ffprobe'):
        return 0
    try:
        out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                              'stream_side_data=rotation:stream_tags=rotate', '-of', 'default=nw=1:nk=1', str(path)],
                             capture_output=True, text=True, timeout=15).stdout
        vals = [float(v) for v in out.split() if v.strip().lstrip('-').replace('.', '', 1).isdigit()]
        return int(round(vals[0])) if vals else 0
    except (subprocess.SubprocessError, OSError, ValueError):
        return 0


def video_rotation(cap, path=None):
    """Clockwise degrees (0/90/180/270) needed to make this capture's frames upright."""
    rot = 0
    try:
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
        if cap.get(cv2.CAP_PROP_ORIENTATION_AUTO):  # this build insists on rotating itself: nothing to add
            return 0
        rot = int(round(cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0))
    except Exception:  # very old OpenCV without the orientation properties
        rot = 0
    if rot % 360 == 0 and path is not None:
        rot = -_ffprobe_rotation(path)  # tag says counter-clockwise; we rotate clockwise
    rot %= 360
    return rot if rot in _ROTATE else 0


def open_video(path):
    """cv2.VideoCapture plus the clockwise rotation that makes its frames upright."""
    cap = cv2.VideoCapture(str(path))
    return cap, video_rotation(cap, path)


def upright(frame, rot):
    return cv2.rotate(frame, _ROTATE[rot]) if rot in _ROTATE else frame


def _ascii(text):
    return (text or '').replace('°', ' deg').replace('–', '-').replace('’', "'").encode('ascii', 'ignore').decode()


def _px(p, w, h):
    return None if p is None else (int(round(p[0] * w)), int(round(p[1] * h)))


def _panel(img, y0, y1, alpha=0.65):
    y0, y1 = max(0, y0), min(img.shape[0], y1)
    roi = img[y0:y1]
    img[y0:y1] = cv2.addWeighted(roi, 1 - alpha, np.full_like(roi, PANEL), alpha, 0)


def _chip(img, text, x, y, s, th, fill=PANEL, fg=WHITE):
    """Draw a left-aligned HUD chip; returns the next x."""
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.62 * s
    (tw, tht), _ = cv2.getTextSize(text, font, scale, max(1, th - 1))
    pad, hpad = int(6 * s), int(8 * s)
    cv2.rectangle(img, (x, y), (x + tw + 2 * hpad, y + tht + 2 * pad), fill, -1, cv2.LINE_AA)
    cv2.putText(img, text, (x + hpad, y + pad + tht), font, scale, fg, max(1, th - 1), cv2.LINE_AA)
    return x + tw + 2 * hpad + int(8 * s)


def draw_overlay(frame, points, elbow_angle=None, label=None, caption=None, time_s=None, badge=None, view='side', body=None):
    """Draw tracked arm, head/shoulder lines, elbow angle, header, optional HUD badge and caption.

    badge: extra chip such as 'Release 120 deg', shown next to the elbow-angle chip so release
    and follow-through angles stay readable even when the joint label is tucked off-frame.
    view='front': draws the vertical shoulder line the arm should stay on and reports the hand's sideways
    offset from it (body = (shoulder_width, outward_sign)) instead of the foreshortened elbow angle.
    """
    img = frame.copy()
    h, w = img.shape[:2]
    s = max(0.6, min(w, h) / 540)  # scale strokes/text with resolution
    th = max(2, int(3 * s))
    P = {k: _px(points.get(k), w, h) for k in ('head', 'shoulder', 'elbow', 'wrist', 'other_shoulder')}
    front = view == 'front'
    if front:
        elbow_angle = None
        if P['shoulder']:  # dashed vertical "line of the throw" through the throwing shoulder
            x = P['shoulder'][0]
            for y in range(0, h, int(18 * s)):
                cv2.line(img, (x, y), (x, min(h, y + int(10 * s))), HEAD, max(1, th - 1), cv2.LINE_AA)
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
    bar = 0
    if label or time_s is not None:
        bar = int(44 * s)
        _panel(img, 0, bar)
        if label:
            cv2.putText(img, _ascii(label), (int(12 * s), int(30 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s, WHITE, max(1, th - 1), cv2.LINE_AA)
        if time_s is not None:
            t = f'{time_s:.2f}s'
            (tw, _), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s, 1)
            cv2.putText(img, t, (w - tw - int(12 * s), int(30 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s, (200, 220, 230), 1, cv2.LINE_AA)
    chips = []
    off = front_offset(points, body) if front else None
    if off is not None:
        chips.append((f"Hand {abs(off) * 100:.0f}% {'out' if off > 0 else 'in'}", PANEL, WHITE))
    if elbow_angle is not None:
        chips.append((f'Elbow {elbow_angle:.0f} deg', PANEL, WHITE))
    if badge:
        chips.append((_ascii(badge), (45, 15, 200) if 'Release' in badge else PANEL, WHITE))
    if chips:
        x, y = int(12 * s), bar + int(8 * s)
        for text, fill, fg in chips:
            x = _chip(img, text, x, y, s, th, fill, fg)
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


def front_offset(points, body):
    """Wrist offset from the shoulder's vertical line in shoulder widths (+ = outward), or None."""
    if not body or not body[0] or not points.get('wrist') or not points.get('shoulder'):
        return None
    return (points['wrist'][0] - points['shoulder'][0]) * body[1] / body[0]


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
    cap, rot = open_video(video_path)
    try:
        while idx <= last:
            if idx not in wanted:  # skip decoding-to-BGR of frames we don't draw
                if not cap.grab():
                    break
                idx += 1
                continue
            ok, frame = cap.read()
            if not ok:
                break
            frame = upright(frame, rot)
            for r in wanted.get(idx, []):
                img = draw_overlay(_fit(frame, FRAME_MAX_SIDE), r['points'], r.get('elbow_angle_deg'),
                                   r.get('label'), r.get('caption'), r.get('time_s'), None, r.get('view', 'side'), r.get('body'))
                if cv2.imwrite(str(out_dir / r['name']), img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
                    written.add(r['name'])
            idx += 1
    finally:
        cap.release()
    return written


def _window_at(windows, t):
    """windows: (throw_no, start_s, end_s) or (throw_no, start_s, end_s, peak_s)."""
    for w in windows:
        n, a, b = w[0], w[1], w[2]
        if a <= t <= b:
            return n, a, b, (w[3] if len(w) > 3 else None)
    return None, None, None, None


def _phase_at(t, start, end, peak):
    if peak is None:
        return None
    if abs(t - peak) <= 0.18:
        return 'Release'
    if t < peak:
        return 'Take-back' if t >= (start + peak) / 2 else 'Set-up'
    return 'Follow-through'


def render_overlay_video(video_path, rows, windows, out_path, view='side', body=None, on_progress=None):
    """Annotated MP4 of the clip (first OVERLAY_MAX_SECONDS). rows need 'frame_idx', 'time_s', 'points',
    'elbow_angle_deg'; windows: [(throw_no, start_s, end_s[, peak_s])]. Returns out_path or None if no H.264 encoder."""
    cap, rot = open_video(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    ok, frame = cap.read()
    if not ok:
        cap.release()
        return None
    frame = upright(frame, rot)
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
    total = min(limit, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or limit)) or 1
    try:
        while ok and idx < limit:
            while j + 1 < len(rows) and sampled[j + 1] <= idx:
                j += 1
            r = rows[j]
            t = idx / fps
            n, start, end, peak = _window_at(windows, t)
            phase = _phase_at(t, start, end, peak) if n is not None else None
            label = f'Throw {n} - {phase}' if n is not None and phase else (f'Throw {n}' if n is not None else 'Oche Lab')
            ang = r.get('elbow_angle_deg')
            if view == 'front':
                off = front_offset(r['points'], body)
                badge = f"Release line {abs(off) * 100:.0f}% {'out' if off > 0 else 'in'}" if phase == 'Release' and off is not None else None
            else:
                badge = f'Release {ang:.0f} deg' if phase == 'Release' and ang is not None else None
            img = draw_overlay(_fit(frame, OVERLAY_MAX_SIDE), r['points'], ang, label, None, t, badge, view, body)
            if img.shape[1] != size[0] or img.shape[0] != size[1]:
                img = cv2.resize(img, size)
            writer.write(img)
            ok, frame = cap.read()
            if ok:
                frame = upright(frame, rot)
            idx += 1
            if on_progress and idx % 15 == 0:
                on_progress(min(1.0, idx / total))
    finally:
        writer.release()
        cap.release()
    if transcode:
        try:
            subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(target), '-c:v', 'libx264', '-preset', 'veryfast',
                            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-an', '-metadata:s:v:0', 'rotate=0', str(out_path)],
                           check=True, timeout=FFMPEG_TIMEOUT_S, capture_output=True)
        except (subprocess.SubprocessError, OSError):
            out_path.unlink(missing_ok=True)
        finally:
            target.unlink(missing_ok=True)
    return out_path if out_path.exists() and out_path.stat().st_size > 0 else None
