"""Branded 9:16 share clip: ffmpeg compose of a skeleton replay, score HUD and end card.

This module is a self-contained FastAPI router so it can land beside other server
work with a one-line include in main.py. Inputs and outputs live in a temp dir
and are deleted as soon as the file has been sent (leftovers are swept after
SHARE_TTL_S). Designed for the 512 MB Render plan: one short H.264 encode,
veryfast preset, a single thread, no Python video decode.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

import guard

W, H = 720, 1280
FPS = 30
END_S = 2.5
MAX_CLIP_S = 18.0          # source footage cap (plus the end card)
MAX_UPLOAD = 40 * 1024 * 1024
SHARE_TTL_S = 10 * 60
MAX_STORED = 4
FFMPEG_TIMEOUT_S = 90
OPEN_LINK = 'https://oche-lab-api.onrender.com/open'
# Neon red / matte black (OpenCV is BGR / BGRA).
RED = (61, 31, 255)
RED_A = (61, 31, 255, 255)
BLACK = (12, 11, 11)
WHITE = (255, 255, 255)
MUTED = (160, 160, 168)
FONT = cv2.FONT_HERSHEY_SIMPLEX

SHARE_ROOT = Path(os.environ.get('OCHELAB_SHARE_DIR', Path(tempfile.gettempdir()) / 'ochelab_shareclips'))
_lock = threading.Lock()
router = APIRouter()


def _ascii(s, n=32):
    t = ''.join(ch if 32 <= ord(ch) < 127 else ' ' for ch in str(s or '')).strip()
    return re.sub(r'\s+', ' ', t)[:n]


def _num(s, lo=None, hi=None):
    if s is None or s == '':
        return None
    try:
        v = float(s)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    if lo is not None:
        v = max(lo, v)
    if hi is not None:
        v = min(hi, v)
    return v


def parse_stats(raw):
    """Up to three {label, value} tiles from a JSON form field or a list."""
    if raw in (None, '', b''):
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    out = []
    if isinstance(raw, list):
        for item in raw[:3]:
            if not isinstance(item, dict):
                continue
            label = _ascii(item.get('label'), 16)
            value = _ascii(item.get('value'), 12)
            if label or value:
                out.append({'label': label or 'Stat', 'value': value or '–'})
    return out


def clip_window(start_s, end_s, duration, pad=0.4, max_s=MAX_CLIP_S):
    """Pick a [start, length] trim in seconds from optional throw bounds and the source duration."""
    dur = max(0.0, float(duration or 0))
    if dur <= 0:
        return 0.0, max_s
    ss, ee = _num(start_s, 0, dur), _num(end_s, 0, dur)
    if ss is not None and ee is not None and ee > ss:
        a = max(0.0, ss - pad)
        b = min(dur, ee + pad)
        if b - a > max_s:
            # Keep the release (end of the throw window) in frame.
            a = max(0.0, b - max_s)
        return a, max(0.2, b - a)
    if dur <= max_s:
        return 0.0, dur
    return 0.0, max_s


def _put(img, text, org, scale, color, thickness=2, align='left'):
    text = _ascii(text, 48)
    if not text:
        return
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thickness)
    x, y = org
    if align == 'center':
        x -= tw // 2
    elif align == 'right':
        x -= tw
    shadow = (0, 0, 0, 255) if img.shape[2] == 4 else (0, 0, 0)
    cv2.putText(img, text, (int(x) + 2, int(y) + 2), FONT, scale, shadow, thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (int(x), int(y)), FONT, scale, color, thickness, cv2.LINE_AA)
    return tw, th


def _wordmark(img, cx, cy, scale=1.15, accent=True):
    if accent:
        r = int(22 * scale)
        cv2.circle(img, (cx - int(118 * scale) - r, cy - int(6 * scale)), r, RED_A if img.shape[2] == 4 else RED, 2, cv2.LINE_AA)
        cv2.circle(img, (cx - int(118 * scale) - r, cy - int(6 * scale)), max(3, r // 4), RED_A if img.shape[2] == 4 else RED, -1, cv2.LINE_AA)
    _put(img, 'OCHE LAB', (cx, cy), scale, (255, 255, 255, 255) if img.shape[2] == 4 else WHITE, max(2, int(round(scale * 2))), align='center')


def draw_hud(path, overall, stats, name='', view='side'):
    """720x1280 BGRA overlay: wordmark at the top, score + tiles at the bottom, clear centre."""
    img = np.zeros((H, W, 4), dtype=np.uint8)
    top, bot = 132, 250
    for y in range(top):
        a = int(235 * (1 - y / top) ** 0.55)
        img[y, :] = (BLACK[0], BLACK[1], BLACK[2], a)
    for i, y in enumerate(range(H - bot, H)):
        a = int(240 * (i / bot) ** 0.45)
        img[y, :] = (BLACK[0], BLACK[1], BLACK[2], min(250, a + 40))
    _wordmark(img, W // 2, 78, scale=1.05)
    tag = 'FRONT-ON' if view == 'front' else 'SIDE-ON'
    _put(img, tag, (W // 2, 112), 0.55, (200, 200, 210, 255), 1, align='center')

    score = _ascii(overall if overall is not None and overall != '' else '–', 4)
    _put(img, 'OVERALL', (36, H - 214), 0.5, (*RED, 255), 2)
    _put(img, str(score), (32, H - 130), 2.2, (255, 255, 255, 255), 5)
    if name:
        _put(img, _ascii(name, 22), (36, H - 92), 0.65, (220, 220, 230, 255), 2)

    tiles = parse_stats(stats) if isinstance(stats, (str, bytes)) else list(stats or [])
    tiles = tiles[:3]
    if tiles:
        tw = 150
        gap = 10
        total = len(tiles) * tw + (len(tiles) - 1) * gap
        x0 = W - 28 - total
        y0 = H - 118
        for i, t in enumerate(tiles):
            x = x0 + i * (tw + gap)
            overlay = img[y0:y0 + 86, x:x + tw]
            overlay[:] = (30, 28, 32, 220)
            cv2.rectangle(img, (x, y0), (x + tw, y0 + 86), (80, 70, 255, 180), 1, cv2.LINE_AA)
            _put(img, t['label'].upper(), (x + 12, y0 + 28), 0.42, (180, 180, 190, 255), 1)
            _put(img, t['value'], (x + 12, y0 + 68), 0.9, (255, 255, 255, 255), 2)

    cv2.imwrite(str(path), img)
    return path


def draw_end_card(path, link=OPEN_LINK):
    """720x1280 matte-black end card with the wordmark, coaching line and /open link."""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:] = BLACK
    # Soft red glow behind the wordmark.
    yy, xx = np.ogrid[:H, :W]
    glow = np.exp(-((xx - W / 2) ** 2 + (yy - H / 2 + 40) ** 2) / (2 * 180 ** 2))
    for c, v in enumerate(RED):
        img[:, :, c] = np.clip(img[:, :, c] + glow * (v * 0.35), 0, 255).astype(np.uint8)
    cv2.line(img, (W // 2 - 80, H // 2 - 70), (W // 2 + 80, H // 2 - 70), RED, 3, cv2.LINE_AA)
    _wordmark(img, W // 2, H // 2 - 20, scale=1.5)
    _put(img, 'Get coached at Oche Lab', (W // 2, H // 2 + 70), 0.95, WHITE, 2, align='center')
    # Drop the scheme so the line stays readable at this size.
    shown = _ascii((link or OPEN_LINK).replace('https://', '').replace('http://', ''), 40)
    _put(img, shown, (W // 2, H // 2 + 130), 0.65, MUTED, 2, align='center')
    cv2.imwrite(str(path), img)
    return path


def _ffprobe_duration(path):
    try:
        r = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=nw=1:nk=1', str(path)],
            capture_output=True, text=True, timeout=15, check=False,
        )
        return float(r.stdout.strip()) if r.returncode == 0 and r.stdout.strip() else 0.0
    except (ValueError, subprocess.TimeoutExpired, FileNotFoundError):
        return 0.0


def _run_ffmpeg(args, cwd=None):
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT_S, cwd=cwd)
    except FileNotFoundError:
        raise RuntimeError('ffmpeg is not installed on the server.')
    except subprocess.TimeoutExpired:
        raise RuntimeError('The share clip took too long to render. Try a shorter throw.')
    if r.returncode != 0:
        err = (r.stderr or r.stdout or 'ffmpeg failed').strip().splitlines()
        raise RuntimeError(err[-1][:240] if err else 'ffmpeg failed')
    return r


def render_share_clip(src, out, overall=None, stats=None, name='', view='side', start_s=None, end_s=None, link=OPEN_LINK):
    """Compose a 9:16 branded MP4 from a skeleton-replay source. Returns the output path."""
    if not shutil.which('ffmpeg'):
        raise RuntimeError('ffmpeg is not installed on the server.')
    src, out = Path(src), Path(out)
    if not src.is_file() or src.stat().st_size < 32:
        raise ValueError('The replay film is missing or unreadable.')
    duration = _ffprobe_duration(src)
    ss, length = clip_window(start_s, end_s, duration)
    work = out.parent
    hud, end = work / 'hud.png', work / 'end.png'
    draw_hud(hud, overall if overall is not None else '', stats or [], name, view)
    draw_end_card(end, link)
    # One graph: trim + 9:16 pad + HUD overlay, then the end card.
    _run_ffmpeg([
        'ffmpeg', '-y', '-loglevel', 'error',
        '-ss', f'{ss:.3f}', '-t', f'{length:.3f}', '-i', str(src),
        '-i', str(hud),
        '-loop', '1', '-t', str(END_S), '-i', str(end),
        '-filter_complex',
        '[0:v]scale={w}:{h}:force_original_aspect_ratio=decrease,'
        'pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=0x0B0B0C,setsar=1,fps={fps},format=rgba[v0];'
        '[v0][1:v]overlay=0:0:format=auto,format=yuv420p[clip];'
        '[2:v]scale={w}:{h},setsar=1,fps={fps},format=yuv420p[end];'
        '[clip][end]concat=n=2:v=1:a=0[out]'.format(w=W, h=H, fps=FPS),
        '-map', '[out]', '-an',
        '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '26',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-threads', '1',
        str(out),
    ])
    if not out.is_file() or out.stat().st_size < 32:
        raise RuntimeError('The share clip could not be written.')
    return out


def cleanup_shares(now=None):
    """Delete expired share jobs and cap how many are kept."""
    if not SHARE_ROOT.exists():
        return
    now = now or time.time()
    dirs = sorted((d for d in SHARE_ROOT.iterdir() if d.is_dir()), key=lambda d: d.stat().st_mtime, reverse=True)
    for i, d in enumerate(dirs):
        if i >= MAX_STORED or now - d.stat().st_mtime > SHARE_TTL_S:
            shutil.rmtree(d, ignore_errors=True)


def _forget(path):
    try:
        shutil.rmtree(Path(path).parent, ignore_errors=True)
    except OSError:
        pass


@router.post('/share-clip')
async def share_clip(
    request: Request,
    video: UploadFile = File(...),
    overall: str = Form(''),
    stats: str = Form('[]'),
    name: str = Form(''),
    view: str = Form('side'),
    start_s: str = Form(''),
    end_s: str = Form(''),
):
    """Upload a skeleton-replay MP4; return a branded 9:16 share clip. Deleted after send."""
    # Same install/app-key check as /analyse, but this does not use the analysis quota.
    guard.authorize(request)
    if not shutil.which('ffmpeg'):
        raise HTTPException(503, 'Share clips need ffmpeg on the analysis server.')
    cleanup_shares()
    view = 'front' if str(view).lower().strip() == 'front' else 'side'
    job = uuid.uuid4().hex[:16]
    folder = SHARE_ROOT / job
    folder.mkdir(parents=True, exist_ok=True)
    src, dest = folder / 'src.mp4', folder / 'share.mp4'
    size = 0
    try:
        with src.open('wb') as f:
            while True:
                chunk = await video.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD:
                    raise HTTPException(413, 'That replay is too large to turn into a share clip.')
                f.write(chunk)
        if size < 32:
            raise HTTPException(400, 'The replay film is missing or unreadable.')
        try:
            render_share_clip(
                src, dest,
                overall=_ascii(overall, 4),
                stats=parse_stats(stats),
                name=_ascii(name, 22),
                view=view,
                start_s=_num(start_s, 0, 600),
                end_s=_num(end_s, 0, 600),
            )
        except ValueError as e:
            raise HTTPException(400, str(e))
        except RuntimeError as e:
            raise HTTPException(500, str(e))
    except HTTPException:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    finally:
        try:
            src.unlink(missing_ok=True)
        except OSError:
            pass
        await video.close()

    return FileResponse(dest, media_type='video/mp4', filename='oche-lab-clip.mp4', background=BackgroundTask(_forget, dest))
