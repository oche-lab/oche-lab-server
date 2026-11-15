"""Oche Lab experimental motion tracker (side-on or front-on camera). Not a validated biomechanics device."""
import os
import re
import shutil
import tempfile
import time
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from analysis import (TIPS_NOTE, body_frame, build_review, front_metrics, front_session_tips, front_throw_tips,
                      pick_phases, segment_throws, session_tips, summarise_throw, throw_tips)
from annotate import render_key_frames, render_overlay_video

VERSION = '0.7.0'
VIEWS = ('side', 'front')   # camera angle: side-on (default) or front-on

app = FastAPI(title='Oche Lab Motion API', version=VERSION)
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])  # local prototype only

# MediaPipe Pose landmark indices. Left/right are the thrower's own sides.
ARM_KEYS = {
    'right': {'head': 0, 'shoulder': 12, 'elbow': 14, 'wrist': 16, 'other_shoulder': 11},
    'left': {'head': 0, 'shoulder': 11, 'elbow': 13, 'wrist': 15, 'other_shoulder': 12},
}
MAX_BYTES = 150 * 1024 * 1024
MAX_SECONDS = 75               # clips up to ~1 minute (with a little slack)

# Per-analysis review assets (annotated JPEGs, optional overlay MP4) live in a temp dir keyed by id.
RESULTS_ROOT = Path(os.environ.get('OCHELAB_RESULTS_DIR', Path(tempfile.gettempdir()) / 'ochelab_results'))
RESULT_TTL_S = 60 * 60          # assets are deleted after an hour
MAX_STORED_ANALYSES = 20        # and only the newest N analyses are kept
ID_RE = re.compile(r'^[0-9a-f]{32}$')
FRAME_RE = re.compile(r'^t\d{1,2}_(setup|backswing|release|follow_through)\.jpg$')


def cleanup_results(now=None):
    """Delete expired analyses and keep at most MAX_STORED_ANALYSES (newest first)."""
    if not RESULTS_ROOT.exists():
        return
    now = now or time.time()
    dirs = sorted((d for d in RESULTS_ROOT.iterdir() if d.is_dir() and ID_RE.match(d.name)),
                  key=lambda d: d.stat().st_mtime, reverse=True)
    for k, d in enumerate(dirs):
        if k >= MAX_STORED_ANALYSES or now - d.stat().st_mtime > RESULT_TTL_S:
            shutil.rmtree(d, ignore_errors=True)


def angle(a, b, c):
    if any(x is None for x in (a, b, c)):
        return None
    v = np.array(a) - np.array(b)
    w = np.array(c) - np.array(b)
    denom = np.linalg.norm(v) * np.linalg.norm(w)
    if denom < 1e-8:
        return None
    return round(float(np.degrees(np.arccos(np.clip(np.dot(v, w) / denom, -1, 1)))), 1)


def dist(a, b):
    return None if a is None or b is None else float(np.linalg.norm(np.array(a) - np.array(b)))


# LOW_MEM=1 (e.g. Render free tier: 512 MB, 0.1 CPU) uses the lite pose model on smaller frames.
LOW_MEM = os.environ.get('LOW_MEM', '').lower() in ('1', 'true', 'yes')
MODEL_COMPLEXITY = 0 if LOW_MEM else 1
MAX_POSE_DIM = 640 if LOW_MEM else 960


def extract_rows(path, arm='right'):
    """Run MediaPipe Pose on sampled frames. Returns (rows, fps). Heavy imports stay local."""
    import mediapipe as mp
    keys = ARM_KEYS[arm]
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if count < 15:
        cap.release()
        raise ValueError('Video too short. Record at least three complete throws.')
    if count / fps > MAX_SECONDS:
        cap.release()
        raise ValueError('Video is longer than 60 seconds. Trim it and try again.')
    # Sample <= 15 frames per second; this is NOT high-speed release analysis.
    stride = max(1, round(fps / 15))
    rows = []
    pose_api = mp.solutions.pose
    with pose_api.Pose(static_image_mode=False, model_complexity=MODEL_COMPLEXITY, min_detection_confidence=0.55,
                       min_tracking_confidence=0.5) as pose:
        idx = 0
        while True:
            if idx % stride:  # grab() skips the colour conversion of frames we don't analyse
                if not cap.grab():
                    break
                idx += 1
                continue
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            if max(h, w) > MAX_POSE_DIM:
                scale = MAX_POSE_DIM / max(h, w)
                frame = cv2.resize(frame, None, fx=scale, fy=scale)
            result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            coords = {}
            for key, landmark_idx in keys.items():
                if result.pose_landmarks:
                    p = result.pose_landmarks.landmark[landmark_idx]
                    coords[key] = [round(p.x, 5), round(p.y, 5)] if p.visibility >= 0.55 else None
                else:
                    coords[key] = None
            rows.append(make_row(idx, fps, coords))
            idx += 1
            if len(rows) > 1200:
                break
    cap.release()
    return rows, fps


def make_row(frame_idx, fps, coords):
    shoulder, elbow, wrist = coords.get('shoulder'), coords.get('elbow'), coords.get('wrist')
    # Head position relative to shoulder mitigates some camera drift.
    head_relative = ([coords['head'][0]-shoulder[0], coords['head'][1]-shoulder[1]]
                     if coords.get('head') and shoulder else None)
    return {'frame_idx': frame_idx, 'time_s': round(frame_idx/fps, 3), 'points': coords,
            'elbow_angle_deg': angle(shoulder, elbow, wrist), 'head_relative': head_relative}


def build_result(path, rows, fps, arm='right', overlay=False, analysis_id=None, view='side'):
    """Segment throws, compute metrics/tips, render annotated key frames (+ optional overlay MP4)."""
    if len(rows) < 15:
        raise ValueError('Not enough video frames could be analysed.')
    detected = sum(r['points']['elbow'] is not None and r['points']['wrist'] is not None for r in rows)
    if detected / len(rows) < 0.55:
        raise ValueError('Throwing arm not visible enough. ' + ('Film from beside the board at eye height with your whole upper body in frame.'
                         if view == 'front' else 'Use a fixed side-on camera and good lighting.'))
    # Automatic throw detection from wrist/elbow speed peaks; equal thirds only as a fallback.
    seg = segment_throws(rows)
    summaries = [summarise_throw(rows, sg['indices'], i+1, sg['peak_index'], seg['wrist_speed'])
                 for i, sg in enumerate(seg['segments'])]
    frame = body_frame(rows) if view == 'front' else None
    if view == 'front':  # sideways alignment/sway metrics; side-on-only elbow metrics stay in but are flagged n/a
        for t, sg in zip(summaries, seg['segments']):
            t.update(front_metrics(rows, sg['indices'], pick_phases(rows, sg['indices'], sg['peak_index'], seg['wrist_speed']), frame))
    for t, tips in zip(summaries, (front_throw_tips if view == 'front' else throw_tips)(summaries)):
        t['tips'] = tips
    if view == 'front':  # elbow angles are foreshortened from the front: not reported
        for t in summaries:
            t['elbow_angle_mean_deg'] = t['elbow_angle_range_deg'] = None
    arm_pct = round(100*detected/len(rows), 1)

    analysis_id = analysis_id or uuid.uuid4().hex
    out_dir = RESULTS_ROOT / analysis_id
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f'/analysis/{analysis_id}/frames/'
    key_frames, timeline = build_review(rows, seg['segments'], summaries, seg['wrist_speed'], prefix, view)
    requests = [{'frame_idx': rows[kf['row_index']]['frame_idx'], 'name': kf['frame'],
                 'points': rows[kf['row_index']]['points'], 'elbow_angle_deg': kf['elbow_angle_deg'],
                 'label': f"Throw {t['throw']} - {kf['label']}", 'caption': kf['caption'], 'time_s': kf['time_s'],
                 'view': view, 'body': frame}
                for t, frames in zip(summaries, key_frames) for kf in frames]
    try:
        written = render_key_frames(path, requests, out_dir)
    except Exception:  # rendering must never break the metrics response
        written = set()
    for t, frames in zip(summaries, key_frames):
        for kf in frames:
            if kf['frame'] not in written:
                kf['frame_url'] = None
            kf.pop('row_index', None); kf.pop('frame', None)
        t['key_frames'] = frames
    ok_urls = {prefix + n for n in written}
    for e in timeline:
        if e['frame_url'] not in ok_urls:
            e['frame_url'] = None
    overlay_url, overlay_note = None, None
    if overlay:
        try:
            done = render_overlay_video(path, rows, [(t['throw'], t['start_s'], t['end_s'], t.get('peak_time_s')) for t in summaries],
                                        out_dir / 'overlay.mp4', view=view, body=frame)
        except Exception:
            done = None
        overlay_url = f'/analysis/{analysis_id}/overlay.mp4' if done else None
        overlay_note = None if done else 'Overlay video skipped: no H.264 encoder (install ffmpeg on the server to enable it).'

    return {'status':'experimental', 'version':VERSION, 'analysis_id':analysis_id, 'expires_in_s':RESULT_TTL_S,
            'arm':arm, 'fps_source':round(fps,2), 'sampled_frames':len(rows),
            'arm_detected_pct':arm_pct,
            'segmentation_method':seg['method'], 'segmentation':seg['sentence'],
            'view':view,
            'session_tips':(front_session_tips if view == 'front' else session_tips)(summaries, seg['method'], arm_pct),
            'tips_note':TIPS_NOTE,
            'measurement_note':('Front-on: sideways offsets are in shoulder widths (+ = out from the body); elbow drop and '
                                'elbow/release angles are side-on measurements and are not reported. ' if view == 'front' else '')
                               + '2D normalized image coordinates; camera perspective and body size affect values. Not validated as coaching advice.',
            'not_measured':(['elbow_drop', 'release_angle', 'follow_through_angle'] if view == 'front' else []),
            'throws':summaries, 'timeline':timeline, 'overlay_url':overlay_url, 'overlay_note':overlay_note,
            'trajectory':[{'time_s':r['time_s'], 'head':r['points']['head'], 'elbow':r['points']['elbow'],
                           'wrist':r['points']['wrist'], 'elbow_angle_deg':r['elbow_angle_deg']}
                          for r in rows[::max(1,len(rows)//120)]]}


def analyse(path, arm='right', overlay=False, view='side'):
    rows, fps = extract_rows(path, arm)
    return build_result(path, rows, fps, arm, overlay, view=view)


@app.get('/health')
def health():
    return {'ok':True, 'version':VERSION, 'views':list(VIEWS)}


@app.get('/analysis/{analysis_id}/frames/{name}')
def get_frame(analysis_id: str, name: str):
    if not ID_RE.match(analysis_id) or not FRAME_RE.match(name):
        raise HTTPException(404, 'Not found.')
    f = RESULTS_ROOT / analysis_id / name
    if not f.is_file():
        raise HTTPException(404, 'Frame expired or not found. Re-run the analysis.')
    return FileResponse(f, media_type='image/jpeg', headers={'Cache-Control': 'private, max-age=3600'})


@app.get('/analysis/{analysis_id}/overlay.mp4')
def get_overlay(analysis_id: str):
    f = RESULTS_ROOT / analysis_id / 'overlay.mp4' if ID_RE.match(analysis_id) else None
    if f is None or not f.is_file():
        raise HTTPException(404, 'Overlay expired or not found.')
    return FileResponse(f, media_type='video/mp4')


# Background jobs: the phone uploads, gets a job id straight away, then polls. This keeps long
# (up to 1 minute) clips from hitting proxy/tunnel request timeouts while analysis runs.
JOBS = {}
JOBS_LOCK = threading.Lock()
JOB_TTL_S = RESULT_TTL_S
EXECUTOR = ThreadPoolExecutor(max_workers=1)


def _set_job(job_id, **fields):
    with JOBS_LOCK:
        JOBS.setdefault(job_id, {}).update(fields, updated=time.time())


def _run_job(job_id, path, arm, overlay, view='side'):
    _set_job(job_id, status='running')
    try:
        _set_job(job_id, status='done', result=analyse(path, arm, overlay, view))
    except ValueError as e:
        _set_job(job_id, status='error', error=str(e))
    except Exception as e:  # pragma: no cover - surfaced to the app as a readable message
        _set_job(job_id, status='error', error=f'Analysis failed: {e}')
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _prune_jobs(now=None):
    now = now or time.time()
    with JOBS_LOCK:
        for k in [k for k, j in JOBS.items() if now - j.get('updated', now) > JOB_TTL_S]:
            del JOBS[k]


async def _save_upload(video, arm):
    arm = (arm or 'right').strip().lower()
    if arm not in ARM_KEYS:
        raise HTTPException(400, "arm must be 'left' or 'right'.")
    suffix = Path(video.filename or 'throw.mp4').suffix.lower()
    if suffix not in {'.mp4','.mov','.m4v'}:
        raise HTTPException(400, 'Please upload MP4 or MOV video.')
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        path = tmp.name
        size = 0
        try:
            while chunk := await video.read(1024*1024):
                size += len(chunk)
                if size > MAX_BYTES:
                    raise HTTPException(413, 'Video exceeds 150 MB prototype limit.')
                tmp.write(chunk)
        except Exception:
            os.unlink(path)
            raise
    return path, arm


@app.post('/analyse')
async def analyse_video(video: UploadFile = File(...), arm: str = Form('right'), overlay: bool = Form(False),
                        mode: str = Form('sync'), view: str = Form('side')):
    view = (view or 'side').strip().lower()
    if view not in VIEWS:
        raise HTTPException(400, "view must be 'side' or 'front'.")
    path, arm = await _save_upload(video, arm)
    cleanup_results()
    if mode == 'job':
        _prune_jobs()
        job_id = uuid.uuid4().hex
        _set_job(job_id, status='queued')
        EXECUTOR.submit(_run_job, job_id, path, arm, overlay, view)
        return {'job_id': job_id, 'status': 'queued'}
    try:
        return await run_in_threadpool(analyse, path, arm, overlay, view)
    except ValueError as e:
        raise HTTPException(422, str(e))
    finally:
        os.unlink(path)


@app.get('/jobs/{job_id}')
def get_job(job_id: str):
    with JOBS_LOCK:
        job = dict(JOBS.get(job_id) or {}) if ID_RE.match(job_id) else None
    if not job:
        raise HTTPException(404, 'Analysis not found. It may have expired; please try again.')
    job.pop('updated', None)
    return job
