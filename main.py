"""Oche Lab experimental motion tracker (side-on or front-on camera). Not a validated biomechanics device."""
import logging
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
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

from analysis import (TIPS_NOTE, body_frame, build_review, front_metrics, front_session_tips, front_throw_tips,
                      pick_phases, segment_throws, session_tips, summarise_throw, throw_details, throw_mechanics, throw_tips)
from annotate import open_video, render_key_frames, render_overlay_video, upright
import guard

VERSION = '0.9.4'
log = logging.getLogger('oche.detect')
logging.basicConfig(level=logging.INFO)
VIEWS = ('side', 'front')   # camera angle: side-on (default) or front-on

app = FastAPI(title='Oche Lab Motion API', version=VERSION)
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])  # local prototype only

# MediaPipe Pose landmark indices. Left/right are the thrower's own sides.
ARM_KEYS = {
    'right': {'head': 0, 'shoulder': 12, 'elbow': 14, 'wrist': 16, 'other_shoulder': 11,
              'hip': 24, 'other_hip': 23, 'ankle': 28, 'other_ankle': 27},
    'left': {'head': 0, 'shoulder': 11, 'elbow': 13, 'wrist': 15, 'other_shoulder': 12,
             'hip': 23, 'other_hip': 24, 'ankle': 27, 'other_ankle': 28},
}
MAX_BYTES = 150 * 1024 * 1024
MAX_SECONDS = 75               # clips up to ~1 minute (with a little slack)

# Per-analysis review assets (annotated JPEGs, optional overlay MP4) live in a temp dir keyed by id.
RESULTS_ROOT = Path(os.environ.get('OCHELAB_RESULTS_DIR', Path(tempfile.gettempdir()) / 'ochelab_results'))
UPLOADS_ROOT = Path(os.environ.get('OCHELAB_UPLOADS_DIR', Path(tempfile.gettempdir()) / 'ochelab_uploads'))
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


def cleanup_disk():
    """If results + uploads exceed the disk budget, drop the oldest stored analyses first."""
    if not guard.cfg('disk_budget'):
        return
    paths = [RESULTS_ROOT, UPLOADS_ROOT]
    if not guard.over_budget(paths) or not RESULTS_ROOT.exists():
        return
    dirs = sorted((d for d in RESULTS_ROOT.iterdir() if d.is_dir() and ID_RE.match(d.name)),
                  key=lambda d: d.stat().st_mtime)
    for d in dirs:
        if not guard.over_budget(paths):
            break
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


def extract_rows(path, arm='right', on_progress=None):
    """Run MediaPipe Pose on sampled frames. Returns (rows, fps). Heavy imports stay local."""
    import mediapipe as mp
    keys = ARM_KEYS[arm]
    cap, rot = open_video(path)  # portrait phone clips: rotate to upright before pose detection
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
            frame = upright(frame, rot)
            h, w = frame.shape[:2]
            if max(h, w) > MAX_POSE_DIM:
                scale = MAX_POSE_DIM / max(h, w)
                frame = cv2.resize(frame, None, fx=scale, fy=scale)
            result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            coords, conf = {}, None
            for key, landmark_idx in keys.items():
                if result.pose_landmarks:
                    p = result.pose_landmarks.landmark[landmark_idx]
                    coords[key] = [round(p.x, 5), round(p.y, 5)] if p.visibility >= 0.55 else None
                else:
                    coords[key] = None
            if result.pose_landmarks:  # arm landmark confidence: throws seen through a blurry/partial pose are rejected
                lm = result.pose_landmarks.landmark
                conf = round(float(np.mean([lm[keys[k]].visibility for k in ('shoulder', 'elbow', 'wrist')])), 3)
            row = make_row(idx, fps, coords)
            row['conf'] = conf
            row['aspect'] = round(w / h, 4) if h else None  # upright frame width / height (front-on body scale)
            rows.append(row)
            idx += 1
            if on_progress and len(rows) % 10 == 0:
                on_progress(min(1.0, idx / max(count, 1)))
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


def frame_aspect(rows):
    """Upright frame width / height (the overlay film and every normalised coordinate use this frame), or None."""
    vals = [r['aspect'] for r in rows if r.get('aspect')]
    return round(float(np.median(vals)), 4) if vals else None


def build_result(path, rows, fps, arm='right', overlay=False, analysis_id=None, view='side', max_throws=None, progress=None):
    """Segment throws, compute metrics/tips, render annotated key frames (+ optional overlay MP4)."""
    if len(rows) < 15:
        raise ValueError('Not enough video frames could be analysed.')
    detected = sum(r['points']['elbow'] is not None and r['points']['wrist'] is not None for r in rows)
    if detected / len(rows) < 0.55:
        raise ValueError('Throwing arm not visible enough. ' + ('Film from beside the board at eye height with your whole upper body in frame.'
                         if view == 'front' else 'Use a fixed side-on camera and good lighting.'))
    progress = progress or (lambda stage, frac=0.0: None)
    progress('throws')
    # Automatic throw detection from wrist/elbow speed peaks; equal thirds only as a fallback.
    seg = segment_throws(rows, view=view, max_throws=max_throws)
    # Per-candidate reasons (times and motion only, no video) so missed / extra throws can be diagnosed from the logs.
    dbg = seg.get('debug') or {}
    log.info('detection %s view=%s %s | %s', analysis_id or '-', view, dbg.get('summary'),
             ' '.join(f"{c['time_s']}s:{(c['reason'] or 'kept').split(' (')[0].replace(' ', '_')}" for c in dbg.get('candidates', [])))
    summaries = [summarise_throw(rows, sg['indices'], i+1, sg['peak_index'], seg['wrist_speed'])
                 for i, sg in enumerate(seg['segments'])]
    frame = body_frame(rows) if view == 'front' else None
    if view == 'front':  # sideways alignment/sway metrics; side-on-only elbow metrics stay in but are flagged n/a
        for t, sg in zip(summaries, seg['segments']):
            t.update(front_metrics(rows, sg['indices'], pick_phases(rows, sg['indices'], sg['peak_index'], seg['wrist_speed']), frame))
    for t, sg in zip(summaries, seg['segments']):  # phase-by-phase angles and timing for the detailed review
        ph = pick_phases(rows, sg['indices'], sg['peak_index'], seg['wrist_speed'])
        t['mechanics'] = throw_mechanics(rows, sg['indices'], ph, view)
        try:  # detailed breakdown is additive: a failure here must never break the analysis
            t['mechanics']['detail'] = throw_details(rows, sg['indices'], ph, view, frame)
        except Exception:  # pragma: no cover
            log.exception('throw_details failed')
            t['mechanics']['detail'] = None
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
    progress('frames')
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
        progress('overlay', 0.0)
        try:
            done = render_overlay_video(path, rows, [(t['throw'], t['start_s'], t['end_s'], t.get('peak_time_s')) for t in summaries],
                                        out_dir / 'overlay.mp4', view=view, body=frame,
                                        on_progress=lambda f: progress('overlay', f))
        except Exception:
            done = None
        overlay_url = f'/analysis/{analysis_id}/overlay.mp4' if done else None
        overlay_note = None if done else 'Overlay video skipped: no H.264 encoder (install ffmpeg on the server to enable it).'

    return {'status':'experimental', 'version':VERSION, 'analysis_id':analysis_id, 'expires_in_s':RESULT_TTL_S,
            'arm':arm, 'fps_source':round(fps,2), 'sampled_frames':len(rows),
            'arm_detected_pct':arm_pct,
            'segmentation_method':seg['method'], 'segmentation':seg['sentence'], 'detection_debug': seg.get('debug'),
            'view':view,
            'session_tips':(front_session_tips if view == 'front' else session_tips)(summaries, seg['method'], arm_pct),
            'tips_note':TIPS_NOTE,
            'measurement_note':('Front-on: sideways offsets are in shoulder widths (+ = out from the body; the width is corrected for an '
                                'angled stance using torso length). Elbow drop, elbow/release angles and take-back timing are '
                                'side-on measurements and are not reported. ' if view == 'front' else '')
                               + '2D normalized image coordinates; camera perspective and body size affect values. Not validated as coaching advice.',
            'not_measured':(['elbow_drop', 'release_angle', 'follow_through_angle', 'tempo'] if view == 'front' else []),
            'throws':summaries, 'timeline':timeline, 'overlay_url':overlay_url, 'overlay_note':overlay_note,
            'frame_aspect':frame_aspect(rows),
            'trajectory':[{'time_s':r['time_s'], 'head':r['points']['head'], 'elbow':r['points']['elbow'],
                           'wrist':r['points']['wrist'], 'elbow_angle_deg':r['elbow_angle_deg']}
                          for r in rows[::max(1,len(rows)//120)]]}


# Job progress: each stage owns a slice of 0..1 so the app can show one honest progress bar.
STAGES = {'pose': (0.0, 0.62), 'throws': (0.62, 0.66), 'frames': (0.66, 0.72), 'overlay': (0.72, 0.99)}
STAGE_LABELS = {'queued': 'Waiting for the server', 'pose': 'Detecting your pose', 'throws': 'Finding throws & scoring',
                'frames': 'Drawing key frames', 'overlay': 'Rendering skeleton replay', 'done': 'Done'}


def analyse(path, arm='right', overlay=False, view='side', max_throws=None, report=None):
    """report(stage, overall_fraction) is called as the analysis moves through STAGES (optional)."""
    def progress(stage, frac=0.0):
        if report:
            lo, hi = STAGES[stage]
            report(stage, round(lo + (hi - lo) * max(0.0, min(1.0, frac)), 3))
    progress('pose')
    rows, fps = extract_rows(path, arm, on_progress=lambda f: progress('pose', f))
    return build_result(path, rows, fps, arm, overlay, view=view, max_throws=max_throws, progress=progress)


# Tester invite page: a tappable https link (WhatsApp won't link exp://) that hands off to Expo Go.
# Expo Go registers the exp:// and exps:// schemes (exps = https). Its only https universal links are
# expo.dev/expo-go (opens Expo Go's home) and legacy exp.host/@owner/slug, so EAS Update URLs need a scheme.
EXPO_UPDATE_PATH = 'u.expo.dev/9627303a-4e67-4ff7-9faa-ce2bc73ee28d?runtime-version=exposdk%3A58.0.0&channel-name=main'
EXPO_GO_URL = 'exp://' + EXPO_UPDATE_PATH     # the form Expo's own QR codes use for EAS Update
EXPO_GO_URL_ALT = 'exps://' + EXPO_UPDATE_PATH  # same manifest, explicitly over https
EXPO_GO_HOME = 'https://expo.dev/expo-go'       # universal link: opens Expo Go, else its store page
EXPO_GO_IOS = 'https://apps.apple.com/app/expo-go/id982107779'
EXPO_GO_ANDROID = 'https://play.google.com/store/apps/details?id=host.exp.exponent'
OPEN_PAGE = """<!doctype html><html lang="en-GB"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Open Oche Lab</title><meta name="theme-color" content="#0a0a0b">
<meta property="og:title" content="Oche Lab: AI darts coach"><meta property="og:description" content="Tap to open the Oche Lab beta in Expo Go.">
<style>
*{box-sizing:border-box}html,body{margin:0;background:#0a0a0b;color:#f4f4f5;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
main{max-width:440px;margin:0 auto;padding:44px 22px 40px;text-align:center}
.mark{font-size:46px;font-weight:900;letter-spacing:-1px;color:#ff2d3d;text-shadow:0 0 12px rgba(255,45,61,.75),0 0 32px rgba(255,45,61,.45);margin:0}
.tag{color:#a1a1aa;margin:10px 0 24px;font-size:15px}
.banner{display:none;background:#2a0b0e;border:1px solid #ff2d3d;border-radius:14px;padding:14px 16px;margin:0 0 20px;text-align:left;font-size:15px;line-height:1.5}
.banner b{color:#ff5c69}
.btn{display:block;width:100%;background:#ff2d3d;color:#0a0a0b;font-size:20px;font-weight:800;text-decoration:none;padding:19px 22px;border-radius:16px;box-shadow:0 0 26px rgba(255,45,61,.55);border:0;font-family:inherit}
.btn2{display:block;width:100%;margin-top:12px;background:transparent;color:#ff5c69;font-size:16px;font-weight:700;text-decoration:none;padding:14px;border-radius:14px;border:1.5px solid #ff2d3d;font-family:inherit}
.btn:active,.btn2:active{transform:scale(.98)}
h2{font-size:16px;margin:30px 0 8px;text-align:left;color:#f4f4f5}
ol{text-align:left;background:#141416;border:1px solid #26262b;border-radius:16px;padding:16px 16px 16px 38px;margin:0;line-height:1.55;font-size:15px}
li{margin:6px 0}ol a{color:#ff5c69;font-weight:600}
.note{font-size:13px;color:#a1a1aa;line-height:1.5;border-left:3px solid #ff2d3d;text-align:left;padding:4px 0 4px 12px;margin-top:22px}
.raw{margin-top:22px;font-size:11px;color:#71717a;word-break:break-all;user-select:all;-webkit-user-select:all}
#copied{color:#4ade80;font-size:14px;min-height:20px;margin-top:8px}
</style></head><body><main>
<h1 class="mark">Oche Lab</h1><p class="tag">AI darts coach &middot; beta</p>
<div class="banner" id="inapp">You&rsquo;re in an app&rsquo;s built-in browser, which blocks opening other apps. Tap <b>&middot;&middot;&middot;</b> (or the compass or globe icon), then <b>Open in Safari</b> (or your browser) first. Then tap the red button.</div>
<a class="btn" id="open" href="__URL__">Open in Expo Go</a>
<a class="btn2" href="__ALT__">Didn&rsquo;t open? Try the alternative link</a>
<button class="btn2" id="copy" type="button">Copy link</button>
<div id="copied" aria-live="polite"></div>

<h2>First time?</h2>
<ol>
<li>Install <b>Expo Go</b>: <a href="__IOS__">App Store</a> or <a href="__ANDROID__">Google Play</a>.</li>
<li>Open Expo Go and sign in with the Expo account you were added with.</li>
<li>Come back here and tap <b>Open in Expo Go</b>.</li>
</ol>

<h2>Still nothing? Paste the link instead</h2>
<ol>
<li>Tap <b>Copy link</b> above.</li>
<li>Open <b>Safari</b>, tap the address bar, paste and tap <b>Go</b>. Choose <b>Open</b> if it asks to open Expo Go.</li>
<li>On Android you can also open Expo Go (<a href="__HOME__">open it here</a>), tap <b>Enter URL manually</b> and paste.</li>
</ol>
<p class="note">If Expo Go says <b>&ldquo;requires authentication&rdquo;</b>, you haven&rsquo;t been added yet, or you&rsquo;re signed in with a different account. Message the person who sent you this link.<br><br>Once it&rsquo;s opened once, Oche Lab stays in Expo Go under <b>Recently opened</b>.</p>
<p class="raw" id="raw">__URL__</p>
</main><script>
(function(){
  var URL="__URL__", ua=navigator.userAgent||"";
  var inApp=/FBAN|FBAV|FB_IAB|FBIOS|Instagram|Twitter|Line\\/|WhatsApp|Snapchat|LinkedInApp|Pinterest|TikTok|musical_ly|BytedanceWebview|GSA\\/|MicroMessenger|Messenger|Gmail|GoogleApp|KAKAOTALK|\\bwv\\b/i.test(ua);
  var iosWebView=/(iPhone|iPod|iPad).*AppleWebKit(?!.*Safari)/i.test(ua) && !/CriOS|FxiOS|EdgiOS/i.test(ua);
  if(inApp||iosWebView){document.getElementById("inapp").style.display="block";}
  function done(ok){document.getElementById("copied").textContent=ok?"Copied. Now paste it into Safari\\u2019s address bar.":"Press and hold the link at the bottom to copy it.";}
  function fallback(){var t=document.createElement("textarea");t.value=URL;t.setAttribute("readonly","");t.style.position="fixed";t.style.opacity="0";document.body.appendChild(t);t.select();t.setSelectionRange(0,URL.length);var ok=false;try{ok=document.execCommand("copy");}catch(e){}document.body.removeChild(t);done(ok);}
  document.getElementById("copy").addEventListener("click",function(){
    if(navigator.clipboard&&window.isSecureContext){navigator.clipboard.writeText(URL).then(function(){done(true);},fallback);}else{fallback();}
  });
})();
</script>
</body></html>""".replace('__URL__', EXPO_GO_URL).replace('__ALT__', EXPO_GO_URL_ALT).replace('__HOME__', EXPO_GO_HOME) \
    .replace('__IOS__', EXPO_GO_IOS).replace('__ANDROID__', EXPO_GO_ANDROID)


@app.get('/open', response_class=HTMLResponse)
@app.get('/try', response_class=HTMLResponse, include_in_schema=False)
def open_in_expo_go():
    return HTMLResponse(OPEN_PAGE, headers={'Cache-Control':'no-cache'})


@app.get('/health')
def health():
    return {'ok':True, 'version':VERSION, 'views':list(VIEWS), 'features':['job_progress', 'mechanics', 'mechanics_detail', 'queue_position', 'job_retry', 'install_id']}


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
        mem = dict(JOBS[job_id])
    guard.save_job(job_id, **mem)
    # Keep RAM small on the 512 MB host: once a result is on disk, drop it from memory.
    if mem.get('status') in ('done', 'error', 'lost') and 'result' in mem:
        with JOBS_LOCK:
            if job_id in JOBS:
                JOBS[job_id].pop('result', None)


def _run_job(job_id, path, arm, overlay, view='side', max_throws=None):
    _set_job(job_id, status='running', stage='pose', stage_label=STAGE_LABELS['pose'], progress=0.0)
    report = lambda stage, frac: _set_job(job_id, stage=stage, stage_label=STAGE_LABELS[stage], progress=frac)
    try:
        result = analyse(path, arm, overlay, view, max_throws, report)
        _set_job(job_id, status='done', stage='done', stage_label=STAGE_LABELS['done'], progress=1.0, result=result)
    except ValueError as e:
        _set_job(job_id, status='error', error=str(e))
    except Exception as e:  # pragma: no cover - surfaced to the app as a readable message
        _set_job(job_id, status='error', error=f'Analysis failed: {e}')
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
        _set_job(job_id, upload_path=None)


def _prune_jobs(now=None):
    now = now or time.time()
    expired = set(guard.prune_jobs(JOB_TTL_S, now))
    with JOBS_LOCK:
        for k in [k for k, j in JOBS.items() if k in expired or now - j.get('updated', now) > JOB_TTL_S]:
            del JOBS[k]


def recover_jobs():
    """Reload job metadata from local SQLite. In-flight work cannot resume after a restart
    (uploads and the worker thread are gone, and Render's disk is ephemeral across instances),
    so those jobs are marked lost and the app resubmits automatically."""
    guard.ensure_configured()
    UPLOADS_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    for job in guard.all_jobs():
        jid = job['id']
        if job.get('status') in ('queued', 'running'):
            path = job.get('upload_path')
            guard.save_job(jid, status='lost', error=guard.LOST_DETAIL['message'], retry=True)
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass
        elif job.get('status') in ('done', 'error'):
            slim = {k: v for k, v in job.items() if k not in ('result', 'id')}
            with JOBS_LOCK:
                JOBS[jid] = slim
    if UPLOADS_ROOT.exists():
        for f in UPLOADS_ROOT.iterdir():
            if f.is_file():
                try:
                    f.unlink()
                except OSError:
                    pass
    _prune_jobs()
    cleanup_results()
    cleanup_disk()


@app.on_event('startup')
def _startup():
    recover_jobs()


async def _save_upload(video, arm):
    arm = (arm or 'right').strip().lower()
    if arm not in ARM_KEYS:
        raise HTTPException(400, "arm must be 'left' or 'right'.")
    suffix = Path(video.filename or 'throw.mp4').suffix.lower()
    if suffix not in {'.mp4','.mov','.m4v'}:
        raise HTTPException(400, 'Please upload MP4 or MOV video.')
    UPLOADS_ROOT.mkdir(parents=True, exist_ok=True)
    path = str(UPLOADS_ROOT / f'{uuid.uuid4().hex}{suffix}')
    size = 0
    try:
        with open(path, 'wb') as tmp:
            while chunk := await video.read(1024*1024):
                size += len(chunk)
                if size > MAX_BYTES:
                    raise HTTPException(413, 'Video exceeds 150 MB prototype limit.')
                tmp.write(chunk)
    except Exception:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise
    return path, arm


def _busy():
    raise HTTPException(429, "The analysis server is busy right now. Please try again in a minute.",
                        headers={'Retry-After': '60'})


@app.post('/analyse')
async def analyse_video(request: Request, video: UploadFile = File(...), arm: str = Form('right'), overlay: bool = Form(False),
                        mode: str = Form('sync'), view: str = Form('side'), max_throws: int = Form(0)):
    client = guard.authorize(request)
    view = (view or 'side').strip().lower()
    if view not in VIEWS:
        raise HTTPException(400, "view must be 'side' or 'front'.")
    max_throws = max_throws if 0 < max_throws <= 60 else None  # app sends darts marked on board photos / 3 per visit
    guard.check_rate(client)
    if mode == 'job':
        guard.check_queue_capacity()
    path, arm = await _save_upload(video, arm)
    cleanup_results()
    cleanup_disk()
    if guard.over_budget([RESULTS_ROOT, UPLOADS_ROOT]):
        try:
            os.unlink(path)
        except OSError:
            pass
        _busy()
    if mode == 'job':
        try:
            _prune_jobs()
            job_id = uuid.uuid4().hex
            guard.record_analysis(client)
            _set_job(job_id, status='queued', stage='queued', stage_label=STAGE_LABELS['queued'], progress=0.0,
                     install_id=client.install_id, ip=client.ip, upload_path=path)
            EXECUTOR.submit(_run_job, job_id, path, arm, overlay, view, max_throws)
            out = {'job_id': job_id, 'status': 'queued'}
            pos = guard.queue_position(job_id)
            if pos:
                out['queue_position'] = pos
            return out
        except Exception:
            try:
                os.unlink(path)
            except OSError:
                pass
            raise
    try:
        guard.record_analysis(client)
        return await run_in_threadpool(analyse, path, arm, overlay, view, max_throws)
    except ValueError as e:
        raise HTTPException(422, str(e))
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


@app.get('/jobs/{job_id}')
def get_job(job_id: str, request: Request):
    client = guard.authorize(request)
    job = None
    if ID_RE.match(job_id or ''):
        with JOBS_LOCK:
            job = dict(JOBS.get(job_id) or {})
        if not job:
            job = guard.load_job(job_id)
    if not job:
        raise guard.lost_http()
    if not guard.owns_job(job, client):
        raise HTTPException(404, 'Analysis not found. It may have expired; please try again.')
    if job.get('status') == 'lost':
        raise guard.lost_http()
    if job.get('status') == 'done' and 'result' not in job:
        stored = guard.load_job(job_id) or {}
        if 'result' in stored:
            job['result'] = stored['result']
    if job.get('status') == 'queued':
        pos = guard.queue_position(job_id)
        if pos:
            job['queue_position'] = pos
    for k in ('updated', 'created', 'install_id', 'ip', 'upload_path', 'id', 'retry'):
        job.pop(k, None)
    return job

# Share clips are a separate module so they can land beside other server work.
from shareclip import router as share_clip_router
app.include_router(share_clip_router)
