"""Pure-numpy throw segmentation and coaching-tip rules for Oche Lab.

Kept free of cv2/mediapipe so it can be unit-tested without the pose stack.
Everything here is experimental: 2D normalised image coordinates, ~15 fps sampling,
heuristic thresholds. Not a validated biomechanics or coaching tool.
"""
import numpy as np

# ---- Throw detection -------------------------------------------------------
MAX_GAP_S = 0.35             # interpolate missing landmarks over gaps up to this long
SMOOTH_FRAMES = 3            # centred moving-average window for positions/angles
ELBOW_WEIGHT = 0.5           # contribution of elbow-angle velocity to the detection signal
PEAK_MIN_SEPARATION_S = 0.8  # two release peaks closer than this are one throw (double peak); fast throwers are ~0.9 s apart
PEAK_HYST_LOW_FRAC = 0.5     # hysteresis: a burst only ends once the signal drops below this fraction of the threshold
PEAK_MAD_K = 4.0             # adaptive threshold = median + k * (1.4826 * MAD)
PEAK_MIN_WRIST_SPEED = 0.25  # absolute floor, normalised image units / second
PEAK_MAX_FRAC = 0.5          # threshold never above this fraction of the tallest peak (busy clips)
PEAK_MIN_REL_HEIGHT = 0.35   # ignore release peaks smaller than this fraction of the tallest release
WINDOW_BEFORE_S = 1.1        # window starts this long before the peak (room for a paused take-back + set-up)...
WINDOW_AFTER_S = 0.4         # ...and ends this long after it
MIN_WINDOW_FRAMES = 4        # drop windows with fewer sampled frames than this
MAX_THROWS = 15             # hard cap; the app also sends max_throws (darts on board photos / 3 per visit)
EXT_WINDOW_S = 0.35          # side-on: look this far either side of a peak for the arm extending forward...
EXT_MIN_ELBOW_DEG = 8        # ...elbow opening by at least this much...
EXT_MIN_REACH = 0.15         # ...or wrist moving away from the shoulder by this fraction of upper-arm length (no elbow angle)
EXT_BIG_REACH = 0.35         # ...or by this much even if the elbow angle didn't open
DIR_WINDOW_S = 0.25          # arm direction at a peak (diagnostics): mean just after minus mean just before
OPEN_SHARE = 0.5             # a burst counts as a release only if its opening part reaches this share of its peak
# Whole-body movement gate (walking to collect darts, stepping in/out of frame). Distances are in torso
# lengths (non-throwing shoulder to hip), on smoothed landmarks, and never below what the clip's own pose
# jitter could produce. Only clear walking is rejected: a lean-in, a weight shift or a wobbly ankle is fine.
BODY_WIN_BEFORE_S = 0.8      # look this far before the release...
BODY_WIN_AFTER_S = 0.5       # ...and this far after it (catches the first steps of walking off)
BODY_SMOOTH_FRAMES = 5       # centred moving average on body landmarks before measuring
WALK_SPAN_S = 0.6            # walking = sustained travel within this long...
WALK_SHIFT_MAX = 0.5         # ...of more than this (hips; walking covers ~2 torso lengths a second)
WALK_SCALE_MAX = 1.25        # body size changing by more than 25% in the window = walking towards / away from camera
STEP_HIP_MIN = 0.25          # stepping = hips travel this much...
STEP_FEET_MAX = 0.5          # ...and both ankles travel this much (ankles alone are too noisy to judge)
JITTER_K = 5.0               # a movement must be this many times the clip's landmark jitter to count
MIN_PRESENT_FRAC = 0.5       # body tracked in at least half the window (not leaving / entering frame)
MIN_POSE_CONF = 0.35         # mean arm-landmark confidence in the set-up (release frames excluded: they blur)
CONF_BEFORE_S = (0.8, 0.25)  # set-up window for the confidence check
AWAY_SHIFT_MAX = 1.2         # body more than this many torso lengths from its usual (oche) spot = at the board
AWAY_SCALE_MAX = 1.35        # body size this far off its usual size = nearer / further than the oche
AWAY_HOME_FRAC = 0.4         # only trust the "usual spot" if at least this share of the clip is spent there
MIN_VALID_SIGNAL_PCT = 0.5   # need wrist coverage on at least half the frames to try auto
FALLBACK_WINDOWS = 3

# ---- Coaching tips ---------------------------------------------------------
LOW_VALID_FRAMES = 5             # fewer clear frames than this in a throw -> camera/lighting tip
HEAD_MOTION_HIGH = 0.045         # head-vs-shoulder spread considered large (normalised)
RELATIVE_HIGH_FACTOR = 1.4       # metric this many times the median of the other throws
ELBOW_RANGE_MIN_DIFF_DEG = 12    # ...and at least this many degrees wider
ELBOW_MEAN_DIFF_DEG = 12         # mean elbow angle this far from the others' median
SESSION_ELBOW_SPREAD_DEG = 10    # spread of per-throw mean elbow angle = inconsistent
SESSION_ELBOW_TIGHT_DEG = 5      # ...or within this = nicely consistent
SESSION_LOW_DETECTION_PCT = 75   # arm visible in fewer % of frames -> framing tip
MAX_TIPS_PER_THROW = 2
MAX_SESSION_TIPS = 3
POSITIVE_TIP = 'Nice one. This throw looks consistent with your others on the tracked metrics. Keep that rhythm!'
TIPS_NOTE = ('Experimental tips generated from simple 2D motion rules. They are suggestions to try, '
             'not validated coaching or medical advice.')


def _series(points):
    """List of [x, y] or None -> (N, 2) float array with NaN for missing."""
    return np.array([p if p is not None else [np.nan, np.nan] for p in points], dtype=float).reshape(-1, 2)


def interpolate_gaps(values, times, max_gap_s=MAX_GAP_S):
    """Linearly fill NaN runs bounded by valid samples and no longer than max_gap_s. 1-D or 2-D."""
    arr = np.array(values, dtype=float)
    flat = arr.reshape(len(arr), -1)
    out = flat.copy()
    t = np.asarray(times, dtype=float)
    for c in range(flat.shape[1]):
        col = flat[:, c]
        valid = ~np.isnan(col)
        if valid.sum() < 2:
            continue
        i = 0
        while i < len(col):
            if valid[i]:
                i += 1
                continue
            j = i
            while j < len(col) and not valid[j]:
                j += 1
            if i > 0 and j < len(col) and t[j] - t[i - 1] <= max_gap_s + 1e-9:
                out[i:j, c] = np.interp(t[i:j], [t[i - 1], t[j]], [col[i - 1], col[j]])
            i = j
    return out.reshape(arr.shape)


def smooth(values, frames=SMOOTH_FRAMES):
    """NaN-aware centred moving average along axis 0."""
    arr = np.array(values, dtype=float)
    if frames <= 1 or len(arr) < frames:
        return arr
    flat = arr.reshape(len(arr), -1)
    out = np.full_like(flat, np.nan)
    half = frames // 2
    for i in range(len(flat)):
        win = flat[max(0, i - half):i + half + 1]
        ok = ~np.isnan(win)
        cnt = ok.sum(axis=0)
        s = np.where(ok, win, 0).sum(axis=0)
        out[i] = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
        out[i][np.isnan(flat[i])] = np.nan  # don't invent values where the raw sample is missing
    return out.reshape(arr.shape)


def speed_signals(wrist_points, elbow_angles, times, shoulder_points=None):
    """Return (wrist_speed, elbow_angular_speed) per sampled frame; NaN where unknown.

    With shoulder_points the wrist is measured relative to the shoulder, so walking (the whole body
    translating) doesn't look like a fast arm."""
    t = np.asarray(times, dtype=float)
    w = _series(wrist_points)
    if shoulder_points is not None:
        sh = interpolate_gaps(_series(shoulder_points), t)
        w = w - sh
    wrist = smooth(interpolate_gaps(w, t))
    ang = np.array([np.nan if a is None else a for a in elbow_angles], dtype=float)
    ang = smooth(interpolate_gaps(ang, t))
    if len(t) < 3:
        return np.full(len(t), np.nan), np.full(len(t), np.nan)
    vel = np.gradient(wrist, t, axis=0)
    wrist_speed = np.linalg.norm(vel, axis=1)
    elbow_speed = np.abs(np.gradient(ang, t))
    return wrist_speed, elbow_speed


def detection_signal(wrist_speed, elbow_speed):
    """Wrist speed plus a scaled elbow angular speed, so both live in wrist-speed units."""
    w = np.nan_to_num(wrist_speed, nan=0.0)
    e = np.nan_to_num(elbow_speed, nan=0.0)
    w90, e90 = np.percentile(w, 90), np.percentile(e, 90)
    if e90 > 1e-9 and ELBOW_WEIGHT > 0:
        return w + ELBOW_WEIGHT * e * (w90 / e90)
    return w


def adaptive_threshold(signal, k=PEAK_MAD_K, floor=PEAK_MIN_WRIST_SPEED):
    """median + k*MAD, capped relative to the tallest value (clips with many throws), floored."""
    s = np.asarray(signal, dtype=float)
    med = float(np.median(s))
    mad = float(np.median(np.abs(s - med))) * 1.4826
    return max(min(med + k * mad, PEAK_MAX_FRAC * float(np.max(s))), floor)


def find_peaks(signal, times, threshold, min_sep_s=PEAK_MIN_SEPARATION_S, low_frac=PEAK_HYST_LOW_FRAC):
    """One peak per motion burst, greedily keeping the tallest within min_sep_s.

    Hysteresis: a burst starts when the signal reaches `threshold` and only ends once it falls below
    low_frac * threshold, so the wobbles of a single throw (take-back, release, follow-through) can't
    count twice by dipping just under the threshold. Each burst contributes its tallest sample.
    """
    s = np.asarray(signal, dtype=float)
    t = np.asarray(times, dtype=float)
    cand = bursts(s, threshold, low_frac)
    if cand:
        tallest = max(s[i] for i in cand)
        cand = [i for i in cand if s[i] >= PEAK_MIN_REL_HEIGHT * tallest]
    return merge_close(cand, s, t, min_sep_s)


def merge_close(cand, s, t, min_sep_s=PEAK_MIN_SEPARATION_S):
    """Greedily keep the tallest peak within min_sep_s (double peaks of one throw)."""
    kept = []
    for i in sorted(cand, key=lambda i: -s[i]):
        if all(abs(t[i] - t[j]) >= min_sep_s for j in kept):
            kept.append(i)
    return sorted(kept)


def bursts(signal, threshold, low_frac=PEAK_HYST_LOW_FRAC, spans=False):
    """Tallest sample of each burst: starts at `threshold`, ends below low_frac * threshold.
    With spans=True returns (start, end, peak) tuples."""
    s = np.asarray(signal, dtype=float)
    if len(s) < 3:
        return []
    low = low_frac * threshold
    cand, i = [], 0
    while i < len(s):
        if s[i] >= threshold:
            j = i
            while j + 1 < len(s) and s[j + 1] >= low:
                j += 1
            p = i + int(np.argmax(s[i:j + 1]))
            cand.append((i, j, p) if spans else p)
            i = j + 1
        else:
            i += 1
    return cand


def windows_around_peaks(peaks, times, before_s=WINDOW_BEFORE_S, after_s=WINDOW_AFTER_S,
                         min_frames=MIN_WINDOW_FRAMES):
    """Index arrays for each peak window, clipped to the clip and split at midpoints if overlapping."""
    t = np.asarray(times, dtype=float)
    out = []
    for n, p in enumerate(peaks):
        lo, hi = t[p] - before_s, t[p] + after_s
        if n > 0:
            lo = max(lo, (t[peaks[n - 1]] + t[p]) / 2)
        if n + 1 < len(peaks):
            hi = min(hi, (t[p] + t[peaks[n + 1]]) / 2)
        idx = np.where((t >= lo) & (t < hi if n + 1 < len(peaks) else t <= hi))[0]
        if len(idx) >= min_frames:
            out.append((p, idx))
    return out


def _window_extreme(values, t, lo, hi, fn):
    m = (t >= lo) & (t <= hi) & ~np.isnan(values)
    return float(fn(values[m])) if m.any() else None


def forward_extension(rows, times, peak):
    """(elbow opening deg, wrist reach / upper-arm length) around a peak; None where not measurable.

    Before = smallest value in the EXT_WINDOW_S before the peak, after = largest value in the window after.
    A release opens the elbow and pushes the wrist away from the shoulder; a take-back or a reach for
    the next dart doesn't.
    """
    t = np.asarray(times, dtype=float)
    lo, mid, hi = t[peak] - EXT_WINDOW_S, t[peak], t[peak] + EXT_WINDOW_S
    ang = smooth(interpolate_gaps(np.array([np.nan if r.get('elbow_angle_deg') is None else r['elbow_angle_deg']
                                            for r in rows], dtype=float), t))
    open_deg = None
    b, a = _window_extreme(ang, t, lo, mid, np.min), _window_extreme(ang, t, mid, hi, np.max)
    if b is not None and a is not None:
        open_deg = a - b
    sh, el, wr = (_series([r['points'].get(k) for r in rows]) for k in ('shoulder', 'elbow', 'wrist'))
    upper = np.nanmedian(np.linalg.norm(el - sh, axis=1)) if np.isfinite(el - sh).any() else np.nan
    reach = None
    if np.isfinite(upper) and upper > 1e-6:
        d = smooth(interpolate_gaps(np.linalg.norm(wr - sh, axis=1), t)) / upper
        b, a = _window_extreme(d, t, lo, mid, np.min), _window_extreme(d, t, mid, hi, np.max)
        if b is not None and a is not None:
            reach = a - b
    return open_deg, reach


def is_forward_throw(rows, times, peak):
    open_deg, reach = forward_extension(rows, times, peak)
    if open_deg is None and reach is None:
        return True  # can't tell: don't throw the peak away
    if open_deg is not None and open_deg >= EXT_MIN_ELBOW_DEG:
        return True
    # Hand moving away without the elbow opening is usually reaching for the next dart; only trust reach on its
    # own when the elbow angle can't be measured, or when it's a big push.
    return reach is not None and (reach >= EXT_BIG_REACH or (open_deg is None and reach >= EXT_MIN_REACH))


def _arm_series(rows, times):
    t = np.asarray(times, dtype=float)
    ang = smooth(interpolate_gaps(np.array([np.nan if r.get('elbow_angle_deg') is None else r['elbow_angle_deg']
                                            for r in rows], dtype=float), t))
    sh, el, wr = (_series([r['points'].get(k) for r in rows]) for k in ('shoulder', 'elbow', 'wrist'))
    upper = np.nanmedian(np.linalg.norm(el - sh, axis=1)) if np.isfinite(el - sh).any() else np.nan
    reach = (smooth(interpolate_gaps(np.linalg.norm(wr - sh, axis=1), t)) / upper
             if np.isfinite(upper) and upper > 1e-6 else np.full(len(t), np.nan))
    return ang, reach


def _side_mean(v, t, lo, hi):
    m = (t >= lo) & (t <= hi) & ~np.isnan(v)
    return float(np.mean(v[m])) if m.any() else None


def arm_direction(rows, times, peak, series=None):
    """(elbow change deg, reach change) across a peak: mean just after minus mean just before. None if unknown."""
    t = np.asarray(times, dtype=float)
    ang, reach = series or _arm_series(rows, times)
    out = []
    for v in (ang, reach):
        b, a = _side_mean(v, t, t[peak] - DIR_WINDOW_S, t[peak]), _side_mean(v, t, t[peak], t[peak] + DIR_WINDOW_S)
        out.append(None if a is None or b is None else a - b)
    return tuple(out)


def opening_mask(rows, times, series=None):
    """Per sample: is the arm opening (elbow angle growing; hand moving away from the shoulder if no angle)?"""
    t = np.asarray(times, dtype=float)
    ang, reach = series or _arm_series(rows, times)
    da = np.gradient(ang, t) if len(t) >= 3 else np.full(len(t), np.nan)
    dr = np.gradient(reach, t) if len(t) >= 3 else np.full(len(t), np.nan)
    return np.where(np.isfinite(da), da > 0, np.where(np.isfinite(dr), dr >= 0, True))


def _mid(a, b):
    if a is None and b is None:
        return None
    if a is None or b is None:
        return a if a is not None else b
    return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]


def _both(rows, a, b):
    """Midpoint of two landmarks, NaN unless both are tracked (one alone jumps by half a body width)."""
    out = []
    for r in rows:
        p, q = r['points'].get(a), r['points'].get(b)
        out.append([(p[0] + q[0]) / 2, (p[1] + q[1]) / 2] if p is not None and q is not None else None)
    return _series(out)


def _dist(rows, a, b):
    out = []
    for r in rows:
        p, q = r['points'].get(a), r['points'].get(b)
        out.append(float(np.hypot(p[0] - q[0], p[1] - q[1])) if p is not None and q is not None else np.nan)
    return np.array(out, dtype=float)


def _jitter(raw, smoothed):
    """Robust frame-to-frame noise of a landmark track (std of raw minus smoothed)."""
    d = (raw - smoothed).ravel()
    d = d[np.isfinite(d)]
    return float(1.4826 * np.median(np.abs(d - np.median(d)))) if len(d) >= 5 else 0.0


def body_track(rows):
    """Smoothed body reference for the whole-body gate.

    centre: mid-hip when both hips are tracked in most frames, else the non-throwing shoulder (the throwing
    shoulder moves with the arm). torso: non-throwing shoulder to hip (falls back to mid-shoulder to mid-hip).
    Also ankles, presence, arm confidence, per-clip jitter and the usual standing spot ("home").
    """
    t = np.array([r['time_s'] for r in rows], dtype=float)
    smooth_t = lambda a: smooth(interpolate_gaps(a, t, max_gap_s=0.5), BODY_SMOOTH_FRAMES)
    hips = _both(rows, 'hip', 'other_hip')
    if np.isfinite(hips).all(axis=1).mean() >= 0.6:
        raw = hips
    else:
        raw = _series([r['points'].get('other_shoulder') or r['points'].get('shoulder') for r in rows])
    centre = smooth_t(raw)
    torso_raw = _dist(rows, 'other_shoulder', 'other_hip')
    if np.isfinite(torso_raw).mean() < 0.5:
        sh, hp = _both(rows, 'shoulder', 'other_shoulder'), _both(rows, 'hip', 'other_hip')
        torso_raw = np.linalg.norm(sh - hp, axis=1)
    torso = smooth_t(torso_raw)
    scale = float(np.nanmedian(torso)) if np.isfinite(torso).any() else None
    if not scale or scale < 0.03:
        ua = [float(np.hypot(*np.subtract(r['points']['elbow'], r['points']['shoulder']))) for r in rows
              if r['points'].get('elbow') is not None and r['points'].get('shoulder') is not None]
        scale = 1.6 * float(np.median(ua)) if ua else 0.25  # torso ~ 1.6 upper arms
    ankles_raw = [_series([r['points'].get(k) for r in rows]) for k in ('ankle', 'other_ankle')]
    ankles = [smooth_t(a) for a in ankles_raw]
    seen = np.isfinite(centre).all(axis=1)
    home = np.median(centre[seen], axis=0) if seen.any() else None
    if home is not None:  # only a real "usual spot" if the thrower spends a good share of the clip there
        near = np.abs(centre[seen, 0] - home[0]) < 0.5 * scale
        home = home if near.mean() >= AWAY_HOME_FRAC else None
    keys = ('shoulder', 'other_shoulder', 'hip', 'other_hip')
    return {'centre': centre, 'torso': torso, 'scale': scale, 'home': home, 'ankles': ankles,
            'jitter': _jitter(raw[:, 0], centre[:, 0]),
            'ankle_jitter': max([_jitter(r[:, 0], s[:, 0]) for r, s in zip(ankles_raw, ankles)] + [0.0]),
            'present': np.array([any(r['points'].get(k) is not None for k in keys) for r in rows]),
            'conf': np.array([np.nan if r.get('conf') is None else r['conf'] for r in rows], dtype=float),
            'has_conf': any('conf' in r for r in rows)}


def _max_shift(x, t, m, span=WALK_SPAN_S):
    """Largest |x[i] - x[k]| for samples i, k in mask m no more than `span` apart (sustained travel)."""
    idx = np.where(m & np.isfinite(x))[0]
    best = None
    for n, i in enumerate(idx):
        k = idx[(t[idx] >= t[i] - span) & (t[idx] < t[i])]
        if len(k):
            d = float(np.max(np.abs(x[i] - x[k])))
            best = d if best is None else max(best, d)
    return best


# short labels for the debug summary
REJECT_LABELS = {'walking (hips moved sideways)': 'walking', 'walking towards / away from the camera': 'walking',
                 'stepping': 'stepping', 'out of frame': 'out of frame', 'low pose confidence': 'low confidence',
                 'away from the throwing position': 'away from the oche',
                 'arm folding back (take-back or raising to aim)': 'take-back/raise',
                 'no forward arm extension': 'no extension', 'too small': 'too small',
                 'double peak of a nearby throw': 'merged', 'over the dart limit': 'over limit'}


def body_movement(body, times, peak):
    """Why the candidate throw at `peak` looks like walking / whole-body movement, or None if it's a throw."""
    t = np.asarray(times, dtype=float)
    win = (t >= t[peak] - BODY_WIN_BEFORE_S) & (t <= t[peak] + BODY_WIN_AFTER_S)
    if win.sum() < 3:
        return None
    sc = body['scale']
    if body['present'][win].mean() < MIN_PRESENT_FRAC:
        return 'out of frame'
    if body['has_conf']:
        pre = (t >= t[peak] - CONF_BEFORE_S[0]) & (t <= t[peak] - CONF_BEFORE_S[1])
        c = body['conf'][pre]
        c = c[np.isfinite(c)]
        if len(c) >= 3 and float(np.mean(c)) < MIN_POSE_CONF:
            return 'low pose confidence'
    hip_shift = _max_shift(body['centre'][:, 0], t, win)
    if hip_shift is not None and hip_shift > max(WALK_SHIFT_MAX * sc, JITTER_K * body['jitter']):
        return 'walking (hips moved sideways)'
    tor = body['torso'][win]
    tor = tor[np.isfinite(tor)]
    if len(tor) >= 3 and tor.min() > 0 and tor.max() / tor.min() > WALK_SCALE_MAX:
        return 'walking towards / away from the camera'
    if hip_shift is not None and hip_shift > STEP_HIP_MIN * sc:
        feet = [_max_shift(a[:, 0], t, win) for a in body['ankles']]
        if all(f is not None and f > max(STEP_FEET_MAX * sc, JITTER_K * body['ankle_jitter']) for f in feet):
            return 'stepping'
    home, here, tp = body.get('home'), body['centre'][peak], body['torso'][peak]
    if home is not None and np.isfinite(here).all() and abs(here[0] - home[0]) > AWAY_SHIFT_MAX * sc:
        return 'away from the throwing position'
    if home is not None and np.isfinite(tp) and tp > 0 and max(tp / sc, sc / tp) > AWAY_SCALE_MAX:
        return 'away from the throwing position'
    return None


def detection_summary(kept, candidates):
    """e.g. 'detected 9, rejected 2: 2 walking'."""
    counts = {}
    for c in candidates:
        if c['reason']:
            lab = REJECT_LABELS.get(c['reason'], c['reason'])
            counts[lab] = counts.get(lab, 0) + 1
    rej = sum(counts.values())
    tail = (': ' + ', '.join(f'{n} {k}' for k, n in sorted(counts.items(), key=lambda kv: -kv[1]))) if rej else ''
    return f'detected {kept}, rejected {rej}{tail}'


def segment_throws(rows, view='side', max_throws=None):
    """Detect throws from wrist/elbow motion. Falls back to equal windows if detection fails.

    rows: dicts with 'time_s', 'points' {'wrist','elbow',...}, 'elbow_angle_deg'.
    view: 'side' also requires a forward arm extension at each peak (front-on 2D angles can't show it).
    max_throws: optional cap from the app (darts marked on board photos, or 3 per visit).
    Returns dict: method, sentence, segments [{'indices', 'peak_index'|None}], wrist_speed.
    """
    times = [r['time_s'] for r in rows]
    wrist_pts = [r['points'].get('wrist') for r in rows]
    wrist_speed, elbow_speed = speed_signals(wrist_pts, [r['elbow_angle_deg'] for r in rows], times,
                                             [r['points'].get('shoulder') for r in rows])
    reason, debug = None, None
    coverage = float(np.mean(~np.isnan(wrist_speed))) if len(rows) else 0.0
    if len(rows) < 3 * MIN_WINDOW_FRAMES:
        reason = 'clip too short'
    elif coverage < MIN_VALID_SIGNAL_PCT:
        reason = 'wrist not tracked consistently'
    else:
        raw_sig = detection_signal(wrist_speed, elbow_speed)
        t = np.asarray(times, dtype=float)
        # 1. Whole-body gate: every sample is checked as if it were a release, and samples where the body was
        #    clearly walking / stepping / out of frame are removed before thresholding, so a big walking burst
        #    can't raise the bar for the real throws either.
        body = body_track(rows)
        why = [body_movement(body, times, i) for i in range(len(rows))]
        moving = np.array([w is not None for w in why])
        sig = np.where(moving, 0.0, raw_sig)
        thr = adaptive_threshold(sig) if sig.max() > 0 else adaptive_threshold(raw_sig)
        candidates = [{'index': int(p), 'time_s': round(float(t[p]), 3), 'speed': round(float(raw_sig[p]), 3),
                       'reason': why[p]} for p in bursts(raw_sig, thr) if moving[p]]
        live = bursts(sig, thr, spans=True) if sig.max() > 0 else []
        # 2. Direction: a release opens the arm; a take-back or raising the hand to aim folds it. Each burst is
        #    placed at its fastest *opening* moment (a quick raise-and-throw is one burst); bursts that only
        #    fold the arm are dropped.
        series = _arm_series(rows, times)
        opening = opening_mask(rows, times, series)
        cand = []
        for i, j, p in live:
            sub = np.where(opening[i:j + 1], sig[i:j + 1], 0.0)
            folding = sub.max() < OPEN_SHARE * sig[p]
            if not folding:
                p = i + int(np.argmax(sub))
            c = {'index': int(p), 'time_s': round(float(t[p]), 3), 'speed': round(float(sig[p]), 3), 'reason': None}
            candidates.append(c)
            if folding:
                c['reason'] = 'arm folding back (take-back or raising to aim)'
            else:
                cand.append(c)
        # 3. Side-on: the arm must extend forward (if nothing passes, tracking is too rough to judge).
        if view != 'front' and cand:
            fwd = [c for c in cand if is_forward_throw(rows, times, c['index'])]
            if fwd:
                for c in cand:
                    if c not in fwd:
                        c['reason'] = 'no forward arm extension'
                cand = fwd
        # 4. Small bumps, double peaks of one throw, and the app's dart limit.
        if cand:
            tallest = max(sig[c['index']] for c in cand)
            for c in cand:
                if sig[c['index']] < PEAK_MIN_REL_HEIGHT * tallest:
                    c['reason'] = 'too small'
            cand = [c for c in cand if not c['reason']]
        kept = set(merge_close([c['index'] for c in cand], sig, t))
        for c in cand:
            if c['index'] not in kept:
                c['reason'] = 'double peak of a nearby throw'
        cap = MAX_THROWS if not max_throws else max(1, min(int(max_throws), MAX_THROWS))
        peaks = sorted(kept)
        if len(peaks) > cap:
            top = set(sorted(peaks, key=lambda i: -sig[i])[:cap])
            for c in cand:
                if c['index'] in kept and c['index'] not in top:
                    c['reason'] = 'over the dart limit'
            peaks = sorted(top)
        candidates.sort(key=lambda c: c['time_s'])
        rejected = sum(1 for c in candidates if c['reason'] in ('walking (hips moved sideways)', 'walking towards / away from the camera',
                                                                 'stepping', 'out of frame', 'away from the throwing position'))
        wins = windows_around_peaks(peaks, times)
        debug = {'summary': detection_summary(len(wins), candidates),
                 'candidates': [{k: c[k] for k in ('time_s', 'speed', 'reason')} for c in candidates]}
        if wins:
            n = len(wins)
            ignored = f' Ignored {rejected} burst{"s" if rejected != 1 else ""} of whole-body movement (walking or stepping).' if rejected else ''
            return {'method': 'auto',
                    'sentence': f"Detected {n} throw{'s' if n != 1 else ''} automatically from wrist and elbow speed peaks.{ignored}",
                    'segments': [{'indices': idx, 'peak_index': int(p)} for p, idx in wins],
                    'wrist_speed': wrist_speed, 'debug': debug}
        still = bool(live)
        reason = 'only whole-body movement found' if rejected and not still else 'no clear motion peaks'
    segs = np.array_split(np.arange(len(rows)), FALLBACK_WINDOWS)
    return {'method': 'equal_windows_fallback',
            'sentence': f'Automatic throw detection failed ({reason}); split the clip into three equal time windows instead.',
            'segments': [{'indices': s, 'peak_index': None} for s in segs],
            'wrist_speed': wrist_speed,
            'debug': {'summary': f"fallback ({reason}); {debug['summary'] if debug else 'no candidates'}",
                      'candidates': debug['candidates'] if debug else []}}


def _spread(points):
    if len(points) < 2:
        return None
    a = np.array(points, dtype=float)
    return round(float(np.max(np.linalg.norm(a - a.mean(axis=0), axis=1))), 4)


def summarise_throw(rows, indices, number, peak_index=None, wrist_speed=None):
    subset = [rows[int(j)] for j in indices]
    angles = [r['elbow_angle_deg'] for r in subset if r['elbow_angle_deg'] is not None]
    head = [r['head_relative'] for r in subset if r.get('head_relative') is not None]
    elbow = [r['points']['elbow'] for r in subset if r['points'].get('elbow') is not None]
    wrist = [r['points']['wrist'] for r in subset if r['points'].get('wrist') is not None]
    peak_t = peak_v = None
    if peak_index is not None:
        peak_t = rows[peak_index]['time_s']
        if wrist_speed is not None and not np.isnan(wrist_speed[peak_index]):
            peak_v = round(float(wrist_speed[peak_index]), 3)
    return {'throw': number, 'start_s': subset[0]['time_s'], 'end_s': subset[-1]['time_s'],
            'peak_time_s': peak_t, 'peak_wrist_speed': peak_v,
            'elbow_angle_mean_deg': round(float(np.mean(angles)), 1) if angles else None,
            'elbow_angle_range_deg': round(float(np.ptp(angles)), 1) if angles else None,
            'head_motion_relative_normalized': _spread(head),
            'elbow_motion_normalized': _spread(elbow),
            'wrist_motion_normalized': _spread(wrist),
            'valid_frames': len(angles),
            'path': throw_path(subset)}


def _pt(p):
    return None if p is None else [round(float(p[0]), 4), round(float(p[1]), 4)]


def throw_path(subset):
    """Compact per-frame arm path for one throw window (normalised image coords, y down).

    Lets the app draw the wrist path and compare upper-arm angle / release height between throws.
    Each item: t (s), s/e/w = shoulder/elbow/wrist [x, y] or None, a = elbow angle (deg) or None.
    """
    out = []
    for r in subset:
        pts = r['points']
        a = r.get('elbow_angle_deg')
        out.append({'t': r['time_s'], 's': _pt(pts.get('shoulder')), 'e': _pt(pts.get('elbow')),
                    'w': _pt(pts.get('wrist')), 'a': None if a is None else round(float(a), 1)})
    return out


def _others_median(throws, i, key):
    vals = [t[key] for j, t in enumerate(throws) if j != i and t.get(key) is not None]
    return float(np.median(vals)) if vals else None


def throw_tips(throws):
    """1-2 short, encouraging, experimental tips per throw (list of lists, same order)."""
    all_tips = []
    for i, t in enumerate(throws):
        tips = []
        if t['valid_frames'] < LOW_VALID_FRAMES:
            tips.append(f"Only {t['valid_frames']} clear frames on this throw. Brighter light and keeping your "
                        "whole arm in frame should give more reliable feedback.")
        head, head_o = t.get('head_motion_relative_normalized'), _others_median(throws, i, 'head_motion_relative_normalized')
        if head is not None and (head > HEAD_MOTION_HIGH or (head_o and head > RELATIVE_HIGH_FACTOR * head_o and head > HEAD_MOTION_HIGH / 2)):
            tips.append('Your head moved more here than on your steadiest throws. Try locking your eyes on the target '
                        'and keeping your head still until the dart lands.')
        rng, rng_o = t.get('elbow_angle_range_deg'), _others_median(throws, i, 'elbow_angle_range_deg')
        if rng is not None and rng_o and rng > RELATIVE_HIGH_FACTOR * rng_o and rng - rng_o >= ELBOW_RANGE_MIN_DIFF_DEG:
            tips.append(f'Your elbow opened through a wider range than usual ({rng:.0f}° vs about {rng_o:.0f}°). '
                        'Try repeating the same smooth arm path each time.')
        mean, mean_o = t.get('elbow_angle_mean_deg'), _others_median(throws, i, 'elbow_angle_mean_deg')
        if mean is not None and mean_o is not None and abs(mean - mean_o) >= ELBOW_MEAN_DIFF_DEG:
            tips.append(f'Average elbow angle was {mean:.0f}° vs about {mean_o:.0f}° on your other throws. '
                        'A consistent set-up position can help repeatability.')
        wr, wr_o = t.get('wrist_motion_normalized'), _others_median(throws, i, 'wrist_motion_normalized')
        if wr is not None and wr_o and wr > RELATIVE_HIGH_FACTOR * wr_o:
            tips.append('Your hand travelled further than on your other throws. Try a compact, repeatable '
                        'take-back and follow-through.')
        if not tips:
            tips.append(POSITIVE_TIP)
        all_tips.append(tips[:MAX_TIPS_PER_THROW])
    return all_tips


def session_tips(throws, method, arm_detected_pct=None):
    tips = []
    if method != 'auto':
        tips.append('Throws were not detected automatically, so these windows may not line up with each throw. '
                    'Trim idle time and keep the camera still, then try again.')
    elif len(throws) != 3:
        tips.append(f"Detected {len(throws)} throw{'s' if len(throws) != 1 else ''}. If you threw a different number, "
                    'keep pauses between throws clear and your arm in frame.')
    means = [t['elbow_angle_mean_deg'] for t in throws if t.get('elbow_angle_mean_deg') is not None]
    if len(means) >= 2:
        spread = max(means) - min(means)
        if spread > SESSION_ELBOW_SPREAD_DEG:
            tips.append(f'Your average elbow angle varied by {spread:.0f}° across throws. Working on the same '
                        'set-up and release position could tighten your grouping.')
        elif spread <= SESSION_ELBOW_TIGHT_DEG and method == 'auto':
            tips.append(f'Great repeatability: your average elbow angle stayed within about {max(1, round(spread))}° '
                        'across throws.')
    heads = [t['head_motion_relative_normalized'] for t in throws if t.get('head_motion_relative_normalized') is not None]
    if heads and float(np.median(heads)) > HEAD_MOTION_HIGH:
        tips.append('Head movement was noticeable on most throws. A still head is a common foundation to experiment with.')
    if arm_detected_pct is not None and arm_detected_pct < SESSION_LOW_DETECTION_PCT:
        tips.append(f'Your arm was only tracked in {arm_detected_pct:.0f}% of frames. A side-on tripod and good '
                    'lighting will make results more reliable.')
    return tips[:MAX_SESSION_TIPS]


# ---- Key phases & timeline (coach video review) -----------------------------
PHASES = ('setup', 'backswing', 'release', 'follow_through')
PHASE_LABELS = {'setup': 'Set-up / aim', 'backswing': 'Take-back', 'release': 'Release',
                'follow_through': 'Follow-through'}
HEAD_DRIFT_HIGH = 0.03          # head-vs-shoulder shift between set-up and release (normalised)
ELBOW_DROP_HIGH = 0.04          # elbow moves this much lower in the image by release (normalised y)
FOLLOW_THROUGH_MIN_DEG = 150    # max elbow extension after release below this = short follow-through
BACKSWING_MIN_FLEX_DEG = 15     # take-back must close the elbow at least this much vs set-up to count


def _angle(rows, i):
    return None if i is None else rows[i].get('elbow_angle_deg')


def pick_phases(rows, indices, peak_index=None, wrist_speed=None):
    """Pick key frame indices for one throw window: set-up, take-back, release, follow-through.

    release = detected peak (or fastest wrist frame in the window); take-back = most-flexed elbow
    before release; set-up = stillest wrist frame before the take-back; follow-through = most
    extended elbow after release. Missing phases are omitted. Returns {phase: row_index}.
    """
    idx = [int(i) for i in indices]
    if not idx:
        return {}
    speed = np.asarray(wrist_speed, dtype=float) if wrist_speed is not None else np.full(len(rows), np.nan)
    rel = int(peak_index) if peak_index is not None and peak_index in idx else None
    if rel is None:
        finite = [i for i in idx if not np.isnan(speed[i])]
        rel = max(finite, key=lambda i: speed[i]) if finite else idx[len(idx) // 2]
    before = [i for i in idx if i < rel]
    after = [i for i in idx if i > rel]
    out = {}
    flex = [i for i in before if _angle(rows, i) is not None]
    back = min(flex, key=lambda i: _angle(rows, i)) if flex else None
    pre = [i for i in before if back is None or i < back]
    setup = min(pre, key=lambda i: (np.inf if np.isnan(speed[i]) else speed[i], i)) if pre else None
    if back is not None and setup is not None and _angle(rows, setup) is not None \
            and _angle(rows, setup) - _angle(rows, back) < BACKSWING_MIN_FLEX_DEG:
        back = None  # no real take-back visible; don't label one
    if setup is None and before:
        setup = before[0]
    ext = [i for i in after if _angle(rows, i) is not None]
    follow = max(ext, key=lambda i: _angle(rows, i)) if ext else (after[-1] if after else None)
    for name, i in (('setup', setup), ('backswing', back), ('release', rel), ('follow_through', follow)):
        if i is not None:
            out[name] = i
    return out


def _r(v, nd=1):
    return None if v is None or not np.isfinite(v) else round(float(v), nd)


def _arm_at(rows, idx, i, k=2):
    """(shoulder, elbow, wrist) at row i, borrowing from the nearest row within k samples when missing."""
    out = []
    for key in ('shoulder', 'elbow', 'wrist'):
        p = None
        for d in range(k + 1):
            for j in (i - d, i + d):
                if j in idx and rows[j]['points'].get(key) is not None:
                    p = rows[j]['points'][key]
                    break
            if p is not None:
                break
        out.append(p)
    return out


def throw_mechanics(rows, indices, phases, view='side'):
    """Phase-by-phase arm angles and timing for one throw (2D, experimental). Additive output for the app's review.

    phases[name] = {t, elbow_deg, upper_arm_deg, forearm_deg, wrist_height} for setup/backswing/release/follow_through.
      upper_arm_deg: upper arm below horizontal (+ = elbow lower than shoulder); forearm_deg: forearm above
      horizontal (+ = hand higher than elbow); wrist_height: hand above the shoulder in upper-arm lengths.
    timing: aim_s (set-up -> take-back), pullback_s (take-back -> release), finish_s (release -> follow-through).
    elbow_drop_deg: upper arm angle change from take-back (or set-up) to release, + = elbow dropped.
    body_shift: shoulder travel during the throw in upper-arm lengths (stance stability).
    Front-on: elbow and arm angles are foreshortened, so they are None; timing, wrist height and shift remain.
    """
    idx = set(int(i) for i in indices)
    front = view == 'front'
    ul = [float(np.linalg.norm(np.subtract(rows[i]['points']['elbow'], rows[i]['points']['shoulder']))) for i in idx
          if rows[i]['points'].get('elbow') is not None and rows[i]['points'].get('shoulder') is not None]
    ulen = float(np.median(ul)) if ul else None
    out_ph = {}
    for name in PHASES:
        i = phases.get(name)
        if i is None or (name == 'setup' and i == phases.get('backswing')):
            continue
        sh, el, wr = _arm_at(rows, idx, i)
        upper = fore = height = None
        if sh is not None and el is not None:
            upper = np.degrees(np.arctan2(el[1] - sh[1], abs(el[0] - sh[0]) + 1e-9))
        if el is not None and wr is not None:
            fore = np.degrees(np.arctan2(el[1] - wr[1], abs(wr[0] - el[0]) + 1e-9))
        if sh is not None and wr is not None and ulen:
            height = (sh[1] - wr[1]) / ulen
        out_ph[name] = {'t': rows[i]['time_s'], 'elbow_deg': None if front else _angle(rows, i),
                        'upper_arm_deg': None if front else _r(upper), 'forearm_deg': None if front else _r(fore),
                        'wrist_height': _r(height, 3)}
    t = {k: v['t'] for k, v in out_ph.items()}
    gap = lambda a, b: _r(t[b] - t[a], 3) if a in t and b in t else None
    ref = out_ph.get('backswing') or out_ph.get('setup')
    rel = out_ph.get('release')
    drop = None
    if not front and ref and rel and ref['upper_arm_deg'] is not None and rel['upper_arm_deg'] is not None:
        drop = _r(rel['upper_arm_deg'] - ref['upper_arm_deg'])
    shoulders = [rows[i]['points']['shoulder'] for i in sorted(idx) if rows[i]['points'].get('shoulder') is not None]
    shift = None
    if len(shoulders) >= 3 and ulen:
        a = np.array(shoulders, dtype=float)
        shift = _r(float(np.max(np.linalg.norm(a - np.median(a, axis=0), axis=1))) / ulen, 3)
    return {'phases': out_ph,
            # Front-on, the take-back moves towards the camera and is picked from a foreshortened elbow angle,
            # so its timing is not reliable: reported as None rather than a misleading number.
            'timing': {'aim_s': None if front else gap('setup', 'backswing'),
                       'pullback_s': None if front else gap('backswing', 'release'),
                       'finish_s': gap('release', 'follow_through')},
            'elbow_drop_deg': drop,
            'follow_through_deg': None if front else (out_ph.get('follow_through') or {}).get('elbow_deg'),
            'body_shift': shift}


# ---- detailed mechanics (additive, 0.9.0) ----------------------------------------------------------------
# Per-throw values for the app's "Mechanics breakdown". Lengths are body-relative: side-on in upper-arm lengths
# ('ul'), front-on in shoulder widths ('sw'), so they don't depend on how far the camera is. Point tracks are
# smoothed with a 3-sample median first. Anything the camera angle can't see is None (never a guess).
HOLD_TOL = 0.15        # wrist within this many units of its take-back / follow-through position = "holding"


def _track(rows, order, key):
    """{row index: smoothed (x, y)} for one landmark over the throw window (3-sample median)."""
    pts = [(i, rows[i]['points'].get(key)) for i in order]
    pts = [(i, p) for i, p in pts if p is not None]
    out = {}
    for k, (i, _) in enumerate(pts):
        win = [p for _, p in pts[max(0, k - 1):k + 2]]
        out[i] = (float(np.median([p[0] for p in win])), float(np.median([p[1] for p in win])))
    return out


def _at(track, i, order, k=2):
    """Track value at row i or the nearest row within k samples."""
    if i is None:
        return None
    pos = {r: n for n, r in enumerate(order)}
    if i not in pos:
        return None
    for d in range(k + 1):
        for j in (pos[i] - d, pos[i] + d):
            if 0 <= j < len(order) and order[j] in track:
                return track[order[j]]
    return None


def _extent(track, rows_in):
    """10th-90th percentile extent of a smoothed track over the given rows: (dx, dy, euclid)."""
    pts = [track[i] for i in rows_in if i in track]
    if len(pts) < 3:
        return None
    a = np.array(pts)
    dx = float(np.percentile(a[:, 0], 90) - np.percentile(a[:, 0], 10))
    dy = float(np.percentile(a[:, 1], 90) - np.percentile(a[:, 1], 10))
    return dx, dy, float(np.hypot(dx, dy))


def _hold(track, order, i, unit, rows, forward=True):
    """Seconds the wrist stays within HOLD_TOL units of its position at row i (contiguous, one direction)."""
    if i is None or i not in track or not unit:
        return None
    p0, pos = track[i], order.index(i)
    seq = order[pos:] if forward else order[:pos + 1][::-1]
    last = i
    for j in seq:
        if j not in track:
            continue
        if np.hypot(track[j][0] - p0[0], track[j][1] - p0[1]) / unit > HOLD_TOL:
            break
        last = j
    return round(abs(rows[last]['time_s'] - rows[i]['time_s']), 3)


def throw_details(rows, indices, phases, view='side', frame=None):
    """Detailed per-throw mechanics (see block comment above). All keys always present; unmeasurable = None."""
    order = sorted(int(i) for i in indices)
    front = view == 'front'
    sh, el, wr = _track(rows, order, 'shoulder'), _track(rows, order, 'elbow'), _track(rows, order, 'wrist')
    hd, osh = _track(rows, order, 'head'), _track(rows, order, 'other_shoulder')
    hip, ohip = _track(rows, order, 'hip'), _track(rows, order, 'other_hip')
    ul = [np.hypot(el[i][0] - sh[i][0], el[i][1] - sh[i][1]) for i in order if i in el and i in sh]
    if front:
        unit = frame[0] if frame and frame[0] else None
        out_sign = frame[1] if frame else 1.0
    else:
        unit = float(np.median(ul)) if ul else None
        out_sign = 1.0
    s_i, b_i, r_i, f_i = (phases.get(k) for k in ('setup', 'backswing', 'release', 'follow_through'))
    T = lambda i: rows[i]['time_s'] if i is not None else None
    U = lambda v: None if v is None or not unit else round(float(v) / unit, 3)
    # which way is "towards the board" side-on: where the hand ends up relative to the shoulder after release
    fwd = 1.0
    a_sh, a_w = _at(sh, f_i if f_i is not None else r_i, order), _at(wr, f_i if f_i is not None else r_i, order)
    if a_sh and a_w and abs(a_w[0] - a_sh[0]) > 1e-6:
        fwd = 1.0 if a_w[0] > a_sh[0] else -1.0
    hsign = out_sign if front else fwd
    pre = [i for i in order if (s_i if s_i is not None else order[0]) <= i <= (r_i if r_i is not None else order[-1])]
    post = [i for i in order if r_i is not None and i >= r_i]

    # pull-back
    pb = {'elbow_deg': None, 'depth_deg': None, 'hand_dist': None, 'duration_s': None, 'pause_s': None, 't': T(b_i)}
    if not front and b_i is not None:
        pb['elbow_deg'] = _r(_angle(rows, b_i))
        a_s = _angle(rows, s_i) if s_i is not None else None
        pb['depth_deg'] = _r(a_s - pb['elbow_deg']) if a_s is not None and pb['elbow_deg'] is not None else None
        p_s, p_w = _at(sh, b_i, order), _at(wr, b_i, order)
        pb['hand_dist'] = U(np.hypot(p_w[0] - p_s[0], p_w[1] - p_s[1])) if p_s and p_w else None
        pb['duration_s'] = _r(T(b_i) - T(s_i), 3) if s_i is not None else None
        h1, h2 = _hold(wr, order, b_i, unit, rows, True), _hold(wr, order, b_i, unit, rows, False)
        pb['pause_s'] = _r((h1 or 0) + (h2 or 0), 3) if h1 is not None or h2 is not None else None

    # elbow height (+ = above the shoulder), side-on only
    eh = {'setup': None, 'backswing': None, 'release': None, 'change': None, 'travel': None, 't': T(r_i)}
    if not front:
        for name, i in (('setup', s_i), ('backswing', b_i), ('release', r_i)):
            a, b = _at(sh, i, order), _at(el, i, order)
            eh[name] = U(a[1] - b[1]) if a and b else None
        ref = eh['backswing'] if eh['backswing'] is not None else eh['setup']
        eh['change'] = _r(eh['release'] - ref, 3) if ref is not None and eh['release'] is not None else None
        ys = [(sh[i][1] - el[i][1]) for i in pre if i in sh and i in el]
        eh['travel'] = U(max(ys) - min(ys)) if len(ys) >= 3 else None

    # head
    hm = {'travel_pre': None, 'travel_post': None, 'dx': None, 'dy': None, 't': T(r_i)}
    e1, e2 = _extent(hd, pre), _extent(hd, post)
    if e1:
        hm['travel_pre'] = U(e1[2])
    if e2:
        hm['travel_post'] = U(e2[2])
    h_s, h_r = _at(hd, s_i if s_i is not None else (pre[0] if pre else None), order), _at(hd, r_i, order)
    if h_s and h_r:
        hm['dx'] = U((h_r[0] - h_s[0]) * hsign)   # side: + = towards the board; front: + = outward
        hm['dy'] = U(h_s[1] - h_r[1])             # + = head rose

    # release
    rl = {'height': None, 'angle_deg': None, 'elbow_deg': None, 'timing_s': None, 'lateral': None, 't': T(r_i)}
    if r_i is not None:
        a, w = _at(sh, r_i, order), _at(wr, r_i, order)
        rl['height'] = U(a[1] - w[1]) if a and w else None
        if not front:
            e = _at(el, r_i, order)
            if e and w:
                rl['angle_deg'] = _r(np.degrees(np.arctan2(e[1] - w[1], abs(w[0] - e[0]) + 1e-9)))
            rl['elbow_deg'] = _r(_angle(rows, r_i))
            rl['timing_s'] = _r(T(r_i) - T(b_i), 3) if b_i is not None else None
        elif a and w:
            rl['lateral'] = U((w[0] - a[0]) * out_sign)

    # body
    def mid(tr1, tr2, i):
        p, q = _at(tr1, i, order), _at(tr2, i, order)
        return ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2) if p and q else (p or q)
    def lean(i):  # torso angle from vertical, + = leaning towards the board (side) / outward (front)
        a, b = mid(sh, osh, i), mid(hip, ohip, i)
        if not a or not b or abs(b[1] - a[1]) < 1e-6:
            return None
        return float(np.degrees(np.arctan2((a[0] - b[0]) * hsign, b[1] - a[1])))
    def hipx(i):
        m = mid(hip, ohip, i)
        return m[0] if m else None
    bd = {'lean_setup_deg': None, 'lean_release_deg': None, 'lean_change_deg': None, 'shoulder_sway': None,
          'hip_sway': None, 'shift_pre': None, 'shift_during': None, 'shift_after': None, 'shoulder_tilt_deg': None,
          't': T(r_i)}
    l_s, l_r = lean(s_i), lean(r_i)
    bd['lean_setup_deg'], bd['lean_release_deg'] = _r(l_s), _r(l_r)
    bd['lean_change_deg'] = _r(l_r - l_s) if l_s is not None and l_r is not None else None
    e = _extent(sh, pre)
    bd['shoulder_sway'] = U(e[0]) if e else None
    hips_track = {i: mid(hip, ohip, i) for i in order if mid(hip, ohip, i)}
    e = _extent(hips_track, pre)
    bd['hip_sway'] = U(e[0]) if e else None
    x0, xs, xr, xe = hipx(order[0]), hipx(s_i), hipx(r_i), hipx(order[-1])
    sh_ = lambda a, b: U((b - a) * hsign) if a is not None and b is not None else None
    bd['shift_pre'], bd['shift_during'], bd['shift_after'] = sh_(x0, xs), sh_(xs, xr), sh_(xr, xe)
    if front:
        bd['shoulder_tilt_deg'] = _r(_tilt(rows, r_i))

    # follow-through
    ft = {'extension_deg': None, 'hold_s': None, 'direction': None, 't': T(f_i)}
    if f_i is not None:
        ft['extension_deg'] = None if front else _r(_angle(rows, f_i))
        ft['hold_s'] = _hold(wr, order, f_i, unit, rows, True)
        if front:
            a, w = _at(sh, f_i, order), _at(wr, f_i, order)
            ft['direction'] = U((w[0] - a[0]) * out_sign) if a and w else None
    return {'unit': 'sw' if front else 'ul', 'pullback': pb, 'elbow': eh, 'head': hm, 'release': rl, 'body': bd,
            'follow': ft}


def _fmt_deg(a):
    return f'{a:.0f}°' if a is not None else 'not tracked'


def phase_observations(rows, phases):
    """Per-phase coaching observations: list of (phase, message). Experimental heuristics."""
    obs = []
    s, r, f = phases.get('setup'), phases.get('release'), phases.get('follow_through')
    if s is not None and r is not None:
        hs, hr = rows[s].get('head_relative'), rows[r].get('head_relative')
        if hs is not None and hr is not None and float(np.linalg.norm(np.subtract(hr, hs))) > HEAD_DRIFT_HIGH:
            obs.append(('release', 'Your head drifted between set-up and release. Try keeping your eyes and head '
                                   'still until the dart has gone.'))
        es, er = rows[s]['points'].get('elbow'), rows[r]['points'].get('elbow')
        if es is not None and er is not None and er[1] - es[1] > ELBOW_DROP_HIGH:
            obs.append(('release', 'Your elbow dropped as you released. Try keeping it at the same height and '
                                   'let the forearm do the work.'))
    fa = _angle(rows, f)
    if fa is not None and fa < FOLLOW_THROUGH_MIN_DEG:
        obs.append(('follow_through', f'Follow-through looks short (arm reached {fa:.0f}°). Try letting your arm '
                                      'extend towards the target after release.'))
    return obs


def phase_caption(rows, phase, i, wrist_speed=None):
    a = _angle(rows, i)
    if phase == 'setup':
        return f'Set-up: elbow {_fmt_deg(a)}. Settle and aim.'
    if phase == 'backswing':
        return f'Take-back: elbow closes to {_fmt_deg(a)}.'
    if phase == 'release':
        v = None if wrist_speed is None or np.isnan(wrist_speed[i]) else float(wrist_speed[i])
        return f'Release: elbow {_fmt_deg(a)}' + (f', wrist speed {v:.2f}/s.' if v is not None else '.')
    return f'Follow-through: arm extends to {_fmt_deg(a)}.'


def build_review(rows, segments, throws, wrist_speed=None, url_prefix=None, view='side'):
    """Key frames per throw plus a time-sorted timeline of phases and tips.

    segments: [{'indices', 'peak_index'}] aligned with throws (which already carry 'tips').
    Phase observations are merged into each throw's 'tips' (replacing the generic positive tip).
    Returns (key_frames_per_throw, timeline). Each key frame: phase, label, row_index, time_s,
    elbow_angle_deg, caption, frame (file name), frame_url. Timeline items: time_s, throw, type
    ('setup'|'backswing'|'release'|'follow_through'|'tip'), message, frame_url.
    """
    all_frames, timeline = [], []
    for seg, t in zip(segments, throws):
        n = t['throw']
        phases = pick_phases(rows, seg['indices'], seg.get('peak_index'), wrist_speed)
        front = view == 'front'
        obs = front_observations(t) if front else phase_observations(rows, phases)
        frames = []
        for phase in PHASES:
            if phase not in phases:
                continue
            i = phases[phase]
            name = f't{n}_{phase}.jpg'
            url = f'{url_prefix}{name}' if url_prefix else None
            base = front_caption(t, phase) if front else phase_caption(rows, phase, i, wrist_speed)
            tip = next((m for p, m in obs if p == phase), None)
            frames.append({'phase': phase, 'label': PHASE_LABELS[phase], 'row_index': i, 'time_s': rows[i]['time_s'],
                           'elbow_angle_deg': None if front else _angle(rows, i), 'caption': tip or base, 'observation': base,
                           'frame': name, 'frame_url': url})
            timeline.append({'time_s': rows[i]['time_s'], 'throw': n, 'type': phase, 'message': base, 'frame_url': url})
        by_phase = {f['phase']: f for f in frames}
        for p, m in obs:
            kf = by_phase.get(p)
            timeline.append({'time_s': kf['time_s'] if kf else t['start_s'], 'throw': n, 'type': 'tip', 'message': m,
                             'frame_url': kf['frame_url'] if kf else None})
        if obs:
            merged = [m for m in t.get('tips', []) if m not in (POSITIVE_TIP, FRONT_POSITIVE_TIP)] + [m for _, m in obs]
            t['tips'] = list(dict.fromkeys(merged))[:MAX_TIPS_PER_THROW]
        anchor = by_phase.get('release') or (frames[0] if frames else None)
        for m in t.get('tips', []):
            if any(m == om for _, om in obs):
                continue
            timeline.append({'time_s': anchor['time_s'] if anchor else t['start_s'], 'throw': n, 'type': 'tip',
                             'message': m, 'frame_url': anchor['frame_url'] if anchor else None})
        all_frames.append(frames)
    order = {p: k for k, p in enumerate(PHASES + ('tip',))}
    timeline.sort(key=lambda e: (e['time_s'], e['throw'], order[e['type']]))
    return all_frames, timeline


# ---- Front-on view ------------------------------------------------------------
# Filmed from in front of the thrower (phone beside the board, angled back). Sideways movement is visible;
# elbow drop and the elbow angle are foreshortened, so side-on-only metrics are left out (None).
# Lateral values are in shoulder widths, signed so + = OUTWARD (away from the body on the throwing side)
# and - = ACROSS the body. That is independent of mirrored selfie footage.
# The "shoulder width" is the real one where possible: darts players stand angled to the board, so the
# shoulders look much narrower from the front than they are. body_frame() floors the apparent width at
# TORSO_TO_SHOULDER x the shoulder-to-hip length (which doesn't shrink when the body turns).
# A good release is NOT on the shoulder line: the dart is aimed under the eye, which sits between the
# throwing shoulder and the middle of the head. FRONT_AIM_IN..FRONT_AIM_OUT is that aiming band.
FRONT_FLARE_HIGH = 0.22         # elbow this far outside the shoulder's vertical line at release = flaring ("chicken wing")
FRONT_AIM_IN = -0.60            # hand released up to this far inside the shoulder line = under the aiming eye (fine)
FRONT_AIM_OUT = 0.15            # ...and up to this far outside it
TORSO_TO_SHOULDER = 0.75        # shoulder width ~= 0.75 x shoulder-to-hip length (adult average)
FRONT_CROSS_HIGH = 0.25         # wrist finishing this far across its own release line after release
FRONT_OUT_HIGH = 0.30           # ...or this far out wide
FRONT_SWAY_HIGH = 0.15          # head moved sideways this many shoulder widths during one throw
FRONT_LEAN_HIGH_DEG = 6.0       # shoulder line tilted this much more at release than at set-up
FRONT_RELEASE_SPREAD_HIGH = 0.15  # release point moved sideways this much between throws
FRONT_RELEASE_SPREAD_TIGHT = 0.06
FRONT_POSITIVE_TIP = ('Nice one. Your arm stayed on its line and your body stayed still on this throw. '
                      'Keep that alignment!')


def body_frame(rows):
    """(shoulder_width, outward_sign) from the clip: median shoulder separation and which way is 'out'.

    The apparent width shrinks when the thrower stands angled (normal in darts), which would inflate every
    offset. When hips and the frame aspect are known, the width is floored at TORSO_TO_SHOULDER x torso length.
    """
    d = [r['points']['shoulder'][0] - r['points']['other_shoulder'][0] for r in rows
         if r['points'].get('shoulder') and r['points'].get('other_shoulder')]
    if len(d) < 3:
        return None, 1.0
    width = float(np.median(np.abs(d)))
    torso = []
    for r in rows:
        p, asp = r['points'], r.get('aspect')
        if asp and all(p.get(k) for k in ('shoulder', 'other_shoulder', 'hip', 'other_hip')):
            sy = (p['shoulder'][1] + p['other_shoulder'][1]) / 2
            hy = (p['hip'][1] + p['other_hip'][1]) / 2
            torso.append(abs(hy - sy) / asp)  # height-normalised -> width-normalised units
    if len(torso) >= 3:
        width = max(width, TORSO_TO_SHOULDER * float(np.median(torso)))
    return (width if width > 0.02 else None), (1.0 if np.median(d) >= 0 else -1.0)


def lateral(p, shoulder, frame):
    """Sideways offset of point p from the vertical line through the shoulder, in shoulder widths (+ = outward)."""
    width, sign = frame
    if p is None or shoulder is None or not width:
        return None
    return round(float((p[0] - shoulder[0]) * sign / width), 3)


def _near(rows, idx, i, key, k=2):
    """Point `key` at row i, or the nearest row in this throw window (within k samples) that has it."""
    if i is None:
        return None, None
    for d in range(k + 1):
        for j in (i - d, i + d):
            if j in idx and rows[j]['points'].get(key) is not None and rows[j]['points'].get('shoulder') is not None:
                return rows[j]['points'][key], rows[j]['points']['shoulder']
    return None, None


def _tilt(rows, i):
    if i is None:
        return None
    s, o = rows[i]['points'].get('shoulder'), rows[i]['points'].get('other_shoulder')
    if s is None or o is None or abs(s[0] - o[0]) < 1e-6:
        return None
    return float(np.degrees(np.arctan2(s[1] - o[1], abs(s[0] - o[0]))))  # + = throwing shoulder lower


def head_sway(rows, idx, start, release, frame):
    """Sideways head travel from set-up to just after release, in shoulder widths. Smoothed (3-sample median)
    and measured as the 10th-90th percentile range, so one jittery pose frame can't fake a big sway."""
    if not frame[0]:
        return None
    order = sorted(idx)
    lo = start if start is not None else order[0]
    hi = (release + 1) if release is not None else order[-1]
    xs = [rows[i]['points']['head'][0] for i in order if lo <= i <= hi and rows[i]['points'].get('head') is not None]
    if len(xs) < 3:
        return None
    sm = [float(np.median(xs[max(0, k - 1):k + 2])) for k in range(len(xs))]
    return round(float(np.percentile(sm, 90) - np.percentile(sm, 10)) / frame[0], 3)


def front_metrics(rows, indices, phases, frame):
    """Front-on metrics for one throw window (see constants above). Missing values are None."""
    idx = set(int(i) for i in indices)
    r, s, f = phases.get('release'), phases.get('setup'), phases.get('follow_through')
    e_r, sh_r = _near(rows, idx, r, 'elbow')
    w_r, sh_w = _near(rows, idx, r, 'wrist')
    w_f, sh_f = _near(rows, idx, f, 'wrist')
    forearm = None
    if e_r is not None and w_r is not None and frame[0]:
        dx, dy = (w_r[0] - e_r[0]) * frame[1], e_r[1] - w_r[1]
        forearm = round(float(np.degrees(np.arctan2(dx, max(abs(dy), 1e-6)))), 1)  # + = hand outside the elbow
    sway = head_sway(rows, idx, s, r, frame)
    ts, tr = _tilt(rows, s), _tilt(rows, r)
    lean = round(tr - ts, 1) if ts is not None and tr is not None else None
    return {'elbow_flare': lateral(e_r, sh_r, frame), 'wrist_line_release': lateral(w_r, sh_w, frame),
            'forearm_deviation_deg': forearm, 'head_sway': sway, 'shoulder_tilt_change_deg': lean,
            'follow_through_lateral': lateral(w_f, sh_f, frame)}


def front_observations(m):
    """Coaching observations (phase, message) for one throw's front-on metrics. Experimental heuristics."""
    obs = []
    fl, wl, ft, sw, ln = (m.get(k) for k in ('elbow_flare', 'wrist_line_release', 'follow_through_lateral',
                                              'head_sway', 'shoulder_tilt_change_deg'))
    if fl is not None and fl > FRONT_FLARE_HIGH:
        obs.append(('release', f'Your elbow flared out at release ({fl * 100:.0f}% of a shoulder width outside your '
                               'shoulder line). Keep the elbow tucked under the dart so the arm swings straight.'))
    elif wl is not None and (wl > FRONT_AIM_OUT or wl < FRONT_AIM_IN):
        side = 'outside your shoulder' if wl > 0 else 'across your body, past your aiming eye'
        obs.append(('release', f'Your hand released {abs(wl) * 100:.0f}% of a shoulder width {side}. '
                               'Line the dart, hand and elbow up under your aiming eye.'))
    fr = ft - wl if ft is not None and wl is not None else ft  # follow-through vs the release line
    if fr is not None and fr < -FRONT_CROSS_HIGH:
        obs.append(('follow_through', 'Your hand finished across your body. Follow through straight at the target '
                                      'rather than pulling it in.'))
    elif fr is not None and fr > FRONT_OUT_HIGH:
        obs.append(('follow_through', 'Your hand finished out wide. Point your follow-through straight at the '
                                      'target, not out to the side.'))
    if sw is not None and sw > FRONT_SWAY_HIGH:
        obs.append(('release', 'Your head and body swayed sideways during the throw. Keep your weight still on '
                               'your front foot until the dart lands.'))
    elif ln is not None and abs(ln) > FRONT_LEAN_HIGH_DEG:
        obs.append(('release', f'Your shoulders tilted {abs(ln):.0f}° between set-up and release. Stay level and '
                               'let only the arm move.'))
    return obs


def front_throw_tips(throws):
    """Front-on version of throw_tips: alignment and sway first, then consistency with the other throws."""
    out = []
    for i, t in enumerate(throws):
        tips = []
        if t['valid_frames'] < LOW_VALID_FRAMES:
            tips.append(f"Only {t['valid_frames']} clear frames on this throw. Keep your head, both shoulders and "
                        'throwing arm in frame, with even light.')
        tips += [m for _, m in front_observations(t)]
        wl, wl_o = t.get('wrist_line_release'), _others_median(throws, i, 'wrist_line_release')
        if not tips and wl is not None and wl_o is not None and abs(wl - wl_o) > FRONT_RELEASE_SPREAD_HIGH:
            tips.append(f"Your release point was {abs(wl - wl_o) * 100:.0f}% of a shoulder width "
                        f"{'further out' if wl > wl_o else 'further in'} than on your other throws. "
                        'Aim to release from the same spot every time.')
        out.append((tips or [FRONT_POSITIVE_TIP])[:MAX_TIPS_PER_THROW])
    return out


def front_session_tips(throws, method, arm_detected_pct=None):
    tips = []
    if method != 'auto':
        tips.append('Throws were not detected automatically, so these windows may not line up with each throw. '
                    'Trim idle time and keep the camera still, then try again.')
    rel = [t['wrist_line_release'] for t in throws if t.get('wrist_line_release') is not None]
    if len(rel) >= 2:  # 10th-90th percentile range: one mis-tracked throw shouldn't decide it
        spread = float(np.percentile(rel, 90) - np.percentile(rel, 10)) if len(rel) >= 5 else max(rel) - min(rel)
        if spread > FRONT_RELEASE_SPREAD_HIGH:
            tips.append(f'Your release point moved sideways by {spread * 100:.0f}% of a shoulder width across throws. '
                        'A repeatable line is what tightens left-right grouping.')
        elif spread <= FRONT_RELEASE_SPREAD_TIGHT and method == 'auto':
            tips.append('Great line: your release point stayed in the same place sideways on every throw.')
    flares = [t['elbow_flare'] for t in throws if t.get('elbow_flare') is not None]
    if flares and float(np.median(flares)) > FRONT_FLARE_HIGH:
        tips.append('Your elbow flared out on most throws (a "chicken wing"). Keeping it under the dart is the '
                    'quickest fix for left-right misses.')
    sways = [t['head_sway'] for t in throws if t.get('head_sway') is not None]
    if sways and float(np.median(sways)) > FRONT_SWAY_HIGH:
        tips.append('Your body swayed sideways on most throws. Work on a balanced stance with your weight forward.')
    if arm_detected_pct is not None and arm_detected_pct < SESSION_LOW_DETECTION_PCT:
        tips.append(f'Your arm was only tracked in {arm_detected_pct:.0f}% of frames. Film from beside the board at '
                    'eye height with your whole upper body in frame.')
    return tips[:MAX_SESSION_TIPS]


def front_caption(m, phase):
    fmt = lambda v: 'not tracked' if v is None else f"{abs(v) * 100:.0f}% {'out' if v > 0 else 'in'}"
    if phase == 'setup':
        return 'Set-up: stand still and line the dart up with your eye.'
    if phase == 'backswing':
        return 'Take-back: keep the hand on the same line as the elbow.'
    if phase == 'release':
        return f"Release: elbow {fmt(m.get('elbow_flare'))}, hand {fmt(m.get('wrist_line_release'))} of the shoulder line (under the eye is about 20-50% in)."
    return f"Follow-through: hand finishes {fmt(m.get('follow_through_lateral'))} of the shoulder line."
