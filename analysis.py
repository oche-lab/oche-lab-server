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
PEAK_MIN_SEPARATION_S = 1.2  # two peaks closer than this are one throw (take-back + release); real throws are 2 s+ apart
PEAK_HYST_LOW_FRAC = 0.5     # hysteresis: a burst only ends once the signal drops below this fraction of the threshold
PEAK_MAD_K = 4.0             # adaptive threshold = median + k * (1.4826 * MAD)
PEAK_MIN_WRIST_SPEED = 0.25  # absolute floor, normalised image units / second
PEAK_MAX_FRAC = 0.5          # threshold never above this fraction of the tallest peak (busy clips)
PEAK_MIN_REL_HEIGHT = 0.4    # ignore peaks smaller than this fraction of the tallest (arm return etc.)
WINDOW_BEFORE_S = 0.7        # window starts this long before the peak...
WINDOW_AFTER_S = 0.4         # ...and ends this long after it
MIN_WINDOW_FRAMES = 4        # drop windows with fewer sampled frames than this
MAX_THROWS = 15             # hard cap; the app also sends max_throws (darts on board photos / 3 per visit)
EXT_WINDOW_S = 0.35          # side-on: look this far either side of a peak for the arm extending forward...
EXT_MIN_ELBOW_DEG = 15       # ...elbow opening by at least this much...
EXT_MIN_REACH = 0.25         # ...or wrist moving away from the shoulder by this fraction of upper-arm length
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


def speed_signals(wrist_points, elbow_angles, times):
    """Return (wrist_speed, elbow_angular_speed) per sampled frame; NaN where unknown."""
    t = np.asarray(times, dtype=float)
    wrist = smooth(interpolate_gaps(_series(wrist_points), t))
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
    if len(s) < 3:
        return []
    low = low_frac * threshold
    cand, i = [], 0
    while i < len(s):
        if s[i] >= threshold:
            j = i
            while j + 1 < len(s) and s[j + 1] >= low:
                j += 1
            cand.append(i + int(np.argmax(s[i:j + 1])))
            i = j + 1
        else:
            i += 1
    if cand:
        tallest = max(s[i] for i in cand)
        cand = [i for i in cand if s[i] >= PEAK_MIN_REL_HEIGHT * tallest]
    kept = []
    for i in sorted(cand, key=lambda i: -s[i]):
        if all(abs(t[i] - t[j]) >= min_sep_s for j in kept):
            kept.append(i)
    return sorted(kept)


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
    return (open_deg is not None and open_deg >= EXT_MIN_ELBOW_DEG) or (reach is not None and reach >= EXT_MIN_REACH)


def segment_throws(rows, view='side', max_throws=None):
    """Detect throws from wrist/elbow motion. Falls back to equal windows if detection fails.

    rows: dicts with 'time_s', 'points' {'wrist','elbow',...}, 'elbow_angle_deg'.
    view: 'side' also requires a forward arm extension at each peak (front-on 2D angles can't show it).
    max_throws: optional cap from the app (darts marked on board photos, or 3 per visit).
    Returns dict: method, sentence, segments [{'indices', 'peak_index'|None}], wrist_speed.
    """
    times = [r['time_s'] for r in rows]
    wrist_pts = [r['points'].get('wrist') for r in rows]
    wrist_speed, elbow_speed = speed_signals(wrist_pts, [r['elbow_angle_deg'] for r in rows], times)
    reason = None
    coverage = float(np.mean(~np.isnan(wrist_speed))) if len(rows) else 0.0
    if len(rows) < 3 * MIN_WINDOW_FRAMES:
        reason = 'clip too short'
    elif coverage < MIN_VALID_SIGNAL_PCT:
        reason = 'wrist not tracked consistently'
    else:
        sig = detection_signal(wrist_speed, elbow_speed)
        thr = adaptive_threshold(sig)
        peaks = find_peaks(sig, times, thr)
        if view != 'front' and peaks:
            forward = [p for p in peaks if is_forward_throw(rows, times, p)]
            peaks = forward or peaks  # if nothing passes, tracking is too rough to judge: keep the motion peaks
        cap = MAX_THROWS if not max_throws else max(1, min(int(max_throws), MAX_THROWS))
        if len(peaks) > cap:
            peaks = sorted(sorted(peaks, key=lambda i: -sig[i])[:cap])
        wins = windows_around_peaks(peaks, times)
        if wins:
            n = len(wins)
            return {'method': 'auto',
                    'sentence': f"Detected {n} throw{'s' if n != 1 else ''} automatically from wrist and elbow speed peaks.",
                    'segments': [{'indices': idx, 'peak_index': int(p)} for p, idx in wins],
                    'wrist_speed': wrist_speed}
        reason = 'no clear motion peaks'
    segs = np.array_split(np.arange(len(rows)), FALLBACK_WINDOWS)
    return {'method': 'equal_windows_fallback',
            'sentence': f'Automatic throw detection failed ({reason}); split the clip into three equal time windows instead.',
            'segments': [{'indices': s, 'peak_index': None} for s in segs],
            'wrist_speed': wrist_speed}


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
FRONT_FLARE_HIGH = 0.22         # elbow this far outside the shoulder's vertical line at release = flaring ("chicken wing")
FRONT_LINE_HIGH = 0.20          # wrist this far off the shoulder line at release
FRONT_CROSS_HIGH = 0.25         # wrist finishing this far across the body after release
FRONT_OUT_HIGH = 0.30           # ...or this far out wide
FRONT_SWAY_HIGH = 0.15          # head moved sideways this many shoulder widths during one throw
FRONT_LEAN_HIGH_DEG = 6.0       # shoulder line tilted this much more at release than at set-up
FRONT_RELEASE_SPREAD_HIGH = 0.15  # release point moved sideways this much between throws
FRONT_RELEASE_SPREAD_TIGHT = 0.06
FRONT_POSITIVE_TIP = ('Nice one. Your arm stayed on its line and your body stayed still on this throw. '
                      'Keep that alignment!')


def body_frame(rows):
    """(shoulder_width, outward_sign) from the clip: median shoulder separation and which way is 'out'."""
    d = [r['points']['shoulder'][0] - r['points']['other_shoulder'][0] for r in rows
         if r['points'].get('shoulder') and r['points'].get('other_shoulder')]
    if len(d) < 3:
        return None, 1.0
    width = float(np.median(np.abs(d)))
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
    heads = [rows[i]['points']['head'][0] for i in sorted(idx) if rows[i]['points'].get('head') is not None]
    sway = round(float(np.ptp(heads)) / frame[0], 3) if len(heads) >= 3 and frame[0] else None
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
    elif wl is not None and abs(wl) > FRONT_LINE_HIGH:
        side = 'outside' if wl > 0 else 'inside'
        obs.append(('release', f'Your hand released {abs(wl) * 100:.0f}% of a shoulder width {side} your shoulder line. '
                               'Line the dart, hand and elbow up under your aiming eye.'))
    if ft is not None and ft < -FRONT_CROSS_HIGH:
        obs.append(('follow_through', 'Your hand finished across your body. Follow through straight at the target '
                                      'rather than pulling it in.'))
    elif ft is not None and ft > FRONT_OUT_HIGH:
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
    if len(rel) >= 2:
        spread = max(rel) - min(rel)
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
        return f"Release: elbow {fmt(m.get('elbow_flare'))}, hand {fmt(m.get('wrist_line_release'))} of the shoulder line."
    return f"Follow-through: hand finishes {fmt(m.get('follow_through_lateral'))} of the shoulder line."
