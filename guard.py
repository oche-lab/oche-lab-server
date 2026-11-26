"""Access control for the analysis API: install id, optional app key, rate limits, job state.

The shared app key is read from OCHELAB_APP_KEY and must never be committed.
Render's disk is ephemeral, so persisted jobs only survive a process restart on
the same instance; a new instance treats in-flight work as lost so the app can retry.
"""
import hmac
import json
import os
import re
import sqlite3
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException, Request

INSTALL_HEADER = 'X-Install-Id'
APP_KEY_HEADER = 'X-Oche-App-Key'
INSTALL_RE = re.compile(r'^[A-Za-z0-9._-]{8,128}$')
HOUR_S = 3600
DAY_S = 86400
LOST_DETAIL = {
    'status': 'lost',
    'retry': True,
    'message': 'The server restarted before your analysis finished. The app will try again automatically.',
}

DEFAULTS = {
    'hour_limit': 6,
    'day_limit': 20,
    'ip_hour_limit': 10,
    'ip_day_limit': 30,
    'max_queue': 3,
    'disk_budget': 400 * 1024 * 1024,
    'allow_legacy': True,
}

_lock = threading.Lock()
_db = None
_cfg = {}


def _flag(name, default=True):
    v = os.environ.get(name)
    if v is None:
        return default
    return v.lower() in ('1', 'true', 'yes')


def _int_env(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _state_path():
    return Path(os.environ.get('OCHELAB_STATE_PATH', Path(tempfile.gettempdir()) / 'ochelab_state.sqlite'))


def configure(**overrides):
    """Load settings from the environment, then apply test overrides."""
    global _cfg, _db
    with _lock:
        if _db is not None:
            try:
                _db.close()
            except sqlite3.Error:
                pass
            _db = None
        _cfg = {
            'hour_limit': _int_env('OCHELAB_RATE_HOUR', DEFAULTS['hour_limit']),
            'day_limit': _int_env('OCHELAB_RATE_DAY', DEFAULTS['day_limit']),
            'ip_hour_limit': _int_env('OCHELAB_IP_RATE_HOUR', DEFAULTS['ip_hour_limit']),
            'ip_day_limit': _int_env('OCHELAB_IP_RATE_DAY', DEFAULTS['ip_day_limit']),
            'max_queue': _int_env('OCHELAB_MAX_QUEUE', DEFAULTS['max_queue']),
            'disk_budget': _int_env('OCHELAB_DISK_BUDGET', DEFAULTS['disk_budget']),
            'allow_legacy': _flag('OCHELAB_ALLOW_LEGACY', DEFAULTS['allow_legacy']),
            'app_key': os.environ.get('OCHELAB_APP_KEY') or '',
            'state_path': _state_path(),
        }
        for k, v in overrides.items():
            if k == 'app_key' and v is None:
                _cfg['app_key'] = ''
            elif k == 'state_path' and v is not None:
                _cfg['state_path'] = Path(v)
            elif v is not None:
                _cfg[k] = v
        Path(_cfg['state_path']).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(_cfg['state_path']), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute('CREATE TABLE IF NOT EXISTS hits (key TEXT NOT NULL, ts REAL NOT NULL)')
        conn.execute('CREATE INDEX IF NOT EXISTS hits_key_ts ON hits (key, ts)')
        conn.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, status TEXT, install_id TEXT, created REAL, updated REAL, payload TEXT)')
        conn.commit()
        _db = conn
    return _cfg


def ensure_configured():
    if not _cfg:
        configure()
    return _cfg


def cfg(name, default=None):
    if not _cfg:
        configure()
    return _cfg.get(name, default)


def _conn():
    if _db is None:
        configure()
    return _db


@dataclass
class Client:
    install_id: str
    ip: str
    legacy: bool


def client_ip(request: Request) -> str:
    forwarded = request.headers.get('x-forwarded-for')
    if forwarded:
        return forwarded.split(',')[0].strip() or 'unknown'
    host = request.client.host if request.client else None
    return host or 'unknown'


def _header(request: Request, name: str) -> str:
    return (request.headers.get(name) or '').strip()


def _keys_match(provided: str, expected: str) -> bool:
    if not expected:
        return True
    if not provided or len(provided) != len(expected):
        hmac.compare_digest(expected, expected)
        return False
    return hmac.compare_digest(provided, expected)


def authorize(request: Request) -> Client:
    """Require an install id (and the shared app key if configured).

    During rollout, OCHELAB_ALLOW_LEGACY=1 (the default) lets older app versions
    through, rate-limited by IP only.
    """
    if not _cfg:
        configure()
    ip = client_ip(request)
    install = _header(request, INSTALL_HEADER)
    key = _header(request, APP_KEY_HEADER)
    expected = _cfg.get('app_key') or ''
    legacy = _cfg.get('allow_legacy', True)
    if expected and not _keys_match(key, expected) and not legacy:
        raise HTTPException(401, 'Please update the Oche Lab app and try again.')
    if install and not INSTALL_RE.match(install):
        raise HTTPException(400, 'Invalid install id.')
    if not install:
        if not legacy:
            raise HTTPException(401, 'Please update the Oche Lab app and try again.')
        return Client(install_id=f'legacy:{ip}', ip=ip, legacy=True)
    return Client(install_id=install, ip=ip, legacy=False)


def _count_hits(key: str, window_s: float, now: float) -> int:
    cur = _conn().execute('SELECT COUNT(*) FROM hits WHERE key = ? AND ts > ?', (key, now - window_s))
    return int(cur.fetchone()[0])


def _oldest_in_window(key: str, window_s: float, now: float) -> float | None:
    cur = _conn().execute('SELECT MIN(ts) FROM hits WHERE key = ? AND ts > ?', (key, now - window_s))
    row = cur.fetchone()
    return float(row[0]) if row and row[0] is not None else None


def _retry_after(key: str, window_s: float, now: float) -> int:
    oldest = _oldest_in_window(key, window_s, now)
    if oldest is None:
        return 60
    return max(60, int(oldest + window_s - now) + 1)


def _limit_message(limit: int, window: str, retry_after: int) -> str:
    mins = max(1, (retry_after + 59) // 60)
    unit = 'minute' if mins == 1 else 'minutes'
    if window == 'hour':
        return f"You've reached the limit of {limit} analyses per hour. Try again in about {mins} {unit}."
    return f"You've reached the daily limit of {limit} analyses. Try again in about {mins} {unit}."


def check_rate(client: Client, now=None):
    """Reject with a friendly 429 if this install or IP is over its cap."""
    now = now or time.time()
    with _lock:
        conn = _conn()
        conn.execute('DELETE FROM hits WHERE ts < ?', (now - DAY_S * 2,))
        conn.commit()
        checks = [
            (f'install:{client.install_id}', cfg('hour_limit'), HOUR_S, 'hour'),
            (f'install:{client.install_id}', cfg('day_limit'), DAY_S, 'day'),
            (f'ip:{client.ip}', cfg('ip_hour_limit'), HOUR_S, 'hour'),
            (f'ip:{client.ip}', cfg('ip_day_limit'), DAY_S, 'day'),
        ]
        for key, limit, window, label in checks:
            if limit and _count_hits(key, window, now) >= limit:
                retry = _retry_after(key, window, now)
                raise HTTPException(429, _limit_message(limit, label, retry), headers={'Retry-After': str(retry)})


def remaining(client: Client, now=None) -> dict:
    """How many analyses this install has left this hour and today (the tighter of its own and its IP's caps).
    Read-only; lets the app show "5 analyses left this hour" before upload instead of a surprise 429."""
    now = now or time.time()
    with _lock:
        out = {}
        for label, window, own, ip_cap in (('hour', HOUR_S, cfg('hour_limit'), cfg('ip_hour_limit')),
                                           ('day', DAY_S, cfg('day_limit'), cfg('ip_day_limit'))):
            lefts = []
            if own:
                lefts.append(max(0, own - _count_hits(f'install:{client.install_id}', window, now)))
            if ip_cap:
                lefts.append(max(0, ip_cap - _count_hits(f'ip:{client.ip}', window, now)))
            out[f'{label}_limit'] = own or None
            out[f'{label}_left'] = min(lefts) if lefts else None
        return out


def record_analysis(client: Client, now=None):
    now = now or time.time()
    with _lock:
        conn = _conn()
        conn.execute('INSERT INTO hits (key, ts) VALUES (?, ?)', (f'install:{client.install_id}', now))
        conn.execute('INSERT INTO hits (key, ts) VALUES (?, ?)', (f'ip:{client.ip}', now))
        conn.commit()


def clear_hits():
    with _lock:
        _conn().execute('DELETE FROM hits')
        _conn().commit()


def queued_count(statuses=('queued',)):
    with _lock:
        q = ','.join('?' * len(statuses))
        cur = _conn().execute(f'SELECT COUNT(*) FROM jobs WHERE status IN ({q})', statuses)
        return int(cur.fetchone()[0])


def check_queue_capacity():
    limit = cfg('max_queue')
    if limit and queued_count(('queued',)) >= limit:
        raise HTTPException(429, "The analysis server is busy right now. Please try again in a minute.",
                            headers={'Retry-After': '60'})


def save_job(job_id: str, **fields):
    """Upsert job metadata. `result` and other large fields go in the JSON payload."""
    now = fields.pop('updated', None) or time.time()
    with _lock:
        conn = _conn()
        row = conn.execute('SELECT payload, created, install_id, status FROM jobs WHERE id = ?', (job_id,)).fetchone()
        payload = json.loads(row['payload']) if row and row['payload'] else {}
        created = fields.pop('created', None) or (row['created'] if row else now)
        install_id = fields.pop('install_id', None) or (row['install_id'] if row else None)
        payload.update(fields)
        status = payload.get('status') or (row['status'] if row else 'queued')
        payload['status'] = status
        conn.execute(
            'INSERT INTO jobs (id, status, install_id, created, updated, payload) VALUES (?, ?, ?, ?, ?, ?) '
            'ON CONFLICT(id) DO UPDATE SET status=excluded.status, install_id=COALESCE(excluded.install_id, jobs.install_id), '
            'updated=excluded.updated, payload=excluded.payload',
            (job_id, status, install_id, created, now, json.dumps(payload, default=str)))
        conn.commit()
        return {**payload, 'install_id': install_id, 'created': created, 'updated': now, 'id': job_id}


def load_job(job_id: str):
    with _lock:
        row = _conn().execute('SELECT * FROM jobs WHERE id = ?', (job_id,)).fetchone()
        if not row:
            return None
        payload = json.loads(row['payload']) if row['payload'] else {}
        payload.update(id=row['id'], status=row['status'] or payload.get('status'),
                       install_id=row['install_id'], created=row['created'], updated=row['updated'])
        return payload


def all_jobs():
    with _lock:
        rows = _conn().execute('SELECT * FROM jobs').fetchall()
        out = []
        for row in rows:
            payload = json.loads(row['payload']) if row['payload'] else {}
            payload.update(id=row['id'], status=row['status'] or payload.get('status'),
                           install_id=row['install_id'], created=row['created'], updated=row['updated'])
            out.append(payload)
        return out


def delete_job(job_id: str):
    with _lock:
        _conn().execute('DELETE FROM jobs WHERE id = ?', (job_id,))
        _conn().commit()


def prune_jobs(ttl_s, now=None):
    now = now or time.time()
    with _lock:
        conn = _conn()
        rows = conn.execute('SELECT id FROM jobs WHERE updated < ?', (now - ttl_s,)).fetchall()
        ids = [r['id'] for r in rows]
        if ids:
            conn.executemany('DELETE FROM jobs WHERE id = ?', [(i,) for i in ids])
            conn.commit()
        return ids


def queue_position(job_id: str):
    """1-based position among queued jobs (not including a running analysis)."""
    with _lock:
        rows = _conn().execute(
            "SELECT id FROM jobs WHERE status = 'queued' ORDER BY created ASC, id ASC").fetchall()
        ids = [r['id'] for r in rows]
    try:
        return ids.index(job_id) + 1
    except ValueError:
        return None


def owns_job(job: dict, client: Client) -> bool:
    owner = (job or {}).get('install_id')
    if not owner:
        # Jobs recorded before this release: allow during rollout, hide otherwise.
        return bool(cfg('allow_legacy'))
    return owner == client.install_id


def lost_http():
    return HTTPException(410, LOST_DETAIL)


def dir_size(path: Path) -> int:
    if not path or not Path(path).exists():
        return 0
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return total


def over_budget(paths, extra=0) -> bool:
    budget = cfg('disk_budget') or 0
    if not budget:
        return False
    return sum(dir_size(Path(p)) for p in paths if p) + extra > budget
