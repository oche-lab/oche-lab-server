"""Public privacy page for the Oche Lab app (store listings link here). Keep in step with the app's
src/logic/privacy.js and with guard.py's retention: rate-limit hits pruned after 2 days, jobs/results after 1 hour,
uploads deleted after analysis, share clips deleted after sending."""
from html import escape

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

UPDATED = '26 November 2026'
SECTIONS = [
    ('No account, no tracking', 'Oche Lab has no sign-up, no adverts and no analytics or tracking tools. We don\u2019t sell or share your data.'),
    ('Stays on your phone', 'Your name, sessions, practice games, matches, Mind Lab journal, settings and replay videos are saved only on your phone. Deleting the app deletes them. Profile \u203a Data lets you export or delete everything at any time.'),
    ('Video analysis', 'When you tap Analyse, the clip is uploaded over HTTPS to the Oche Lab analysis server, which measures your throwing action. The uploaded video is deleted as soon as the analysis finishes (or fails). The generated report and replay are kept on the server for up to one hour so your phone can download them, then deleted.'),
    ('Fair-use limits', 'To keep the free service fair, the app sends a random install code (not linked to you) with each analysis. The server keeps the install code and your IP address with the time of each analysis for up to two days, only to apply the hourly and daily limits.'),
    ('Board photos and sharing', 'Photos you take to mark your darts are read on your phone. If you share a throw video, the replay is sent to the server to add the Oche Lab frame and both files are deleted once the video is sent back. Share cards, challenges and data exports go only where you choose to send them.'),
    ('Notifications', 'Reminders are scheduled on your phone. Nothing is sent from a server.'),
    ('Children', 'Oche Lab is not aimed at children under 13 and doesn\u2019t knowingly collect their data.'),
]

router = APIRouter()


def page() -> str:
    body = ''.join(f'<h2>{escape(t)}</h2><p>{escape(b)}</p>' for t, b in SECTIONS)
    return ('<!doctype html><html lang="en-GB"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Oche Lab privacy</title><style>body{font:16px/1.55 -apple-system,system-ui,sans-serif;max-width:680px;margin:0 auto;padding:32px 20px;'
            'background:#09090B;color:#F5F5F7}h1{font-size:28px;margin:0 0 4px}h2{font-size:18px;margin:28px 0 6px}p{color:#C9C9D2;margin:0}'
            '.u{color:#8B8B95;font-size:14px}</style></head><body>'
            f'<h1>Oche Lab privacy</h1><p class="u">Updated {escape(UPDATED)}</p>{body}</body></html>')


@router.get('/privacy', response_class=HTMLResponse)
def privacy():
    return HTMLResponse(page(), headers={'Cache-Control': 'public, max-age=3600'})
