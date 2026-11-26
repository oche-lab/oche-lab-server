"""Public support page for the Oche Lab app (App Store / Google Play 'Support URL').

Contact route: GitHub issues on this public repo (no personal email exposed). If the owner later sets the
OCHELAB_SUPPORT_EMAIL environment variable on the server, a mail link is shown as well."""
import os
from html import escape

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

ISSUES_URL = 'https://github.com/oche-lab/oche-lab-server/issues/new?title=Oche%20Lab%20support%3A%20'
UPDATED = '26 November 2026'
FAQ = [
    ('How does throw analysis work?',
     'Film one to three throws on your phone (side-on or front-on), then tap Analyse. The clip is uploaded to the Oche Lab '
     'analysis server, which tracks your arm through the throw and returns measurements such as elbow position, release '
     'and follow-through, plus drills to work on. It usually takes under a minute; the first analysis after a quiet spell '
     'can take longer while the server wakes up.'),
    ('Tips for a good clip',
     'Prop the phone so your whole throwing arm and the dart are in frame, keep it still, film in good light and keep clips '
     'under a minute. Choose side-on or front-on and your throwing arm before you record so the coach reads the right angle.'),
    ('Is it free?',
     'Yes. Oche Lab is free with no adverts, no subscriptions and no in-app purchases. To keep the free analysis server fair '
     'for everyone there is an hourly and daily analysis limit; the Record screen shows how many you have left.'),
    ('Do I need an account?',
     'No. There is no sign-up. Your sessions, games, Mind Lab journal and settings are stored only on your phone.'),
    ('How do I export or delete my data?',
     'In the app open Profile, then Data, to export or delete everything. Uploaded videos are deleted from the server as soon '
     'as analysis finishes, and reports are deleted within an hour. Deleting the app removes everything stored on your phone.'),
    ('Analysis failed or says no thrower found',
     'Check that your arm is clearly visible for the whole throw, the phone is steady, and the clip is under about a minute. '
     'If the server is busy or waking up, wait a minute and tap Try again.'),
]

router = APIRouter()


def page() -> str:
    email = (os.environ.get('OCHELAB_SUPPORT_EMAIL') or '').strip()
    faq = ''.join(f'<h3>{escape(q)}</h3><p>{escape(a)}</p>' for q, a in FAQ)
    mail = (f'<p>Or email <a href="mailto:{escape(email)}">{escape(email)}</a>.</p>' if email else '')
    return ('<!doctype html><html lang="en-GB"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Oche Lab support</title><style>body{font:16px/1.55 -apple-system,system-ui,sans-serif;max-width:680px;margin:0 auto;padding:32px 20px;'
            'background:#09090B;color:#F5F5F7}h1{font-size:28px;margin:0 0 4px}h2{font-size:20px;margin:32px 0 8px}h3{font-size:17px;margin:22px 0 4px}'
            'p{color:#C9C9D2;margin:0}a{color:#FF5C77}.u{color:#8B8B95;font-size:14px}.btn{display:inline-block;margin:12px 0 6px;padding:12px 18px;'
            'border-radius:12px;background:#17171D;border:1px solid #2A2A33;color:#F5F5F7;text-decoration:none;font-weight:600}</style></head><body>'
            f'<h1>Oche Lab support</h1><p class="u">Don\u2019t just play. Evolve. \u00b7 Updated {escape(UPDATED)}</p>'
            '<h2>Contact us</h2><p>Found a bug or have a question or idea? Open a support request and we\u2019ll reply there. '
            'Requests are public, so please don\u2019t include personal details or videos.</p>'
            f'<a class="btn" href="{ISSUES_URL}">Open a support request</a>{mail}'
            f'<h2>Common questions</h2>{faq}'
            '<h2>Privacy</h2><p>Read how Oche Lab handles your data: <a href="/privacy">Privacy policy</a>.</p>'
            '</body></html>')


@router.get('/support', response_class=HTMLResponse)
def support():
    return HTMLResponse(page(), headers={'Cache-Control': 'public, max-age=3600'})
