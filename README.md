# Oche Lab analysis API

FastAPI + MediaPipe Pose service behind the Oche Lab darts coaching app's video coach review.
It finds throws in a clip of up to 60 seconds, measures the throwing arm, returns coaching tips and
annotated key frames, and can render a skeleton-overlay MP4.

**This repo is auto-synced from the private Oche Lab app repo. Don't edit it directly**; changes
here are overwritten on the next sync.

## Run

```bash
docker build -t oche-lab-api .
docker run -p 7860:7860 -e LOW_MEM=1 oche-lab-api
curl localhost:7860/health
```

Hosted on Render (free web service, Docker). The container listens on `$PORT` (default 7860).
`LOW_MEM=1` uses the lite pose model on smaller frames for small hosts (512 MB RAM).

## Endpoints

- `GET /health`
- `POST /analyse` (multipart: `video`, `arm`=right|left, `overlay`=true|false, `mode`=sync|job)
- `GET /jobs/{id}`: poll a `mode=job` analysis (includes `queue_position` while queued)
- `GET /analysis/{id}/frames/{name}` and `/analysis/{id}/overlay.mp4`: generated media

`POST /analyse` and `GET /jobs/{id}` expect an `X-Install-Id` header (a per-install random id from the app).
If the server is configured with `OCHELAB_APP_KEY`, they also expect `X-Oche-App-Key`. Older app versions
are accepted during a short rollout window (`OCHELAB_ALLOW_LEGACY=1`, the default). Analyses are rate
limited per install and per IP. If the process restarts mid-job, `GET /jobs/{id}` returns HTTP 410 so
the app can resubmit automatically.

Uploaded videos are deleted after analysis; generated results are cleaned up automatically and kept
within a disk budget on small hosts.
