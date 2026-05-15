# LangLoc annotation website

Small FastAPI app that crowdsources two kinds of human input for the
LangLoc evaluation:

1. **First-person descriptions** of ScanNet / 3RScan keyframes
   ("I'm standing next to the bookshelf with a blue sofa to my left…").
2. **First-person localizations** — given someone else's description,
   walk through the 3D mesh with WASD + mouse-look and stand where
   you think the description was written from. Compares against the
   GT camera pose; computes position error, angular error, and 3-D
   View IoU at the paper's per-dataset FoV.

Both feed the rebuttal §6 reviewer-WxoL closed-loop-bias and
reviewer-EEmB human-performance experiments. See
[`tools/baselines/human/README.md`](../baselines/human/README.md)
for how the collected descriptions get extracted back into the
pipeline.

## What it serves

| Route | Page | Purpose |
|---|---|---|
| `/` | landing | dataset chooser |
| `/datasets` | dataset chooser | pick ScanNet or 3RScan |
| `/annotate` | describe | one keyframe at a time + textarea |
| `/localize` | localize | Three.js viewer, WASD walk, submit pose |
| `/localize/scenes` | scene browser | per-scene progress; pick which scene to work on |
| `/history` | "my work" | mixed list of own descriptions + localizations, with edit links |
| `/done` | thanks | final-state landing when the pool runs out |
| `/admin/coverage` | admin | gated on `LANGLOC_ADMIN_TOKEN` |
| `/meshes/<dataset>/<scene>.ply` | static | mesh file (full-res when available, decimated fallback) |
| `/api/save`, `/api/submit`, `/api/edit` | api | description endpoints |
| `/api/localize/submit`, `/api/localize/skip` | api | localizer endpoints |

## Quick start (local dev)

```bash
cd tools/annotation_website
python -m pip install -r requirements.txt

cp .env.example .env
# edit .env to fill in LANGLOC_COOKIE_SECRET and LANGLOC_ADMIN_TOKEN
# (generate with: python -c "import secrets; print(secrets.token_urlsafe(48))")

# one-time pool prep per dataset (~2 min each)
python scripts/prepare_keyframes.py \
    --dataset scannet \
    --manifest ../../manifests/scannet_table4_first_100.txt \
    --data-root ../../data/scans \
    --out data/scenes_keyframes_scannet.json \
    --keyframes-dir static/keyframes/scannet

python scripts/compute_difficulty.py \
    --dataset scannet \
    --keyframes-json data/scenes_keyframes_scannet.json \
    --metrics-json ../../eval/new_data/eval_metrics_table4_scannet_parsed_NEW.json \
    --out data/scenes_scannet.json

# (repeat for --dataset 3rscan with the matching manifest + metrics)

# dev server — local only, no tunnel
python -m uvicorn server.main:app --host 127.0.0.1 --port 8000
```

Browse to <http://127.0.0.1:8000/>.

## Public deployment

```bash
./launch.sh                       # defaults to Cloudflare Tunnel
./launch.sh --tunnel ngrok        # opt-out: use ngrok instead
./launch.sh --no-tunnel           # local bind only
```

`launch.sh` reads `.env`, boots uvicorn on `0.0.0.0:8000` with
`--forwarded-allow-ips '*'`, opens the tunnel, and prints the public
URL. Stop with Ctrl-C; both processes are torn down.

Cloudflare Tunnel is the default because ngrok's free tier has a
~1 GB/month bandwidth cap, which we hit fast once we started serving
80–150 MB ScanNet meshes. Cloudflare's ephemeral tunnels are
unmetered. Install with:

```bash
mkdir -p ~/.local/bin && cd ~/.local/bin
curl -sSL -o cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x cloudflared
```

Logs: `logs/{uvicorn,cloudflared,ngrok}.log`.

## Data flow

- SQLite at `data/annotations.db` (gitignored). WAL mode + 5 s busy
  timeout. Schema: `annotators`, `scenes`, `keyframes`, `leases`,
  `frame_completions`, `descriptions`, `human_localizations`,
  `localization_skips`.
- Append-only JSONL at `data/annotations.jsonl` written on every
  submit / edit / skip — belt-and-braces against DB corruption.
- Mesh files served from `data/scans/<scene>/<scene>_vh_clean.ply`
  (full-res) with `_vh_clean_2.ply` (decimated) as fallback. ScanNet
  ships both variants; 3RScan ships just `labels.instances.annotated.v2.ply`.

## Identity & concurrency

- Cookie-based annotator UUID, set on first visit; signed with
  `LANGLOC_COOKIE_SECRET`.
- Per-(scene, frame) lease for descriptions, 20-minute TTL,
  auto-renewed on each save. `BEGIN IMMEDIATE` transactions; race
  windows handled by an IntegrityError retry loop.
- Per-frame global cap for localizations (one per frame, ever);
  skip recorded in `localization_skips`.

## Description assignment policy

3-phase, configured via `LANGLOC_REDUNDANCY`:

1. Continue mine — if I have in-progress completions in some scene,
   keep me there.
2. Close partial — help close a scene someone else started.
3. Open fresh — lowest-`difficulty_rank` scene I haven't touched.

The localization side uses a stricter policy: per-frame global cap,
no scene exclusion (annotators are free to do all 10 frames of a
scene if they want).

## Admin

`GET /admin/coverage`, `/admin/annotators`, `/admin/export.jsonl`
are gated on `LANGLOC_ADMIN_TOKEN`. Pass
`Authorization: Bearer <token>` to view.

## Persistence

For multi-week deployments, set up
[Litestream](https://litestream.io/) to replicate the SQLite WAL to
S3 / B2. See `litestream.yml.example`. The append-only JSONL is the
backup of last resort.

## Configuration

See `server/config.py` for every tunable (lease TTL, redundancy
target, JPEG quality, port, etc.). Environment variables override
the dataclass defaults — see `.env.example` for the canonical set.
