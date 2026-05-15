# GPT-5.5 vision describer

Calls OpenAI's GPT-5.5 vision model once per keyframe with the JPEG
image and a fixed prompt; saves a 2–4-sentence first-person
description in the same JSON schema as
`data/scans/<scene>/output/descriptions/<frame>.json`. Used in the
rebuttal §6 closed-loop-bias defense as a "describer that has zero
access to the GT scene-graph metadata".

## Quick start

```bash
# requires OPENAI_API_KEY in .env (see ../../../.env.example)
python tools/baselines/gpt_vlm/run_descriptions.py \
    --dataset scannet \
    --data-root data/scans \
    --manifest manifests/scannet_table4_first_100.txt \
    --out-root eval/gpt55_vlm/scannet \
    --num-scenes 100 \
    --concurrency 8 \
    --max-cost-usd 20 \
    --seed 42
```

`postrun.sh` runs the full chain (parse → localize → compute
difficulty) for both datasets — invoke once after both
`run_descriptions.py` runs finish.

## What the runner does

1. **Resume by default.** Per-frame JSONs that already exist and pass
   the validity check are skipped. Failed calls write a sibling
   `<frame>.error.json` sentinel that's retried only on
   `--retry-errors`.
2. **Cost cap.** Every call's input + output token count is priced
   in real-time and the run aborts cleanly when `--max-cost-usd` is
   hit (also via `LANGLOC_GPT_MAX_USD` env var).
3. **Async + concurrency.** Uses `openai.AsyncOpenAI` with a
   semaphore; default 10 concurrent in-flight requests.
4. **Backoff.** 5 retries per call with exponential delay +
   `Retry-After` header parsing + jitter.
5. **Run manifest.** Writes a JSON manifest at
   `<out_root>/run_manifest.json` before and after the run, capturing
   git SHA, prompt id, model id, seed, and final cost.

## Reproducing the rebuttal numbers

Full ScanNet 100 + 3RScan 97 (both datasets) is ~1915 calls, ~$31:

```bash
LANGLOC_GPT_MAX_USD=20 python tools/baselines/gpt_vlm/run_descriptions.py \
    --dataset scannet --data-root data/scans \
    --manifest manifests/scannet_table4_first_100.txt \
    --out-root eval/gpt55_vlm/scannet --concurrency 8 --seed 42
LANGLOC_GPT_MAX_USD=20 python tools/baselines/gpt_vlm/run_descriptions.py \
    --dataset 3rscan --data-root data/3RScan \
    --manifest manifests/3rscan_table4_subset_100.txt \
    --out-root eval/gpt55_vlm/3rscan --concurrency 8 --seed 42
bash tools/baselines/gpt_vlm/postrun.sh
```

Outputs:
- `eval/gpt55_vlm/{scannet,3rscan}/<scene>/output/descriptions/*.json` — per-frame raw descriptions
- `eval/gpt55_vlm/{scannet,3rscan}/<scene>/output/descriptions/*_parsed.json` — parser output
- `eval/gpt55_vlm_full_{scannet,3rscan}_metrics.json` — localizer per-frame metrics

## Prompt

The exact prompt used is the constant `PROMPT_VLM_FIRST_PERSON` in
[`run_descriptions.py`](run_descriptions.py). It is matched-stimulus
with the prompt the human annotation website shows (`I'm standing in…`),
modulo an explicit "mention distinctive objects" instruction that the
website conveys via examples.
