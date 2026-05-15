# Scene manifests

Text files (one scene id per line) that pin which scenes participate
in each paper table. Tracked in git so the reproducibility chain is
deterministic. Pair these with the Hydra config files in
[`../configs/manifests/`](../configs/manifests/), which list the
underlying dataset releases.

| File | Purpose | Used by |
|---|---|---|
| `scannet_table4_first_100.txt` | First 100 ScanNet validation scenes (sorted), used by paper **Tab. 4(b)** and Tab. 5(b). | [`scripts/localization/reproduce_table4.sh`](../scripts/localization/reproduce_table4.sh) (with `scannet`) |
| `3rscan_table4_subset_100.txt` | Paper subset of 100 3RScan scenes used for **Tab. 4(a)** and Tab. 5(a). Three of the 100 are skipped at runtime because of missing mesh/anchor data — that's why downstream metric JSONs typically have 97 rows. | [`scripts/localization/reproduce_table4.sh`](../scripts/localization/reproduce_table4.sh) (with `3rscan`) |
| `3rscan_table5_full.txt` | Full 1,319-scan 3RScan release used by **Tab. 5** (no-dialog only). | [`scripts/localization/reproduce_table5.sh`](../scripts/localization/reproduce_table5.sh) |
| `scannet_run2_10.txt` | 10 ScanNet scenes used as the "anchor" set for the human-annotation pilot (rebuttal §6). 6 of these had perfect 0-error position in the colleague's run-2 Qwen-A3-MAP evaluation. The annotation site promotes them to ranks 1–10 so newly-arriving annotators always see them first. | [`tools/annotation_website/scripts/promote_run2_first.py`](../tools/annotation_website/scripts/promote_run2_first.py) |

## Canonicality

- For paper Tab. 4 / Tab. 5 reproduction, the `_first_100` and
  `_subset_100` / `_full` files are authoritative. Don't add a new
  file with a similar name — extend or comment instead.
- ScanNet has more scenes available; the `_first_100` is just the
  alphabetically-first 100 of `configs/manifests/scannetv2_all.txt`.

## Why these are tracked but `configs/manifests/` already exists

`configs/manifests/` holds Hydra-style references to **dataset
releases** (e.g. `scannetv2_all.txt` lists every ScanNet scene that
was released). `manifests/` (this directory) holds the **subsets of
those releases that the paper actually evaluated**. Keeping them in
the repo root makes them easy to find and easy to override from a
shell script without going through Hydra config composition.
