# Who Is Talking? Auditing Lip-Sync Metrics for Two-Person Video by Fault Injection

Code for the fault-injection audit of lip-sync metrics on real two-person conversations and for
**SLT**, a combined speaker / listener / timing check.

The idea: take real conversations, where the answer to "who is talking, and is the mouth in sync?" is known
to be correct, inject one controlled fault at a time (wrong speaker, listener moving the mouth, speaker frozen,
audio delayed, ...) and measure whether a metric notices. A metric that does not separate a faulted clip from
its original cannot be trusted to judge two-person generated video.

## What is in this repository

```
wvaq/
  config.py             all paths (environment variables, no hard-coded locations)
  prepare_clips.py      step 1  cut side-by-side two-person clips from Seamless Interaction
  extract_landmarks.py  step 2  per-frame mouth landmarks of both people (insightface + MediaPipe FaceMesh)
  faults.py             fault definitions and landmark metrics (library, shared by steps 3 and 4)
  landmark_metrics.py   step 3  Bind / Lis / Spk (and reference metrics) for every fault
  make_variants.py      step 4  render faulted videos / audio for the image-based metrics
  syncnet_score.py      SyncNet scoring (official geometry and scoring, insightface detection)
  syncnet_batch.py      step 5  per-person SyncNet + distance matrices, conventional single-track SyncNet
  sync_cstar.py         step 6  Sync-C* (per person, own speaking periods only)
  slt.py                step 7  SLT parts, common-quantile lines (combined false alarm <= 5 %), detection
  auc_faults.py         step 8  threshold-free AUC per fault with conversation bootstrap
clips/
  clips.json, clips.csv the 147 clips (27 conversations): Seamless file ids, start / end time, crop, conversation
  syncnet100.txt        the 100 clips of the SyncNet set (SLT and AUC)
  synccstar138.txt      the 138 original clips of the paper's standalone Sync-C* table (see notes)
```

No video, audio, landmark data, model weights or results are distributed.

## Data (download yourself)

**Seamless Interaction** (Meta), licence **CC BY-NC 4.0** (non-commercial):
<https://github.com/facebookresearch/seamless_interaction> / <https://huggingface.co/datasets/facebook/seamless-interaction>.
Each participant of a conversation is a file id `V<vendor>_S<session>_I<interaction>_P<participant>` with
`.mp4` (own camera), `.wav` (own mic) and `.json` (`metadata:vad`, `metadata:transcript`).
We used the `improvised` and `naturalistic` `train` split; `clips/clips.csv` lists, per clip, the relative
path of both participants' files (`left_source`, `right_source`, e.g. `improvised/train/0000/0000/V00_S0059_I00000479_P0089`).
Download those participants (or the whole split) and point `WVAQ_DATA_ROOT` at the directory that contains
`improvised/` and `naturalistic/`.

Clip ids encode their origin: `sbs_<conversation>_<t0 with . -> p>`, e.g. `sbs_V00_S0059_I00000479_0p5` is
conversation `V00_S0059_I00000479` starting at 0.5 s. The id rounds t0 to 0.1 s; the exact start time is in
`clips.csv` (`t0_s`; each clip lasts 49/12 = 4.083 s, `t_end_s`). The left person is `left_participant`
(audio track s1), the right person is `right_participant` (s2).

## Third-party models (obtain yourself)

| model | used for | where | licence |
|---|---|---|---|
| SyncNet (`syncnet_v2.model`) and `SyncNetModel.py` | SyncNet, Sync-C* | clone <https://github.com/joonson/syncnet_python> and run its `download_model.sh` | code MIT; check the weights' terms on the project page |
| insightface `buffalo_l` (detection only) | face detection | downloaded automatically by `insightface` on first use | insightface code MIT; the pretrained models are for non-commercial research use only |
| MediaPipe FaceMesh (legacy `mp.solutions.face_mesh`, refined landmarks) | mouth landmarks | bundled with the `mediapipe` pip package | Apache-2.0 |

## Install

Python 3.12 and ffmpeg / ffprobe on `PATH`.

```bash
python -m venv venv && . venv/bin/activate
pip install -r requirements.txt
git clone https://github.com/joonson/syncnet_python third_party/syncnet_python
(cd third_party/syncnet_python && sh download_model.sh)

export WVAQ_DATA_ROOT=/path/to/seamless_interaction      # contains improvised/ and naturalistic/
export WVAQ_WORK_DIR=/path/to/work                        # all outputs go here
export WVAQ_SYNCNET_DIR=$PWD/third_party/syncnet_python  # WVAQ_SYNCNET_WEIGHTS defaults to <dir>/data/syncnet_v2.model
```

Everything runs on CPU. Run the commands from the repository root (`python -m wvaq.<step>`).

## Pipeline

```bash
# 1. clips: recreate the 147 listed clips (704x1280, 12 fps, 49 frames; per-person 16 kHz tracks)
python -m wvaq.prepare_clips
#    (alternative: search new windows on whatever is downloaded: python -m wvaq.prepare_clips --search --max-per-conv 12)

# 2. landmarks of both people, 49 frames per clip
python -m wvaq.extract_landmarks

# 3. landmark metrics for every fault (Bind / Lis / Spk ...)        -> results/landmark_metrics.json, _summary.json
python -m wvaq.landmark_metrics

# 4. faulted variants for SyncNet, 100 clips round-robin over conversations  -> variants/manifest.json
python -m wvaq.make_variants --n 100

# 5. SyncNet (per person with distance matrices; conventional mixed track)  -> syncnet/results.jsonl, syncnet/dmat/
python -m wvaq.syncnet_batch --workers 2
python -m wvaq.syncnet_score --selftest        # optional: official example clip + two negative controls

# 6. Sync-C* detection table                                         -> results/sync_cstar.json
python -m wvaq.sync_cstar

# 7. SLT: calibrate the three lines (combined false alarm <= 5 %) and report      -> results/slt.json
python -m wvaq.slt
#    apply existing lines to every item of a manifest                          -> results/slt_decisions.json
python -m wvaq.slt --apply-lines $WVAQ_WORK_DIR/results/slt.json

# 8. AUC per fault (design faults and held-out fault types), plus conventional SyncNet  -> results/auc_faults.json
python -m wvaq.auc_faults
```

`--n 100` reproduces the paper's SyncNet set (identical to `--clip-list clips/syncnet100.txt`).

Step 3 is independent of steps 4 to 6; steps 7 and 8 need 3, 5 and 6 (step 6's table is not needed; they
recompute Sync-C* from the distance matrices). Step 5 is the slow one (about 1,600 items); it appends to the
`.jsonl` file, so it can be stopped and restarted, and `--slice i/n` splits it across processes or machines.

## Faults

A fault name is `<kind>@<p>`. Kinds ending in `_t` work on whole speaking turns: speech activity of each
track (energy > 0.25 x the 95th percentile of both tracks pooled, 12 fps) is closed over word gaps of up to
6 frames (0.5 s), and replacement frames come from the longest contiguous run, played back and forth.
`_c` is the same without closing gaps; no suffix joins non-adjacent source frames (an early version, kept
for the landmark tables). Video faults re-order the frames of one person's half only; the audio is untouched
unless stated.

| fault | definition |
|---|---|
| `none` | unchanged clip (control; should not alarm) |
| `swap` | the two per-person audio tracks are exchanged (the mixed track is unchanged) |
| `wrong_spk_t@1.0` | the person who speaks keeps still and the person who listens moves the mouth (both edits below at once) |
| `lis_talk_t@p` | during the other person's speech, the listener's frames are replaced by his own speaking frames (first fraction p) |
| `spk_mute_t@1.0` | during his own speech, the speaker's frames are replaced by his own quiet frames |
| `lis_quiet_t@1.0` | cut-only control: the listener's frames are replaced by another quiet stretch of himself |
| `shift@k` | both audio tracks delayed by k frames (12 fps: 2 = 167 ms, 8 = 667 ms) |
| `vlead@k` | both video halves advanced by k frames (= audio delayed by k frames relative to video) |
| `rev_spk_t@1.0` | each speaker's frames are reversed inside each of his turns (held-out type) |
| `frz_lis_t@1.0` | the listener is frozen on the first frame of each of the other person's turns (held-out type) |
| `one_lag_t@8` | only the left person's video is delayed by 8 frames (held-out type) |
| `voice_drop_t@1.0` | the left person's own track is muted during each of his turns, video unchanged (held-out type) |
| `rot_spk_t@1.0` | inside each of the left person's turns, his frames are circularly shifted by half a turn (held-out type) |

`landmark_metrics.py` additionally scores `lis_talk@{0.33,0.66,1.0}`, `spk_mute@{0.33,0.66,1.0}`, `shift@4`,
`wrong_spk@1.0` and the `_c` versions.

## Metrics

* **Spk**: mouth activity (|frame-to-frame change of the inner-lip / mouth-width ratio|, 3-frame smoothing) of a
  person during his own solo speech; **Lis**: the listener's mouth activity during the other's solo speech;
  **Bind** = Lis / Spk. Reference columns: LisOpen, E2b, SLS, E2c_h, negSpk.
* **Sync-C\*** (AnyTalker): per-person SyncNet confidence restricted to that person's speaking periods, weighted by
  period length; left face x s1, right face x s2.
* **SLT**: S = -Sync-C\*, L = Bind, T = mean per-person |SyncNet offset|. Alarm if any part exceeds its line; the
  three lines are the same quantile q of the original clips, found by bisection (40 steps in [90, 100]) so that the
  combined false-alarm rate on the original clips is <= 5 %.
* Confidence intervals: bootstrap over conversations (clips of one conversation are resampled together),
  2000 resamples, `numpy.random.default_rng(0)`.

## Expected outputs

* `results/landmark_metrics.json`: per clip, `orig` and one entry per fault with all landmark metrics;
  `_summary.json`: paired and absolute (P95 line) detection rates with intervals.
* `syncnet/results.jsonl`: per item `per_left`, `per_right` (offset, conf, min_dist, n_frames), `std_L`, `std_R`,
  `face_w`; `syncnet/dmat/<key>.npz`: distance matrices `L`, `R` (offsets x 25 fps time steps).
* `results/sync_cstar.json`: alarm lines (`_thr`), detection rates (`real`) and paired comparison (`paired`).
* `results/slt.json`: `q` (about 99.0 in the paper), `lines` [S, L, T], combined detection per fault, each part
  alone at P95, lowest combined detection over the full-strength faults, held-out-conversation check.
* `results/auc_faults.json`: per fault, AUC [point, 2.5 %, 97.5 %, n] for SLT, S, L, T and conventional SyncNet (MIX).

Re-running steps 3 and 6 to 8 on the same landmarks and SyncNet outputs reproduces the numbers exactly. Steps 1,
2 and 5 depend on ffmpeg, onnxruntime, MediaPipe and PyTorch builds, so their outputs can differ slightly across
versions.

## Notes on reproducing the paper's tables

* SLT (`slt.json`) and the AUC table (`auc_faults.json`) use the 100-clip SyncNet set.
* The standalone Sync-C* detection table in the paper used a superset of originals: the 138 clips in
  `clips/synccstar138.txt`, with faults rendered on 100 to 108 of them. Running step 6 on the default 100-clip
  manifest applies the same protocol to the 100-clip set; for the 138-clip version, render variants with
  `--clip-list clips/synccstar138.txt`.
* The landmark tables (`landmark_metrics.py`) use all 147 clips.

## Licence

Code: MIT (see `LICENSE`). The clip list in `clips/` is derived from Seamless Interaction and inherits its
CC BY-NC 4.0 licence; the data itself is not included.
