#!/usr/bin/env python3
"""SyncNet lip-sync scoring (Sync-C confidence / Sync-D minimum distance / audio-video offset).

Follows the official syncnet_python (MIT, joonson):
  - video at 25 fps, audio at 16 kHz (the network was trained at these rates)
  - face crop geometry as in run_pipeline.crop_video: box size = max(w,h)/2, centre and size median-filtered
    (window 13), crop_scale = 0.4, padding value 110, resized to 224
  - scoring as in SyncNetInstance.evaluate: 5-frame lip windows x 20-frame MFCC windows, offsets searched in
    +-vshift, conf = median(dist) - min(dist)
Differences from the official code: faces are detected with insightface (buffalo_l) instead of S3FD, and the
crops are not re-encoded with XVID. Values are therefore not bit-identical to the official tool; --selftest runs
the official example clip plus two negative controls.

Usage:
  python -m wvaq.syncnet_score --video clip.mp4 [--audio speech.wav]
  python -m wvaq.syncnet_score --selftest
"""
import argparse, os, subprocess, sys, tempfile
import cv2, numpy as np, torch
from scipy import signal
import python_speech_features
from wvaq import config

SN = config.SYNCNET_DIR
sys.path.insert(0, SN)
from SyncNetModel import S

_M = {}
DEV = os.environ.get("SN_DEVICE", "cpu")   # SN_DEVICE=cuda puts the network on the GPU (face detection stays on CPU)


def model():
    if "m" not in _M:
        m = S(num_layers_in_fc_layers=1024)
        sd = torch.load(config.SYNCNET_WEIGHTS, map_location="cpu")
        own = m.state_dict()
        # the official weights were saved by an old PyTorch without BatchNorm num_batches_tracked (the official
        # loadParameters also only copies the keys present in the file) - only check that real parameters are present
        miss = [k for k in own if k not in sd and not k.endswith("num_batches_tracked")]
        assert not miss, f"SyncNet weights miss keys {miss[:3]} (incomplete file? expected 54,573,114 bytes)"
        m.load_state_dict({k: sd[k] for k in own if k in sd}, strict=False); m.eval(); _M["m"] = m.to(DEV)
    return _M["m"]


_F = {}


def detector():
    if "f" not in _F:
        from insightface.app import FaceAnalysis
        if os.environ.get("ORT_THREADS"):   # fixed onnxruntime thread count (useful in CPU-limited containers); unchanged when unset
            import onnxruntime as ort; _IS = ort.InferenceSession
            def _is(*a, **k):
                so = k.get("sess_options") or ort.SessionOptions(); so.intra_op_num_threads = so.inter_op_num_threads = int(os.environ["ORT_THREADS"]); k["sess_options"] = so; return _IS(*a, **k)
            ort.InferenceSession = _is
        fa = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"], providers=["CPUExecutionProvider"])
        fa.prepare(ctx_id=-1, det_size=(640, 640)); _F["f"] = fa
    return _F["f"]


def read_25fps(video, t0=None, dur=None):
    """Video (any frame rate) -> 25 fps frames + 16 kHz mono audio."""
    with tempfile.TemporaryDirectory() as td:
        ss = ["-ss", f"{t0:.3f}"] if t0 is not None else []
        tt = ["-t", f"{dur:.3f}"] if dur is not None else []
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", *ss, "-i", video, *tt, "-vf", "fps=25", "-an",
                        os.path.join(td, "%06d.png")], check=True)
        r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", *ss, "-i", video, *tt, "-vn", "-ac", "1", "-ar", "16000",
                            "-f", "s16le", "-"], capture_output=True)
        fr = [cv2.imread(os.path.join(td, f)) for f in sorted(os.listdir(td))]
    return fr, np.frombuffer(r.stdout, np.int16)


def crops(frames, crop_scale=0.4, pick="largest"):
    """As run_pipeline.crop_video. Frames without a face reuse the nearest valid box; returns None if more than 30 %
    of the frames have no face. pick: largest (single person) / left / right (two people: the two largest faces,
    split by x centre)."""
    fa = detector(); boxes = []
    for f in frames:
        fs = sorted(fa.get(f), key=lambda z: -(z.bbox[2] - z.bbox[0]) * (z.bbox[3] - z.bbox[1]))
        if pick in ("left", "right"):
            two = sorted(fs[:2], key=lambda z: (z.bbox[0] + z.bbox[2]) / 2)
            boxes.append(None if len(two) < 2 else two[0 if pick == "left" else 1].bbox)
        else:
            boxes.append(None if not fs else fs[0].bbox)
    ok = [i for i, b in enumerate(boxes) if b is not None]
    if len(ok) < 0.7 * len(frames): return None
    boxes = [b if b is not None else boxes[min(ok, key=lambda j: abs(j - i))] for i, b in enumerate(boxes)]
    s = signal.medfilt([max(b[3] - b[1], b[2] - b[0]) / 2 for b in boxes], kernel_size=13)
    y = signal.medfilt([(b[1] + b[3]) / 2 for b in boxes], kernel_size=13)
    x = signal.medfilt([(b[0] + b[2]) / 2 for b in boxes], kernel_size=13)
    out = []
    for i, f in enumerate(frames):
        bs = s[i]; bsi = int(bs * (1 + 2 * crop_scale))
        fp = np.pad(f, ((bsi, bsi), (bsi, bsi), (0, 0)), "constant", constant_values=(110, 110))
        my, mx = y[i] + bsi, x[i] + bsi
        face = fp[int(my - bs):int(my + bs * (1 + 2 * crop_scale)), int(mx - bs * (1 + crop_scale)):int(mx + bs * (1 + crop_scale))]
        out.append(cv2.resize(face, (224, 224)))
    return out


def calc_pdist(feat1, feat2, vshift=15):
    win = vshift * 2 + 1
    f2p = torch.nn.functional.pad(feat2, (0, 0, vshift, vshift))
    return [torch.nn.functional.pairwise_distance(feat1[[i], :].repeat(win, 1), f2p[i:i + win, :]) for i in range(len(feat1))]


@torch.no_grad()
def score(frames25, audio16k, vshift=15):
    """dict(offset in frames, conf=Sync-C, min_dist=Sync-D, n_frames) or None (not enough faces)."""
    return score_crops(crops(frames25), audio16k, vshift)


@torch.no_grad()
def score_crops(cr, audio16k, vshift=15, keep=False):
    """Same as score, on already cropped faces (one detection pass can be scored against several audio tracks).
    keep=True also returns the offset x time distance matrix dmat (n_offsets x n_times, float16), used to recompute
    time-restricted variants such as Sync-C* (speaking periods only)."""
    if cr is None: return None
    im = torch.from_numpy(np.stack(cr, axis=3)[None].transpose(0, 3, 4, 1, 2).astype(np.float32))   # (1,3,T,224,224) BGR 0..255, as the official code
    mf = np.stack([np.array(i) for i in zip(*python_speech_features.mfcc(audio16k, 16000))])
    cc = torch.from_numpy(mf[None, None].astype(np.float32))
    n = min(len(cr), len(audio16k) // 640) - 5
    if n < 2 * vshift + 1: vshift = max(1, n // 3)
    m = model(); im_f, cc_f = [], []
    for i in range(0, n, 20):
        idx = range(i, min(n, i + 20))
        im_f.append(m.forward_lip(torch.cat([im[:, :, v:v + 5] for v in idx], 0).to(DEV)).cpu())
        cc_f.append(m.forward_aud(torch.cat([cc[:, :, :, v * 4:v * 4 + 20] for v in idx], 0).to(DEV)).cpu())
    full = torch.stack(calc_pdist(torch.cat(im_f), torch.cat(cc_f), vshift), 1); d = full.mean(1)
    mn, mi = torch.min(d, 0)
    out = dict(offset=int(vshift - mi), conf=float(torch.median(d) - mn), min_dist=float(mn), n_frames=len(cr))
    if keep: out["dmat"] = full.numpy().astype(np.float16); out["vshift"] = vshift
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video"); ap.add_argument("--audio"); ap.add_argument("--t0", type=float); ap.add_argument("--dur", type=float)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        ex = os.path.join(SN, "data", "example.avi")
        if not os.path.exists(ex):
            subprocess.run(["curl", "-sL", "-m", "120", "-o", ex, "http://www.robots.ox.ac.uk/~vgg/software/lipsync/data/example.avi"], check=True)
        fr, au = read_25fps(ex)
        r = score(fr, au); print(f"official example (official README: offset 3, conf ~10.0): {r}")
        r2 = score(fr, np.roll(au, 16000)); print(f"negative control 1, audio shifted by 1 s: {r2}")
        r3 = score(fr, np.zeros_like(au) + (np.random.RandomState(0).randn(len(au)) * 300).astype(np.int16)); print(f"negative control 2, white noise: {r3}")
        sys.exit(0)
    fr, au = read_25fps(a.video, a.t0, a.dur)
    if a.audio:
        r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", a.audio, "-ac", "1", "-ar", "16000", "-f", "s16le", "-"], capture_output=True)
        au = np.frombuffer(r.stdout, np.int16)
    print(score(fr, au))
