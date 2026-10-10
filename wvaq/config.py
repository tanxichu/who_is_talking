"""All filesystem locations used by the pipeline.

Every path is taken from an environment variable so that nothing machine-specific
is hard-coded in the scripts. Set at least WVAQ_DATA_ROOT and WVAQ_WORK_DIR.

  WVAQ_DATA_ROOT        root of the Seamless Interaction download (the directory that
                        contains e.g. improvised/train/0000/0000/<file_id>.{mp4,wav,json})
  WVAQ_WORK_DIR         where all intermediate files and results are written
  WVAQ_SYNCNET_DIR      a clone of https://github.com/joonson/syncnet_python
                        (we import SyncNetModel.py from it)
  WVAQ_SYNCNET_WEIGHTS  syncnet_v2.model (default: $WVAQ_SYNCNET_DIR/data/syncnet_v2.model)
"""
import os

DATA_ROOT = os.environ.get("WVAQ_DATA_ROOT", "./data/seamless_interaction")
WORK_DIR = os.environ.get("WVAQ_WORK_DIR", "./work")
SYNCNET_DIR = os.environ.get("WVAQ_SYNCNET_DIR", "./third_party/syncnet_python")
SYNCNET_WEIGHTS = os.environ.get("WVAQ_SYNCNET_WEIGHTS", os.path.join(SYNCNET_DIR, "data", "syncnet_v2.model"))

RELEASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIP_LIST = os.path.join(RELEASE_DIR, "clips", "clips.json")       # the 147 clips used in the paper

CLIPS_DIR = os.path.join(WORK_DIR, "clips")            # <id>.mp4 (silent), <id>_s1.wav (left), <id>_s2.wav (right), <id>.wav (mix)
LANDMARK_DIR = os.path.join(WORK_DIR, "landmarks")     # <id>.npz
VARIANT_DIR = os.path.join(WORK_DIR, "variants")       # v/*.mp4, a/*.wav, manifest.json
SYNCNET_RESULTS = os.path.join(WORK_DIR, "syncnet", "results.jsonl")
DMAT_DIR = os.path.join(WORK_DIR, "syncnet", "dmat")   # per-item offset x time distance matrices
RESULTS_DIR = os.path.join(WORK_DIR, "results")


def clip_paths(cid, clips_dir=None):
    """Video and the two per-person audio tracks of one clip."""
    d = clips_dir or CLIPS_DIR
    return {"video": os.path.join(d, f"{cid}.mp4"), "s1": os.path.join(d, f"{cid}_s1.wav"), "s2": os.path.join(d, f"{cid}_s2.wav"), "mix": os.path.join(d, f"{cid}.wav")}
