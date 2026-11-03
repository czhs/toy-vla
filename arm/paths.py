"""Where the arm pipeline reads and writes. TOYVLA_DATA (default <repo>/data) holds everything large; SO101_REPO is the clone of
aviadarn/so101-lerobot made by setup_external.sh (default $TOYVLA_DATA/so101-lerobot)."""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
ASSETS = HERE / "assets"
DATA = Path(os.environ.get("TOYVLA_DATA", REPO_ROOT / "data")).expanduser()
SO101_REPO = Path(os.environ.get("SO101_REPO", DATA / "so101-lerobot")).expanduser()
DATASET = DATA / "so101_redcube_angled_shadow"
FEATURE_CACHE = DATA / "feature_cache"
RUNS = DATA / "arm_runs"
POLICY = DATA / "checkpoints/arm_policy"
FOLD = DATA / "arm_fold"
EVAL = DATA / "arm_eval"
TEXTURES = DATA / "textures"
RESULTS = REPO_ROOT / "results/arm"
SMOLVLA_BASE_REVISION = "d9f33c94a60fb382c90dea2164c96845bd955e28"
