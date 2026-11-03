"""Paths shared by the figure scripts: read results/, write figures/out/. Needs numpy, matplotlib and pillow only."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; RES = ROOT / 'results'; OUT = ROOT / 'figures' / 'out'; OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT))
