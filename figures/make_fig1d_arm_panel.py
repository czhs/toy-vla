"""Panel (d) of the overview figure: the first observation of layout seed 950002 through the arm policy's own camera with its shadow lighting.
Writes results/arm/fig1_arm_panel.png, which figures/make_fig1.py composes. Runs in the arm environment (requirements/arm.txt), on CPU; MuJoCo needs a
rendering backend: MUJOCO_GL=egl (the default of arm/scene.py) on a headless Linux machine, MUJOCO_GL=cgl on macOS."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "arm"))
import eval_closed_loop as E, paths, scene as S
from PIL import Image
L = E.load_layouts(None, str(paths.ASSETS / "camera.json"), "shadow"); env = S.make_env(S.SceneSpec.from_dict(L["conditions"]["plain"]))
obs, _ = env.reset(seed=950002); Image.fromarray(obs["overhead_camera"]).save(paths.RESULTS / "fig1_arm_panel.png"); env.close(); print("wrote", paths.RESULTS / "fig1_arm_panel.png")
