"""Controllable SO-101 red-cube pick-and-place scene (MuJoCo, so101-nexus 0.7), matched to the toy task.

Base: aviadarn/so101-lerobot (so101-nexus ``PickAndPlaceEnv``, its scripted DLS-IK expert, LeRobot
so101_follower units). The training scene is changed to match the toy pivot task as closely as possible
(defaults in ``assets/scene_defaults.json``):

- plain untextured table RGB (214,202,178); pure matte red cube RGB (220,30,30), 25 mm, yaw 0;
- the whole arm (gripper included) dark grey RGB (60,60,60), no specular;
- a FIXED place target: a flat low-saliency square "tray" (9 x 9 cm, RGB 194,182,158) at (0.22, 0);
  success = cube resting, released, within 4 cm of the tray centre;
- flat even lighting by default (ambient 0.8 + straight-down diffuse 0.2, no shadows, no specular); the policy of
  the paper is trained and evaluated with the ``shadow`` preset and the camera in ``assets/camera.json``;
- one fixed overhead camera at 512 x 512 (default top-down, fovy 30, 1.111 m up), cube ~22 px wide;
- cube uniform (by area) over the reachable sector rho 0.14-0.31 m, bearing -60..60 deg from the base,
  outside the tray; arm start pose random (rejection-sampled joint box, TCP 9-30 cm up), independent
  of the cube.

Perturbations (all SceneSpec fields): table (plain, dark, wood, clutter, checker, and the toy's
structured-table elements ``struct`` / ``struct:checker+tape`` ... painted at the toy's pixel positions
and sizes), distractors (visual-only cubes / discs of chosen colours), lighting presets and jitter,
overhead / wrist camera jitter, random cube yaw, tray kind / colour / position.

Layout determinism: ``env.reset(seed=s)`` draws the arm start pose and the cube position from the env's
own RNG. Everything else this module adds per episode (distractors, jitter, lights) comes from a
separate generator seeded with ``[s, 7771]``: a seed gives the same arm + cube layout in every condition."""

from __future__ import annotations

import atexit
import colorsys
import dataclasses
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402

import so101_nexus.mujoco  # noqa: F401,E402
import so101_nexus.mujoco.pick_and_place as _pnp  # noqa: E402
from so101_nexus.config import PickAndPlaceConfig  # noqa: E402
from so101_nexus.observations import JointPositions, OverheadCamera, WristCamera  # noqa: E402

HERE = Path(__file__).resolve().parent
import paths  # noqa: E402
REPO = paths.SO101_REPO
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
TEX_DIR = paths.TEXTURES
DEF = json.loads((HERE / "assets/scene_defaults.json").read_text())

TASK = DEF["task"]
SIM_HZ = DEF["sim_hz"]
RECORD_EVERY = DEF["record_every"]
FPS = SIM_HZ // RECORD_EVERY           # dataset / policy rate (10 Hz)
EXTRA_SEED = 7771
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def _c(rgb) -> tuple:
    return tuple(float(x) / 255.0 for x in rgb)


# --------------------------------------------------------------------------- colours


def _hsv(h_deg, s, v):
    return colorsys.hsv_to_rgb((h_deg % 360.0) / 360.0, s, v)


def sample_color(kind, rng: np.random.Generator) -> tuple[float, float, float]:
    """RGB in [0,1]. nonred = the toy's rand_nonred (20 % greys 20-235, else hue 40-320, s 0.5-1,
    v 0.45-1); nonredblue: hue 40-180 or 270-320; bluish: hue 200-250; reddish: hue +-(5..25) deg,
    s 0.7-1, v 0.6-1; red = the cube's (220,30,30); 'toyblue' = the toy distractor blue (30,70,220);
    nexus colour names; or an explicit 0-255 RGB triple."""
    if isinstance(kind, (tuple, list)):
        return _c(kind[:3])
    if kind in ("nonred", "nonredblue"):
        if rng.random() < 0.2:
            g = rng.uniform(20, 235) / 255.0
            return (g, g, g)
        if kind == "nonred":
            h = rng.uniform(40, 320)
        else:
            h = rng.uniform(40, 180) if rng.random() < 140 / 190 else rng.uniform(270, 320)
        return _hsv(h, rng.uniform(0.5, 1.0), rng.uniform(0.45, 1.0))
    if kind == "bluish":
        return _hsv(rng.uniform(200, 250), rng.uniform(0.6, 1.0), rng.uniform(0.5, 1.0))
    if kind == "reddish":
        h = rng.uniform(5, 25) * (1 if rng.random() < 0.5 else -1)
        return _hsv(h, rng.uniform(0.7, 1.0), rng.uniform(0.6, 1.0))
    if kind == "red":
        return _c(DEF["cube_rgb"])
    if kind == "toyblue":
        return _c((30, 70, 220))
    from so101_nexus.constants import COLOR_MAP

    if kind in COLOR_MAP:
        return tuple(COLOR_MAP[kind][:3])
    raise ValueError(f"unknown colour spec {kind!r}")


# --------------------------------------------------------------------------- spec


@dataclass
class Distractor:
    """``n`` visual-only objects (no collisions). near: None (anywhere in the cube workspace, clear of
    cube and tray), "tray" or "cube" (centre at a distance drawn from near_range metres)."""
    kind: str = "cube"                  # cube | disc
    color: object = "nonred"
    n: int = 1
    near: str | None = None
    near_range: tuple = (0.07, 0.10)
    size: float | None = None           # cube half-size or disc radius (m); default cube 0.0125, disc 0.03


def _d(k):
    v = DEF[k]
    return tuple(v) if isinstance(v, list) else v


@dataclass
class SceneSpec:
    table: str = _d("table")            # plain | dark | wood | clutter | checker | struct | struct:<part+part>
    table_rgb: tuple = _d("table_rgb")
    texture_seed: int = 0
    cube_rgb: tuple = _d("cube_rgb")
    cube_mm: float = _d("cube_mm")
    cube_yaw_deg: float = _d("cube_yaw_deg")
    cube_yaw_random: bool = _d("cube_yaw_random")
    arm_rgb: tuple = _d("arm_rgb")
    tray_kind: str = _d("tray_kind")    # square | disc
    tray_xy: tuple = _d("tray_xy")
    tray_half: float = _d("tray_half")  # half side (square) or radius (disc)
    tray_rgb: tuple = _d("tray_rgb")
    goal_thresh: float = _d("goal_thresh")
    ws_rho: tuple = _d("ws_rho")
    ws_bearing_deg: tuple = _d("ws_bearing_deg")
    ws_tray_clear: float = _d("ws_tray_clear")
    random_start: bool = _d("random_start")
    distractors: list = field(default_factory=list)
    lighting: str = _d("lighting")      # flat | shadow | shaded | dim | bright | warm | cool | side
    light_jitter: float = 0.0           # intensity x U(1-j,1+j), key direction +-30j deg
    overhead_jitter: float = 0.0        # 1.0 = position +-1 cm (xy) / +-3 % height, tilt +-1.5 deg, yaw +-2 deg
    wrist_jitter: float = 0.0           # 1.0 = pitch +-2 deg, fov +-3 deg, position +-3 mm
    camera_pos: tuple = _d("camera_pos")       # top-down camera; also the anchor of the table textures (world_to_pixel)
    camera_fovy: float = _d("camera_fovy")
    overhead_cam: dict | None = None           # override of the policy camera: {"pos", "quat" [w x y z], "fovy"} (e.g.
                                               # assets/camera.json["camera"]); textures stay world-anchored
    ov_w: int = _d("ov_w")
    ov_h: int = _d("ov_h")
    wr_w: int = _d("wr_w")
    wr_h: int = _d("wr_h")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "SceneSpec":
        d = dict(d)
        d["distractors"] = [Distractor(**x) if isinstance(x, dict) else x for x in d.get("distractors", [])]
        for k, v in list(d.items()):
            if isinstance(v, list) and k != "distractors":
                d[k] = tuple(v)
        return SceneSpec(**d)


# --------------------------------------------------------------------------- camera geometry


def camera_scale(spec: SceneSpec, z: float = 0.0) -> float:
    """Image pixels per metre for a horizontal surface at height z (top-down pinhole)."""
    f = (spec.ov_h / 2) / np.tan(np.radians(spec.camera_fovy / 2))
    return f / (spec.camera_pos[2] - z)


def world_to_pixel(spec: SceneSpec, x, y, z=0.0):
    """(u, v) image coordinates (u right, v down, continuous; pixel i covers [i, i+1)) of a world point
    for the UNJITTERED overhead camera. Image up = world +x, image right = world -y."""
    s = camera_scale(spec, z)
    cx, cy = spec.camera_pos[:2]
    return spec.ov_w / 2 - s * (np.asarray(y) - cy), spec.ov_h / 2 - s * (np.asarray(x) - cx)


def pixel_to_world(spec: SceneSpec, u, v, z=0.0):
    s = camera_scale(spec, z)
    cx, cy = spec.camera_pos[:2]
    return cx - (np.asarray(v) - spec.ov_h / 2) / s, cy - (np.asarray(u) - spec.ov_w / 2) / s


# --------------------------------------------------------------------------- table textures
# Textured tables are a visual-only overlay plane covering 1.5x the camera footprint, with the texture
# defined in IMAGE-PIXEL coordinates of the unjittered camera (2 texels per image pixel).

OV_PAD = 0.25            # overlay extends 25 % of the image beyond each edge
OV_TEX = 1536            # texels per side

# The toy's structured table (scene_pivot.py / bg_common.py), in toy-image pixels.
TOY = dict(base_uv=(256.0, 288.0), R=176.0, tile=48.0, tile_off=(7.0, 13.0), tape_y=100.0, tape_w=5.0,
           green_r=26.0, yellow=(120.0, -75.0, 200.0, -35.0),
           col=dict(table=(214, 202, 178), tile=(158, 142, 116), tape=(248, 248, 248), green=(60, 150, 80),
                    yellow=(225, 195, 50)))
TOY_PARTS = ("checker", "tape", "green", "yellow")


def _texel_uv(spec: SceneSpec):
    n = OV_TEX
    k = spec.ov_w * (1 + 2 * OV_PAD) / n
    g = -OV_PAD * spec.ov_w + (np.arange(n) + 0.5) * k
    return np.meshgrid(g, g)   # U (cols), V (rows)


def _toy_struct(spec: SceneSpec, parts) -> np.ndarray:
    U, V = _texel_uv(spec)
    x, y = U - TOY["base_uv"][0], TOY["base_uv"][1] - V          # toy world (pixel units)
    col = {k: np.array(v, float) for k, v in TOY["col"].items()}
    im = np.empty(U.shape + (3,))
    im[:] = np.array(spec.table_rgb, float)
    if "checker" in parts:
        t = ((np.floor((x - TOY["tile_off"][0]) / TOY["tile"]) + np.floor((y - TOY["tile_off"][1]) / TOY["tile"])) % 2).astype(bool)
        im[t] = col["tile"]
    if "tape" in parts:
        im[np.abs(y - TOY["tape_y"]) <= TOY["tape_w"] / 2] = col["tape"]
    if "green" in parts:
        gx, gy = TOY["R"] * np.cos(np.radians(-135)), TOY["R"] * np.sin(np.radians(-135))
        im[np.hypot(x - gx, y - gy) <= TOY["green_r"]] = col["green"]
    if "yellow" in parts:
        x0, y0, x1, y1 = TOY["yellow"]
        im[(x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)] = col["yellow"]
    return np.clip(np.round(im), 0, 255).astype(np.uint8)


def _world_grid(spec: SceneSpec):
    U, V = _texel_uv(spec)
    return pixel_to_world(spec, U, V)   # X, Y in metres


def _wood(spec: SceneSpec, seed: int) -> np.ndarray:
    rng = np.random.default_rng([seed, 11])
    X, Y = _world_grid(spec)
    y, x = Y + 1.0, X + 1.0
    plank = np.floor(y / 0.1).astype(int)
    shift = rng.uniform(0, 1, 64)[plank % 64]
    tone = rng.uniform(0.85, 1.12, 64)[plank % 64]
    warp = np.zeros_like(x)
    for k in range(1, 5):
        warp += rng.normal(0, 0.0025 / k) * np.sin(2 * np.pi * (k * 3 * x + rng.uniform())) \
            + rng.normal(0, 0.0012 / k) * np.sin(2 * np.pi * (k * 5 * (x + shift)))
    grain = 0.5 + 0.5 * np.sin(2 * np.pi * ((y + warp) * 90 + 0.6 * np.sin(2 * np.pi * (x * 1.5 + shift))))
    base, dark = np.array([0.56, 0.38, 0.22]), np.array([0.40, 0.25, 0.13])
    g = (grain ** 3)[..., None]
    img = (base * (1 - g) + dark * g) * tone[..., None] + rng.normal(0, 0.03, x.shape)[..., None]
    img = np.where((np.abs(y / 0.1 - np.round(y / 0.1)) < 0.012)[..., None], img * 0.55, img)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def _clutter(spec: SceneSpec, seed: int) -> np.ndarray:
    """Table colour printed with random non-red, non-blue flat shapes (1-6 cm)."""
    rng = np.random.default_rng([seed, 12])
    X, Y = _world_grid(spec)
    img = np.empty(X.shape + (3,))
    img[:] = _c(spec.table_rgb)
    x0, x1, y0, y1 = X.min(), X.max(), Y.min(), Y.max()
    area = (x1 - x0) * (y1 - y0)
    for _ in range(int(140 * area)):
        c = np.array(sample_color("nonredblue", rng))
        cx, cy = rng.uniform(x0, x1), rng.uniform(y0, y1)
        r = rng.random()
        if r < 0.5:
            w, h = rng.uniform(0.01, 0.06, 2)
            m = (np.abs(X - cx) < w / 2) & (np.abs(Y - cy) < h / 2)
        elif r < 0.85:
            rr = rng.uniform(0.006, 0.03)
            m = (X - cx) ** 2 + (Y - cy) ** 2 < rr * rr
        else:
            ang, L = rng.uniform(0, np.pi), rng.uniform(0.03, 0.12)
            d = np.abs(-(X - cx) * np.sin(ang) + (Y - cy) * np.cos(ang))
            t = (X - cx) * np.cos(ang) + (Y - cy) * np.sin(ang)
            m = (d < 0.003) & (np.abs(t) < L / 2)
        img[m] = c
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def _checker(spec: SceneSpec, square: float = 0.04) -> np.ndarray:
    X, Y = _world_grid(spec)
    t = ((np.floor(X / square) + np.floor(Y / square)) % 2).astype(bool)
    a = np.array(spec.table_rgb, float)
    im = np.where(t[..., None], a * 0.74, a)
    return np.clip(np.round(im), 0, 255).astype(np.uint8)


def table_texture(spec: SceneSpec) -> np.ndarray | None:
    """Overlay texture (rows = image top to bottom, cols = image left to right), or None."""
    t = spec.table
    if t in ("plain", "dark"):
        return None
    if t == "struct":
        return _toy_struct(spec, TOY_PARTS)
    if t.startswith("struct:"):
        parts = tuple(t.split(":", 1)[1].split("+"))
        assert set(parts) <= set(TOY_PARTS), parts
        return _toy_struct(spec, parts)
    if t == "wood":
        return _wood(spec, spec.texture_seed)
    if t == "clutter":
        return _clutter(spec, spec.texture_seed)
    if t == "checker":
        return _checker(spec)
    raise ValueError(f"unknown table {t!r}")


# Orientation fix for MuJoCo's plane texture mapping, found by rendering.
TEX_FLIP = dict(flipud=False, fliplr=False, rot90=0)   # identity: mean |render - toy| = 0.93 / 255 on table pixels


def _texture_file(spec: SceneSpec) -> Path | None:
    import hashlib

    from PIL import Image

    arr_needed = spec.table not in ("plain", "dark")
    if not arr_needed:
        return None
    key = json.dumps([spec.table, spec.texture_seed, list(spec.table_rgb), list(spec.camera_pos), spec.camera_fovy,
                      spec.ov_w, spec.ov_h, OV_PAD, OV_TEX, TEX_FLIP, 3])
    p = TEX_DIR / f"tex_{spec.table.replace(':', '_').replace('+', '-')}_{hashlib.md5(key.encode()).hexdigest()[:10]}.png"
    if p.exists():
        return p
    arr = table_texture(spec)
    if TEX_FLIP["rot90"]:
        arr = np.rot90(arr, TEX_FLIP["rot90"])
    if TEX_FLIP["flipud"]:
        arr = arr[::-1]
    if TEX_FLIP["fliplr"]:
        arr = arr[:, ::-1]
    TEX_DIR.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".{os.getpid()}.tmp.png")
    Image.fromarray(np.ascontiguousarray(arr)).save(tmp)
    os.replace(tmp, p)
    return p


# --------------------------------------------------------------------------- lighting

LIGHT_PRESETS = {
    # amb = headlight ambient; fill = directional light straight down; key = spot light (nexus key);
    # shadow = key casts shadows; spec = light specular; tint; keypos = key position override
    "flat": dict(amb=DEF["flat_ambient"], fill=DEF["flat_diffuse"], key=0.0, shadow=False, spec=0.0),
    "shadow": dict(amb=0.6, fill=0.1, key=0.35, shadow=True, spec=0.0),
    "shaded": dict(amb=0.3, fill=0.5, key=0.5, shadow=True, spec=0.3),        # the stock so101-nexus lights
    "dim": dict(amb=0.5, fill=0.12, key=0.0, shadow=False, spec=0.0),
    "bright": dict(amb=0.95, fill=0.25, key=0.0, shadow=False, spec=0.0),
    "warm": dict(amb=0.8, fill=0.2, key=0.0, shadow=False, spec=0.0, tint=(1.0, 0.85, 0.65)),
    "cool": dict(amb=0.8, fill=0.2, key=0.0, shadow=False, spec=0.0, tint=(0.75, 0.87, 1.0)),
    "side": dict(amb=0.55, fill=0.0, key=0.6, shadow=True, spec=0.0, keypos=(0.15, 1.6, 0.9)),
}


# --------------------------------------------------------------------------- env


def _robot_masks(m):
    base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
    rb = np.zeros(m.nbody, bool)
    rb[base] = True
    for b in range(base + 1, m.nbody):
        rb[b] = rb[m.body_parentid[b]]
    return rb[m.geom_bodyid], rb


class SceneEnv(_pnp.PickAndPlaceEnv):
    """PickAndPlaceEnv in the toy-matched training scene, with perturbation parameters."""

    def __init__(self, spec: SceneSpec, config: PickAndPlaceConfig, **kw):
        self.spec = spec
        self._episode_seed = 0
        self.extra_info: dict = {}
        orig = _pnp.build_object_scene_xml
        tex = _texture_file(spec)
        dxml, self._dslots = self._distractor_xml(spec)
        cam = spec.camera_pos
        s2 = float(np.sqrt(0.5))
        # image up = world +x, image right = world -y  ->  camera frame = Rz(-90 deg)
        cam_xml = (f'    <camera name="overhead_fixed" mode="fixed" pos="{cam[0]} {cam[1]} {cam[2]}" '
                   f'quat="{s2} 0 0 {-s2}" fovy="{spec.camera_fovy}"/>\n')
        ov_xml, ov_asset = "", ""
        if tex is not None:
            half = spec.ov_w * (0.5 + OV_PAD) / camera_scale(spec, 0.0)
            ov_asset = (f'    <texture name="table_tex" type="2d" file="{tex.as_posix()}"/>\n'
                        f'    <material name="table_mat" texture="table_tex" texrepeat="1 1" texuniform="false" '
                        f'specular="0" shininess="0" reflectance="0"/>\n')
            # plane frame = camera frame (plane +x = image right, plane +y = image up)
            ov_xml = (f'    <geom name="table_overlay" type="plane" size="{half} {half} 0.001" material="table_mat" '
                      f'pos="{cam[0]} {cam[1]} 0.0003" quat="{s2} 0 0 {-s2}" contype="0" conaffinity="0" group="0"/>\n')
        th = spec.tray_half
        tray_geom = (f'type="box" size="{th} {th} 0.0005"' if spec.tray_kind == "square"
                     else f'type="cylinder" size="{th} 0.0005"')
        t_rgb = _c(spec.tray_rgb)
        floor_rgb = _c(spec.table_rgb) if spec.table != "dark" else _c((60, 56, 50))

        def patched(*a, **k):
            k["extra_bodies"] = k.get("extra_bodies", "") + dxml + ov_xml + cam_xml
            s = orig(*a, **k)
            i0 = s.index('<geom name="floor"')
            i1 = s.index("/>", i0) + 2
            s = s[:i0] + (f'<geom name="floor" type="plane" size="0 0 0.01" pos="0 0 0" contype="1" conaffinity="1" '
                          f'rgba="{floor_rgb[0]} {floor_rgb[1]} {floor_rgb[2]} 1"/>') + s[i1:]
            i0 = s.index('<geom name="target_disc"')
            i1 = s.index("/>", i0) + 2
            s = s[:i0] + (f'<geom name="target_disc" {tray_geom} rgba="{t_rgb[0]} {t_rgb[1]} {t_rgb[2]} 1" '
                          f'contype="0" conaffinity="0"/>') + s[i1:]
            s = s.replace("<visual>", '<visual>\n    <global offwidth="1024" offheight="1024"/>', 1)
            if ov_asset:
                if "<asset>" in s:
                    s = s.replace("<asset>\n", "<asset>\n" + ov_asset, 1)
                else:
                    s = s.replace("  <visual>", f"  <asset>\n{ov_asset}  </asset>\n  <visual>", 1)
            return s

        _pnp.build_object_scene_xml = patched
        try:
            super().__init__(config, **kw)
        finally:
            _pnp.build_object_scene_xml = orig
        m = self.model
        for sl in self._dslots:
            sl["bid"] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, sl["name"])
            sl["gid"] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, sl["name"] + "_g")
        self._cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "overhead_fixed")
        self._cam0 = dict(pos=m.cam_pos[self._cam_id].copy(), quat=m.cam_quat[self._cam_id].copy(),
                          fovy=float(m.cam_fovy[self._cam_id].item()))
        if self._overhead_obs_renderer is not None:
            self._overhead_obs_cam = self._cam_id     # render the fixed MJCF camera
        self._robot_geom_mask, self._robot_body_mask = _robot_masks(m)
        self._colour_scene()
        self._light0 = dict(pos=m.light_pos.copy(), dir=m.light_dir.copy())
        self._floor_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self._base_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
        self._fk = mujoco.MjData(m)
        self._wrist0 = None

    # -- static appearance ---------------------------------------------------
    def _colour_scene(self):
        m, spec = self.model, self.spec
        arm = np.array([*_c(spec.arm_rgb), 1.0])
        g = np.where(self._robot_geom_mask & (m.geom_group <= 2))[0]
        for i in set(int(i) for i in m.geom_matid[g] if i >= 0):
            m.mat_rgba[i] = arm
            m.mat_specular[i] = 0.0
            m.mat_shininess[i] = 0.0
            m.mat_reflectance[i] = 0.0
        m.geom_rgba[g] = arm
        for gid in self._slots[0].geom_ids:
            m.geom_rgba[gid] = [*_c(spec.cube_rgb), 1.0]

    @staticmethod
    def _distractor_xml(spec: SceneSpec):
        xml, slots = "", []
        half = spec.cube_mm / 2000.0
        for gi, d in enumerate(spec.distractors):
            for j in range(d.n):
                name = f"distr_{gi}_{j}"
                if d.kind == "cube":
                    h = d.size or half
                    g = f'type="box" size="{h} {h} {h}"'
                    z, rad = h, h * 1.42
                elif d.kind == "disc":
                    r = d.size or 0.03
                    g = f'type="cylinder" size="{r} 0.0008"'
                    z, rad = 0.0016, r
                else:
                    raise ValueError(d.kind)
                xml += (f'    <body name="{name}" pos="0 0 -1"><geom name="{name}_g" {g} rgba="0.5 0.5 0.5 0" '
                        f'contype="0" conaffinity="0"/></body>\n')
                slots.append(dict(name=name, kind=d.kind, z=z, radius=rad, spec=d))
        return xml, slots

    # -- reset hooks -----------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        if seed is None:
            seed = int(np.random.SeedSequence().generate_state(1)[0])
        self._episode_seed = int(seed)
        self.extra_info = {"seed": int(seed)}
        return super().reset(seed=seed, options=options)

    def _start_pose_ok(self, q) -> bool:
        d, m = self._fk, self.model
        d.qpos[:] = self.data.qpos
        for i, jid in enumerate(self._joint_ids):
            d.qpos[m.jnt_qposadr[jid]] = q[i]
        mujoco.mj_fwdPosition(m, d)
        tcp = d.site_xpos[self._tcp_site_id]
        z0, z1 = DEF["start_tcp_z"]
        r0, r1 = DEF["start_tcp_rho"]
        if not (z0 <= tcp[2] <= z1 and r0 <= np.hypot(tcp[0], tcp[1]) <= r1):
            return False
        arm_b = np.where(self._robot_body_mask)[0]
        if np.any(d.xpos[arm_b[arm_b != self._base_bid], 2] < 0.03):
            return False
        for c in d.contact[: d.ncon]:
            g1, g2 = int(c.geom[0]), int(c.geom[1])
            ra, rb = self._robot_geom_mask[g1], self._robot_geom_mask[g2]
            if ra and rb:
                return False                                      # self-collision
            if (ra and g2 == self._floor_gid and m.geom_bodyid[g1] != self._base_bid) or \
               (rb and g1 == self._floor_gid and m.geom_bodyid[g2] != self._base_bid):
                return False
        return True

    def _reset_robot_joints(self, init_qpos=None):
        if init_qpos is not None or not self.spec.random_start:
            return super()._reset_robot_joints(init_qpos)
        box = DEF["start_box_rad"]
        lo = np.array([box[j][0] for j in JOINTS])
        hi = np.array([box[j][1] for j in JOINTS])
        q = None
        for k in range(1000):
            cand = self.np_random.uniform(lo, hi)
            if self._start_pose_ok(cand):
                q = cand
                break
        self.extra_info["start_tries"] = k + 1
        if q is None:
            return super()._reset_robot_joints(None)
        applied = np.clip(q, self._target_low, self._target_high)
        for i, jid in enumerate(self._joint_ids):
            self.data.qpos[self.model.jnt_qposadr[jid]] = applied[i]
        self.data.ctrl[self._actuator_ids] = applied
        self.extra_info["start_qpos"] = applied.round(4).tolist()
        return applied

    def sample_workspace_xy(self, rng, avoid=(), clear=0.0, tray_clear=None):
        """Uniform-by-area point in the cube sector, Chebyshev-clear of the tray, and ``clear`` + radius
        from each ((x, y), radius) in ``avoid``."""
        sp = self.spec
        r0, r1 = sp.ws_rho
        b0, b1 = np.radians(sp.ws_bearing_deg)
        tc = sp.ws_tray_clear if tray_clear is None else tray_clear
        tx, ty = sp.tray_xy
        for _ in range(1000):
            rho = np.sqrt(rng.uniform(r0 * r0, r1 * r1))
            b = rng.uniform(b0, b1)
            x, y = rho * np.cos(b), rho * np.sin(b)
            if max(abs(x - tx), abs(y - ty)) < tc:
                continue
            if all(np.hypot(x - ax, y - ay) >= clear + ar for (ax, ay), ar in avoid):
                return np.array([x, y])
        return None

    def _task_reset(self) -> None:
        super()._task_reset()                      # nexus draws (disc colour/pose, cube pose), then overridden
        m, sp = self.model, self.spec
        m.body_pos[self._target_body_id] = [sp.tray_xy[0], sp.tray_xy[1], 0.001]
        m.geom_rgba[self._target_geom_id] = [*_c(sp.tray_rgb), 1.0]
        self.task_description = TASK
        slot = self._slots[self._target_slot_idx]
        xy = self.sample_workspace_xy(self.np_random)
        yaw = self.np_random.uniform(0, np.pi / 2) if sp.cube_yaw_random else np.radians(sp.cube_yaw_deg)
        self.data.qpos[slot.qpos_addr: slot.qpos_addr + 3] = [xy[0], xy[1], slot.spawn_z]
        self.data.qpos[slot.qpos_addr + 3: slot.qpos_addr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self._initial_obj_z = slot.spawn_z
        for gid in slot.geom_ids:
            m.geom_rgba[gid] = [*_c(sp.cube_rgb), 1.0]
        rng = np.random.default_rng([self._episode_seed, EXTRA_SEED])
        self.extra_info.update(cube_xy=xy.round(5).tolist(), cube_yaw=float(yaw))
        self.extra_info["distractors"] = self._place_distractors(rng, xy)
        self.extra_info["lighting"] = self._apply_lights(rng)
        self.extra_info["overhead_cam"] = self._jitter_overhead(rng)

    def _randomize_wrist_camera(self) -> None:
        super()._randomize_wrist_camera()          # fixed camera (ranges collapsed in make_env)
        rng = np.random.default_rng([self._episode_seed, EXTRA_SEED, 2])
        j, cid, m = self.spec.wrist_jitter, self._wrist_cam_id, self.model
        if self._wrist0 is None:
            self._wrist0 = dict(pos=m.cam_pos[cid].copy(), quat=m.cam_quat[cid].copy(), fovy=float(m.cam_fovy[cid].item()))
        w0 = self._wrist0
        if j <= 0:
            m.cam_pos[cid], m.cam_quat[cid], m.cam_fovy[cid] = w0["pos"], w0["quat"], w0["fovy"]
            return
        dp = rng.uniform(-0.003, 0.003, 3) * j
        dpitch = np.radians(rng.uniform(-2, 2) * j)
        dfov = rng.uniform(-3, 3) * j
        q, qq = np.zeros(4), np.zeros(4)
        mujoco.mju_euler2Quat(q, np.array([dpitch, 0.0, 0.0]), "XYZ")
        mujoco.mju_mulQuat(qq, w0["quat"], q)
        m.cam_pos[cid], m.cam_quat[cid], m.cam_fovy[cid] = w0["pos"] + dp, qq, w0["fovy"] + dfov
        self.extra_info["wrist_cam"] = dict(dpos=dp.tolist(), dpitch_deg=float(np.degrees(dpitch)), dfov=float(dfov))

    def _place_distractors(self, rng, cube_xy) -> list:
        out = []
        if not self._dslots:
            return out
        sp = self.spec
        tray = np.array(sp.tray_xy, float)
        placed = [(tuple(cube_xy), 0.018)]
        for sl in self._dslots:
            d: Distractor = sl["spec"]
            xy = None
            for _ in range(400):
                if d.near in ("tray", "cube"):
                    c = tray if d.near == "tray" else np.asarray(cube_xy)
                    rr, th = rng.uniform(*d.near_range), rng.uniform(0, 2 * np.pi)
                    cand = c + rr * np.array([np.cos(th), np.sin(th)])
                    rho = np.hypot(*cand)
                    if rho < 0.12 or rho > sp.ws_rho[1] + 0.03:
                        continue
                    if max(abs(cand[0] - tray[0]), abs(cand[1] - tray[1])) < sp.tray_half + sl["radius"] + 0.005:
                        continue
                    if any(np.hypot(cand[0] - a[0], cand[1] - a[1]) < sl["radius"] + ar + 0.01 for a, ar in placed):
                        continue
                    xy = cand
                    break
                cand = self.sample_workspace_xy(rng, avoid=placed, clear=sl["radius"] + 0.01,
                                                tray_clear=sp.tray_half + sl["radius"] + 0.005)
                if cand is not None:
                    xy = cand
                    break
            col = sample_color(d.color, rng)
            if xy is None:
                self.model.body_pos[sl["bid"]] = [0, 0, -1]
                self.model.geom_rgba[sl["gid"]] = [*col, 0.0]
                out.append(dict(name=sl["name"], kind=sl["kind"], placed=False))
                continue
            placed.append((tuple(xy), sl["radius"]))
            yaw = rng.uniform(0, np.pi / 2) if sl["kind"] == "cube" else 0.0
            self.model.body_pos[sl["bid"]] = [xy[0], xy[1], sl["z"]]
            self.model.body_quat[sl["bid"]] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            self.model.geom_rgba[sl["gid"]] = [*col, 1.0]
            out.append(dict(name=sl["name"], kind=sl["kind"], placed=True, xy=[float(xy[0]), float(xy[1])],
                            rgb=[round(float(c) * 255) for c in col], near=d.near))
        return out

    def _apply_lights(self, rng) -> dict:
        P = LIGHT_PRESETS[self.spec.lighting]
        m, L, j = self.model, self._light0, self.spec.light_jitter
        tint = np.array(P.get("tint", (1.0, 1.0, 1.0)))
        km = rng.uniform(1 - j, 1 + j) if j > 0 else 1.0
        fm = rng.uniform(1 - j, 1 + j) if j > 0 else 1.0
        am = rng.uniform(1 - j, 1 + j) if j > 0 else 1.0
        # light 0 = spot key (nexus), light 1 = directional fill straight down
        m.light_diffuse[0] = P["key"] * km * tint
        m.light_diffuse[1] = P["fill"] * fm * tint
        m.light_specular[:] = P["spec"]
        m.light_ambient[:] = 0.0
        m.light_castshadow[0] = bool(P["shadow"])
        m.light_castshadow[1] = False
        m.light_active[0] = P["key"] > 0
        m.vis.headlight.ambient[:] = P["amb"] * am * tint
        m.vis.headlight.diffuse[:] = 0.0
        m.vis.headlight.specular[:] = 0.0
        pos, dirv = L["pos"].copy(), L["dir"].copy()
        dirv[1] = [0.0, 0.0, -1.0]
        tgt = np.array([0.2, 0.0, 0.0])
        if P.get("keypos") is not None:
            pos[0] = P["keypos"]
            v = tgt - pos[0]
            dirv[0] = v / np.linalg.norm(v)
        if j > 0:
            a = np.radians(rng.uniform(-30, 30) * j)
            R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
            pos[0] = tgt + R @ (pos[0] - tgt)
            dirv[0] = R @ dirv[0]
        m.light_pos[:] = pos
        m.light_dir[:] = dirv
        return dict(preset=self.spec.lighting, key_mult=float(km), fill_mult=float(fm), amb_mult=float(am))

    def _jitter_overhead(self, rng) -> dict:
        m, c0, j = self.model, self._cam0, self.spec.overhead_jitter
        cid = self._cam_id
        if j <= 0:
            m.cam_pos[cid], m.cam_quat[cid], m.cam_fovy[cid] = c0["pos"], c0["quat"], c0["fovy"]
            return {}
        dxy = rng.uniform(-0.01, 0.01, 2) * j
        dz = c0["pos"][2] * rng.uniform(-0.03, 0.03) * j
        tilt = np.radians(rng.uniform(-1.5, 1.5, 2) * j)
        yaw = np.radians(rng.uniform(-2, 2) * j)
        q, qq = np.zeros(4), np.zeros(4)
        mujoco.mju_euler2Quat(q, np.array([tilt[0], tilt[1], yaw]), "XYZ")
        mujoco.mju_mulQuat(qq, c0["quat"], q)
        m.cam_pos[cid] = c0["pos"] + np.array([dxy[0], dxy[1], dz])
        m.cam_quat[cid] = qq
        return dict(dxy=dxy.tolist(), dz=float(dz), tilt_deg=np.degrees(tilt).tolist(), yaw_deg=float(np.degrees(yaw)))


def make_env(spec: SceneSpec | None = None) -> SceneEnv:
    """Env as in the repo's collect_sim.make_env (fixed wrist camera, IK orientation weight 0.05)
    plus the toy-matched scene."""
    spec = spec or SceneSpec()
    wrist = WristCamera(width=spec.wr_w, height=spec.wr_h, fov_deg_range=(75.0, 75.0),
                        pitch_deg_range=(-17.2, -17.2), pos_x_noise=0.0, pos_y_noise=0.0, pos_z_noise=0.0)
    cfg = PickAndPlaceConfig(
        cube_side_length_mm=spec.cube_mm, goal_thresh=spec.goal_thresh,
        spawn_min_radius=0.04, spawn_max_radius=0.17, target_colors="blue", cube_colors="red",
        obs_mode="visual", observations=[JointPositions(), wrist, OverheadCamera(width=spec.ov_w, height=spec.ov_h)],
        terminate_on_success=False,
    )
    env = SceneEnv(spec, cfg, render_mode="rgb_array", control_mode="pd_joint_pos")
    atexit.register(env.close)
    env.config.robot.ee_orientation_weight = 0.05
    if spec.overhead_cam:
        set_camera(env, spec.overhead_cam)
    if env._ik_data is None:
        env._ik_data = mujoco.MjData(env.model)
    return env


def set_camera(env, cam: dict) -> None:
    """Make cam ({"pos", "quat", "fovy"}) the policy (overhead_fixed) camera and the jitter baseline"""
    m, cid = env.model, env._cam_id
    env._cam0 = dict(pos=np.asarray(cam["pos"], float), quat=np.asarray(cam["quat"], float), fovy=float(cam["fovy"]))
    m.cam_pos[cid], m.cam_quat[cid], m.cam_fovy[cid] = env._cam0["pos"], env._cam0["quat"], env._cam0["fovy"]


def load_camera(path) -> dict:
    """{"pos", "quat", "fovy", ...} from a camera.json-style file (key "camera") or a bare camera dict."""
    d = json.loads(Path(path).read_text())
    d = d.get("camera", d)
    return {k: d[k] for k in ("name", "pos", "quat", "fovy") if k in d}


def grip_limits(env) -> tuple[float, float]:
    return float(env.action_space.low[-1]), float(env.action_space.high[-1])


# --------------------------------------------------------------------------- named conditions


def conditions() -> dict[str, SceneSpec]:
    D = Distractor
    return {
        "plain": SceneSpec(),
        # the toy's structured-table elements at the toy's pixel positions / sizes / colours
        "struct": SceneSpec(table="struct"),
        "struct_checker": SceneSpec(table="struct:checker"),
        "struct_tape": SceneSpec(table="struct:tape"),
        "struct_green": SceneSpec(table="struct:green"),
        "struct_yellow": SceneSpec(table="struct:yellow"),
        "struct_no_yellow": SceneSpec(table="struct:checker+tape+green"),
        # other tables
        "dark": SceneSpec(table="dark"),
        "checker4cm": SceneSpec(table="checker"),
        "wood": SceneSpec(table="wood"),
        "clutter": SceneSpec(table="clutter"),
        # distractors (visual only)
        "d_nonred_cube": SceneSpec(distractors=[D("cube", "nonred", 1)]),
        "d_blue_cube": SceneSpec(distractors=[D("cube", "toyblue", 1)]),
        "d_blue_cube_near_tray": SceneSpec(distractors=[D("cube", "toyblue", 1, near="tray", near_range=(0.07, 0.10))]),
        "d_reddish_cube": SceneSpec(distractors=[D("cube", "reddish", 1)]),
        "d_red_cube": SceneSpec(distractors=[D("cube", "red", 1)]),
        "d_several": SceneSpec(distractors=[D("cube", "nonred", 3), D("disc", "nonred", 1)]),
        # lighting
        "light_shadow": SceneSpec(lighting="shadow"),
        "light_shaded": SceneSpec(lighting="shaded"),
        "light_dim": SceneSpec(lighting="dim"),
        "light_warm": SceneSpec(lighting="warm"),
        "light_side": SceneSpec(lighting="side"),
        "light_jitter": SceneSpec(light_jitter=0.3),
        # camera / object pose
        "cam_jitter": SceneSpec(overhead_jitter=1.0),
        "cube_yaw": SceneSpec(cube_yaw_random=True),
    }
