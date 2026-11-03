"""Top-down background for Figure 3b: the plain-table scene rendered from the unjittered top-down overhead camera (camera_pos, camera_fovy; image up =
world +x, image right = world -y), arm at its nominal (non-random) start pose, cube moved out of view. Saves results/arm/bg.png (512x512) and
results/arm/map.json with the world->pixel mapping.
    python arm/make_figure_map.py [OUT_DIR]"""
import json, os, sys
import numpy as np
import eval_closed_loop as E, paths, scene as S, mujoco
from PIL import Image
OUT = sys.argv[1] if len(sys.argv) > 1 else str(paths.RESULTS); os.makedirs(OUT, exist_ok=True)
sd = dict(E.load_layouts(None, str(paths.ASSETS / "camera.json"), "shadow")["conditions"]["plain"]); sd["overhead_cam"] = None; sd["random_start"] = False; sd["overhead_jitter"] = 0.0
spec = S.SceneSpec.from_dict(sd); env = S.make_env(spec)
env.reset(seed=940000)
u = env.unwrapped if hasattr(env, "unwrapped") else env
slot = u._slots[u._target_slot_idx]; u.data.qpos[slot.qpos_addr: slot.qpos_addr + 3] = [2.0, 2.0, slot.spawn_z]; mujoco.mj_forward(u.model, u.data)
Image.fromarray(np.asarray(u._get_obs()["overhead_camera"]).astype(np.uint8)).save(f"{OUT}/bg.png")
pts = {"origin": (0.0, 0.0), "tray": tuple(spec.tray_xy), "x0.3": (0.3, 0.0), "y0.2": (0.0, 0.2)}
mp = {k: [float(a) for a in S.world_to_pixel(spec, x, y)] for k, (x, y) in pts.items()}
tcp = u.data.site_xpos[u._tcp_site_id].copy(); mp["tcp_start_world"] = tcp.round(4).tolist(); mp["tcp_start_px"] = [float(a) for a in S.world_to_pixel(spec, tcp[0], tcp[1], tcp[2])]
mp.update(camera_pos=list(spec.camera_pos), fovy=spec.camera_fovy, scale_px_per_m=float(S.camera_scale(spec, 0.0)), ws_rho=list(spec.ws_rho), ws_bearing_deg=list(spec.ws_bearing_deg), tray_xy=list(spec.tray_xy), tray_half=spec.tray_half, note="image up = world +x, image right = world -y; u = column, v = row")
json.dump(mp, open(f"{OUT}/map.json", "w"), indent=1); print(json.dumps(mp)); env.close()
