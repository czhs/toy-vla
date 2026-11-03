"""Download the trained policies from this repository's GitHub release into data/checkpoints and check their SHA-256.
    python scripts/download_checkpoints.py toy          # the red policy (0.4 GB): everything on the toy except the native-colour reference
    python scripts/download_checkpoints.py toy-colours  # the policies trained on a green and on a blue block (0.8 GB)
    python scripts/download_checkpoints.py arm          # the arm policy as a delta on smolvla_base, the two folds' q_proj weights and the fold's fit data (0.2 GB)"""
import hashlib, os, sys, tarfile, urllib.request
from pathlib import Path
RELEASE = 'https://github.com/czhs/toy-vla/releases/download/v1.0'
DATA = Path(os.environ.get('TOYVLA_DATA', Path(__file__).resolve().parents[1] / 'data')); CKPT = DATA / 'checkpoints'
FILES = dict(
    toy={'toy_policy_red.pt': 'c220d7a6c295408b49afdebd7185acd45141260d6fd9aa2870a4eb91d9e50b35'},
    **{'toy-colours': {'toy_policy_green.pt': 'f1c6b01f1d3034d53a6039e486a12f1b927d76ecc4ca479bb2bd35d8d16a1b10', 'toy_policy_blue.pt': '846720ba2525ac85093e70fbf3ae7cafaf94e06acf9577a2829ea1dcb16e766e'}})
FILES['arm'] = {n: '%s' % h for n, h in {'arm_policy.tar': 'e681c0fb08b97be6fd4b1444feb160067697769cdb2d61d1e7bf2586d030640d', 'arm_fold_fit.tar': '6eb1154a2a3116d43e02eeb86c80a899b5ee7db3548d0eb2e8184f7fc38dc22a', 'fold_q_proj_plus60.safetensors': '70c14634ad60550ac12cb7e140540b9835dc540e0459a6b194b61a7b9fc1010b', 'fold_q_proj_minus60.safetensors': '4959972b41ff7d1edaa484f977eed88881b0d7a710f64a731e43eb0c3aa63b99'}.items()}
UNPACK = {'arm_policy.tar': CKPT / 'arm_policy', 'arm_fold_fit.tar': DATA / 'arm_fold'}
def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''): h.update(b)
    return h.hexdigest()
for group in sys.argv[1:] or ['toy']:
    for name, want in FILES[group].items():
        p = CKPT / name; p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            print('downloading', name, flush=True); urllib.request.urlretrieve(f'{RELEASE}/{name}', p)
        got = sha256(p); assert got == want, f'{name}: sha256 {got}, expected {want}'; print('ok', name)
        if name in UNPACK: UNPACK[name].mkdir(parents=True, exist_ok=True); tarfile.open(p).extractall(UNPACK[name], filter='data')
