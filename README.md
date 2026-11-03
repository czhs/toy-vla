# A mechanistic account of a fine-tuned SmolVLA: reproduction package

Code and per-scene outputs for "Toward a Mechanistic Understanding of a Fine-Tuned SmolVLA". The paper studies one SmolVLA policy fine-tuned on a toy task (a drawn one-link arm pointing at a red block) and one simulated SO-101 pick-and-place policy. This repository remakes every figure and number from the stored outputs on any machine, and retrains and reruns everything behind those outputs on a GPU.

There are three ways in, in order of cost:

| goal | what it needs | time |
|---|---|---|
| every number and figure from the stored per-scene outputs in `results/` | Python with numpy, matplotlib, pillow | under a minute |
| rerun the toy experiments on the trained policies | toy environment, one GPU with 10 GB, the checkpoints | about 1 h in all |
| retrain and rerun everything | both environments, one 24 GB GPU, about 200 GB of disk | about 1 day |

## 1. Numbers and figures from the stored outputs

```bash
python -m venv .venv-figures && .venv-figures/bin/pip install numpy==2.2.6 matplotlib==3.10.9 pillow==12.3.0
```

```bash
.venv-figures/bin/python numbers/recompute_numbers.py
```

```bash
cd figures && for f in make_fig1 make_fig2 make_fig3 make_fig4a make_fig4b; do ../.venv-figures/bin/python $f.py; done
```

`recompute_numbers.py` recomputes each number from the per-scene outputs, compares it with the value printed in the paper (`numbers/paper_values.json`) and writes `numbers/recomputed.json`. It exits non-zero if a value differs at the paper's printed precision. Figures go to `figures/out/`.

## 2. Environments

The toy and the arm use different stacks, because the arm's simulator needs a newer lerobot. Both pins come from the environments that produced the paper's outputs.

```bash
uv venv --python 3.10 .venv-toy && uv pip install --python .venv-toy/bin/python -r requirements/toy.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

The toy stack is Python 3.10, torch 2.7.1, lerobot 0.3.3 and transformers 4.51.3. The installed lerobot was checked file by file against its release: it is unmodified. The base policy (`lerobot/smolvla_base`) and the backbone (`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`) are fetched from Hugging Face at pinned revisions the first time they are needed.

Paths and device come from environment variables, all optional: `TOYVLA_DATA` (default `data/`, not tracked by git), `TOYVLA_RESULTS` (default `results/`) and `TOYVLA_DEVICE` (default `cuda` when available, else `cpu`).

```bash
.venv-toy/bin/python scripts/download_checkpoints.py toy toy-colours
```

The checkpoints are on the [v1.0 release](https://github.com/czhs/toy-vla/releases/tag/v1.0) and are checked by SHA-256. To check that the code runs at all without a GPU, `tests/smoke_toy.sh .venv-toy/bin/python` runs every toy experiment on a few scenes and writes to a temporary directory.

## 3. Retraining the toy policies

Training needs a GPU; the feature cache is 61.4 GB per colour. Runtimes here and below are from one RTX 4090.

```bash
.venv-toy/bin/python train/build_feature_cache.py
```

```bash
.venv-toy/bin/python train/train_policy.py --run toy_policy_red --mask_pad --max_steps 134400 --const_steps 112000 --cooldown 22400 --seed 20260930
```

The cache takes about 25 min and training about 3 h. For the green and blue reference policies add `--block_colour green` or `--block_colour blue` to both commands and change `--run`. Copy `data/runs/<run>/final.pt` to `data/checkpoints/<run>.pt` to use a retrained policy.

A retrained policy will not match the released checkpoint bit for bit: the released red policy was resumed once from step 80,000 after a machine restart, and the feature cache and compile choices affect bf16 rounding. The experiments below read the released checkpoint unless you replace it.

## 4. The toy experiments

Run from the repository root with `.venv-toy/bin/python`. Each script writes its per-scene output to `results/`, so its output replaces the stored one. The order matters where a row says it needs an earlier step.

| script | what it does | needs | time, GPU memory |
|---|---|---|---|
| `experiments/gradient_directions.py` | the directions the expert reads; writes `results/gradient_directions.json`, `data/vlm_projection_weights.npz` | | 1 min, 10 GB |
| `experiments/even_layer_skip.py` | removes groups of expert layers | | 1 min |
| `experiments/keep_erase.py` | keep-only and erase, random and PCA controls | directions | 1 min |
| `experiments/stream_gradients.py` | gradient of the answer with respect to the residual stream; 5.4 GB in `data/` | | 1 min, 10 GB |
| `experiments/stream_gradient_overlap.py` | share of that gradient inside the pulled-back directions; CPU | directions, stream gradients | 1 min |
| `experiments/block_calibration.py` | what a red block produces in the read projections | directions | 1 min |
| `experiments/pointer_layer_sweep.py` | pointer test by layer and strength, 200 scenes | calibration | 40 min on an RTX 3060, 4.4 GB |
| `experiments/pointer_test.py` | pointer test at layer 7, 4,000 scenes | calibration | 20 min, 4.3 GB |
| `experiments/distractor_one_square.py` | the one-square pull and its read-direction share | directions | 1 min, 2.3 GB |
| `experiments/colour_calibration.py` | colour directions and reference rows | | under 1 min |
| `experiments/colour_switch.py` | the detector and the edit, 2,048 + 512 scenes | directions, colour calibration | a few min |
| `experiments/colour_native_reference.py green`, then `blue` | policies trained on those colours, no edit | their checkpoints | under 1 min each |
| `experiments/query_replacement.py` | the estimate replaces the query input; writes `data/estimate_fit.npz` | | 1 min, 2.1 GB |
| `experiments/estimate_gate_scan.py` | the estimate set to an absolute angle, per-head attention | | 2 min, 2.1 GB |
| `experiments/head_gating.py` | per-head gating; CPU | gate scan, `results/head_ranking_top16.json` | seconds |
| `experiments/weight_fold.py` | the fixed estimate folded into the query weights | `data/estimate_fit.npz` | 1 min |

Times come from the original run logs on an RTX 4090 where one exists; the sweep ran on an RTX 3060. Every toy experiment has fixed scene and noise seeds. On a CUDA device it draws the same noise as the stored outputs; CPU generators draw differently.

## 5. The simulated SO-101 arm

The arm environment is Python 3.12, lerobot 0.6.1, so101-nexus 0.7.0, mujoco 3.14.0, transformers 5.5.4 and torch 2.11. Commands run from the repository root with that environment active.

```bash
uv venv --python 3.12 .venv-arm && source .venv-arm/bin/activate
```

```bash
uv pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu126 && uv pip install -r requirements/arm.txt
```

```bash
bash arm/setup_external.sh
```

The last command clones [aviadarn/so101-lerobot](https://github.com/aviadarn/so101-lerobot) at commit 7fe686c for its scripted expert. That repository has no licence file, so it is fetched and not copied here. Rendering uses MuJoCo EGL on a headless Linux machine; on macOS set `MUJOCO_GL=cgl`.

To start from the released policy, download it and rebuild the full weights from the base policy, then go to step 4 (the release also holds what `export` needs, so that step runs on a CPU) or step 5:

```bash
python scripts/download_checkpoints.py arm
```

```bash
BASE=$(python -c "from huggingface_hub import snapshot_download as s; print(s('lerobot/smolvla_base', revision='d9f33c94a60fb382c90dea2164c96845bd955e28'))"); python arm/apply_delta.py apply data/checkpoints/arm_policy $BASE data/checkpoints/arm_policy/delta.safetensors
```

| step | command | time | needs |
|---|---|---|---|
| 1. Dataset: 3,000 expert episodes, 110,479 frames, 0.6 GB. The seeds in `arm/assets/dataset_episodes.jsonl` are the successful attempts of the scripted expert | `for k in 0 ... 9: python arm/make_dataset.py --camera arm/assets/camera.json --lighting shadow --seed-list arm/assets/dataset_episodes.jsonl --shard $k --nshards 10 --root $TOYVLA_DATA/so101_shards/shard_0$k --repo-id local/shard_0$k`, then `python arm/merge_shards.py --out $TOYVLA_DATA/so101_redcube_angled_shadow $TOYVLA_DATA/so101_shards/shard_0*` | 8 min with 10 parallel shards | CPU + EGL |
| 2. Feature cache (13.6 GB) | `python arm/build_feature_cache.py --dataset $TOYVLA_DATA/so101_redcube_angled_shadow --image-key observation.images.overhead --workers 14` | 7 min | GPU |
| 3. Policy: 40,000 constant steps plus a 10,000-step cooldown | `python arm/train_policy.py --run arm_policy --dataset $TOYVLA_DATA/so101_redcube_angled_shadow --camera arm/assets/camera.json --lighting shadow --const-steps 40000 --cooldown 10000 --fixed-schedule --eval_conds plain,struct,struct_yellow,d_several`, then copy `$TOYVLA_DATA/arm_runs/arm_policy/checkpoints/final/pretrained_model` to `$TOYVLA_DATA/checkpoints/arm_policy` | 50 min | GPU, 6 GB |
| 4. The fold | `python arm/fit_fold.py panels`; `... streams`; `... fit_g`; `... export 60 1.25`; `... export -60 1.25` (and `export 60 0.75`, `export 60 1.5` for the other candidates) | under 5 min; export is CPU | GPU, 3 GB, except export |
| 5. Closed loop | `bash arm/run_closed_loop.sh` (`dev`: 200 layouts x 4 checkpoints; `fresh`: 200 x 2; `big`: 3,000 x 3 in 9 parallel processes) | 3 min per 200 layouts alone; 1.1 to 1.7 h for `big` | GPU, 0.9 GiB and up to 8 GB host RAM per process |
| 6. Reduce to what the paper uses | `python arm/reduce_eval_outputs.py` writes `results/arm/closed_loop.npz` | seconds | none |
| 7. Figure 3b background | `python arm/make_figure_map.py` writes `results/arm/bg.png`, `map.json` | seconds | CPU + renderer |

Retraining in step 3 gives a different policy from the released one, because torch.compile and bf16 on another machine are not bit-reproducible; the paper's numbers come from the released policy. Simulated episodes are also not bitwise across platforms, because of physics rounding. An episode succeeds on the first control step where the so101-nexus environment reports the cube released, static, at table height and within 4 cm of the tray centre, within 12 s.

| file | role |
|---|---|
| `scene.py`, `assets/scene_defaults.json`, `assets/camera.json`, `assets/eval_layouts.json` | scene, policy camera, frozen condition specs and seed sets |
| `expert.py`, `make_dataset.py`, `merge_shards.py`, `assets/dataset_episodes.jsonl` | scripted expert and dataset |
| `fast_core.py`, `build_feature_cache.py`, `train_policy.py` | expert-only fine-tune on cached image features |
| `policy_tools.py`, `fit_fold.py`, `apply_delta.py` | the fold of Equation (fold), scaled by lambda |
| `eval_closed_loop.py`, `run_closed_loop.sh`, `reduce_eval_outputs.py` | closed-loop evaluation. An episode succeeds on the first control step where the so101-nexus environment reports the cube released, static, at table height and within 4 cm (xy) of the tray centre, within 12 s; ten actions of each chunk are executed before replanning |
| `make_figure_map.py` | top-down render and world-to-pixel map for the figure |

## Paper figures and tables

Numbering follows the 3 November 2026 draft.

| in the paper | remade by | data in `results/` | data produced by |
|---|---|---|---|
| Figure 1, overview | `figures/make_fig1.py` | `arm/fig1_arm_panel.png` | `figures/make_fig1d_arm_panel.py` (arm panel) |
| Figure 2, colour switch | `figures/make_fig3.py` | `colour_switch.npz`, `colour_native_green.npz`, `colour_native_blue.npz` | `experiments/colour_switch.py`, `experiments/colour_native_reference.py` |
| Figure 3a, toy weight fold | `figures/make_fig4a.py` | `weight_fold.npz` | `experiments/weight_fold.py` |
| Figure 3b, arm weight fold | `figures/make_fig4b.py` | `arm/closed_loop.npz`, `arm/map.json`, `arm/bg.png` | `arm/run_closed_loop.sh`, `arm/reduce_eval_outputs.py`, `arm/make_figure_map.py` |
| Table 1, direction counts and energy shares | `numbers/recompute_numbers.py` | `gradient_directions.json` | `experiments/gradient_directions.py` |
| Table 2, keep-only and erase | `numbers/recompute_numbers.py` | `keep_erase.npz` | `experiments/keep_erase.py` |
| Table 3, even-layer removal | `numbers/recompute_numbers.py` | `even_layer_skip.json` | `experiments/even_layer_skip.py` |
| Table 4, pointer test | `numbers/recompute_numbers.py` | `pointer_layer_sweep.npz`, `pointer_test.npz` | `experiments/pointer_layer_sweep.py`, `experiments/pointer_test.py` |
| Table 5, one-square distractor | `numbers/recompute_numbers.py` | `distractor_one_square.npz` | `experiments/distractor_one_square.py` |
| Table 6, fixed angle in the query input | `numbers/recompute_numbers.py` | `query_replacement.npz` | `experiments/query_replacement.py` |
| Table 7, toy weight fold | `numbers/recompute_numbers.py` | `weight_fold.npz` | `experiments/weight_fold.py` |
| Table 8, arm | `numbers/recompute_numbers.py` | `arm/closed_loop.npz` | `arm/run_closed_loop.sh`, `arm/reduce_eval_outputs.py` |

`figures/make_fig2.py` draws the pointer test as a figure; the paper reports the same data in Table 4 instead. The script names keep the order in which the figures were first made.

Numbers in the running text that no table holds:

| number | data in `results/` | produced by |
|---|---|---|
| share of the stream gradient inside the pulled-back directions | `stream_gradient_overlap.json` | `experiments/stream_gradients.py`, `experiments/stream_gradient_overlap.py` |
| variance fraction and line fit of the estimate | `estimate_fit.json` | `experiments/query_replacement.py` |
| head counts (gated, open, reversed) | `head_gating.npz` | `experiments/estimate_gate_scan.py`, `experiments/head_gating.py` |

## Not regenerated by this repository

- `results/head_ranking_top16.json` is a fixed list. It comes from a single-head scan of the query path that is not part of the paper's results, and `experiments/head_gating.py` uses it, with a firing count, to choose the heads it analyses.
- The scale of the arm fold was chosen on 200 development layouts among more candidates than `arm/run_closed_loop.sh` reruns. The other candidates used other frame and flow-step sets; their code and outputs are not here.
- The arm policy's training log no longer exists. The training flags in step 3 come from the project's setup notes, and the released weights are the ones the paper used.

## What has been checked

- Every value in `numbers/paper_values.json` was compared with the per-scene outputs by `numbers/recompute_numbers.py`, and the figures were compared with the paper's.
- `tests/check_scene_seeds.py` regenerates the scenes of every toy experiment from their seeds and compares them with the stored outputs.
- `tests/smoke_toy.sh` runs every toy experiment on a few scenes on a CPU.
- On the arm, `arm/fit_fold.py export` on the recorded streams gives q_proj weights bit-identical to the checkpoints behind the 3,000-layout runs. Layouts, the expert and small dataset shards match the original records apart from physics rounding across platforms.
- The cleaned toy trainer, the arm trainer, `arm/fit_fold.py panels`, `streams`, `fit_g` and `arm/run_closed_loop.sh` have not yet been rerun on a GPU from this repository. They are mechanical ports of the scripts that produced the stored outputs.

## Layout

```
toyvla/        scene, policy loading and hooks, readouts, cache edits, pointer test, colour scenes
train/         feature cache and trainer for the toy policies
experiments/   one script per toy experiment
arm/           SO-101 simulator, dataset, trainer, weight fold, closed-loop evaluation
results/       per-scene outputs read by the number and figure scripts
numbers/       recompute_numbers.py and the values printed in the paper
figures/       one script per figure
scripts/       checkpoint download
tests/         short runs of the experiment scripts
```

## Licence

The code in this repository is under the [Apache License 2.0](LICENSE). The policies build on `lerobot/smolvla_base` and SmolVLM2, under their own licences. `arm/setup_external.sh` fetches aviadarn/so101-lerobot, which is not part of this repository and has no licence file.
