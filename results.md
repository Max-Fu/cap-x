# Results

Last updated: 2026-04-21

## Setup

These comparisons assume the following services are already running:

- NVIDIA-compatible LLM proxy on `http://127.0.0.1:8110/chat/completions`
- OpenPI server on `127.0.0.1:8000`
- SAM3 on `127.0.0.1:8114`
- Contact-GraspNet on `127.0.0.1:8115`
- PyRoKi on `127.0.0.1:8116`
- Molmo2 on `127.0.0.1:8122` when available

The coding / visual model used in current comparisons is:

- `openai/openai/gpt-5.4`

## Scripts

Three Python entrypoints are used for the main comparison matrix:

- Pure VLA / direct OpenPI:
  - `capx/envs/scripts/run_openpi_libero_eval.py`
- Clean coding-agent baseline:
  - `capx/envs/scripts/run_libero_batch.py`
- Hybrid coding-agent + VLA:
  - `capx/envs/launch.py`
- Direct helper-only hybrid ablation:
  - `capx/envs/scripts/run_openpi_local_helper_eval.py`

## One-Task End-to-End Walkthrough

The most complete single-task example right now is orange juice:

- suite: `libero_object_swap`
- task id: `9`
- task prompt: `Pick the orange juice and place it in the basket`

Files and directories you need:

- CaP-X checkout:
  - this repo
- OpenPI checkout:
  - a separate checkout, referenced by `OPENPI_ROOT`
- OpenPI LIBERO checkpoint:
  - config: `pi05_libero`
  - checkpoint: `gs://openpi-assets/checkpoints/pi05_libero`
- LIBERO config file:
  - `~/.libero/config.yaml`
- Python envs:
  - `.venv`
  - `.venv-libero`

### Annotated Bash Walkthrough

```bash
# 1. Activate the LIBERO-capable environment used for eval scripts.
source .venv-libero/bin/activate

# 2. Export the OpenPI checkout. This checkout must contain:
#    - scripts/serve_policy.py
#    - the OpenPI Python package and its env
export OPENPI_ROOT=/path/to/openpi

# 3. Launch the NVIDIA-compatible LLM proxy used by the coding agent.
#    This is the endpoint used by clean CaP-X and hybrid CaP-X+VLA.
uv run --active --no-sync capx/serving/nv_server.py \
  --host 127.0.0.1 \
  --port 8110 \
  --model openai/openai/gpt-5.4 \
  --key-file .nvinferencekey

# 4. Launch OpenPI serving the LIBERO checkpoint.
#    The first run may download/cache model assets from:
#    gs://openpi-assets/checkpoints/pi05_libero
OPENPI_ROOT=$OPENPI_ROOT \
uv run --active --no-sync capx/serving/launch_openpi_server.py \
  --policy-config pi05_libero \
  --policy-dir gs://openpi-assets/checkpoints/pi05_libero \
  --port 8000

# 5. Launch the primitive perception / motion servers used by CaP-X.
#    These are needed by clean CaP-X and hybrid CaP-X+VLA.
uv run --active --no-sync capx/serving/launch_pyroki_server.py \
  --host 127.0.0.1 \
  --port 8116 \
  --robot panda_description \
  --target-link panda_hand

uv run --active --no-sync capx/serving/launch_contact_graspnet_server.py \
  --host 127.0.0.1 \
  --port 8115

uv run --active --no-sync capx/serving/launch_sam3_server.py \
  --host 127.0.0.1 \
  --port 8114 \
  --device cuda

# 6. Optional but useful: Molmo2 pointing server for object-centric prompts.
#    This helps with object grounding in the hybrid prompts.
CUDA_VISIBLE_DEVICES=0 uv run --active --no-sync vllm serve allenai/Molmo2-8B \
  --trust-remote-code \
  --port 8122 \
  --max-num-batched-tokens 36864 \
  --dtype bfloat16 \
  --limit-mm-per-prompt.image 2

# 7. Pure VLA baseline on orange juice.
.venv-libero/bin/python3 capx/envs/scripts/run_openpi_libero_eval.py \
  --task-suite-name libero_object_swap \
  --task-id 9 \
  --num-trials-per-task 20 \
  --port 8000

# 8. Clean coding-agent baseline on orange juice.
.venv-libero/bin/python3 capx/envs/scripts/run_libero_batch.py \
  --args.base-config-path env_configs/libero/hillclimb_object_swap_0_clean_fast.yaml \
  --args.suites libero_object_swap \
  --args.task-id-start 9 \
  --args.task-id-end 9 \
  --args.models openai/openai/gpt-5.4 \
  --args.server-url http://127.0.0.1:8110/chat/completions \
  --args.max-tokens 4096 \
  --args.reasoning-effort low \
  --args.total-trials 20 \
  --args.output-dir ./outputs/object_swap_task9_clean_20trials

# 9. Hybrid coding-agent + VLA evaluation on orange juice.
#    This uses the explicit orange-juice hybrid prompt config.
.venv-libero/bin/python3 capx/envs/launch.py \
  --config-path env_configs/libero/hillclimb_object_swap_9_vla_minimal_v4.yaml \
  --model openai/openai/gpt-5.4 \
  --server-url http://127.0.0.1:8110/chat/completions \
  --visual-differencing-model openai/openai/gpt-5.4 \
  --visual-differencing-model-server-url http://127.0.0.1:8110/chat/completions \
  --max-tokens 4096 \
  --reasoning-effort low \
  --total-trials 20 \
  --output-dir ./outputs/object_swap_task9_hybrid_20trials_v4
```

## Core Launch Commands

### Pure VLA / Direct OpenPI

```bash
.venv-libero/bin/python3 capx/envs/scripts/run_openpi_libero_eval.py \
  --task-suite-name libero_object_swap \
  --task-id 9 \
  --num-trials-per-task 20 \
  --port 8000
```

### Clean CaP-X Coding Agent

```bash
.venv-libero/bin/python3 capx/envs/scripts/run_libero_batch.py \
  --args.base-config-path env_configs/libero/hillclimb_object_swap_0_clean_fast.yaml \
  --args.suites libero_object_swap \
  --args.task-id-start 9 \
  --args.task-id-end 9 \
  --args.models openai/openai/gpt-5.4 \
  --args.server-url http://127.0.0.1:8110/chat/completions \
  --args.max-tokens 4096 \
  --args.reasoning-effort low \
  --args.total-trials 20 \
  --args.output-dir ./outputs/object_swap_task9_clean_20trials
```

### Hybrid CaP-X + VLA

```bash
.venv-libero/bin/python3 capx/envs/launch.py \
  --config-path env_configs/libero/hillclimb_object_swap_9_vla_minimal_v4.yaml \
  --model openai/openai/gpt-5.4 \
  --server-url http://127.0.0.1:8110/chat/completions \
  --visual-differencing-model openai/openai/gpt-5.4 \
  --visual-differencing-model-server-url http://127.0.0.1:8110/chat/completions \
  --max-tokens 4096 \
  --reasoning-effort low \
  --total-trials 20 \
  --output-dir ./outputs/object_swap_task9_hybrid_20trials_v4
```

## Current Best Hybrid Configs

- Orange juice:
  - `env_configs/libero/hillclimb_object_swap_9_vla_minimal_v4.yaml`
- Orange juice, stricter placement gate:
  - `env_configs/libero/hillclimb_object_swap_9_vla_minimal_v5.yaml`
- Salad dressing:
  - `env_configs/libero/hillclimb_object_swap_2_vla_minimal_v1.yaml`
- Butter:
  - `env_configs/libero/hillclimb_object_swap_6_vla_minimal_v4.yaml`

## Latest Direct VLA Scout Results

These are pure OpenPI / direct LIBERO numbers from local GPU-backed sweeps:

| Task | Suite / task id | Pure VLA |
|---|---|---:|
| Orange juice | `libero_object_swap:9` | `3/20` |
| Butter | `libero_object_swap:6` | `9/20` |
| Salad dressing | `libero_object_swap:2` | `0/20` |
| Milk | `libero_object_swap:7` | `0/20` |
| Spatial swap task 1 | `libero_spatial_swap:1` | `2/20` |
| Spatial swap task 6 | `libero_spatial_swap:6` | `0/20` |

Important consequence:

- `butter` is now the strongest VLA task we have found for LIBERO-PRO-style object swap.
- `orange juice` is no longer the best hillclimb target.
- `salad dressing`, `milk`, and the sampled spatial-swap tasks are poor VLA targets under the current OpenPI checkpoint.

## Latest Targeted Comparison Snapshot

### LIBERO-PRO Object Swap

| Task | Pure VLA | Clean CaP-X | Hybrid / Helper |
|---|---:|---:|---:|
| Orange juice (`task_id=9`) | `3/20` | `13/20` clean `20`-trial run, `9/10` earlier targeted run | `3/10` with `v4`; prompt-only/helper reruns did not improve |
| Salad dressing (`task_id=2`) | `0/20` | `8/20` clean `20`-trial run, `6/10` earlier targeted run | `0/10` with `v1` |
| Butter (`task_id=6`) | `9/20`; matched seeds `1/5` on GPU 7 | `17/20` clean `20`-trial run, `8/10` earlier targeted run | `0/10` with `v3`; helper ablations still `0/5` on matched seeds |

Notes:

- The earlier one-trial broad scout made orange juice look like the strongest hybrid candidate, but the larger reruns did not hold that up.
- Butter is now the best target because the direct VLA policy is materially stronger there than on other sampled tasks.
- The remaining problem is integration: the helper / hybrid path is still worse than direct VLA on matched seeds.

## Current Best Performance

- Best clean CaP-X result so far on the targeted 20-trial LIBERO-PRO runs:
  - Butter (`libero_object_swap:6`): `17/20`
- Best pure direct VLA result so far on the targeted 20-trial LIBERO-PRO runs:
  - Butter (`libero_object_swap:6`): `9/20`
- Best completed hybrid result so far on the targeted reruns:
  - Orange juice (`libero_object_swap:9`): `3/10` with `hillclimb_object_swap_9_vla_minimal_v4.yaml`
- Best completed direct-VLA budget point on butter in the latest max-step sweep:
  - `400` steps: `3/5`

## Latest Helper Ablation Snapshot

These runs remove the coding model entirely and test only the local helper pattern:

| Task | Helper setting | Result |
|---|---|---:|
| Orange juice (`task_id=9`) | staged helper, `3/3` OpenPI burst | `0/1` |
| Orange juice (`task_id=9`) | staged helper, `5/5` OpenPI burst | `0/1` |
| Butter (`task_id=6`) | staged helper, `3/3` OpenPI burst | `0/1` |
| Butter (`task_id=6`) | no-stage helper, `3/3` OpenPI burst | `0/5` |
| Butter (`task_id=6`) | no-stage helper, alias-fixed, `5/5` OpenPI burst | `0/5` |

Interpretation:

- The helper path is still breaking the policy structure even after:
  - removing pre-staging
  - fixing object aliases
  - matching the direct baseline's `5/5` OpenPI horizon
- On matched seeds for butter, direct VLA gets `1/5` while the helper gets `0/5`, so the helper remains worse but only slightly.

## Latest Long-Horizon Butter Ablations

These runs test whether OpenPI simply needed a longer rollout budget on the strongest VLA task, butter (`libero_object_swap:6`).

### Direct OpenPI / pure VLA

| Max steps | Result |
|---|---:|
| `10` | `0/5` |
| `50` | `0/5` |
| `100` | `0/5` |
| `200` | `1/5` |
| `400` | `3/5` |
| `600` | `0/5` |
| `800` | `2/5` |
| `1000` | `2/5` |

Logs:

- `outputs/openpi_libero/object_swap_task6_vla_budget10_5trials.log`
- `outputs/openpi_libero/object_swap_task6_vla_budget50_5trials.log`
- `outputs/openpi_libero/object_swap_task6_vla_budget100_5trials.log`
- `outputs/openpi_libero/object_swap_task6_vla_budget200_5trials.log`
- `outputs/openpi_libero/object_swap_task6_vla_budget400_5trials.log`
- `outputs/openpi_libero/object_swap_task6_vla_budget600_5trials.log`
- `outputs/openpi_libero/object_swap_task6_vla_budget800_5trials.log`
- `outputs/openpi_libero/object_swap_task6_vla_budget1000_5trials_rerun3.log`

### CaP-X native OpenPI rollout wrapper

This is the no-LLM rollout-only wrapper inside CaP-X, using native OpenPI execution.

| Wrapper setting | Result |
|---|---:|
| coarse wrapper, `200` steps | `1/5` |
| coarse wrapper, `400` steps | `2/5` |

Artifacts:

- `outputs/openpi_rollout_task6_budget200_5trials/aggregate.json`
- `outputs/openpi_rollout_task6_budget400_5trials/aggregate.json`

Interpretation:

- Longer OpenPI horizon definitely matters on butter.
- But more steps are not monotonically better; the best completed direct point so far is still `400` steps.
- The CaP-X wrapper improves with longer horizons too, but still trails the direct baseline at the same budget (`2/5` vs `3/5` at `400`).
- The most plausible remaining mismatch is control granularity:
  - direct baseline: replan a `5`-step chunk, execute `1` raw action, reobserve, repeat
  - older wrapper: replan a `5`-step chunk, execute all `5` actions before reobserving
- A more faithful wrapper rerun is now in progress using `execute_actions_per_plan=1`.

## Current Artifact Pointers

### Hybrid Orange Juice

- Success:
  - `outputs/openai_openai_gpt-5.4/object_swap_task9_hybrid_10trials_v4/trial_06_sandboxrc_0_reward_1.000_taskcompleted_1`
- Failure:
  - `outputs/openai_openai_gpt-5.4/object_swap_task9_hybrid_10trials_v4/trial_01_sandboxrc_0_reward_0.000_taskcompleted_0`

### Clean Orange Juice

- Success:
  - `outputs/object_swap_task9_clean_10trials/libero_object_swap/pick_up_the_orange_juice_and_place_it_in_the_basket/openai_openai_gpt-5.4/run/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1`

### Clean Salad Dressing

- Success:
  - `outputs/object_swap_task2_clean_10trials/libero_object_swap/pick_up_the_salad_dressing_and_place_it_in_the_basket/openai_openai_gpt-5.4/run/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1`

### Clean Butter

- Success:
  - `outputs/object_swap_task6_clean_10trials/libero_object_swap/pick_up_the_butter_and_place_it_in_the_basket/openai_openai_gpt-5.4/run/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1`

### Direct Helper Ablations

- Orange juice, alias-fixed helper:
  - `outputs/openpi_helper_eval_task9_short_targetfix`
- Orange juice, longer `5/5` helper burst:
  - `outputs/openpi_helper_eval_task9_long`
- Butter, alias-fixed no-stage helper `5` trials:
  - `outputs/openpi_helper_eval_task6_nostage_5trials_aliasfix_long`

## Current Interpretation

- On the current targeted reruns, no hybrid configuration is yet better than both pure VLA and clean CaP-X.
- Butter is the best current hillclimb target because direct VLA is already strong there (`9/20`).
- The main remaining issue is not just prompt wording. It is helper structure:
  - OpenPI itself can work on butter.
  - The current helper / hybrid path still degrades that policy.
- The next change should be a butter-specific helper pattern:
  - direct-first OpenPI
  - reobserve
  - optional single aligned retry
  - only then explicit fallback

## Next 20-Trial Matrix

The next intended comparisons are:

- Butter:
  - pure VLA
  - clean CaP-X
  - helper / hybrid variants, starting with a direct-first butter-specific helper
- Optional follow-up only if butter stalls:
  - orange juice clean vs hybrid
  - spatial-swap task `1`

## Example 20-Trial Commands

### Orange Juice

```bash
.venv-libero/bin/python3 capx/envs/scripts/run_openpi_libero_eval.py \
  --task-suite-name libero_object_swap \
  --task-id 9 \
  --num-trials-per-task 20 \
  --port 8000

.venv-libero/bin/python3 capx/envs/scripts/run_libero_batch.py \
  --args.base-config-path env_configs/libero/hillclimb_object_swap_0_clean_fast.yaml \
  --args.suites libero_object_swap \
  --args.task-id-start 9 \
  --args.task-id-end 9 \
  --args.models openai/openai/gpt-5.4 \
  --args.server-url http://127.0.0.1:8110/chat/completions \
  --args.max-tokens 4096 \
  --args.reasoning-effort low \
  --args.total-trials 20 \
  --args.output-dir ./outputs/object_swap_task9_clean_20trials

.venv-libero/bin/python3 capx/envs/launch.py \
  --config-path env_configs/libero/hillclimb_object_swap_9_vla_minimal_v4.yaml \
  --model openai/openai/gpt-5.4 \
  --server-url http://127.0.0.1:8110/chat/completions \
  --visual-differencing-model openai/openai/gpt-5.4 \
  --visual-differencing-model-server-url http://127.0.0.1:8110/chat/completions \
  --max-tokens 4096 \
  --reasoning-effort low \
  --total-trials 20 \
  --output-dir ./outputs/object_swap_task9_hybrid_20trials_v4
```

### Salad Dressing

```bash
.venv-libero/bin/python3 capx/envs/scripts/run_openpi_libero_eval.py \
  --task-suite-name libero_object_swap \
  --task-id 2 \
  --num-trials-per-task 20 \
  --port 8000

.venv-libero/bin/python3 capx/envs/scripts/run_libero_batch.py \
  --args.base-config-path env_configs/libero/hillclimb_object_swap_0_clean_fast.yaml \
  --args.suites libero_object_swap \
  --args.task-id-start 2 \
  --args.task-id-end 2 \
  --args.models openai/openai/gpt-5.4 \
  --args.server-url http://127.0.0.1:8110/chat/completions \
  --args.max-tokens 4096 \
  --args.reasoning-effort low \
  --args.total-trials 20 \
  --args.output-dir ./outputs/object_swap_task2_clean_20trials

.venv-libero/bin/python3 capx/envs/launch.py \
  --config-path env_configs/libero/hillclimb_object_swap_2_vla_minimal_v1.yaml \
  --model openai/openai/gpt-5.4 \
  --server-url http://127.0.0.1:8110/chat/completions \
  --visual-differencing-model openai/openai/gpt-5.4 \
  --visual-differencing-model-server-url http://127.0.0.1:8110/chat/completions \
  --max-tokens 4096 \
  --reasoning-effort low \
  --total-trials 20 \
  --output-dir ./outputs/object_swap_task2_hybrid_20trials_v1
```

### Butter

```bash
.venv-libero/bin/python3 capx/envs/scripts/run_openpi_libero_eval.py \
  --task-suite-name libero_object_swap \
  --task-id 6 \
  --num-trials-per-task 20 \
  --port 8000

.venv-libero/bin/python3 capx/envs/scripts/run_libero_batch.py \
  --args.base-config-path env_configs/libero/hillclimb_object_swap_0_clean_fast.yaml \
  --args.suites libero_object_swap \
  --args.task-id-start 6 \
  --args.task-id-end 6 \
  --args.models openai/openai/gpt-5.4 \
  --args.server-url http://127.0.0.1:8110/chat/completions \
  --args.max-tokens 4096 \
  --args.reasoning-effort low \
  --args.total-trials 20 \
  --args.output-dir ./outputs/object_swap_task6_clean_20trials

.venv-libero/bin/python3 capx/envs/launch.py \
  --config-path env_configs/libero/hillclimb_object_swap_6_vla_minimal_v4.yaml \
  --model openai/openai/gpt-5.4 \
  --server-url http://127.0.0.1:8110/chat/completions \
  --visual-differencing-model openai/openai/gpt-5.4 \
  --visual-differencing-model-server-url http://127.0.0.1:8110/chat/completions \
  --max-tokens 4096 \
  --reasoning-effort low \
  --total-trials 20 \
  --output-dir ./outputs/object_swap_task6_hybrid_20trials_v4
```
