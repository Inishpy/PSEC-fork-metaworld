import os
import sys
import time
import json
import logging
import multiprocessing as mp

# Ensure project root is in sys.path for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# TensorBoard is now used in train_diffusion_psec.py, no wandb import needed here

import gymnasium as gym
import metaworld

from dotenv import load_dotenv
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Meta-World task names
env_names = [
    'assembly-v3', 'basketball-v3', 'bin-picking-v3', 'box-close-v3', 'button-press-topdown-v3',
    'button-press-topdown-wall-v3', 'button-press-v3', 'button-press-wall-v3', 'coffee-button-v3',
    'coffee-pull-v3', 'coffee-push-v3', 'dial-turn-v3', 'disassemble-v3', 'door-close-v3',
    'door-lock-v3', 'door-open-v3', 'door-unlock-v3', 'drawer-close-v3', 'drawer-open-v3',
    'faucet-close-v3', 'faucet-open-v3', 'hammer-v3', 'hand-insert-v3', 'handle-press-side-v3',
    'handle-press-v3', 'handle-pull-side-v3', 'handle-pull-v3', 'lever-pull-v3', 'peg-insert-side-v3',
    'peg-unplug-side-v3', 'pick-out-of-hole-v3', 'pick-place-v3', 'pick-place-wall-v3',
    'plate-slide-back-side-v3', 'plate-slide-back-v3', 'plate-slide-side-v3', 'plate-slide-v3',
    'push-back-v3', 'push-v3', 'push-wall-v3', 'reach-v3', 'reach-wall-v3', 'shelf-place-v3',
    'soccer-v3', 'stick-pull-v3', 'stick-push-v3', 'sweep-into-v3', 'sweep-v3', 'window-close-v3',
    'window-open-v3'
]

def run_task(idx, env_name, base_results_dir, constant_parameters, gpu_id):
    """
    Spawned worker that pins a single GPU and launches training for one task.
    The import of call_main happens *after* CUDA visibility is set to ensure
    JAX enumerates only the chosen device in this process.
    """
    # Pin GPU for this process before importing JAX
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Avoid preallocating all memory on the device
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    from examples.states.train_diffusion_psec import call_main

    task_dir = os.path.join(base_results_dir, env_name)
    os.makedirs(task_dir, exist_ok=True)

    details = constant_parameters.copy()
    details['env_name'] = (idx, env_name)
    details['results_dir'] = task_dir

    config_path = os.path.join(task_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(details, f, indent=4)
    logging.info(f"[GPU {gpu_id}] Config saved for task {env_name} (idx={idx}) at {config_path}")

    try:
        logging.info(f"[GPU {gpu_id}] === [Task {idx}] Starting training for {env_name} ===")
        call_main(details)
        logging.info(f"[GPU {gpu_id}] === [Task {idx}] Finished training for {env_name} ===")
    except Exception as e:
        logging.error(f"[GPU {gpu_id}] Exception during training for task {env_name} (idx={idx}): {e}", exc_info=True)
        raise

def main():
    info = 'pretrain'
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    base_results_dir = f"./results/{info}/{timestamp}"
    os.makedirs(base_results_dir, exist_ok=True)

    # Constant parameters for all tasks
    constant_parameters = dict(
        project='PSEC',
        group='metaworld',
        experiment_name='ddpm_lora',
        max_steps=2000000,
        batch_size=4096,
        eval_episodes=10,
        log_interval=1000,
        save_steps=1999999,
        eval_interval=4000,
        save_video=False,
        filter_threshold=None,
        take_top=None,
        online_max_steps=0,
        unsquash_actions=False,
        normalize_returns=True,
        ratio=1.0,
        training_time_inference_params=dict(
            N=64,
            clip_sampler=True,
            M=1,
        ),
        rl_config=dict(
            model_cls='Pretrain',
            actor_lr=3e-4,
            T=5,
            N=64,
            M=0,
            actor_dropout_rate=0.1,
            actor_num_blocks=3,
            decay_steps=int(3e6),
            actor_layer_norm=True,
            actor_tau=0.001,
            beta_schedule='vp',
        ),
        benchmark='Meta-World/MT1',
        inference_variants=[dict(N=1, clip_sampler=True, M=0)],
        seed=0,
        timestamp=timestamp,
    )

    # GPU scheduling
    available_gpus = [1]#, 1]
    per_gpu_limit = 1
    max_concurrent = per_gpu_limit * len(available_gpus)

    # Spawn strategy: keep up to max_concurrent processes alive; each worker pins to a GPU.
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # start method already set
        pass
    ctx = mp.get_context("spawn")

    pending = [(idx, env_name) for idx, env_name in enumerate(env_names)]
    active = []  # list of (process, args)
    active_counts = {gpu: 0 for gpu in available_gpus}

    logging.info("========== Starting Meta-World Pretrain Loop (parallel) ==========")

    def launch_next():
        """Launch the next pending task on a GPU with available capacity."""
        if not pending:
            return False

        # Pick the GPU with the lowest load that still has capacity
        candidate_gpus = sorted(available_gpus, key=lambda g: active_counts[g])
        target_gpu = None
        for gpu in candidate_gpus:
            if active_counts[gpu] < per_gpu_limit:
                target_gpu = gpu
                break
        if target_gpu is None:
            return False

        idx, env_name = pending.pop(0)
        args = (idx, env_name, base_results_dir, constant_parameters, target_gpu)
        p = ctx.Process(target=run_task, args=args, daemon=False)
        p.start()
        active.append((p, args))
        active_counts[target_gpu] += 1
        logging.info(f"Launched task {env_name} (idx={idx}) on GPU {target_gpu} with PID {p.pid}")
        return True

    # Prime the pool
    while len(active) < max_concurrent and pending:
        if not launch_next():
            break

    # Monitor and keep launching until done
    while active:
        time.sleep(1.0)
        still_running = []
        for proc, args in active:
            idx, env_name, _, _, gpu_id = args
            if proc.is_alive():
                still_running.append((proc, args))
                continue
            active_counts[gpu_id] = max(0, active_counts[gpu_id] - 1)
            exit_code = proc.exitcode
            if exit_code != 0:
                logging.error(f"Task {env_name} (idx={idx}) on GPU {gpu_id} exited with code {exit_code}")
            else:
                logging.info(f"Task {env_name} (idx={idx}) on GPU {gpu_id} completed successfully")
        active = still_running

        # Fill available slots (respect per-GPU limits)
        while len(active) < max_concurrent and pending:
            if not launch_next():
                break

    logging.info("========== Finished Meta-World Pretrain Loop (parallel) ==========")

if __name__ == '__main__':
    main()
