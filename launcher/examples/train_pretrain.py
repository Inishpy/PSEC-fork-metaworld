import os
import sys
import time
import json
import logging

# Ensure project root is in sys.path for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from examples.states.train_diffusion_psec import call_main

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

    logging.info("========== Starting Meta-World Pretrain Loop ==========")
    try:
        for idx, env_name in enumerate(env_names):
            # Create a subdirectory for each task
            task_dir = os.path.join(base_results_dir, env_name)
            os.makedirs(task_dir, exist_ok=True)

            # Prepare details dict for this task
            details = constant_parameters.copy()
            details['env_name'] = (idx, env_name)
            details['results_dir'] = task_dir  # Pass the directory for saving outputs

            # Save config for this task
            config_path = os.path.join(task_dir, "config.json")
            with open(config_path, "w") as f:
                json.dump(details, f, indent=4)
            logging.info(f"Config saved for task {env_name} (idx={idx}) at {config_path}")


            print("details", details)
            # Call the main training routine for this task
            logging.info(f"=== [Task {idx}] Starting training for {env_name} ===")
            try:
                call_main(details)
                logging.info(f"=== [Task {idx}] Finished training for {env_name} ===")
                
            except Exception as e:
                logging.error(f"Exception during training for task {env_name} (idx={idx}): {e}", exc_info=True)
            
    except Exception as main_e:
        logging.critical(f"Fatal error in main training loop: {main_e}", exc_info=True)
    logging.info("========== Finished Meta-World Pretrain Loop ==========")

if __name__ == '__main__':
    main()
