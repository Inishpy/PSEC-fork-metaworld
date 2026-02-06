
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
# Meta-World task exclusion dictionary: key = task, value = list of tasks to exclude as priors
# env_names = {
#     'assembly-v3': ['disassemble-v3'],
#     'basketball-v3': ['peg-unplug-side-v3'],
#     'bin-picking-v3': ['sweep-v3'],
#     'box-close-v3': ['disassemble-v3'],
#     'button-press-topdown-v3': ['button-press-topdown-wall-v3'],
#     'button-press-topdown-wall-v3': ['button-press-topdown-v3'],
#     'button-press-v3': ['drawer-open-v3'],
#     'button-press-wall-v3': ['faucet-open-v3'],
#     'coffee-button-v3': ['button-press-v3'],
#     'coffee-pull-v3': ['push-wall-v3'],
#     'coffee-push-v3': ['push-v3'],
#     'dial-turn-v3': ['peg-unplug-side-v3'],
#     'disassemble-v3': ['pick-out-of-hole-v3'],
#     'door-close-v3': ['faucet-open-v3'],
#     'door-lock-v3': ['button-press-v3'],
#     'door-open-v3': ['dial-turn-v3'],
#     'door-unlock-v3': ['faucet-open-v3'],
#     'drawer-close-v3': ['push-v3'],
#     'drawer-open-v3': ['peg-unplug-side-v3'],
#     'faucet-close-v3': ['hand-insert-v3'],
#     'faucet-open-v3': ['door-unlock-v3'],
#     'hammer-v3': ['dial-turn-v3'],
#     'hand-insert-v3': ['faucet-open-v3'],
#     'handle-press-side-v3': ['handle-press-v3'],
#     'handle-press-v3': ['handle-press-v3'],
#     'handle-pull-side-v3': ['push-v3'],
#     'handle-pull-v3': ['push-v3'],
#     'lever-pull-v3': ['coffee-button-v3'],
#     'peg-insert-side-v3': ['shelf-place-v3'],
#     'peg-unplug-side-v3': ['sweep-v3'],
#     'pick-out-of-hole-v3': ['peg-unplug-side-v3'],
#     'pick-place-v3': ['push-v3'],
#     'pick-place-wall-v3': ['push-v3'],
#     'plate-slide-back-side-v3': ['push-back-v3'],
#     'plate-slide-back-v3': ['push-back-v3'],
#     'plate-slide-side-v3': ['push-v3'],
#     'plate-slide-v3': ['push-wall-v3'],
#     'push-back-v3': ['push-v3'],
#     'push-v3': ['push-v3'],
#     'push-wall-v3': ['push-wall-v3'],
#     'reach-v3': ['reach-v3'],
#     'reach-wall-v3': ['reach-wall-v3'],
#     'shelf-place-v3': ['shelf-place-v3'],
#     'soccer-v3': ['soccer-v3'],
#     'stick-pull-v3': ['stick-pull-v3'],
#     'stick-push-v3': ['stick-push-v3'],
#     'sweep-into-v3': ['sweep-into-v3'],
#     'sweep-v3': ['sweep-v3'],
#     'window-close-v3': ['window-close-v3'],
#     'window-open-v3': ['window-open-v3'],
# }

env_names = {
    'assembly-v3': [''],
    'basketball-v3': [''],
    'bin-picking-v3': [''],
    'box-close-v3': [''],
    'button-press-topdown-v3': [''],
    'button-press-topdown-wall-v3': [''],
    'button-press-v3': [''],
    'button-press-wall-v3': [''],
    'coffee-button-v3': [''],
    'coffee-pull-v3': [''],
    'coffee-push-v3': [''],
    'dial-turn-v3': [''],
    'disassemble-v3': [''],
    'door-close-v3': [''],
    'door-lock-v3': [''],
    'door-open-v3': [''],
    'door-unlock-v3': [''],
    'drawer-close-v3': [''],
    'drawer-open-v3': [''],
    'faucet-close-v3': [''],
    'faucet-open-v3': [''],
    'hammer-v3': [''],
    'hand-insert-v3': [''],
    'handle-press-side-v3': [''],
    'handle-press-v3': [''],
    'handle-pull-side-v3': [''],
    'handle-pull-v3': [''],
    'lever-pull-v3': [''],
    'peg-insert-side-v3': [''],
    'peg-unplug-side-v3': [''],
    'pick-out-of-hole-v3': [''],
    'pick-place-v3': [''],
    'pick-place-wall-v3': [''],
    'plate-slide-back-side-v3': [''],
    'plate-slide-back-v3': [''],
    'plate-slide-side-v3': [''],
    'plate-slide-v3': [''],
    'push-back-v3': [''],
    'push-v3': [''],
    'push-wall-v3': [''],
    'reach-v3': [''],
    'reach-wall-v3': [''],
    'shelf-place-v3': [''],
    'soccer-v3': [''],
    'stick-pull-v3': [''],
    'stick-push-v3': [''],
    'sweep-into-v3': [''],
    'sweep-v3': [''],
    'window-close-v3': [''],
    'window-open-v3': [''],
}

def main():
    import argparse
    parser = argparse.ArgumentParser(description="RL script with seed option (no demonstration data)")
    parser.add_argument('--seed', type=int, default=0, help="Random seed for training")
    parser.add_argument('--env_name', type=str, default='reach-v3', help="Meta-World task name, e.g. reach-v3")
    parser.add_argument('--algo', type=str, default='SAC', choices=['SAC', 'TD3', 'IQL', 'REDQ'], help="RL algorithm")
    args = parser.parse_args()

    info = 'rl'
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    base_results_dir = f"./results/{info}/{timestamp}"
    os.makedirs(base_results_dir, exist_ok=True)

    # RL config selection
    if args.algo == 'SAC':
        from examples.states.configs.sac_config import get_config as get_algo_config
    elif args.algo == 'TD3':
        from examples.states.configs.td3_config import get_config as get_algo_config
    elif args.algo == 'IQL':
        from examples.states.configs.iql_mujoco_config import get_config as get_algo_config
    elif args.algo == 'REDQ':
        from examples.states.configs.redq_config import get_config as get_algo_config
    else:
        raise ValueError("Unknown RL algorithm")

    algo_config = get_algo_config()
    # Set env_name in details for Meta-World
    details = dict(
        project='PSEC',
        group='metaworld',
        experiment_name=f'rl_{args.algo.lower()}',
        max_steps=1000000,  # RL usually needs more steps
        batch_size=256,
        eval_episodes=10,
        log_interval=1000,
        save_steps=999999,
        eval_interval=50000,
        save_video=False,
        filter_threshold=None,
        take_top=None,
        online_max_steps=0,
        unsquash_actions=False,
        normalize_returns=True,
        ratio=1.0,
        training_time_inference_params=dict(
            N=1,
            clip_sampler=True,
            M=0,
        ),
        rl_config=algo_config,
        benchmark='Meta-World/MT1',
        inference_variants=[dict(N=1, clip_sampler=True, M=0)],
        seed=args.seed,
        timestamp=timestamp,
        env_name=(0, args.env_name),
        results_dir=base_results_dir,
        
    )

    logging.info("========== Starting Meta-World RL Training ==========")
    try:
        # For RL, just train on the selected task (no priors, no demonstration data)
        details["online_rl"] = True
        call_main(details)
    except Exception as main_e:
        logging.critical(f"Fatal error in RL training loop: {main_e}", exc_info=True)
    logging.info("========== Finished Meta-World RL Training ==========")

if __name__ == '__main__':
    main()
