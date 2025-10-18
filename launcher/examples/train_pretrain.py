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
    parser = argparse.ArgumentParser(description="Pretrain script with seed option")
    parser.add_argument('--seed', type=int, default=0, help="Random seed for training")
    args = parser.parse_args()

    info = 'pretrain'
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    base_results_dir = f"./results/{info}/{timestamp}"
    os.makedirs(base_results_dir, exist_ok=True)

    # Constant parameters for all tasks
    constant_parameters = dict(
        project='PSEC',
        group='metaworld',
        experiment_name='ddpm_lora',
        max_steps=20000,
        batch_size=4096,
        eval_episodes=10,
        log_interval=1000,
        save_steps=19999,
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
            model_cls='PretrainWithComposition',
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
        seed=args.seed,
        timestamp=timestamp,
    )

    logging.info("========== Starting Meta-World Pretrain Loop ==========")
    try:
        for idx, (env_name, exclude_list) in enumerate(env_names.items()):
            # For each task, determine all possible priors (all other tasks except itself and its exclusion list)
            all_possible_priors = [k for k in env_names.keys() if k != env_name and k not in exclude_list]
            for kept_prior in all_possible_priors:
                # Exclude all priors except the one being kept
                exclude_for_this_run = [k for k in env_names.keys() if k != kept_prior and k != env_name]
                for seed in range(4):
                    # Check if prior model exists for kept_prior
                    prior_model_dir = os.path.join("./results/pretrain/20251009-015311", kept_prior)
                    prior_model_exists = False
                    if os.path.isdir(prior_model_dir):
                        model_files = [f for f in os.listdir(prior_model_dir) if f.startswith('model') and f.endswith('.pickle')]
                        if len(model_files) > 0:
                            prior_model_exists = True
                    if not prior_model_exists:
                        logging.warning(f"Skipping training for task {env_name} with kept_prior={kept_prior}, seed={seed} because no prior model found in {prior_model_dir}")
                        continue

                    # Create a subdirectory for each (task, prior, seed)
                    run_dir = os.path.join(base_results_dir, env_name, f"kept_prior_{kept_prior}", f"seed_{seed}")
                    os.makedirs(run_dir, exist_ok=True)

                    # Prepare details dict for this run
                    details = constant_parameters.copy()
                    details['env_name'] = (idx, env_name)
                    details['results_dir'] = run_dir  # Pass the directory for saving outputs
                    details['exclude_tasks'] = exclude_for_this_run
                    details['kept_prior'] = kept_prior
                    details['seed'] = seed

                    # Save config for this run
                    config_path = os.path.join(run_dir, "config.json")
                    with open(config_path, "w") as f:
                        json.dump(details, f, indent=4)
                    logging.info(f"Config saved for task {env_name} (idx={idx}), kept_prior={kept_prior}, seed={seed} at {config_path}")

                    print("details", details)
                    # Call the main training routine for this run
                    logging.info(f"=== [Task {idx}] Starting training for {env_name} with kept_prior={kept_prior}, seed={seed} ===")
                    try:
                        call_main(details)
                        logging.info(f"=== [Task {idx}] Finished training for {env_name} with kept_prior={kept_prior}, seed={seed} ===")
                        import gc
                        import jax
                        gc.collect()
                        jax.clear_caches()
                    except Exception as e:
                        logging.error(f"Exception during training for task {env_name} (idx={idx}), kept_prior={kept_prior}, seed={seed}: {e}", exc_info=True)

            
            
    except Exception as main_e:
        logging.critical(f"Fatal error in main training loop: {main_e}", exc_info=True)
    logging.info("========== Finished Meta-World Pretrain Loop ==========")

if __name__ == '__main__':
    main()
