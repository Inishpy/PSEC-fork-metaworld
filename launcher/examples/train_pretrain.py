# train_pretrain.py (updated to enumerate Meta-World tasks and call call_main per task)
import os
import numpy as np
from absl import app, flags
import sys
current_work_path = os.getcwd()
sys.path.insert(0, current_work_path)
from examples.states.train_diffusion_psec import call_main
from launcher.hyperparameters import set_hyperparameters
from jax import config
from ml_collections import config_flags, ConfigDict
import json
import time
import gymnasium as gym
import metaworld

import os
from dotenv import load_dotenv

# This loads the environment variables from .env into os.environ
load_dotenv()  

# Now you can access them
api_key = os.getenv("WANDB_API_KEY")
env_names = ['assembly-v3', 'basketball-v3', 'bin-picking-v3', 'box-close-v3', 'button-press-topdown-v3', 'button-press-topdown-wall-v3', 'button-press-v3', 'button-press-wall-v3', 'coffee-button-v3', 'coffee-pull-v3', 'coffee-push-v3', 'dial-turn-v3', 'disassemble-v3', 'door-close-v3', 'door-lock-v3', 'door-open-v3', 'door-unlock-v3', 'drawer-close-v3', 'drawer-open-v3', 'faucet-close-v3', 'faucet-open-v3', 'hammer-v3', 'hand-insert-v3', 'handle-press-side-v3', 'handle-press-v3', 'handle-pull-side-v3', 'handle-pull-v3', 'lever-pull-v3', 'peg-insert-side-v3', 'peg-unplug-side-v3', 'pick-out-of-hole-v3', 'pick-place-v3', 'pick-place-wall-v3', 'plate-slide-back-side-v3', 'plate-slide-back-v3', 'plate-slide-side-v3', 'plate-slide-v3', 'push-back-v3', 'push-v3', 'push-wall-v3', 'reach-v3', 'reach-wall-v3', 'shelf-place-v3', 'soccer-v3', 'stick-pull-v3', 'stick-push-v3', 'sweep-into-v3', 'sweep-v3', 'window-close-v3', 'window-open-v3']   

# for debug
# config.update('jax_disable_jit', True)
os.environ["WANDB_MODE"] = "offline"

FLAGS = flags.FLAGS
flags.DEFINE_integer('variant', 1, 'Which variant index to run (0..N-1).')
flags.DEFINE_integer('seed', 0, 'Choose seed')
config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

def to_dict(config):
    if isinstance(config, ConfigDict):
        return {k: to_dict(v) for k, v in config.items()}
    return config

def discover_metaworld_tasks(seed=0):
    """
    Create a temporary MT50 vec env and read its .ids to discover task names.
    Returns a list of task ids like 'reach-v3', 'push-v3', ...
    """
    try:
        vec = gym.make_vec('Meta-World/MT50', vector_strategy='sync', seed=seed)
        
        ids = getattr(vec, 'ids', None)
        
        if ids is not None and len(ids) > 0:
            # ids are usually of the form 'reach-v3', 'push-v3', ...
            return list(ids)
        else:
            
            # fallback: try MT10 or ML45 if MT50 doesn't expose ids
            vec = gym.make_vec('Meta-World/MT10', vector_strategy='sync', seed=seed)
            ids = getattr(vec, 'ids', None)
         
            if ids:
                return list(ids)
    except Exception as e:
        print("Warning: could not auto-discover Meta-World task IDs:", e)
    # fallback list (small subset) if discovery fails:
    return [
        "reach-v3", "push-v3", "pick-place-v3", "open-drawer-v3", "open-door-v3",
        "close-drawer-v3", "press-button-topdown-v3", "insert-peg-v3", "open-window-v3", "open-box-v3"
    ]


def main(_):
    info = 'Pretrain'
    timestamp = f'{info}'
    # Base parameters
    constant_parameters = dict(project='PSEC',
                               group='metaworld',
                               experiment_name='ddpm_lora',
                               timestamp=timestamp,
                               max_steps=1,
                               batch_size=2048,
                               eval_episodes=10,
                               log_interval=1000,
                               save_steps=250000,
                               eval_interval=250000,
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
                                M=1,),
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
                               )

    # Discover Meta-World tasks programmatically
    task_list = discover_metaworld_tasks(seed=FLAGS.seed)
    print(f"Discovered {len(task_list)} Meta-World tasks (using MT50/MT10 discovery).")

    # We'll run one variant per task (single-task MT1 runs).
    sweep_parameters = dict(seed=[FLAGS.seed],
                            # we will set env_name to each discovered task id
                            env_name=task_list,
                            )

    # assemble variants
    variants = [constant_parameters]
    name_keys = ['env_name','experiment_name']
    variants = set_hyperparameters(sweep_parameters, variants, name_keys)

    inference_sweep_parameters = dict(
                            N=[1],
                            clip_sampler=[True],
                            M=[0],
                            )

    inference_variants = [{}]
    inference_variants = set_hyperparameters(inference_sweep_parameters, inference_variants)

    filtered_variants = []
    for variant in variants:
        # When running per task we set benchmark to MT1 for single-task training
        variant['benchmark'] = 'Meta-World/MT1'
        variant['inference_variants'] = inference_variants
        filtered_variants.append(variant)

    print(f"Prepared {len(filtered_variants)} variants (one per task).")
    variant_idx = FLAGS.variant
    if variant_idx < 0 or variant_idx >= len(filtered_variants):
        raise ValueError(f"variant index {variant_idx} out of range (0..{len(filtered_variants)-1})")

    variant = filtered_variants[variant_idx]
    variant['seed'] = FLAGS.seed

    # ensure directories
    if not os.path.exists(f"./results/{info}/{variant['timestamp']}_{variant['group']}/{variant['experiment_name']}"):
        os.makedirs(f"./results/{info}/{variant['timestamp']}_{variant['group']}/{variant['experiment_name']}", exist_ok=True)
    if not os.path.exists(f"./results/{info}/{variant['timestamp']}_{variant['group']}/{variant['experiment_name']}_bc"):
        os.makedirs(f"./results/{info}/{variant['timestamp']}_{variant['group']}/{variant['experiment_name']}_bc", exist_ok=True)

    with open(f"./results/{info}/{variant['timestamp']}_{variant['group']}/{variant['experiment_name']}/config.json", "w") as f:
        json.dump(to_dict(variant), f, indent=4)
    with open(f"./results/{info}/{variant['timestamp']}_{variant['group']}/{variant['experiment_name']}_bc/config.json", "w") as f:
        json.dump(to_dict(variant), f, indent=4)

    # call the main training routine (single task, MT1 style)
    print(variant, "variant")

    import copy
    for idx, env_name in enumerate(env_names):
        variant_copy = copy.deepcopy(variant)
        variant_copy["env_name"] = (idx, env_name)
        call_main(variant_copy)


if __name__ == '__main__':
    app.run(main)
