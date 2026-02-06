import os
import sys
import json
import argparse
import logging
import numpy as np

# Ensure project root is in sys.path for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from examples.load_model import load_diffusion_model
from jaxrl5.evaluation_dsrl import evaluate_bc

import gymnasium as gym
import metaworld
from dotenv import load_dotenv

load_dotenv()

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
tasks_easy = [
    "button-press-topdown-v3",
    "button-press-topdown-wall-v3",
    "button-press-v3",
    "button-press-wall-v3",
    "coffee-button-v3",
    "dial-turn-v3",
    "disassemble-v3",
    "door-lock-v3",
    "door-open-v3",
    "door-unlock-v3",
    "drawer-close-v3",
    "drawer-open-v3",
    "faucet-close-v3",
    "faucet-open-v3",
    "hammer-v3",
    "hand-insert-v3",
    "handle-press-side-v3",
    "handle-press-v3",
    "peg-insert-side-v3",
    "peg-unplug-side-v3",
    "pick-out-of-hole-v3",
    "push-wall-v3",
    "reach-v3",
    "reach-wall-v3",
    "shelf-place-v3",
    "sweep-into-v3",
    "sweep-v3",
    "window-close-v3",
    "window-open-v3"
]

task_medium = [
    "coffee-push-v3",  #hardest in list
    "pick-place-wall-v3",
    "soccer-v3",
    "handle-pull-side-v3",
    "stick-pull-v3",
    "push-back-v3"
]


def create_env(details):
    import gymnasium as gym
    logging.info("[ENV] Creating environment for: %s", str(details.get('env_name')))
   
    
    try:
        if 'benchmark' in details and details['benchmark'].startswith('Meta-World'):
            benchmark = details['benchmark']
            env_name = details.get('env_name', None)
            if env_name is None:
                raise ValueError("For Meta-World runs you must set details['env_name'] to a task name like 'reach-v3'.")
            logging.info("[ENV] Using Meta-World benchmark: %s, task: %s", benchmark, env_name)
            env = gym.make(benchmark, env_name=env_name, seed=details.get('seed', None))
        else:
            env = gym.make(details['env_name'])
    except TypeError:
        env = gym.make(details['env_name'])

    env_max_steps = getattr(env.unwrapped, '_max_episode_steps', None)
    if env_max_steps is None:
        env_max_steps = getattr(env, 'max_episode_steps', None)
    if env_max_steps is None:
        env_max_steps = 150
    logging.info("[ENV] Created environment: %s, max_steps: %s", str(details.get('env_name')), str(env_max_steps))
    return env, env_max_steps

def main():
    parser = argparse.ArgumentParser(description="Evaluate pretrained models trained on tasks_easy against task_medium envs.")
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Path to the results directory containing subdirectories for each task (each with config.json and model checkpoints)."
    )
    parser.add_argument(
        "--pickle_ext",
        type=str,
        default=".pickle",
        help="Extension of the model checkpoint files (default: .pickle)."
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=None,
        help="Number of evaluation episodes (overrides config if set)."
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment during evaluation."
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Path to write cross-eval results JSON (default: <results_dir>/cross_eval_results.json)."
    )
    args = parser.parse_args()

    results_log = {}
    num_seeds = 5

    for source_env in tasks_easy:
        subdir = os.path.join(args.results_dir, source_env)
        config_path = os.path.join(subdir, "config.json")
        if not os.path.exists(config_path):
            logging.warning(f"[{source_env}] Config file not found at {config_path}, skipping.")
            continue

        with open(config_path, "r") as f:
            details = json.load(f)

        for target_env in tasks_easy:
            seed_results = []

            for seed in range(num_seeds):
                details_eval = details.copy()
                details_eval['env_name'] = target_env
                details_eval['seed'] = seed

                try:
                    env, _ = create_env(details_eval)
                except Exception as e:
                    logging.error(f"[{source_env} -> {target_env} | seed {seed}] Failed to create environment: {e}")
                    continue

                try:
                    _, agent = load_diffusion_model(subdir, args.pickle_ext, env)
                except Exception as e:
                    logging.error(f"[{source_env} -> {target_env} | seed {seed}] Failed to load model: {e}")
                    try:
                        env.close()
                    except Exception:
                        pass
                    continue

                eval_episodes = args.num_episodes if args.num_episodes is not None else details_eval.get("eval_episodes", 10)

                logging.info(f"[{source_env} -> {target_env} | seed {seed}] Evaluating agent from {subdir} for {eval_episodes} episodes...")
                try:
                    results = evaluate_bc(agent, env, eval_episodes, train_lora=False)
                    logging.info(f"[{source_env} -> {target_env} | seed {seed}] Evaluation results: {results}")
                    print(f"[{source_env} -> {target_env} | seed {seed}] Evaluation results: {results}")

                    seed_results.append({"seed": seed, **results})
                except Exception as e:
                    logging.error(f"[{source_env} -> {target_env} | seed {seed}] Evaluation failed: {e}")
                finally:
                    try:
                        env.close()
                    except Exception:
                        pass

            if not seed_results:
                logging.warning(f"[{source_env} -> {target_env}] No successful evaluations across seeds, skipping aggregation.")
                continue

            metric_keys = [k for k in seed_results[0].keys() if k != "seed"]
            avg_results = {k: float(np.mean([r[k] for r in seed_results])) for k in metric_keys}
            seed_count = len(seed_results)
            logging.info(f"[{source_env} -> {target_env}] Average over {seed_count} seeds: {avg_results}")
            print(f"[{source_env} -> {target_env}] Average over {seed_count} seeds: {avg_results}")

            results_log.setdefault(source_env, {})[target_env] = {
                "seeds": seed_results,
                "average": avg_results
            }

    # Persist aggregated results
    output_path = args.output_json or os.path.join(args.results_dir, "cross_eval_results.json")
    try:
        with open(output_path, "w") as f:
            json.dump(results_log, f, indent=2, default=float)
        logging.info(f"[SAVE] Cross-eval results written to {output_path}")
        print(f"[SAVE] Cross-eval results written to {output_path}")
    except Exception as e:
        logging.error(f"[SAVE] Failed to write results to {output_path}: {e}")

if __name__ == "__main__":
    main()
