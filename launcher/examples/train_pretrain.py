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
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

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
avg_embs = dict()

def create_graph(emb_dict, save_path=None):
    """
    Plots tasks in 2D based on cosine similarity to a reference embedding.
    Args:
        emb_dict: dict of {task_name: np.ndarray}
        save_path: if provided, saves the plot to this path
    """
    if not emb_dict:
        print("No embeddings to plot.")
        return

    # Pick the first task as reference
    ref_task, ref_emb = next(iter(emb_dict.items()))
    ref_emb = ref_emb / np.linalg.norm(ref_emb)

    similarities = {}
    for task, emb in emb_dict.items():
        emb_norm = emb / np.linalg.norm(emb)
        sim = np.dot(ref_emb, emb_norm)
        similarities[task] = sim

    # Sort by similarity for better visualization
    sorted_tasks = sorted(similarities.items(), key=lambda x: -x[1])

    # Place reference at (0,0), others on a circle with radius = 1 - similarity
    angles = np.linspace(0, 2 * np.pi, len(sorted_tasks), endpoint=False)
    xs, ys, labels, colors = [], [], [], []
    for i, (task, sim) in enumerate(sorted_tasks):
        if task == ref_task:
            xs.append(0)
            ys.append(0)
            labels.append(task)
            colors.append('red')
        else:
            r = 1 - sim  # closer if similarity is high
            x = r * np.cos(angles[i])
            y = r * np.sin(angles[i])
            xs.append(x)
            ys.append(y)
            labels.append(task)
            colors.append('blue')

    plt.figure(figsize=(10, 10))
    plt.scatter(xs, ys, c=colors)
    for x, y, label in zip(xs, ys, labels):
        plt.text(x, y, label, fontsize=8, ha='right' if x < 0 else 'left', va='bottom')
    plt.title(f"Task Embeddings: Cosine Similarity to '{ref_task}'")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis('equal')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Embedding similarity plot saved to {save_path}")
    else:
        plt.show()

def plot_tsne(emb_dict, save_path=None):
    """
    Plots a t-SNE projection of the embeddings.
    Args:
        emb_dict: dict of {task_name: np.ndarray}
        save_path: if provided, saves the plot to this path
    """
    if not emb_dict:
        print("No embeddings to plot.")
        return

    labels = list(emb_dict.keys())
    X = np.stack([emb_dict[k] for k in labels])
    # Perplexity must be less than n_samples
    n_samples = X.shape[0]
    perplexity = min(30, max(2, (n_samples - 1) // 2))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    X_proj = tsne.fit_transform(X)

    plt.figure(figsize=(10, 10))
    plt.scatter(X_proj[:, 0], X_proj[:, 1], c='blue')
    for i, label in enumerate(labels):
        plt.text(X_proj[i, 0], X_proj[i, 1], label, fontsize=8, ha='right' if X_proj[i, 0] < 0 else 'left', va='bottom')
    plt.title("Task Embeddings: t-SNE Projection")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"t-SNE plot saved to {save_path}")
    else:
        plt.show()

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
        max_steps=20000,
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
            # if idx >= 3:
            #     break
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
                avg_embedding = call_main(details)
                avg_embs[env_name] = avg_embedding
                logging.info(f"=== [Task {idx}] Finished training for {env_name} ===")
                
            except Exception as e:
                logging.error(f"Exception during training for task {env_name} (idx={idx}): {e}", exc_info=True)
            
    except Exception as main_e:
        logging.critical(f"Fatal error in main training loop: {main_e}", exc_info=True)
    logging.info("========== Finished Meta-World Pretrain Loop ==========")
    # Plot embedding similarities if available
    if avg_embs:
        # Save embeddings dictionary for later use
        emb_save_path = os.path.join(base_results_dir, "embeddings_dict.npz")
        # Convert to arrays for saving
        np.savez(emb_save_path, **avg_embs)
        print(f"Embeddings dictionary saved to {emb_save_path}")
        # Save plot
        save_path = os.path.join(base_results_dir, "embedding_similarity.png")
        create_graph(avg_embs, save_path=save_path)
        # Save t-SNE plot
        tsne_path = os.path.join(base_results_dir, "embedding_tsne.png")
        plot_tsne(avg_embs, save_path=tsne_path)
        # To load later: data = np.load(emb_save_path); emb_dict = {k: data[k] for k in data}

if __name__ == '__main__':
    main()
