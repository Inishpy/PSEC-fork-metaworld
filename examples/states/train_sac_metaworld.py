import gym
import numpy as np
import jax
import jax.numpy as jnp
import logging
from tqdm import trange
import time
import os

from jaxrl5.agents.sac.sac_learner import SACLearner
from jaxrl5.data.replay_buffer import ReplayBuffer

try:
    import metaworld
except ImportError:
    raise ImportError("Please install metaworld: pip install metaworld")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def make_metaworld_env(env_name="reach-v3", seed=42):
    """
    Create a MetaWorld environment using gym interface.
    """
    # Use MT1 for single-task, or MT10/MT50 for multi-task
    ml1 = metaworld.ML1(env_name)
    env = ml1.train_classes[env_name]()
    task = ml1.train_tasks[0]
    env.set_task(task)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=env.max_path_length)
    env.seed(seed)
    return env

def main():
    # Configurations
    env_name = "hammer-v3"  # Change to any ML1 task, e.g., "push-v2"
    seed = 42
    max_steps = 1_000_000
    start_steps = 5000
    eval_interval = 6000
    batch_size = 256
    replay_buffer_size = 1_000_000
    log_dir = f"./results/sac_metaworld_{env_name}_{int(time.time())}"
    os.makedirs(log_dir, exist_ok=True)

    # Environment
    env = make_metaworld_env(env_name, seed=seed)
    eval_env = make_metaworld_env(env_name, seed=seed+100)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Replay buffer
    # print("DEBUG: observation_space type:", type(env.observation_space))
    # print("DEBUG: observation_space repr:", repr(env.observation_space))
    replay_buffer = ReplayBuffer(
        observation_space=env.observation_space,
        action_space=env.action_space,
        capacity=replay_buffer_size,
    )

    # SAC agent
    agent = SACLearner.create(
        seed=seed,
        observation_space=env.observation_space,
        action_space=env.action_space,
        actor_lr=3e-4,
        critic_lr=3e-4,
        temp_lr=3e-4,
        hidden_dims=(256, 256),
        discount=0.99,
        tau=0.005,
        num_qs=2,
        num_min_qs=None,
        critic_dropout_rate=None,
        critic_layer_norm=False,
        target_entropy=None,
        init_temperature=1.0,
        backup_entropy=True,
    )

    obs, _ = env.reset() if isinstance(env.reset(), tuple) else (env.reset(), None)
    episode_reward = 0
    episode_length = 0
    episode = 0

    for step in trange(1, max_steps + 1):
        # Sample random actions for initial exploration
        if step < start_steps:
            action = env.action_space.sample()
        else:
            action = np.asarray(
                agent.actor.apply_fn(
                    {"params": agent.actor.params},
                    obs[None]
                ).sample(seed=jax.random.PRNGKey(np.random.randint(1e6)))
            )[0]
        
        # print(env.step(action))
        # next_obs, reward, done, info = env.step(action)
        step_result = env.step(action)
        # Handle both Gym and Gymnasium step API
        if isinstance(step_result, tuple) and len(step_result) == 5:
            next_obs, reward, done, truncations, info = step_result
        elif isinstance(step_result, tuple) and len(step_result) == 4:
            next_obs, reward, done, info = step_result
            truncations = False
        else:
            raise ValueError("Unexpected number of elements returned by env.step()")
        mask = 0.0 if done else 1.0

        # print("DEBUG: obs shape:", np.shape(obs), "type:", type(obs))
        # print("DEBUG: action shape:", np.shape(action), "type:", type(action))
        # print("DEBUG: reward shape:", np.shape(reward), "type:", type(reward))
        # print("DEBUG: next_obs shape:", np.shape(next_obs), "type:", type(next_obs))
        # print("DEBUG: mask shape:", np.shape(mask), "type:", type(mask))
        # print("DEBUG: done shape:", np.shape(done), "type:", type(done))
        replay_buffer.insert({
            'observations': obs,
            'actions': action,
            'rewards': reward,
            'next_observations': next_obs,
            'masks': mask,
            'dones': done,
        })

        obs = next_obs
        # If next_obs is a tuple (obs, info), take only obs
        if isinstance(obs, tuple):
            obs = obs[0]
        episode_reward += reward
        episode_length += 1

        if done or truncations:
            logging.info(f"Episode {episode} | Step {step} | Reward: {episode_reward:.2f} | Length: {episode_length}")
            obs, _ = env.reset() if isinstance(env.reset(), tuple) else (env.reset(), None)
            episode_reward = 0
            episode_length = 0
            episode += 1

        # Update agent after collecting enough data
        if step >= start_steps:
            batch = replay_buffer.sample(batch_size)
            # Convert to jax arrays
            batch = {k: jnp.asarray(v) for k, v in batch.items()}
            agent, info = agent.update(batch, utd_ratio=1)

            if step % 1000 == 0:
                log_str = f"[Step {step}] " + ", ".join([f"{k}: {float(v):.4f}" for k, v in info.items()])
                logging.info(log_str)

        # Periodic evaluation
        if step % eval_interval == 0:
            eval_rewards = []
            eval_successes = []
            for _ in range(10):
                eval_obs, _ = eval_env.reset() if isinstance(eval_env.reset(), tuple) else (eval_env.reset(), None)
                eval_ep_reward = 0
                eval_ep_success = 0
                for _ in range(env.max_path_length):
                    eval_action = np.asarray(
                        agent.actor.apply_fn(
                            {"params": agent.actor.params},
                            eval_obs[None]
                        ).sample(seed=jax.random.PRNGKey(np.random.randint(1e6)))
                    )[0]
                    step_result = eval_env.step(eval_action)
                    if isinstance(step_result, tuple) and len(step_result) == 5:
                        eval_obs, r, d, trunc, info = step_result
                    elif isinstance(step_result, tuple) and len(step_result) == 4:
                        eval_obs, r, d, info = step_result
                        trunc = False
                    else:
                        raise ValueError("Unexpected number of elements returned by eval_env.step()")
                    eval_ep_reward += r
                    # Collect success if available
                    if isinstance(info, dict) and "success" in info:
                        if info["success"]:
                            eval_ep_success = 1
                    if d or trunc:
                        break
                eval_rewards.append(eval_ep_reward)
                eval_successes.append(eval_ep_success)
            avg_eval_reward = np.mean(eval_rewards)
            avg_success_rate = np.mean(eval_successes)
            logging.info(f"[EVAL] Step {step} | Avg Reward: {avg_eval_reward:.2f} | Success Rate: {avg_success_rate:.2f}")

if __name__ == "__main__":
    main()