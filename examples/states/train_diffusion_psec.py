# train_diffusion_psec.py (updated for Meta-World)
import gymnasium as gym
import jax
import wandb
import logging
from tqdm import tqdm
from flax.core import frozen_dict
from jaxrl5.agents import Pretrain
from jaxrl5.data.dsrl_datasets import DSRLDataset, Toy_dataset
from jaxrl5.evaluation_dsrl import evaluate_bc
from jaxrl5.wrappers import wrap_gym
from jaxrl5.wrappers.wandb_video import WANDBVideo
from jaxrl5.data import ReplayBuffer, BinaryDataset
import jax.numpy as jnp
import numpy as np
from jax import config
import time
#from dataset.utils import get_imitation_data
import os
from dotenv import load_dotenv
import glob
# This loads the environment variables from .env into os.environ
load_dotenv()  
from metaworld.evaluation import evaluation, metalearning_evaluation
import glob              # <--- FIX: ensure glob is imported
import traceback         # <--- helpful for printing full exception info

# Now you can access them
api_key = os.getenv("WANDB_API_KEY")

timestamp = int(time.time())

# Set up basic logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

class SimpleOfflineDataset:
    """
    Minimal dataset class to stand in for DSRLDataset when environment doesn't provide get_dataset().
    Provides:
      - seed(rng_seed)
      - sample_jax(batch_size, keys=None) -> dict of jnp arrays
      - normalize_returns(max_reward, min_reward, env_max_steps)
    """
    def __init__(self, arrays_dict):
        import numpy as _np
        # Expected keys: observations, actions, rewards, next_observations, dones
        self._np = _np
        self.observations = arrays_dict['observations']
        self.actions = arrays_dict['actions']
        self.rewards = arrays_dict['rewards']
        self.next_observations = arrays_dict['next_observations']
        self.dones = arrays_dict['dones']
        self.episode_returns = arrays_dict.get('episode_returns', None)
        self.size = self.observations.shape[0]
        self._rng = _np.random.RandomState(0)
        # optional normalization info
        self._return_shift = 0.0
        self._return_scale = 1.0

    def seed(self, s):
        self._rng = self._np.random.RandomState(int(s))

    def sample_jax(self, batch_size, keys=None):
        """
        Return a dict of JAX arrays of shape (batch_size, ...).
        If keys is provided (list), only return those keys. Typical keys: ['observations','actions'].
        """
        import jax.numpy as _jnp
        idx = self._rng.randint(0, self.size, size=(batch_size,))
        out = {}
        if keys is None or 'observations' in keys:
            out['observations'] = _jnp.asarray(self.observations[idx])
        if keys is None or 'actions' in keys:
            out['actions'] = _jnp.asarray(self.actions[idx])
        # include other useful fields
        if keys is None or 'rewards' in keys:
            out['rewards'] = _jnp.asarray(self.rewards[idx] * self._return_scale + self._return_shift)
        if keys is None or 'next_observations' in keys:
            out['next_observations'] = _jnp.asarray(self.next_observations[idx])
        if keys is None or 'dones' in keys:
            out['dones'] = _jnp.asarray(self.dones[idx])
        return out

    def normalize_returns(self, max_reward, min_reward, env_max_steps):
        """
        Scale returns in dataset to be in a consistent range expected by training.
        This implementation uses (reward - min) / (max - min) as a simple scaling of per-step rewards.
        """
        # avoid divide-by-zero
        if max_reward == min_reward:
            self._return_scale = 1.0
            self._return_shift = 0.0
            return
        # map reward r to (r - min) / (max - min)
        self._return_scale = 1.0 / float(max_reward - min_reward)
        self._return_shift = - float(min_reward) * self._return_scale

def multi_task_eval(agent, envs, num_evaluation_episodes = 50, episode_horizon = 500):
    success_rate = 0.0
    for episode in range(num_evaluation_episodes):
        obs = envs.reset()


        
        for step in range(episode_horizon):
            action, _ = agent.eval_actions(obs)
            next_obs, _, _, _, info = env.step(action)
            obs = next_obs

            if info["success"] == 1:
                success_rate += 1
                break
    success_rate /= (num_evaluation_episodes * envs.num_envs)

    return success_rate

def _unwrap_env_for_dataset(env, max_unwrap=20):
    """
    Try to unwrap common gym / gymnasium wrappers to reach the base env
    that may implement get_dataset(). Returns the unwrapped candidate.
    """
    candidate = env
    for _ in range(max_unwrap):
        # If the object itself has get_dataset, return it immediately
        if hasattr(candidate, "get_dataset"):
            return candidate

        # Common attributes that hold the inner env depending on wrapper types
        next_candidate = None
        # Gym/Gymnasium common patterns:
        if hasattr(candidate, "unwrapped") and getattr(candidate, "unwrapped") is not candidate:
            next_candidate = candidate.unwrapped
        elif hasattr(candidate, "env") and getattr(candidate, "env") is not candidate:
            next_candidate = candidate.env
        elif hasattr(candidate, "wrapped_env") and getattr(candidate, "wrapped_env") is not candidate:
            next_candidate = candidate.wrapped_env
        elif hasattr(candidate, "inner_env") and getattr(candidate, "inner_env") is not candidate:
            next_candidate = candidate.inner_env

        if next_candidate is None:
            break
        candidate = next_candidate

    # Final check: maybe `.spec` or `.unwrapped` contains the dataset provider
    if hasattr(candidate, "get_dataset"):
        return candidate

    # as a final attempt, try .env if available (some wrappers nest deeper)
    try:
        if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "get_dataset"):
            return env.unwrapped
    except Exception:
        pass

    return candidate  # best-effort return (may not have get_dataset)  


@jax.jit
def merge_batch(batch1, batch2):
    merge = {}
    for k in batch1.keys():
        merge[k] = jnp.concatenate([batch1[k], batch2[k]], axis = 0)
    return frozen_dict.freeze(merge)

def call_main(details):
    # wandb.init(project=details['project'], name=details['group'], mode="offline")
    # wandb.config.update(details)
    logging.info("=========================================================")
    logging.info(f"[LOG] Initialized project: {details['project']}, group: {details['group']}, env: {details['env_name']} ")
    

    # Special toy dataset (kept from original)
    if details.get('env_name') == '8gaussians-multitarget':
        assert details['dataset_kwargs']['pr_data'] is not None, "No data for Point Robot"
        env = details['env_name']
        ds = Toy_dataset(env)
    else:
        # If the config contains a Meta-World benchmark key, use that API:
        # e.g. details['benchmark'] = "Meta-World/MT1", details['env_name'] = "reach-v3"
        try:
            if 'benchmark' in details and details['benchmark'].startswith('Meta-World'):
                benchmark = details['benchmark']  # e.g. "Meta-World/MT1" or "Meta-World/ML1"
                env_name = details.get('env_name', None)  # e.g. 'reach-v3'
                if env_name is None:
                    raise ValueError("For Meta-World runs you must set details['env_name'] to a task name like 'reach-v3'.")
                # create the single-task benchmark env (MT1 / ML1 style)
                print(benchmark, env_name, "env")
                           
                
                env = gym.make(benchmark, env_name=env_name[1], seed=details.get('seed', None))
            else:
                # default (non-MetaWorld)
                env = gym.make(details['env_name'])
        except TypeError:
            # fallback for older gym versions or unexpected args
            env = gym.make(details['env_name'])

    
    # make sure env_max_steps is defined before branches that use it
    env_max_steps = getattr(env.unwrapped, '_max_episode_steps', None)
    if env_max_steps is None:
        env_max_steps = getattr(env, 'max_episode_steps', None)
    if env_max_steps is None:
        env_max_steps = 150

    dataset_env_candidate = _unwrap_env_for_dataset(env)

    hub_error = None  # ensure this exists in case we reference it later
    built_ds = None

    try:
        if hasattr(dataset_env_candidate, "get_dataset"):
            ds = DSRLDataset(dataset_env_candidate, ratio=details.get('ratio', 1.0))
        else:
            print("Info: env does not expose get_dataset(). Attempting to load ml-jku/meta-world from Hugging Face...")

            use_hf = details.get('use_hf_meta_world', True)
            hf_local_dir = details.get('hf_local_dir', None)
            hf_max_files = details.get('hf_max_files', None)

            if use_hf:
                repo_dir = None
                try:
                    # try snapshot_download if huggingface_hub is available
                    from huggingface_hub import snapshot_download
                    print("Attempting snapshot_download from Hugging Face (may be large)...")
                    # allow_patterns may be ignored in some versions, still okay
                    repo_dir = snapshot_download(
                        repo_id="ml-jku/meta-world",
                        repo_type="dataset",
                        allow_patterns=["2M/*", "2M_separate/*"],
                        local_dir=details.get("hf_download_dir", None),
                        use_auth_token=details.get("hf_token", None)  # optional token
                    )
                except Exception as ex_hub:
                    # keep the exception for diagnostics but continue to other fallbacks
                    hub_error = ex_hub
                    print("snapshot_download failed or unavailable:", str(ex_hub))
                    traceback.print_exc()

                # If we have a local dir (or downloaded repo_dir), try to find 2M/ and parse .npz files
                try:
                    if hf_local_dir is not None and os.path.isdir(hf_local_dir):
                        repo_dir = hf_local_dir

                    if repo_dir:
                        # find 2M folder
                        possible = [
                            os.path.join(repo_dir, "2M"),
                            os.path.join(repo_dir, "meta-world", "2M"),
                            os.path.join(repo_dir, "2M_separate"),
                            repo_dir
                        ]
                        npz_folder = None
                        for p in possible:
                            if p and os.path.isdir(p):
                                # if it's a repo root, check for *.npz
                                if len(glob.glob(os.path.join(p, "*.npz"))) > 0:
                                    npz_folder = p
                                    break

                        if npz_folder is None:
                            raise RuntimeError(f"Could not find folder with .npz files under {repo_dir} (checked {possible}).")

                        npz_files = sorted(glob.glob(os.path.join(npz_folder, "*.npz")))
                        if not npz_files:
                            raise RuntimeError(f"No .npz files found in {npz_folder}")

                        if hf_max_files:
                            npz_files = npz_files[:hf_max_files]

                        # tolerant loader for .npz trajectory files
                        def _get_from_npz(npz_dict, candidates):
                            for k in candidates:
                                if k in npz_dict:
                                    return npz_dict[k]
                            return None

                        obs_buf = []
                        act_buf = []
                        rew_buf = []
                        next_obs_buf = []
                        done_buf = []
                        episode_returns = []

                        
                        for fpath in npz_files[env_name[0]:env_name[0]+1]:
                            print(fpath,"fpath", env_name)
                            try:
                                d = np.load(fpath, allow_pickle=True)
                                obs = _get_from_npz(d, ['observations', 'obs', 'observations_raw', 'o'])
                                actions = _get_from_npz(d, ['actions', 'acts', 'a'])
                                rewards = _get_from_npz(d, ['rewards', 'r', 'reward'])
                                dones = _get_from_npz(d, ['dones', 'done', 'terminals', 'terminated'])
                                next_obs = _get_from_npz(d, ['next_observations', 'next_obs', 'observations_next'])
                                if obs is None or actions is None or rewards is None:
                                    print(f"Skipping {fpath}: missing required arrays. Keys: {list(d.keys())}")
                                    continue

                                # build next_obs if not present
                                if next_obs is None:
                                    if getattr(obs, "shape", None) and obs.shape[0] >= 2:
                                        next_obs = np.concatenate([obs[1:], obs[-1:]], axis=0)
                                    else:
                                        next_obs = obs.copy()

                                if dones is None:
                                    dones = np.zeros((obs.shape[0],), dtype=np.bool_)
                                    dones[-1] = True

                                # Patch: ensure obs and actions are always stacked as (N, dim)
                                if isinstance(obs, np.ndarray) and obs.ndim == 2:
                                    obs_buf.extend(obs.astype(np.float32))
                                else:
                                    obs_buf.extend([np.asarray(x, dtype=np.float32) for x in obs])
                                
                                if isinstance(actions, np.ndarray) and actions.ndim == 2:
                                    act_buf.extend(actions.astype(np.float32))
                                else:
                                    act_buf.extend([np.asarray(x, dtype=np.float32) for x in actions])
                                rew_buf.extend([float(x) for x in rewards])
                                next_obs_buf.extend([np.asarray(x, dtype=np.float32) for x in next_obs])
                                done_buf.extend([bool(x) for x in dones])
                                episode_returns.append(float(np.sum(rewards)))
                            except Exception as e_traj:
                                print(f"Warning: failed to parse {fpath}: {e_traj}. Skipping.")
                                traceback.print_exc()
                        
                        if len(obs_buf) == 0:
                            raise RuntimeError("No usable transitions parsed from .npz files.")

                        arrays = dict(
                            observations=np.stack(obs_buf, axis=0),
                            actions=np.stack(act_buf, axis=0),
                            rewards=np.asarray(rew_buf, dtype=np.float32),
                            next_observations=np.stack(next_obs_buf, axis=0),
                            dones=np.asarray(done_buf, dtype=np.float32),
                            episode_returns=episode_returns,
                        )
                       
                        built_ds = SimpleOfflineDataset(arrays)
                        
                    else:
                        # fallback using datasets library (streaming small amount)
                        from datasets import load_dataset
                        print("Falling back to datasets.load_dataset streaming approach (may be heavy).")
                        hf_ds = load_dataset("ml-jku/meta-world", split="train")
                        # Simple streaming extraction (stop early if hf_max_files set)
                        obs_buf = []; act_buf = []; rew_buf = []; next_obs_buf = []; done_buf = []; episode_returns = []
                        def _safe_get(sample, keys):
                            for k in keys:
                                if k in sample:
                                    return sample[k]
                            return None
                        for i, sample in enumerate(hf_ds):
                            obs = _safe_get(sample, ['observations','obs','o','observation'])
                            actions = _safe_get(sample, ['actions','acts','a','action'])
                            rewards = _safe_get(sample, ['rewards','r','reward'])
                            if obs is None or actions is None or rewards is None:
                                continue
                            # flatten trajectory if necessary
                            if hasattr(obs, "__len__") and hasattr(rewards, "__len__") and len(obs) == len(rewards):
                                T = len(obs)
                                next_obs = list(obs[1:]) + [obs[-1]]
                                dones = [False]*(T-1) + [True]
                                obs_buf.extend([np.asarray(x, dtype=np.float32) for x in obs])
                                act_buf.extend([np.asarray(x, dtype=np.float32) for x in actions])
                                rew_buf.extend([float(x) for x in rewards])
                                next_obs_buf.extend([np.asarray(x, dtype=np.float32) for x in next_obs])
                                done_buf.extend(dones)
                                episode_returns.append(float(np.sum(rewards)))
                            else:
                                # treat as single transition
                                obs_buf.append(np.asarray(obs, dtype=np.float32))
                                act_buf.append(np.asarray(actions, dtype=np.float32))
                                rew_buf.append(float(rewards) if rewards is not None else 0.0)
                                next_obs_buf.append(np.asarray(obs, dtype=np.float32))
                                done_buf.append(False)
                            if hf_max_files and i >= hf_max_files:
                                break

                        if len(obs_buf) == 0:
                            raise RuntimeError("No usable data extracted from Hugging Face dataset object.")

                        arrays = dict(
                            observations=np.stack(obs_buf, axis=0),
                            actions=np.stack(act_buf, axis=0),
                            rewards=np.asarray(rew_buf, dtype=np.float32),
                            next_observations=np.stack(next_obs_buf, axis=0),
                            dones=np.asarray(done_buf, dtype=np.float32),
                            episode_returns=episode_returns,
                        )
                        built_ds = SimpleOfflineDataset(arrays)

                except Exception as ex_inner:
                    print("Warning: error while trying to parse Hugging Face repo or dataset:", str(ex_inner))
                    traceback.print_exc()

            # finalize: choose built_ds or fallback to collecting rollouts
            
            if built_ds is not None:
                ds = built_ds
                print("Loaded offline dataset from ml-jku/meta-world and built SimpleOfflineDataset.")
                if details.get('normalize_returns', True):
                    if 'max_reward' in details and 'min_reward' in details:
                        ds.normalize_returns(details['max_reward'], details['min_reward'], env_max_steps)
                    else:
                        obs_min = float(ds.rewards.min())
                        obs_max = float(ds.rewards.max())
                        ds.normalize_returns(obs_max, obs_min, env_max_steps)
            else:
                print("Notice: could not load Hugging Face dataset. Collecting rollouts to build an offline dataset (this may take time).")
                arrays = collect_demonstrations_from_env(env,
                                                        num_episodes=details.get('collect_episodes', 200),
                                                        max_steps_per_episode=env_max_steps,
                                                        policy_fn=details.get('collect_policy_fn', None),
                                                        seed=details.get('seed', 0))
                ds = SimpleOfflineDataset(arrays)
                if details.get('normalize_returns', True):
                    obs_min = float(arrays['rewards'].min())
                    obs_max = float(arrays['rewards'].max())
                    ds.normalize_returns(obs_max, obs_min, env_max_steps)

    except Exception as e_outer:
        print("Fatal error in dataset loader:", str(e_outer))
        traceback.print_exc()
        raise
    # --- END PATCH --- #
    # make sure dataset has seed method
    if hasattr(ds, "seed"):
        ds.seed(details.get("seed", 0))

    # --- END of replacement --- #           
        

    if details.get('save_video', False):
        env = WANDBVideo(env)

    config_dict = details['rl_config']
    model_cls = config_dict.pop("model_cls")

    if "BC" in model_cls:
        agent_bc = globals()[model_cls].create(
            details['seed'], env.observation_space, env.action_space, **config_dict
        )
        keys = ["observations", "actions"]
    

    else:
        agent_bc = globals()[model_cls].create(
            details['seed'], env.observation_space, env.action_space, **config_dict
        )
        keys = None
    # Debug: print batch shapes
    
    for i in tqdm(range(2000), smoothing=0.1):
        sample = ds.sample_jax(details['batch_size'], keys=keys)
        # print("DEBUG batch shapes: obs", sample['observations'].shape, "actions", sample['actions'].shape)
        # Squeeze singleton dimension if present
        if sample['observations'].ndim == 3 and sample['observations'].shape[1] == 1:
            sample['observations'] = sample['observations'].squeeze(1)
        if sample['actions'].ndim == 3 and sample['actions'].shape[1] == 1:
            sample['actions'] = sample['actions'].squeeze(1)
        # print("DEBUG batch shapes: obs", sample['observations'].shape, "actions", sample['actions'].shape)    
        agent_bc, info_bc = agent_bc.update_bc(sample)

        if i % details['log_interval'] == 0:
            # wandb.log({f"train_bc/{k}": v for k, v in info_bc.items()}, step=i)
            print(f"[LOG][Step {i}] train_bc: " + ", ".join([f"{k}: {v}" for k, v in info_bc.items()]))

        if i % details['save_steps'] == 0:
            agent_bc.save(f"./results/{details['timestamp']}/{details['timestamp']}_{details['group']}/{details['experiment_name']}_bc", i)

        if i % 1000 == 0 or i == 0:
            for inference_params in details['inference_variants']:
                agent_bc = agent_bc.replace(**inference_params)
                eval_info_bc = evaluate_bc(agent_bc, env, details['eval_episodes'], train_lora=False)
                
                # Meta-World: try to normalize using env helper if present
                # try:
                #     eval_info_bc["normalized_return"], eval_info_bc["normalized_cost"] = env.get_normalized_score(eval_info_bc["return"], eval_info_bc["cost"])
                # except Exception:
                #     pass
                # wandb.log({f"eval_bc/{inference_params}_{k}": v for k, v in eval_info_bc.items()}, step=i)
                logging.info(f"[Step {i}] eval_bc ({inference_params}): " + ", ".join([f"{k}: {v}" for k, v in eval_info_bc.items()]))
            agent_bc.replace(**details['training_time_inference_params'])

            
            