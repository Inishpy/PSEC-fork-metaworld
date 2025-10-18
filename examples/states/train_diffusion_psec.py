# train_diffusion_psec.py (refactored and modularized for Meta-World)
import gymnasium as gym
import jax
import logging
from tqdm import tqdm
from flax.core import frozen_dict
from jaxrl5.data.dsrl_datasets import DSRLDataset, Toy_dataset
from jaxrl5.evaluation_dsrl import evaluate_bc
import jax.numpy as jnp
import numpy as np
from jax import config
import time
import os
from dotenv import load_dotenv
import glob
from metaworld.evaluation import evaluation, metalearning_evaluation
import traceback
from jaxrl5.agents.psec.psec_pretrain import PretrainWithComposition
import pandas as pd
# Load environment variables from .env into os.environ
load_dotenv()
# api_key = os.getenv("WANDB_API_KEY")
# logging.info("wandb login started")
# wandb.login(key=api_key)
# logging.info("wandb login done")
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
        self._np = _np
        self.observations = arrays_dict['observations']
        self.actions = arrays_dict['actions']
        self.rewards = arrays_dict['rewards']
        self.next_observations = arrays_dict['next_observations']
        self.dones = arrays_dict['dones']
        self.episode_returns = arrays_dict.get('episode_returns', None)
        self.size = self.observations.shape[0]
        self._rng = _np.random.RandomState(0)
        self._return_shift = 0.0
        self._return_scale = 1.0

    def seed(self, s):
        self._rng = self._np.random.RandomState(int(s))

    def sample_jax(self, batch_size, keys=None):
        import jax.numpy as _jnp
        idx = self._rng.randint(0, self.size, size=(batch_size,))
        out = {}
        if keys is None or 'observations' in keys:
            out['observations'] = _jnp.asarray(self.observations[idx])
        if keys is None or 'actions' in keys:
            out['actions'] = _jnp.asarray(self.actions[idx])
        if keys is None or 'rewards' in keys:
            out['rewards'] = _jnp.asarray(self.rewards[idx] * self._return_scale + self._return_shift)
        if keys is None or 'next_observations' in keys:
            out['next_observations'] = _jnp.asarray(self.next_observations[idx])
        if keys is None or 'dones' in keys:
            out['dones'] = _jnp.asarray(self.dones[idx])
        return out

    def normalize_returns(self, max_reward, min_reward, env_max_steps):
        if max_reward == min_reward:
            self._return_scale = 1.0
            self._return_shift = 0.0
            return
        self._return_scale = 1.0 / float(max_reward - min_reward)
        self._return_shift = - float(min_reward) * self._return_scale

import os
import re

def find_parquet_files(repo_dir: str, max_files: int | None = None):
    """
    Recursively find parquet (.parquet or .pq) files under repo_dir.
    Prefer repo_dir/data if it exists. Sort results by chunk & file numeric order
    when filenames use patterns like chunk-000/file-012.parquet.
    Returns a list of absolute file paths (maybe truncated to max_files).
    """
    if not repo_dir or not os.path.isdir(repo_dir):
        return []

    # prefer repo_dir/data if present, else search repo_dir recursively
    start_dirs = []
    data_dir = os.path.join(repo_dir, "data")
    if os.path.isdir(data_dir):
        start_dirs.append(data_dir)
    else:
        start_dirs.append(repo_dir)

    parquet_paths = []
    for start in start_dirs:
        for root, dirs, files in os.walk(start):
            for fn in files:
                if fn.lower().endswith((".parquet", ".pq")):
                    parquet_paths.append(os.path.join(root, fn))

    # If filenames look like chunk-XXX/file-YYY.parquet, sort numerically by chunk then file.
    # Fallback: natural sort by full path.
    pattern = re.compile(r".*chunk-(\d+).*file-(\d+)\.parquet$", re.IGNORECASE)
    parsed = []
    for p in parquet_paths:
        m = pattern.match(p.replace(os.sep, "/"))  # unify separators for regex
        if m:
            chunk_num = int(m.group(1))
            file_num = int(m.group(2))
            parsed.append((chunk_num, file_num, p))
        else:
            parsed.append((None, None, p))

    # Sort: entries with numeric chunk/file first (by chunk,file), then the rest alphabetically
    with_nums = [t for t in parsed if t[0] is not None]
    without_nums = [t for t in parsed if t[0] is None]

    with_nums.sort(key=lambda t: (t[0], t[1]))
    without_nums.sort(key=lambda t: t[2].lower())

    sorted_paths = [t[2] for t in with_nums] + [t[2] for t in without_nums]

    if max_files is not None:
        sorted_paths = sorted_paths[:max_files]

    return sorted_paths

def multi_task_eval(agent, envs, num_evaluation_episodes=50, episode_horizon=500):
    success_rate = 0.0
    for episode in range(num_evaluation_episodes):
        obs = envs.reset()
        for step in range(episode_horizon):
            action, _ = agent.eval_actions(obs)
            next_obs, _, _, _, info = envs.step(action)
            obs = next_obs
            if info["success"] == 1:
                success_rate += 1
                break
    success_rate /= (num_evaluation_episodes * envs.num_envs)
    return success_rate

def _unwrap_env_for_dataset(env, max_unwrap=20):
    candidate = env
    for _ in range(max_unwrap):
        if hasattr(candidate, "get_dataset"):
            return candidate
        next_candidate = None
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
    if hasattr(candidate, "get_dataset"):
        return candidate
    try:
        if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "get_dataset"):
            return env.unwrapped
    except Exception:
        pass
    return candidate

@jax.jit
def merge_batch(batch1, batch2):
    merge = {}
    for k in batch1.keys():
        merge[k] = jnp.concatenate([batch1[k], batch2[k]], axis=0)
    return frozen_dict.freeze(merge)

def load_offline_dataset(details, env, env_max_steps):
    """
    Modularized dataset loading logic.
    Returns: ds (dataset object)
    """
    logging.info("[DATA] Starting offline dataset loading for env: %s", str(details.get('env_name')))
    dataset_env_candidate = _unwrap_env_for_dataset(env)
    hub_error = None
    built_ds = None
    try:
        if hasattr(dataset_env_candidate, "get_dataset"):
            logging.info("[DATA] Using environment's get_dataset() method.")
            ds = DSRLDataset(dataset_env_candidate, ratio=details.get('ratio', 1.0))
        else:
            logging.info("[DATA] Env does not expose get_dataset(). Attempting to load ml-jku/meta-world from Hugging Face...")
            use_hf = details.get('use_hf_meta_world', True)
            hf_local_dir = details.get('hf_local_dir', None)
            hf_max_files = details.get('hf_max_files', None)
            if use_hf:
                repo_dir = None
                try:
                    from huggingface_hub import snapshot_download
                    logging.info("[DATA] Attempting snapshot_download from Hugging Face (may be large)...")
                    repo_dir = snapshot_download(
                        repo_id="ml-jku/meta-world",
                        repo_type="dataset", 
                        allow_patterns=["2M/*", "2M_separate/*"],
                        local_dir=details.get("hf_download_dir", None),
                        use_auth_token=details.get("hf_token", None)
                    )
                    # this is for libero download uncomment this and remove the raise from second try and add to first
                    # repo_dir = snapshot_download(
                    #     repo_id="HuggingFaceVLA/libero",
                    #     repo_type="dataset",
                    #     local_dir="libero_partial",
                    #     allow_patterns=["data/chunk-000/file-0*.parquet"],
                    #     use_auth_token=details.get("hf_token", None)
                    # )
                except Exception as ex_hub:
                    hub_error = ex_hub
                    logging.warning("[DATA] snapshot_download failed or unavailable: %s", str(ex_hub), exc_info=True)
                try:
                    
                    if hf_local_dir is not None and os.path.isdir(hf_local_dir):
                        repo_dir = hf_local_dir
                    if repo_dir:
                        possible = [
                            os.path.join(repo_dir, "2M"),
                            os.path.join(repo_dir, "meta-world", "2M"),
                            os.path.join(repo_dir, "2M_separate"),
                            repo_dir
                        ]
                        npz_folder = None
                        for p in possible:
                            if p and os.path.isdir(p):
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
                        def _get_from_npz(npz_dict, candidates):
                            for k in candidates:
                                if k in npz_dict:
                                    return npz_dict[k]
                            return None
                        obs_buf, act_buf, rew_buf, next_obs_buf, done_buf, episode_returns = [], [], [], [], [], []
                        env_name = details.get('env_name', None)
                        for fpath in npz_files[env_name[0]:env_name[0]+1]:
                            logging.info("[DATA] Loading .npz file: %s for env %s", fpath, env_name)
                            try:
                                d = np.load(fpath, allow_pickle=True)
                                obs = _get_from_npz(d, ['observations', 'obs', 'observations_raw', 'o'])
                                actions = _get_from_npz(d, ['actions', 'acts', 'a'])
                                rewards = _get_from_npz(d, ['rewards', 'r', 'reward'])
                                dones = _get_from_npz(d, ['dones', 'done', 'terminals', 'terminated'])
                                next_obs = _get_from_npz(d, ['next_observations', 'next_obs', 'observations_next'])
                                if obs is None or actions is None or rewards is None:
                                    logging.warning("[DATA] Skipping %s: missing required arrays. Keys: %s", fpath, list(d.keys()))
                                    continue
                                if next_obs is None:
                                    if getattr(obs, "shape", None) and obs.shape[0] >= 2:
                                        next_obs = np.concatenate([obs[1:], obs[-1:]], axis=0)
                                    else:
                                        next_obs = obs.copy()
                                if dones is None:
                                    dones = np.zeros((obs.shape[0],), dtype=np.bool_)
                                    dones[-1] = True
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
                                logging.warning("[DATA] Failed to parse %s: %s. Skipping.", fpath, e_traj, exc_info=True)
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
                        logging.info("[DATA] Built SimpleOfflineDataset from .npz files.")
                    else:
                        from datasets import load_dataset
                        logging.info("[DATA] Falling back to datasets.load_dataset streaming approach (may be heavy).")
                        hf_ds = load_dataset("ml-jku/meta-world", split="train")
                        obs_buf, act_buf, rew_buf, next_obs_buf, done_buf, episode_returns = [], [], [], [], [], []
                        def _safe_get(sample, keys):
                            for k in keys:
                                if k in sample:
                                    return sample[k]
                            return None
                        hf_max_files = details.get('hf_max_files', None)
                        for i, sample in enumerate(hf_ds):
                            obs = _safe_get(sample, ['observations','obs','o','observation'])
                            actions = _safe_get(sample, ['actions','acts','a','action'])
                            rewards = _safe_get(sample, ['rewards','r','reward'])
                            if obs is None or actions is None or rewards is None:
                                continue
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
                        logging.info("[DATA] Built SimpleOfflineDataset from Hugging Face streaming dataset.")
                
                except Exception as ex_inner:
                    logging.warning("[DATA] Error while trying to parse Hugging Face repo or dataset: %s", str(ex_inner), exc_info=True)
                try:
                    raise
                    # prefer provided local dir if available
                    if hf_local_dir is not None and os.path.isdir(hf_local_dir):
                        repo_dir = hf_local_dir

                    parquet_files = []
                    if repo_dir:
                        # find up to e.g. first 150 parquet files (set max_files=None to get all)
                        parquet_files = find_parquet_files(repo_dir, max_files=150)

                    # parquet_files now contains absolute paths (up to max_files)
                    print(f"Found {len(parquet_files)} parquet files (showing up to requested max).")
                    for p in parquet_files[:10]:
                        print(" ", p)
                        # if we found parquet files, keep going; otherwise drop through to datasets.load_dataset
                    # If no local parquet found, try to load with datasets.load_dataset (this can stream/parquet-backed)
                    if not parquet_files:
                        logging.info("[DATA] No local parquet found under repo_dir. Falling back to datasets.load_dataset(...)")
                        try:
                            hf_ds = load_dataset("VLA/libero", split="train")
                            # datasets returns an Arrow table-like object; convert to pandas in chunks or full table
                            try:
                                df = hf_ds.to_pandas()
                            except Exception:
                                # sometimes direct conversion fails; iterate to build a pandas.DataFrame (may be heavy)
                                rows = []
                                for i, sample in enumerate(hf_ds):
                                    rows.append(dict(sample))
                                    if hf_max_files and i >= hf_max_files:
                                        break
                                df = pd.DataFrame(rows)
                        except Exception as ex_ds:
                            logging.warning("[DATA] datasets.load_dataset('VLA/libero') failed: %s", str(ex_ds), exc_info=True)
                            df = None
                    else:
                        # read parquet files into a single DataFrame (may be heavy)
                        dfs = []
                        for pfile in parquet_files[:hf_max_files] if hf_max_files else parquet_files:
                            logging.info("[DATA] Reading parquet: %s", pfile)
                            try:
                                df_part = pd.read_parquet(pfile)
                                dfs.append(df_part)
                            except Exception as e_p:
                                logging.warning("[DATA] Failed reading parquet %s: %s", pfile, e_p, exc_info=True)
                        df = pd.concat(dfs, ignore_index=True) if dfs else None

                    if df is None or df.shape[0] == 0:
                        raise RuntimeError("Could not load parquet / HF dataset into a DataFrame.")

                    logging.info("[DATA] Loaded parquet/dataframe with %d rows", len(df))

                    # --- Group by task_index and build per-task datasets ---
                    # try possible column names for task index and task name:
                    task_index_col = None
                    for cand in ['task_index', 'task_idx', 'task', 'task_id', 'taskIndex']:
                        if cand in df.columns:
                            task_index_col = cand
                            break
                    if task_index_col is None:
                        logging.warning("[DATA] Could not find a 'task_index' column. Defaulting all rows to task_index=0.")
                        df['_task_index_internal'] = 0
                        task_index_col = '_task_index_internal'

                    # optional task name column:
                    task_name_col = None
                    for cand in ['task_name', 'task_label', 'task', 'task_str', 'task_description']:
                        if cand in df.columns and cand != task_index_col:
                            task_name_col = cand
                            break

                    # candidate names for obs/actions/rewards/dones/next_obs in a row
                    def _safe_get_from_row(row, keys):
                        for k in keys:
                            if k in row and row[k] is not None:
                                return row[k]
                        return None

                    obs_keys = ['observations', 'obs', 'observation', 'frames', 'states', 'observation.state']
                    act_keys = ['actions', 'acts', 'a', 'action']
                    rew_keys = ['rewards', 'r', 'reward', 'rewards_sum']
                    dones_keys = ['dones', 'done', 'terminals', 'terminated']
                    next_obs_keys = ['next_observations', 'next_obs', 'observations_next', 'next_observation']

                    # Group rows by task index
                    groups = df.groupby(task_index_col)
                    task_datasets = {}
                    merged_buffers = dict(observations=[], actions=[], rewards=[], next_observations=[], dones=[], episode_returns=[])

                    for task_idx, subdf in groups:
                        logging.info("[DATA] Building task %s with %d rows", str(task_idx), len(subdf))
                        obs_buf, act_buf, rew_buf, next_obs_buf, done_buf, episode_returns = [], [], [], [], [], []

                        # iterate rows; each row might itself contain an episode (list/array) or a single transition
                        for _, row in subdf.iterrows():
                            try:
                                obs = _safe_get_from_row(row, obs_keys)
                                acts = _safe_get_from_row(row, act_keys)
                                rews = _safe_get_from_row(row, rew_keys)
                                dones = _safe_get_from_row(row, dones_keys)
                                next_obs = _safe_get_from_row(row, next_obs_keys)

                                # Only skip if obs or acts are missing
                                if obs is None or acts is None:
                                    logging.warning("[DATA] Skipping row: missing obs/acts keys. Available keys: %s", list(row.index))
                                    continue
                                # If rewards are missing, set to 0.0 (for imitation learning)
                                if rews is None:
                                    rews = 0.0

                                # If obs is sequence-like with length matching rewards -> treat as episode
                                is_seq = hasattr(obs, "__len__") and hasattr(rews, "__len__") and len(obs) == len(rews) and len(obs) > 1

                                if is_seq:
                                    T = len(obs)
                                    # next_obs: shift or use provided
                                    if next_obs is None:
                                        next_obs_seq = list(obs[1:]) + [obs[-1]]
                                    else:
                                        # if provided as sequence use it, otherwise try to infer
                                        next_obs_seq = next_obs if hasattr(next_obs, "__len__") and len(next_obs) == T else (list(obs[1:]) + [obs[-1]])
                                    dones_seq = dones if (hasattr(dones, "__len__") and len(dones) == T) else ([False] * (T-1) + [True])

                                    for t in range(T):
                                        obs_buf.append(np.asarray(obs[t], dtype=np.float32))
                                        act_buf.append(np.asarray(acts[t], dtype=np.float32) if hasattr(acts, "__len__") else np.asarray(acts, dtype=np.float32))
                                        rew_buf.append(float(rews[t]))
                                        next_obs_buf.append(np.asarray(next_obs_seq[t], dtype=np.float32))
                                        done_buf.append(bool(dones_seq[t]))
                                    episode_returns.append(float(np.sum(rews)))
                                else:
                                    # single transition row (or aggregated scalar)
                                    obs_arr = np.asarray(obs, dtype=np.float32)
                                    act_arr = np.asarray(acts, dtype=np.float32)
                                    rew_val = float(rews) if not (hasattr(rews, "__len__")) else float(np.sum(rews))
                                    if next_obs is None:
                                        next_obs_arr = obs_arr.copy()
                                        done_val = True
                                    else:
                                        next_obs_arr = np.asarray(next_obs, dtype=np.float32) if not (hasattr(next_obs, "__len__") and len(next_obs) != obs_arr.shape[0]) else np.asarray(next_obs, dtype=np.float32)
                                        done_val = bool(dones) if dones is not None else True

                                    obs_buf.append(obs_arr)
                                    act_buf.append(act_arr)
                                    rew_buf.append(rew_val)
                                    next_obs_buf.append(next_obs_arr)
                                    done_buf.append(done_val)
                                    episode_returns.append(rew_val)

                            except Exception as e_row:
                                logging.warning("[DATA] Failed to parse row for task %s: %s", task_idx, e_row, exc_info=True)
                                continue

                        if len(obs_buf) == 0:
                            logging.warning("[DATA] No usable transitions parsed for task %s, skipping.", task_idx)
                            continue

                        arrays = dict(
                            observations=np.stack(obs_buf, axis=0),
                            actions=np.stack(act_buf, axis=0),
                            rewards=np.asarray(rew_buf, dtype=np.float32),
                            next_observations=np.stack(next_obs_buf, axis=0),
                            dones=np.asarray(done_buf, dtype=np.float32),
                            episode_returns=episode_returns,
                        )
                        task_name = None
                        if task_name_col and task_name_col in subdf.columns:
                            # take the first non-null name in the group
                            first_name = subdf[task_name_col].dropna()
                            task_name = str(first_name.iloc[0]) if len(first_name) > 0 else None
                        if task_name is None:
                            task_name = f"task_{int(task_idx)}"

                        task_ds = SimpleOfflineDataset(arrays)
                        task_datasets[int(task_idx)] = dict(name=task_name, dataset=task_ds)

                        # extend merged buffers
                        merged_buffers['observations'].extend(arrays['observations'])
                        merged_buffers['actions'].extend(arrays['actions'])
                        merged_buffers['rewards'].extend(arrays['rewards'])
                        merged_buffers['next_observations'].extend(arrays['next_observations'])
                        merged_buffers['dones'].extend(arrays['dones'])
                        merged_buffers['episode_returns'].extend(arrays['episode_returns'])

                    # Build merged dataset
                    if len(merged_buffers['observations']) == 0:
                        raise RuntimeError("No usable transitions parsed from parquet dataset.")

                    merged_arrays = dict(
                        observations=np.stack(merged_buffers['observations'], axis=0),
                        actions=np.stack(merged_buffers['actions'], axis=0),
                        rewards=np.asarray(merged_buffers['rewards'], dtype=np.float32),
                        next_observations=np.stack(merged_buffers['next_observations'], axis=0),
                        dones=np.asarray(merged_buffers['dones'], dtype=np.float32),
                        episode_returns=merged_buffers['episode_returns'],
                    )
                    built_ds = SimpleOfflineDataset(merged_arrays)
                    logging.info("[DATA] Built merged SimpleOfflineDataset from parquet / VLA/libero with %d transitions across %d tasks.", 
                                merged_arrays['observations'].shape[0], len(task_datasets))

                    # normalize returns if requested
                    if built_ds is not None and details.get('normalize_returns', True):
                        if 'max_reward' in details and 'min_reward' in details:
                            built_ds.normalize_returns(details['max_reward'], details['min_reward'], env_max_steps)
                        else:
                            obs_min = float(built_ds.rewards.min())
                            obs_max = float(built_ds.rewards.max())
                            built_ds.normalize_returns(obs_max, obs_min, env_max_steps)

                    # attach metadata: merged dataset remains the return 'ds' for compatibility
                    # but we also return per-task mapping (task_datasets) so caller can use it.
                    # We'll set built_ds.task_datasets for convenience (duck-typing)
                    built_ds.task_datasets = task_datasets  # dict: task_idx -> {name, dataset}
            
                except Exception as ex_inner:
                    logging.info("skipping loading libero")
                    #logging.warning("[DATA] Error while trying to parse Hugging Face repo or dataset: %s", str(ex_inner), exc_info=True)
            
            if built_ds is not None:
                ds = built_ds
                logging.info("[DATA] Loaded offline dataset from ml-jku/meta-world and built SimpleOfflineDataset.")
                if details.get('normalize_returns', True):
                    if 'max_reward' in details and 'min_reward' in details:
                        ds.normalize_returns(details['max_reward'], details['min_reward'], env_max_steps)
                    else:
                        obs_min = float(ds.rewards.min())
                        obs_max = float(ds.rewards.max())
                        ds.normalize_returns(obs_max, obs_min, env_max_steps)
            else:
                logging.info("[DATA] Could not load Hugging Face dataset. Collecting rollouts to build an offline dataset (this may take time).")
                # --- STUB: collect_demonstrations_from_env ---
                def collect_demonstrations_from_env(env, num_episodes, max_steps_per_episode, policy_fn=None, seed=0):
                    raise NotImplementedError(
                        "collect_demonstrations_from_env is not implemented. "
                        "You must provide a function to collect demonstrations from the environment."
                    )
                arrays = collect_demonstrations_from_env(
                    env,
                    num_episodes=details.get('collect_episodes', 200),
                    max_steps_per_episode=env_max_steps,
                    policy_fn=details.get('collect_policy_fn', None),
                    seed=details.get('seed', 0)
                )
                ds = SimpleOfflineDataset(arrays)
                if details.get('normalize_returns', True):
                    obs_min = float(arrays['rewards'].min())
                    obs_max = float(arrays['rewards'].max())
                    ds.normalize_returns(obs_max, obs_min, env_max_steps)
    except Exception as e_outer:
        logging.error("[DATA] Fatal error in dataset loader: %s", str(e_outer), exc_info=True)
        traceback.print_exc()
        raise
    if hasattr(ds, "seed"):
        ds.seed(details.get("seed", 0))
    logging.info("[DATA] Finished offline dataset loading for env: %s", str(details.get('env_name')))
    return ds

def create_env(details):
    """
    Modularized environment creation logic.
    Returns: env, env_max_steps
    """
    logging.info("[ENV] Creating environment for: %s", str(details.get('env_name')))
    if details.get('env_name') == '8gaussians-multitarget':
        env = details['env_name']
        env_max_steps = 150
    else:
        try:
            if 'benchmark' in details and details['benchmark'].startswith('Meta-World'):
                benchmark = details['benchmark']
                env_name = details.get('env_name', None)
                if env_name is None:
                    raise ValueError("For Meta-World runs you must set details['env_name'] to a task name like 'reach-v3'.")
                logging.info("[ENV] Using Meta-World benchmark: %s, task: %s", benchmark, env_name)
                env = gym.make(benchmark, env_name=env_name[1], seed=details.get('seed', None))
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

def create_agent(details, env, config_dict):
    """
    Modularized agent creation logic.
    Returns: agent_bc, keys
    """
    model_cls = config_dict.pop("model_cls")
    logging.info("[AGENT] Creating agent of class: %s", model_cls)
    if "BC" in model_cls:
        agent_bc = globals()[model_cls].create(
            details['seed'], env.observation_space, env.action_space, **config_dict
        )
        keys = ["observations", "actions"]
    else:
        print("model class", model_cls)
        agent_bc = globals()[model_cls].create(
            details['seed'],
            env.observation_space,
            env.action_space,
            current_task=details["env_name"][1],
            exclude_tasks=details.get('exclude_tasks', None),
            **config_dict
        )
        keys = None
    logging.info("[AGENT] Agent created: %s", model_cls)
    return agent_bc, keys

def squeeze_sample_batch(sample):
    """
    Squeeze singleton dimension from observations/actions if present.
    """
    if sample['observations'].ndim == 3 and sample['observations'].shape[1] == 1:
        sample['observations'] = sample['observations'].squeeze(1)
    if sample['actions'].ndim == 3 and sample['actions'].shape[1] == 1:
        sample['actions'] = sample['actions'].squeeze(1)
    return sample

from torch.utils.tensorboard import SummaryWriter

def train_agent(agent_bc, ds, details, keys, env, log_dir=None):
    """
    Modularized training loop.
    """
    logging.info("[TRAIN] Starting training loop for env: %s", str(details.get('env_name')))

    writer = None
    if log_dir is not None:
        writer = SummaryWriter(log_dir=log_dir)

    for i in tqdm(range(details["max_steps"]), smoothing=0.1):
        sample = ds.sample_jax(details['batch_size'], keys=keys)
        sample = squeeze_sample_batch(sample)
        
        agent_bc, info_bc = agent_bc.update_bc(sample)

        if i % details['log_interval'] == 0:
            log_msg = "[TRAIN][Step %d] train_bc: %s" % (i, ", ".join([f"{k}: {v}" for k, v in info_bc.items()]))
            logging.info(log_msg)
            # Log training info to TensorBoard
            if writer is not None:
                for k, v in info_bc.items():
                    # Convert JAX/NumPy/PyTorch 0-d arrays to Python float for TensorBoard logging
                    if hasattr(v, "item"):
                        v = v.item()
                    elif isinstance(v, (jnp.ndarray, np.ndarray)) and v.shape == ():
                        v = float(v)
                    writer.add_scalar(f"train/{k}", v, i)
        
        success_flag = False
        if i % details["eval_interval"] == 0 or i == 0:
            for inference_params in details['inference_variants']:
                agent_bc = agent_bc.replace(**inference_params)
                eval_info_bc = evaluate_bc(agent_bc, env, details['eval_episodes'], train_lora=False)
                print(eval_info_bc)

                # Log evaluation metrics to TensorBoard
                if writer is not None and isinstance(eval_info_bc, dict):
                    for k, v in eval_info_bc.items():
                        if hasattr(v, "item"):
                            v = v.item()
                        elif isinstance(v, (jnp.ndarray, np.ndarray)) and getattr(v, "shape", None) == ():
                            v = float(v)
                        writer.add_scalar(f"eval/{k}", v, i)

                if eval_info_bc.get("success", 0.0) == 1.0:
                    
                    logging.info(f"[TRAIN][Step {i}] Perfect success rate achieved, stopping training early.")
                    save_dir = details.get('results_dir', f"./results/{details['timestamp']}/{details['timestamp']}_{details['group']}/{details['experiment_name']}_bc")
                    # agent_bc.save(save_dir, i)
                    logging.info("[TRAIN][Step %d] Model saved to %s", i, save_dir)
                    success_flag = True
                    break

                logging.info(f"[TRAIN][Step {i}] eval_bc ({inference_params}): " + ", ".join([f"{k}: {v}" for k, v in eval_info_bc.items()]))
            if success_flag:
                break
                
                
            agent_bc.replace(**details['training_time_inference_params'])

    if writer is not None:
        writer.close()

    logging.info("info, eval_info", info_bc, eval_info_bc)
    logging.info("[TRAIN] Training loop completed for env: %s", str(details.get('env_name')))


    return agent_bc

def call_main(details):
    """
    Main entry point for training diffusion PSEC.
    """
    logging.info("=========================================================")
    logging.info("[MAIN] Initialized project: %s, group: %s, env: %s", details['project'], details['group'], details['env_name'])

    # --- Write init log, seed, and task details to a txt file in results_dir ---
    try:
        import json
        results_dir = details.get('results_dir', None)
        seed = details.get('seed', None)
        env_name = details.get('env_name', None)
        log_msg = f"[MAIN] Initialized project: {details['project']}, group: {details['group']}, env: {env_name}"
        if results_dir is not None and seed is not None:
            os.makedirs(results_dir, exist_ok=True)
            txt_path = os.path.join(results_dir, f"init_seed_{seed}.txt")
            with open(txt_path, "w") as f:
                f.write(log_msg + "\n")
                f.write(f"Seed: {seed}\n")
                f.write("Task details:\n")
                f.write(json.dumps(details, indent=2, default=str) + "\n")
    except Exception as e:
        logging.warning(f"Failed to write init log file for seed {seed} in {results_dir}: {e}")

    try:
        # Special toy dataset
        if details.get('env_name') == '8gaussians-multitarget':
            assert details['dataset_kwargs']['pr_data'] is not None, "No data for Point Robot"
            env = details['env_name']
            ds = Toy_dataset(env)
            env_max_steps = 150
        else:
            env, env_max_steps = create_env(details)
            ds = load_offline_dataset(details, env, env_max_steps)

        # Set up TensorBoard log directory
        log_dir = None
        if 'results_dir' in details:
            log_dir = os.path.join(details['results_dir'], "tensorboard")
            os.makedirs(log_dir, exist_ok=True)

        config_dict = details['rl_config'].copy()
        agent_bc, keys = create_agent(details, env, config_dict)
        agent_bc = train_agent(agent_bc, ds, details, keys, env, log_dir=log_dir)

        logging.info("[MAIN] Training completed for project: %s, group: %s, env: %s", details['project'], details['group'], details['env_name'])
    except Exception as e:
        logging.error("[MAIN] Exception in call_main for env: %s: %s", details.get('env_name'), str(e), exc_info=True)
        raise

# If this script is run directly, you can add a CLI or test entry here if needed.
