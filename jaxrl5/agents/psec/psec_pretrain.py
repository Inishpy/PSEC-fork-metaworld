"""Implementations of algorithms for continuous control with parameter-level composition."""
import os
from functools import partial
from typing import Dict, Optional, Tuple, Union, List
import flax.linen as nn
import gym
import gym.spaces
import jax
import jax.numpy as jnp
import optax
import flax
import pickle
from flax.training.train_state import TrainState
from flax import struct
import numpy as np
from jaxrl5.agents.agent import Agent
from jaxrl5.data.dataset import DatasetDict
from jaxrl5.networks import MLP, DDPM, FourierFeatures, ddpm_sampler_eval_bc, cosine_beta_schedule, MLPResNet, vp_beta_schedule
import tensorflow as tf
from tensorboardX import SummaryWriter
import logging
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

def mish(x):
    return x * jnp.tanh(nn.softplus(x))


class CompositionWeightNetwork(nn.Module):
    """Network to learn composition weights for each prior model."""
    hidden_dims: Tuple[int, ...] = (256, 256)
    num_priors: int = 1
    
    @nn.compact
    def __call__(self, observations):
        x = observations
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = mish(x)
        
        # Output weights for each prior model
        weights = nn.Dense(self.num_priors)(x)
        # Apply softmax to get normalized weights
        weights = nn.softmax(weights, axis=-1)
        return weights


class PretrainWithComposition(Agent):
    # Diffusion actor
    score_model: TrainState
    target_score_model: TrainState
    # Online RL critics
    critic: TrainState
    target_critic: TrainState
    # Composition
    composition_weights: TrainState  # Learnable weights for composition
    # RL hyperparameters
    discount: float
    tau: float
    actor_tau: float
    N: int #How many samples per observation
    ddpm_temperature: float
    betas: jnp.ndarray
    alphas: jnp.ndarray
    alpha_hats: jnp.ndarray
    composition_temperature: float

    # All fields below have no default values and must come first
    prior_models: List[Dict] = struct.field(pytree_node=False)  # List of loaded prior models
    prior_env_names: List[str] = struct.field(pytree_node=False)  # Names of prior tasks
    act_dim: int = struct.field(pytree_node=False)
    T: int = struct.field(pytree_node=False)
    M: int = struct.field(pytree_node=False) #How many repeat last steps
    clip_sampler: bool = struct.field(pytree_node=False)
    use_composition: bool = struct.field(pytree_node=False)

    # Fields with default values must come after all non-default fields
    temp: Optional[TrainState] = None  # For entropy regularization (optional)
    logger: Optional[SummaryWriter] = struct.field(pytree_node=False, default=None)
    update_step: int = 0

    @classmethod
    def create(

        cls,
        seed: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Box,
        actor_lr: Union[float, optax.Schedule] = 3e-4,
        critic_lr: float = 3e-4,
        discount: float = 0.99,
        tau: float = 0.005,
        ddpm_temperature: float = 1.0,
        actor_num_blocks: int = 2,
        actor_tau: float = 0.001,
        actor_dropout_rate: Optional[float] = None,
        actor_layer_norm: bool = False,
        T: int = 5,
        time_dim: int = 64,
        N: int = 64,
        M: int = 0,
        clip_sampler: bool = True,
        beta_schedule: str = 'vp',
        decay_steps: Optional[int] = int(2e6),
        results_dir: str = './results/pretrain/20251015-124601',
        current_task: str = None,
        use_composition: bool = True,
        composition_lr: float = 1e-3,
        composition_temperature: float = 1.0,
        logger_path: str = './logs',
        exclude_tasks: Optional[List[str]] = None,
    ):

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, comp_key = jax.random.split(rng, 4)
        actions = action_space.sample()
        observations = observation_space.sample()
        action_dim = action_space.shape[0]
        
        # Load prior models
        prior_models = []
        prior_env_names = []

        if use_composition and current_task:
            # Build exclusion set: current_task + any in exclude_tasks
            exclusion_set = set()
            if exclude_tasks is not None:
                exclusion_set.update(exclude_tasks)
            exclusion_set.add(current_task)

            for env_name in env_names:
                if env_name in exclusion_set:
                    continue  # Exclude current and any specified tasks
                model_path = os.path.join(results_dir, env_name)
                if os.path.exists(model_path):
                    # Find the latest model file
                    model_files = [f for f in os.listdir(model_path) if f.startswith('model') and f.endswith('.pickle')]
                    if model_files:
                        # Sort by timestamp in filename
                        model_files.sort(key=lambda x: int(x.replace('model', '').replace('.pickle', '')))
                        latest_model = os.path.join(model_path, model_files[-1])
                        try:
                            with open(latest_model, 'rb') as f:
                                prior_model = pickle.load(f)
                            prior_models.append(prior_model)
                            prior_env_names.append(env_name)
                            print(f"Loaded prior model from {env_name}")
                        except Exception as e:
                            print(f"Failed to load model from {env_name}: {e}")
        
        num_priors = len(prior_models)
        print(f"Loaded {num_priors} prior models for composition")
        
        preprocess_time_cls = partial(FourierFeatures,
                                      output_size=time_dim,
                                      learnable=True)
        
        cond_model_cls = partial(MLP,
                                hidden_dims=(128, 128),
                                activations=mish,
                                activate_final=False)
        
        if decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)

        base_model_cls = partial(MLPResNet,
                                    use_layer_norm=actor_layer_norm,
                                    num_blocks=actor_num_blocks,
                                    dropout_rate=actor_dropout_rate,
                                    out_dim=action_dim,
                                    activations=mish,
                                    )

        actor_def = DDPM(time_preprocess_cls=preprocess_time_cls,
                            cond_encoder_cls=cond_model_cls,
                            reverse_encoder_cls=base_model_cls)
        
        time = jnp.zeros((1, 1))
        observations = jnp.expand_dims(observations, axis=0)
        actions = jnp.expand_dims(actions, axis = 0)
        actor_params = actor_def.init(actor_key, observations, actions, time)['params']

        actor_optimiser = optax.adamw(learning_rate=actor_lr)

        score_model = TrainState.create(apply_fn=actor_def.apply,
                                        params=actor_params,
                                        tx=actor_optimiser)
        
        target_score_model = TrainState.create(apply_fn=actor_def.apply,
                                               params=actor_params,
                                               tx=optax.GradientTransformation(
                                                    lambda _: None, lambda _: None))
        
        # === Critic and Target Critic for Online RL ===
        from jaxrl5.networks.state_action_value import StateActionValue
        from jaxrl5.networks.mlp import MLP as CriticMLP

        critic_base_cls = partial(CriticMLP, hidden_dims=(256, 256), activate_final=True)
        critic_def = StateActionValue(base_cls=critic_base_cls)
        critic_params = critic_def.init(critic_key, observations, actions)['params']
        critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=optax.adam(learning_rate=critic_lr),
        )
        target_critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        # Initialize composition weight network
        composition_weights = None
        if use_composition and num_priors > 0:
            comp_def = CompositionWeightNetwork(num_priors=num_priors + 1)  # +1 for the base model
            comp_params = comp_def.init(comp_key, observations)
            comp_optimizer = optax.adam(learning_rate=composition_lr)
            composition_weights = TrainState.create(
                apply_fn=comp_def.apply,
                params=comp_params,
                tx=comp_optimizer
            )
        
        # Initialize logger
        logger = None
        if use_composition and logger_path:
            logger = SummaryWriter(logger_path)

        if beta_schedule == 'cosine':
            betas = jnp.array(cosine_beta_schedule(T))
        elif beta_schedule == 'linear':
            betas = jnp.linspace(1e-4, 2e-2, T)
        elif beta_schedule == 'vp':
            betas = jnp.array(vp_beta_schedule(T))
        else:
            raise ValueError(f'Invalid beta schedule: {beta_schedule}')

        alphas = 1 - betas
        alpha_hat = jnp.array([jnp.prod(alphas[:i + 1]) for i in range(T)])

        return cls(
            actor=None, # Base class attribute
            score_model=score_model,
            target_score_model=target_score_model,
            critic=critic,
            target_critic=target_critic,
            composition_weights=composition_weights,
            prior_models=prior_models,
            prior_env_names=prior_env_names,
            tau=tau,
            discount=discount,
            rng=rng,
            betas=betas,
            alpha_hats=alpha_hat,
            act_dim=action_dim,
            T=T,
            N=N,
            M=M,
            alphas=alphas,
            ddpm_temperature=ddpm_temperature,
            actor_tau=actor_tau,
            clip_sampler=clip_sampler,
            use_composition=use_composition,
            composition_temperature=composition_temperature,
            logger=logger,
            update_step=0,
        )

    # === Online RL methods for PretrainWithComposition ===

    def update_critic(self, batch: DatasetDict) -> Tuple["PretrainWithComposition", Dict[str, float]]:
        masks = 1.0 - batch["dones"]
        rng = self.rng
        key, rng = jax.random.split(rng)
        time = jnp.zeros((batch["next_observations"].shape[0], 1))
        next_actions, _ = self.eval_actions_bc(batch["next_observations"][0], train_lora=False)
        next_actions = jnp.array(next_actions)
        if next_actions.ndim == 1:
            next_actions = next_actions[None, :]
        key, rng = jax.random.split(rng)
        next_q = self.target_critic.apply_fn(
            {"params": self.target_critic.params},
            batch["next_observations"],
            next_actions,
            True,
            rngs={"dropout": key},
        )
        if next_q.ndim == 2 and next_q.shape[-1] == 1:
            next_q = jnp.squeeze(next_q, axis=-1)
        target_q = batch["rewards"] + self.discount * masks * next_q

        key, rng = jax.random.split(rng)
        def critic_loss_fn(critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qs = self.critic.apply_fn(
                {"params": critic_params},
                batch["observations"],
                batch["actions"],
                True,
                rngs={"dropout": key},
            )
            critic_loss = ((qs - target_q[None, :]) ** 2).mean()
            return critic_loss, {"critic_loss": critic_loss, "q": qs.mean()}

        grads, info = jax.grad(critic_loss_fn, has_aux=True)(self.critic.params)
        critic = self.critic.apply_gradients(grads=grads)

        target_critic_params = optax.incremental_update(
            critic.params, self.target_critic.params, self.tau
        )
        target_critic = self.target_critic.replace(params=target_critic_params)

        return self.replace(critic=critic, target_critic=target_critic, rng=rng), info

    def update_actor(self, batch: DatasetDict) -> Tuple["PretrainWithComposition", Dict[str, float]]:
        rng = self.rng
        key, rng = jax.random.split(rng)
        time = jnp.zeros((batch["observations"].shape[0], 1))

        def actor_loss_fn(score_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            actions = self.score_model.apply_fn({"params": score_params},
                                                batch["observations"],
                                                batch["actions"],
                                                time,
                                                rngs={"dropout": key},
                                                training=True)
            qs = self.critic.apply_fn(
                {"params": self.critic.params},
                batch["observations"],
                actions,
                True,
                rngs={"dropout": key},
            )
            q = qs.mean(axis=0)
            actor_loss = -q.mean()
            return actor_loss, {"actor_loss": actor_loss}

        grads, actor_info = jax.grad(actor_loss_fn, has_aux=True)(self.score_model.params)
        score_model = self.score_model.apply_gradients(grads=grads)

        return self.replace(score_model=score_model, rng=rng), actor_info

    @partial(jax.jit, static_argnames="utd_ratio")
    def update(self, batch: DatasetDict, utd_ratio: int):
        new_agent = self
        for i in range(utd_ratio):
            def slice(x):
                assert x.shape[0] % utd_ratio == 0
                batch_size = x.shape[0] // utd_ratio
                return x[batch_size * i : batch_size * (i + 1)]
            mini_batch = jax.tree_util.tree_map(slice, batch)
            new_agent, critic_info = new_agent.update_critic(mini_batch)
        new_agent, actor_info = new_agent.update_actor(mini_batch)
        return new_agent, {**actor_info, **critic_info}
    
    def compose_parameters(self, params, observation, weights):
        """Compose parameters from base model and prior models using learned weights."""
        if not self.use_composition or len(self.prior_models) == 0:
            return params
        
        # Stack all parameters (base + priors)
        all_params = [params] + [prior['score_model']['params'] for prior in self.prior_models]
        
        # Apply weighted combination
        composed_params = jax.tree_util.tree_map(
            lambda *ps: sum(w * p for w, p in zip(weights, ps)),
            *all_params
        )
        
        return composed_params
    
    def update_actor(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        rng = agent.rng
        key, rng = jax.random.split(rng, 2)
        time = jax.random.randint(key, (batch['actions'].shape[0], ), 0, agent.T)
        key, rng = jax.random.split(rng, 2)
        noise_sample = jax.random.normal(
            key, (batch['actions'].shape[0], agent.act_dim))
        
        alpha_hats = agent.alpha_hats[time]
        time = jnp.expand_dims(time, axis=1)
        alpha_1 = jnp.expand_dims(jnp.sqrt(alpha_hats), axis=1)
        alpha_2 = jnp.expand_dims(jnp.sqrt(1 - alpha_hats), axis=1)
        observation = batch['observations']
        noisy_actions = alpha_1 * batch['actions'] + alpha_2 * noise_sample

        key, rng = jax.random.split(rng, 2)
        
        if agent.use_composition and agent.composition_weights is not None:
            # Joint loss for both base model and composition weights
            def joint_loss_fn(score_params, comp_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
                # Get composition weights
                weights = agent.composition_weights.apply_fn(comp_params, observation)
                weights = weights * agent.composition_temperature
                
                # Compose parameters
                composed_params = agent.compose_parameters(score_params, observation, weights[0])
                
                # Forward pass with composed parameters
                eps_pred = agent.score_model.apply_fn({'params': composed_params},
                                           observation,
                                           noisy_actions,
                                           time,
                                           rngs={'dropout': key},
                                           training=True,
                                           )
                
                actor_loss = (((eps_pred - noise_sample) ** 2).sum(axis=-1)).mean()
                
                # Add entropy regularization for weights to encourage exploration
                entropy_reg = -0.01 * (weights * jnp.log(weights + 1e-8)).sum(axis=-1).mean()
                total_loss = actor_loss + entropy_reg
                
                # Prepare info dict with individual weights
                info_dict = {
                    'actor_loss': actor_loss, 
                    'entropy_reg': entropy_reg,
                    f'weight_base': weights[0, 0].mean()
                }
                
                # Add weights for each prior
                for i, env_name in enumerate(agent.prior_env_names[:]):  # Limit to first 10 for logging
                    info_dict[f'weight_{env_name}'] = weights[0, i+1].mean()
                
                return total_loss, info_dict
            
            # Compute gradients for both networks
            (loss, info), (score_grads, comp_grads) = jax.value_and_grad(
                joint_loss_fn, argnums=(0, 1), has_aux=True
            )(agent.score_model.params, agent.composition_weights.params)
            
            # Update both networks
            score_model = agent.score_model.apply_gradients(grads=score_grads)
            composition_weights = agent.composition_weights.apply_gradients(grads=comp_grads)
            
            agent = agent.replace(score_model=score_model, composition_weights=composition_weights)
            
        else:
            # Standard update without composition
            def actor_loss_fn(score_model_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
                eps_pred = agent.score_model.apply_fn({'params': score_model_params},
                                           observation,
                                           noisy_actions,
                                           time,
                                           rngs={'dropout': key},
                                           training=True,
                                           )

                actor_loss = (((eps_pred - noise_sample) ** 2).sum(axis=-1)).mean()
                return actor_loss, {'actor_loss': actor_loss}

            grads, info = jax.grad(actor_loss_fn, has_aux=True)(agent.score_model.params)
            score_model = agent.score_model.apply_gradients(grads=grads)
            agent = agent.replace(score_model=score_model)
        
        # Update target network
        target_score_params = optax.incremental_update(
            score_model.params, agent.target_score_model.params, agent.actor_tau
        )
        target_score_model = agent.target_score_model.replace(params=target_score_params)
        
        # Log weights to TensorBoard
        # Avoid Python boolean logic on JAX tracers inside JIT-compiled functions.
        # Only log if update_step is a regular Python int (i.e., not inside JIT).
        if (
            agent.use_composition
            and agent.logger is not None
            and isinstance(agent.update_step, int)
            and agent.update_step % 100 == 0
        ):
            for key, value in info.items():
                if 'weight' in key:
                    agent.logger.add_scalar(f'composition_weights/{key}', float(value), agent.update_step)
        
        new_agent = agent.replace(
            score_model=score_model, 
            target_score_model=target_score_model, 
            rng=rng,
            update_step=agent.update_step + 1
        )
        
        return new_agent, info
    
    def eval_actions_bc(self, observations: jnp.ndarray, train_lora: bool = False):
        rng = self.rng

        assert len(observations.shape) == 1
        observations = jax.device_put(observations)
        observations = jnp.expand_dims(observations, axis=0).repeat(self.N, axis = 0)

        # Get composed parameters for evaluation
        if self.use_composition and self.composition_weights is not None:
            weights = self.composition_weights.apply_fn(
                self.composition_weights.params, observations[0:1]
            )
            weights = weights * self.composition_temperature
            score_params = self.compose_parameters(
                self.target_score_model.params, observations[0:1], weights[0]
            )
        else:
            score_params = self.target_score_model.params

        actions, rng = ddpm_sampler_eval_bc(
            self.score_model.apply_fn, score_params, self.T, rng, self.act_dim, 
            observations, self.alphas, self.alpha_hats, self.betas, 
            self.ddpm_temperature, self.M, self.clip_sampler, training=False
        )
        rng, _ = jax.random.split(rng, 2)
        idx = 0

        action = actions[idx]
        new_rng = rng
        return np.array(action.squeeze()), self.replace(rng=new_rng)
    
    @jax.jit
    def update_bc(self, batch: DatasetDict):
        new_agent = self

        new_agent, actor_info = new_agent.update_actor(batch)

        return new_agent, actor_info
    
    def save(self, modeldir, save_time):
        file_name = 'model' + str(save_time) + '.pickle'
        state_dict = flax.serialization.to_state_dict(self)
        
        # Save composition weights separately if they exist
        if self.use_composition and self.composition_weights is not None:
            comp_file_name = 'composition_weights_' + str(save_time) + '.pickle'
            comp_dict = flax.serialization.to_state_dict(self.composition_weights)
            pickle.dump(comp_dict, open(os.path.join(modeldir, comp_file_name), 'wb'))
            
            # Also save the list of prior env names for reference
            prior_file_name = 'prior_tasks_' + str(save_time) + '.pickle'
            pickle.dump(self.prior_env_names, open(os.path.join(modeldir, prior_file_name), 'wb'))
        
        pickle.dump(state_dict, open(os.path.join(modeldir, file_name), 'wb'))

    def load(self, model_location):
        pkl_file = pickle.load(open(model_location, 'rb'))
        new_agent = flax.serialization.from_state_dict(target=self, state=pkl_file)
        
        # Try to load composition weights if they exist
        model_dir = os.path.dirname(model_location)
        model_name = os.path.basename(model_location)
        timestamp = model_name.replace('model', '').replace('.pickle', '')
        
        comp_location = os.path.join(model_dir, f'composition_weights_{timestamp}.pickle')
        if os.path.exists(comp_location):
            comp_file = pickle.load(open(comp_location, 'rb'))
            new_agent.composition_weights = flax.serialization.from_state_dict(
                target=new_agent.composition_weights, state=comp_file
            )
        
        return new_agent