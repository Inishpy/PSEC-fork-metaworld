# Launcher for RL training on diffusion model using default config

from examples.states.configs.train_config import get_config
from examples.states.train_diffusion_psec import call_main

def main():
    # Get default config for "fisor" (diffusion RL)
    config = get_config("fisor")
    # Convert ConfigDict to a plain dict for compatibility
    details = dict(config)
    # Add required fields for call_main if missing
    details.setdefault("group", "default_group")
    details.setdefault("experiment_name", "diffusion_rl_default")
    details.setdefault("project", "Lora-FISOR")
    details.setdefault("env_name", "8gaussians-multitarget")  # Default from config, change if needed
    details.setdefault("rl_config", config.agent_kwargs)
    details.setdefault("dataset_kwargs", config.dataset_kwargs)
    details.setdefault("results_dir", "./results/diffusion_rl_default")
    details.setdefault("inference_variants", [{}])
    details.setdefault("training_time_inference_params", {})
    details.setdefault("online_rl", False)
    details.setdefault("timestamp", None)

    call_main(details)

if __name__ == "__main__":
    main()