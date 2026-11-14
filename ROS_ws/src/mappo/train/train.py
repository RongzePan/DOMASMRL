#!/usr/bin/env python
# -*- coding: utf-8 -*-

##################################################
### Module Imports and Initialization
##################################################

import sys
# System operations
import os
# OS interface
import numpy as np
# Numerical computing
from pathlib import Path
# Path object operations
import torch
# PyTorch deep learning framework

try:  # cosmetic only: name the training process in `top`/`ps`
    import setproctitle as _setproctitle
except ImportError:  # pragma: no cover - optional dependency
    _setproctitle = None

from mappo.envs.VirtualTurtlebot import EnvCore
# Custom DOMAS environment class

# Import configuration tools and environment wrappers
from mappo.config import get_config
from mappo.envs.env_wrappers import DummyVecEnv

"""Training entry point for DOMAS multi-agent TurtleBot navigation."""
##################################################
### Environment Creation Module
##################################################

def make_train_env(all_args):
    """Factory function to create training environments"""

    def get_env_fn(rank):
        """Closure function to generate individual environment instances"""

        def init_env():
            """Actual environment initialization logic"""
            # TODO: Important - Choose between continuous/discrete action space by
            # uncommenting either the above or below lines
            # Pass agent_num through constructor parameters
            env = EnvCore(num_agents=all_args.num_agents)
            # Key parameter passing point

            # Debug output (observation space validation)
            print(f"Env {rank} initialized. Observation space: {env.observation_space[0]}")

            # Alternative environment options:
            # from mappo.envs.env_continuous import ContinuousActionEnv
            # env = ContinuousActionEnv()

            # from envs.env_discrete import DiscreteActionEnv
            # env = DiscreteActionEnv()

            # env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    # Validate number of environment threads (must be > 0)
    assert all_args.n_rollout_threads > 0, "At least 1 environment thread required"

    # Create vectorized environment (parallel environment execution)
    return DummyVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    """Factory function to create evaluation environments (currently not fully implemented)"""

    def get_env_fn(rank):
        def init_env():
            # TODO: Important - Choose between continuous/discrete action space by
            # uncommenting either the above or below lines
            # Reserved interface for different environment type selection

            from mappo.envs.env_continuous import ContinuousActionEnv

            env = ContinuousActionEnv()
            # Alternative discrete action environment:
            # from envs.env_discrete import DiscreteActionEnv
            # env = DiscreteActionEnv()

            env.seed(all_args.seed + rank * 1000)

            return env

        return init_env

    return DummyVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


##################################################
### Argument Parsing Module
##################################################
def parse_args(args, parser):
    """Extend command line argument parsing with custom parameters"""

    # Add custom environment parameters
    parser.add_argument("--scenario_name", type=str, default="DOMAS",
                        help="Specify which scenario to run")
    parser.add_argument("--num_landmarks", type=int, default=3,
                        help="Number of landmarks in the environment")
    parser.add_argument("--num_agents", type=int, default=4,
                        help="Number of agent players (4 TurtleBots in DOMAS)")
    parser.add_argument("--obs_dim", type=int, default=28,
                        help="Observation dimension per agent")
    parser.add_argument("--act_dim", type=int, default=2,
                        help="Action dimension per agent")
    parser.add_argument("--config_path", type=str, default=None,
                        help="Optional YAML config (mappo/config/train_config.yaml)")
    parser.add_argument("--env", type=str, default=None,
                        help="Alias for --env_name used by train_mpe.sh")
    parser.add_argument("--train_steps", type=int, default=None,
                        help="Alias for --num_env_steps used by train_mpe.sh")

    # Parse known arguments (maintains compatibility with undefined args)
    all_args = parser.parse_known_args(args)[0]

    # Apply the YAML configuration as defaults, then the CLI aliases.
    if all_args.config_path:
        _apply_yaml_config(all_args, all_args.config_path)
    if all_args.env:
        all_args.env_name = all_args.env
    if all_args.train_steps:
        all_args.num_env_steps = int(all_args.train_steps)

    return all_args


def _strip_yaml_comment(line):
    """Remove a trailing YAML comment, respecting quoted strings."""
    out, quote = [], None
    for ch in line:
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
            out.append(ch)
        elif ch == "#":
            break
        else:
            out.append(ch)
    return "".join(out)


def _parse_yaml_scalar(value):
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        return [_parse_yaml_scalar(item) for item in value[1:-1].split(",") if item.strip()]
    if len(value) >= 2 and value[0] in "'\"" and value[-1] == value[0]:
        return value[1:-1]
    lowered = value.lower()
    if lowered in ("true", "yes"):
        return True
    if lowered in ("false", "no"):
        return False
    if lowered in ("null", "none", "~"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _simple_yaml_load(text):
    """Parse the small YAML subset used by ``train_config.yaml``.

    Keeps the training config functional on interpreters without PyYAML
    (e.g. a freshly created project venv).  Handles nested mappings by
    indentation, scalars, and inline ``[a, b]`` lists.
    """
    result = {}
    stack = [(-1, result)]
    for raw in text.splitlines():
        line = _strip_yaml_comment(raw).rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        content = line.strip()
        if ":" not in content:
            continue
        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        key, _, value = content.partition(":")
        key = key.strip().strip("'\"")
        value = value.strip()
        if value == "":
            child = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_yaml_scalar(value)
    return result


def _apply_yaml_config(all_args, config_path):
    """Overlay ``train_config.yaml`` values onto already-parsed defaults.

    Only keys that were left at their parser defaults are overridden, so an
    explicit command-line flag always wins over the YAML file.  Falls back to a
    built-in parser for the config's simple YAML subset when PyYAML is absent.
    """
    try:
        import yaml
        loader = yaml.safe_load
    except ImportError:  # pragma: no cover - optional dependency
        loader = _simple_yaml_load
    path = Path(config_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / config_path
    if not path.is_file():
        print(f"[train] config file not found: {path}; using CLI defaults")
        return
    with path.open("r", encoding="utf-8") as handle:
        cfg = loader(handle.read()) or {}
    defaults = {
        "num_agents": cfg.get("num_agents"),
        "obs_dim": cfg.get("obs_dim"),
        "act_dim": cfg.get("act_dim"),
    }
    training = cfg.get("training") or {}
    defaults["num_env_steps"] = training.get("max_steps")
    defaults["eval_interval"] = training.get("eval_interval")
    defaults["save_interval"] = training.get("save_interval")
    agent = cfg.get("agent") or {}
    defaults["lr"] = agent.get("learning_rate")
    defaults["gamma"] = agent.get("gamma")
    defaults["entropy_coef"] = agent.get("entropy_coef")
    defaults["clip_param"] = agent.get("clip_param")
    defaults["ppo_epoch"] = agent.get("update_epochs")
    parser = get_config()
    base = parser.parse_args([])
    for key, value in defaults.items():
        if value is None:
            continue
        # Override only when the caller did not change the value explicitly.
        if getattr(all_args, key, None) == getattr(base, key, None):
            setattr(all_args, key, value)


##################################################
### Main Program Module
##################################################
def main(args):
    """Main training workflow"""
    # Initialize argument parser
    parser = get_config()  # Get base configuration parser
    all_args = parse_args(args, parser)  # Parse extended arguments

    ########## Algorithm Configuration Validation ##########
    # Check algorithm and recurrent policy compatibility
    if all_args.algorithm_name == "rmappo":
        assert all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy, \
            "Recurrent policy required for RMAPPO!"
    elif all_args.algorithm_name == "mappo":
        assert (not all_args.use_recurrent_policy and not all_args.use_naive_recurrent_policy), \
            "Recurrent policy not allowed for MAPPO!"
    else:
        raise NotImplementedError

    # Special scenario policy check
    assert not (all_args.share_policy and all_args.scenario_name == "simple_speaker_listener"), \
        "Shared policy not allowed for simple_speaker_listener scenario. Check config.py."

    ########## Hardware Configuration ##########
    # CUDA device initialization
    if all_args.cuda and torch.cuda.is_available():
        print("Using GPU acceleration...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            # Enable deterministic mode
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("Using CPU...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    ########## Experiment Path Management ##########
    # Build result saving path
    # run dir
    # All experiment artifacts live under ROS_ws/results/.  parents[3] of
    # src/mappo/train/train.py is the ROS_ws root.
    run_dir = (
        Path(__file__).resolve().parents[3] / "results"
        / all_args.env_name
        / all_args.scenario_name
        / all_args.algorithm_name
        / all_args.experiment_name
    )
    # Auto-generate run number (run1, run2...)

    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if not run_dir.exists():
        curr_run = "run1"
    else:
        exst_run_nums = [
            int(str(folder.name).split("run")[1])
            for folder in run_dir.iterdir()
            if str(folder.name).startswith("run")
        ]
        if len(exst_run_nums) == 0:
            curr_run = "run1"
        else:
            curr_run = "run%i" % (max(exst_run_nums) + 1)
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))
    # Recursive directory creation

    # Set process name (for monitoring); optional dependency.
    if _setproctitle is not None:
        _setproctitle.setproctitle(
            str(all_args.algorithm_name)
            + "-"
            + str(all_args.env_name)
            + "-"
            + str(all_args.experiment_name)
            + "@"
            + str(all_args.user_name)
        )
    ########## Random Seed Configuration ##########

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    ########## Environment Initialization ##########
    envs = make_train_env(all_args)  # Training environment
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = all_args.num_agents

    ########## Training Configuration ##########
    config = {
        "all_args": all_args,  # Complete parameters
        "envs": envs,  # Vectorized training envs
        "eval_envs": eval_envs,  # Evaluation envs (optional)
        "num_agents": num_agents,  # Number of agents
        "device": device,  # Computation device
        "run_dir": run_dir,  # Result saving path
    }

    ########## Training Execution ##########
    # Select runner based on policy type
    if all_args.share_policy:
        from mappo.runner.shared.env_runner import EnvRunner as Runner
    else:
        from mappo.runner.separated.env_runner import EnvRunner as Runner

    runner = Runner(config)  # Initialize training runner
    runner.run()  # Start training loop

    ########## Post-processing ##########
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    # Export log data. ``writter`` is only created in non-render mode.
    writer = getattr(runner, "writter", None)
    if writer is not None:
        if hasattr(writer, "export_scalars_to_json"):
            writer.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
        writer.close()


if __name__ == "__main__":
    main(sys.argv[1:])
    # Skip the first argument (the script itself)
