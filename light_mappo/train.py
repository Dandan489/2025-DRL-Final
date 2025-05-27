import sys
import os
import setproctitle
import numpy as np
from pathlib import Path
import torch

from config import get_config
from envs.MicroRTS_Env import MicroRTSVecEnv

def parse_args(args, parser):
    parser.add_argument(
        "--scenario_name", 
        type = str, 
        default = "MyEnv",
        help = "Which scenario to run on"
    )
    parser.add_argument(
        "--num_landmarks", 
        type = int, 
        default = 3
    )
    parser.add_argument(
        "--num_agents",
        type = int,
        default = 4,
        help = "number of players"
    )
    parser.add_argument(
        "--num_selfplay_envs",
        type = int,
        default = 0
    )
    parser.add_argument(
        "--ai2s",
        type = list,
        default = []
    )
    parser.add_argument(
        "--partial_obs",
        type = bool,
        default = False
    )
    parser.add_argument(
        "--max_steps",
        type = int,
        default = 2000
    )
    parser.add_argument(
        "--render_theme",
        type = int,
        default = 2
    )
    parser.add_argument(
        "--frame_skip",
        type = int,
        default = 0
    )
    parser.add_argument(
        "--map_paths",
        type = list,
        default = ["maps/AllLight.xml"]
    )
    parser.add_argument(
        "--reward_weight",
        type = list,
        default = [0, 0, 0, 0, 1, 5]
    )
    parser.add_argument(
        "--cycle_maps",
        type = list,
        default = []
    )
    parser.add_argument(
        "--autobuild",
        type = bool,
        default = False
    )
    parser.add_argument(
        "--jvm_args",
        type = list,
        default = []
    )

    all_args = parser.parse_known_args(args)[0]

    return all_args

def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo":
        assert all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy, "check recurrent policy!"
    elif all_args.algorithm_name == "mappo":
        assert (
            all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False
        ), "check recurrent policy!"
    else:
        raise NotImplementedError

    assert (
        all_args.share_policy == True and all_args.scenario_name == "simple_speaker_listener"
    ) == False, "The simple_speaker_listener scenario can not use shared policy. Please check the config.py."

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = (
        Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results")
        / all_args.env_name
        / all_args.scenario_name
        / all_args.algorithm_name
        / all_args.experiment_name
    )
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

    setproctitle.setproctitle(
        str(all_args.algorithm_name)
        + "-"
        + str(all_args.env_name)
        + "-"
        + str(all_args.experiment_name)
        + "@"
        + str(all_args.user_name)
    )

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = MicroRTSVecEnv(all_args)
    eval_envs = MicroRTSVecEnv(all_args) if all_args.use_eval else None
    all_args.n_rollout_threads = envs.env.num_envs
    all_args.n_eval_rollout_threads = envs.eval_envs.num_envs if all_args.use_eval else None
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
    }

    # run experiments
    if all_args.share_policy:
        from runner.shared.env_runner import EnvRunner as Runner
    else:
        from runner.separated.env_runner import EnvRunner as Runner

    runner = Runner(config)
    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    runner.writter.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
    runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
