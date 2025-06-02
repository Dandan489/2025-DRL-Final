# Lightweight RL Agent in Full Length Real-Time Strategy Game - PPO GridNet / MAPPO

This repository is the PPO GridNet baseline and MAPPO implementation of our DRL 2025 final project. Note MAPPO approach can only trained on combat only map.

MAPPO implementation are modified from [light_mappo](https://github.com/tinyzqh/light_mappo)

## Requirements

The following is for WSL environment, there is also a guild for installing environment in [windows_env_install.md](/windows_env/windows_env_install.md)

### Prereq:

- Python 3.8+
- Poetry
- Java 8.0+
- FFmpeg

```setup
# create conda env with python 3.9 and activate
git clone --recursive https://github.com/Farama-Foundation/MicroRTS-Py.git
cd MicroRTS-Py
poetry install
poetry run pip install "torch==1.12.1" --upgrade --extra-index-url https://download.pytorch.org/whl/cu113
python hello_world.py
cd ..
python random_agent.py
```

## Training

First put the [maps](/maps) into `MicroRTS-Py/gym_microrts/microrts/maps`

### PPO GridNet

Then, to train the PPO GridNet models, run this command:

```train
```

### MAPPO

Then, to train the MAPPO models, run this command:

```train
python light_mappo/train.py # default trains AllLight (light only)
python light_mappo/train.py --map_paths maps/<map_name> --num_agents <num-agents> # to train a different map
```

For more training options, see `light_mappo/train.py` and `light_mappo/config.py`

## Evaluation

### PPO GridNet

To evaluate PPO GridNet model against AI, run:

```eval
```

### MAPPO

To evaluate MAPPO model against AI, run:

```eval
# default test 100 episode on AllLight2
python light_mappo/test.py --model_dir <path_to_model> --use_eval True
# Evaluate on another map
python light_mappo/test.py --model_dir <path_to_model> --use_eval True --map_paths maps/<map_name> --num_agents <num-agents>
```

For more testing options, see `light_mappo/test.py` and `light_mappo/config.py`

## Pre-trained Models

### PPO GridNet

PPO GridNet Pre-trained models are in

### MAPPO

MAPPO Pre-trained models are in `ALLlight/models` and `LHR2/models`

## Results

For full results and detailed analyze, see [report]()

### MAPPO

MAPPO results are shown in win rate/win/loss/tie (in 100 games)

| | coacAI | lightRushAI | workerRushAI | randomBiasedAI |
|--:|:-:|:-:|:-:|:-:|
| light only | 0/0/0/100 | 0/0/100/0 | 0/0/100/0 | 0.63/63/0/27 |
| LHR2 | 0/0/100/0 | 0/0/100/0 | 0/0/100/0 | 0/0/100/0 |
| light only 2 | 0/0/0/100 | 0/0/0/100 | - | 0.53/53/0/47 |
| light only 3 | 0/0/53/47 | 0/0/0/100  | - | 0.7/70/0/30 |

vs caoc

![](/assets/mappo/caoc.gif)

vs lightRush

![](/assets/mappo/lightRush.gif)

vs workerRush

![](/assets/mappo/workerRush.gif)

vs randomBiased

![](/assets/mappo/randomBiased.gif)