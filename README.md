# Lightweight RL Agent in Full Length Real-Time Strategy Game

This repository is the official implementation of our DRL 2025 final project. Note MAPPO approach can only trained on combat only map.

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

First put the [maps](/maps) into `MicroRTS-Py\gym_microrts\microrts\maps`

Then, to train the models in the paper, run this command:

```train
python light_mappo/train.py # default trains AllLight (light only)
python light_mappo/train.py --map_paths maps/<map_name> --num_agents <num-agents> # to train a different map
```

For more training options, see `light_mappo/train.py` and `light_mappo/config.py`

## Evaluation

To evaluate model against AI, run:

```eval
# default test 100 episode on AllLight2
python light_mappo/test.py --model_dir <path_to_model> --use_eval True
# Evaluate on another map
python light_mappo/test.py --model_dir <path_to_model> --use_eval True --map_paths maps/<map_name> --num_agents <num-agents>
```

For more testing options, see `light_mappo/test.py` and `light_mappo/config.py`

## Pre-trained Models

Pre-trained models are in `ALLlight/models` and `LHR2/models`

## Results

Shown in win rate/win/loss/tie (in 100 games)

| | coacAI | lightRushAI | workerRushAI | randomBiasedAI |
|--:|:-:|:-:|:-:|:-:|
| light only | 0/0/0/100 | 0/0/100/0 | 0/0/100/0 | 0.63/63/0/27 |
| LHR2 | 0/0/100/0 | 0/0/100/0 | 0/0/100/0 | 0/0/100/0 |
| light only 2 | 0/0/0/100 | 0/0/0/100 | - | 0.53/53/0/47 |
| light only 3 | 0/0/53/47 | 0/0/0/100  | - | 0.7/70/0/30 |