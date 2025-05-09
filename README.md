# 2025-DRL-Final

https://github.com/Farama-Foundation/MicroRTS-Py

### Prereq (In WSL):
- Python 3.8+
- Poetry
- Java 8.0+
- FFmpeg

create conda env with python 3.9 and activate 
git clone --recursive https://github.com/Farama-Foundation/MicroRTS-Py.git 
cd MicroRTS-Py 
poetry install 
poetry run pip install "torch==1.12.1" --upgrade --extra-index-url https://download.pytorch.org/whl/cu113
python hello_world.py