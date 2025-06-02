```
# clone repository
git clone --recursive https://github.com/Farama-Foundation/MicroRTS-Py.git 
conda env create -f .\windows_env\environment-windows.yml
conda activate drl-final-windows
cd MicroRTS-Py
# copy windows_env\build.ps1 to MicroRTS-Py\build.ps1
# modify MicroRTS-Py\gym_microrts\envs\vec_env.py line 94 to run ["powershell", "-ExecutionPolicy", "Bypass", "-File", "build.ps1"]
poetry install
pip install torch==1.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
python hello_world.py
cd ..
python random_agent.py
```