# Residual-MPPI
Residual-MPPI implementation of the MuJoCo experiments part.

## Installation
### Using Conda
```
conda create -n residual-mppi python=3.9
conda activate residual-mppi
pip install setuptools==66.0.0 wheel==0.38.4
pip install -r requirements.txt
```

### MuJoCo Installation
You could download and unzip the file of mujoco210 via:
```
cd ~
mkdir .mujoco/
wget https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
tar -zxvf ~/.mujoco/mujoco210-linux-x86_64.tar.gz -C ~/.mujoco
```

Add the corresponding lines to your `.bashrc` file:
```
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia' >> ~/.bashrc
```

## Src
The core algorithm part of our method is in the `mppi.py`.

## Running
We provide a script `main.py` for the our experiments on all the MuJoCo environment, which includes the evaluation of RL prior, IL prior, and customized policies. 
You could switch to any prior policies by modifying the `parameters/hyperpara.yml`.
