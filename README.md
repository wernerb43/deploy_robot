# Hardware Deploy
Repo for deploying on hardware. 

## Installation
### ROS2
Use `ROS2 Humble` to communicate across different pieces of code. Install instrictions are here:
```bash
https://docs.ros.org/en/humble/Installation.html
```

### Conda Environment
Set the directory of this repo as a enviorment variable in your `~/.bashrc` file, for example:
```bash
export DEPLOY_ROOT_DIR="/home/sergio/projects/deploy_robot"
```

Use `conda` and install via:
```bash
conda env create -f environment.yml
conda activate env_deploy
```