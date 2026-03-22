# Booster Gym

Booster Gym is a reinforcement learning (RL) framework designed for humanoid robot locomotion developed by [Booster Robotics](https://boosterobotics.com/).

This project builds on the published [Booster Gym](https://github.com/BoosterRobotics/booster_gym) for an environment to learn stand up motions for the Booster T1 and Booster K1.

[![sim_stand_up](https://b-human.informatik.uni-bremen.de/public/MachineLearning/IsaacGym/Stand_Up_Env.gif)](https://b-human.informatik.uni-bremen.de/public/MachineLearning/IsaacGym/Stand_Up_Env.gif)

[![real_stand_up](https://b-human.informatik.uni-bremen.de/public/MachineLearning/IsaacGym/K1_Stand_Up.gif)](https://b-human.informatik.uni-bremen.de/public/MachineLearning/IsaacGym/K1_Stand_Up.gif)

## Features

- Training a stand up policy for the K1 and T1.
- One policy that handles both standing up from the back and front.
- The learned motion is inspired by Booster Robotics K1 and T1 stand up.
- The environment includes simulation fixes for IsaacGym, like fixes for the ground friction.
- The environment includes sim2real fixes.

## Overview

The framework supports the following stages for reinforcement learning:

1. **Training**: 

    - Train reinforcement learning policies using Isaac Gym with parallelized environments.

2. **Playing**:

    - **In-Simulation Testing**: Evaluate the trained policy in the same environment with training to ensure it behaves as expected.

## Installation

Follow these steps to set up your environment:

1. Create an environment with Python 3.8:

    ```sh
    $ conda create --name <env_name> python=3.8
    $ conda activate <env_name>
    ```

2. Install PyTorch with CUDA support:

    ```sh
    $ conda install numpy=1.21.6 pytorch=2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
    ```

3. Install Isaac Gym

    Download Isaac Gym from [NVIDIA’s website](https://developer.nvidia.com/isaac-gym/download).

    Extract and install:

    ```sh
    $ tar -xzvf IsaacGym_Preview_4_Package.tar.gz
    $ cd isaacgym/python
    $ pip install -e .
    ```

    Configure the environment to handle shared libraries, otherwise cannot found shared library of `libpython3.8`:

    ```sh
    $ cd $CONDA_PREFIX
    $ mkdir -p ./etc/conda/activate.d
    $ vim ./etc/conda/activate.d/env_vars.sh  # Add the following line
    export OLD_LD_LIBRARY_PATH=${LD_LIBRARY_PATH}
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
    $ mkdir -p ./etc/conda/deactivate.d
    $ vim ./etc/conda/deactivate.d/env_vars.sh  # Add the following line
    export LD_LIBRARY_PATH=${OLD_LD_LIBRARY_PATH}
    unset OLD_LD_LIBRARY_PATH
    ```
    
    Afterwards restart your Conda Environment.

4. Install Python dependencies:

    ```sh
    $ pip install -r requirements.txt
    ```

### Installation for newer GPUs

Newer GPUs need a newer pytorch version, which requires a newer python verion > 3.8. But IsaacGym needs Python <= 3.8.
To solve this problem, a custom pytorch (with cuda) wheel for python 3.8 is necessary. This can be done manually or a pre-build can
be downloaded from [here](https://b-human.informatik.uni-bremen.de/public/pytorch-cuda/torch-2.3.0a0+git63d5e92-cp38-cp38-linux_x86_64.whl). Note, that this wheel is **not** build by us, instead was found it in this [chinese blogpost](https://blog.csdn.net/m0_56706433/article/details/148902144). We are **not** responsible for unwanted side effects!

Before installing, make sure [cuda 12.8](https://developer.nvidia.com/cuda-12-8-0-download-archive) is installed and your nvidia drivers (below on the same page) are installed too.

Afterwards the wheel can be installed with pip:

    ```sh
    $ pip install torch-2.3.0a0+git63d5e92-cp38-cp38-linux_x86_64.whl
    ```

If mpi or libstd are missing:

    ```sh
    $ sudo apt install libopenmpi-dev
    $ conda install -c conda-forge libstdcxx-ng
    ```

In case there are problems with numpy or torchvision, install them once again and execute the wheel one more time:

    ```sh
    $ pip install torchvision==0.18.1
    $ pip install numpy==1.23.5
    $ pip install torch-2.3.0a0+git63d5e92-cp38-cp38-linux_x86_64.whl
    ```

## Usage

### 1. Training

To start training a policy, run the following command for K1:

```sh
$ python train.py --task=K1_Stand_Up --headless=0
```

Or the following command for T1:

```sh
$ python train.py --task=T1_Stand_Up --headless=0
```

Training logs and saved models will be stored in `logs/<date-time>/`.

#### Configurations

Training settings are loaded from `envs/<task>.yaml`. You can also override config values using command-line arguments:

- `--checkpoint`: Path of the model checkpoint to load (set to `-1` to use the most recent model).
- `--num_envs`: Number of environments to create.
- `--headless`: Run headless without creating a viewer window.
- `--sim_device`: Device for physics simulation (e.g., `cuda:0`, `cpu`). 
- `--rl_device`: Device for the RL algorithm (e.g., `cuda:0`, `cpu`). 
- `--seed`: Random seed.
- `--max_iterations`: Maximum number of training iterations.

To add a new task, create a config file in `envs/` and register the environment in `envs/__init__.py`.

#### Additional Optimization

The training itself can be optimized by using two different sets of parameters for the block `algorithm` in the `.yaml` file.
First train with a more aggressive set of values to speed up training and ensure local stuck policy states can be resolved for 10k - 20k episodes and 1024 robots.
Afterwards use a checkpoint that can stand up and use the second set of parameters and more robots for further fine tuning.

You can also test arround with lower entropy values during training to get higher success rate for the stand up, judged by the metric `fall`.
The ideal metric value for the `fall` reward should be between `[-0.2, 0.0]`. Unfortunatly, the current provided version does not reach this reproducible, only values between `[-0.6, -0.3]`.

#### Progress Tracking

To visualize training progress with [TensorBoard](https://www.tensorflow.org/tensorboard), run:

```sh
$ tensorboard --logdir logs
```

To use [Weights & Biases](https://wandb.ai/) for tracking, log in first:

```sh
$ wandb login
```

You can disable W&B tracking by setting `use_wandb` to `false` in the config file.

---

### 2. Playing

#### In-Simulation Testing

To test the trained policy in Isaac Gym, run:

```sh
$ python play.py --task=T1 --checkpoint=-1
```

Videos of the evaluation are automatically saved in `videos/<date-time>.mp4`. You can disable video recording by setting `record_video` to `false` in the config file.

### 3. Deployment

To deploy a trained policy through the Booster Robotics SDK in simulation or in the real world, export the model using:

```sh
$ python export_model.py --task=T1 --checkpoint=-1
```

This gives you the policy as a `.pt`. Follow Booster Robotics [instruction](https://github.com/BoosterRobotics/booster_gym) for the actual deployment.

### 4. Pre-Trained Policies and Reference Code

In the folder `/pre-trained` we provide working stand up policies for the T1 as well as K1.
Also, we provide our c++ code which we use to execute those policies. Note, that this code is **not** a stand-alone script.
It is only meant as a reference to show, how to correctly use the policies. On the real robot it is expected that the correct inputs for the policy are used.
This includes the joint sequence as well as the sensor data.
Additionally, to prevent unsafe states, like a stand up try that failed but keeps going, we define torso orientations which we interpolate inbetween. If the real robot leaves this defined state we break up the stand up to prevent damage to the robot hardware.

We are **not** responsible for any damage or wrong usage. It is **your** obligation to ensure safety and the correct execution of the policy. It is **your** obligation to test everything in simulation before deploying on the real robot.

This project is **not** meant as a full solution, instead it is meant as a starting point for reinforcement learning and as a very simple solution for stand up motions.

