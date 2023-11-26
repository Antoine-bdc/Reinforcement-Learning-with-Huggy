# Deep learning project: Training Huggy the dog

## Introduction 

### Project Description

This is a toy project which aims to train a virtual dog (Huggy) to fetch a virtual stick. The dog can perceive the environment (i.e. where the stick is) and perform certain actions (move its legs). Reinforcement learning is here used to train the artificial neural network that decides what action Huggy will perform at each step.

### What is in this repository 

This repository contains various files:
- The python notebook with the Huggy's training routine
- The training results in their mlagents output raw format
- The project report


### Technical implementation

Huggy the dog and the environment are simulated using the [Unity engine](https://unity.com). The reinforcement learning is performed by the [ML-Agents Toolkit](https://github.com/Unity-Technologies/ml-agents), which itself implements reinforcement learning algorithms using [Pytorch](https://pytorch.org/). The training is handled using their python API.

## How to use

### Training 

To train Huggy, multiple things are needed:
- A linux environment (necessary to run the Huggy scene)
- A Python 3.10 installation with mlagents and mlagents-envs library installed
- A GPU and its appropriate drivers to perform the training

While it is possible to have all these locally, the easiest way to satisfy all these requirement is to use a Google Colab Python Notebook instance connected to a T4 GPU

Note: we have provided a requirements.txt file which includes `ml-agents` and `ml-agents-envs`. These libraries however should be installed directly from the [ml-agents github repository](https://github.com/Unity-Technologies/ml-agents/releases):

```bash
git clone --branch release_21 https://github.com/Unity-Technologies/ml-agents.git
pip3 install torch~=1.13.1 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install ./ml-agents-envs
python -m pip install ./ml-agents
```

For more information on mlagents installation, please refer to their [documentation](https://unity-technologies.github.io/ml-agents/Installation/).

In `src/huggy_training.ipynb` is a copy of the notebook containing all installation and training routines which should run without issue on a Google colab instance. If it is still accessible by the time you read this, it is possible to see our notebook directly on [Google Colab](https://colab.research.google.com/drive/1I_j43naO60VNvIlT4UX4ekfR7nZcM7Jd?usp=sharing)

### Analysing the results

We have used this notebook to train multiple Huggy models using various hyperparameters. The results of these training can be found in the `results` directory.

To assess the training performance, we have used tensorboard to visualise the training logs. In particular we were interested in the cumulative reward over time.

To visualise these results, tensorboard needs to be installed, and one must run:

```bash
tensorboard --logdir results
```

For more convenience, we have hosted a tensorboard instance showing these exact results. If the server is still running, this hosted board can be found here: http://matrix-antoine-bdc.duckdns.org:6006

Alternatively, some of these cumulative reward function plots can be found in `report/plots/ppo_results`.


### Playing with Huggy the dog

Finally, once the model is trained, it is possible to load it in a special unity scene and see how well it fetches the stick. This special scene is hosted on Thomas Simonini's HuggingFace page, the creator of the Huggy project .

In order to play with the models we trained:
- Go to this page: https://huggingface.co/spaces/ThomasSimonini/Huggy
- Select our HuggingFace repository: `Cereale/trained_huggy_models`
- Select the model you want and click 'Play'

According to our analysis:
- PPO_baseline is among our best performing model
- SAC_baseline is a test of the SAC algorithm in our case, it doesn't perform well

Note: while we have loaded all 17 trained models to the [Hugging Face repository](https://huggingface.co/Cereale/trained_huggy_models/tree/main/models), it seems that the hugging face webapp doesn't offer all of them to play with.

## Credits

Project made by:
- [Attoumassa Samaké](https://github.com/Attoumassa)
- [Vixra Keo](https://github.com/Vixk2021)
- [Antoine Bedouch](https://github.com/Antoine-bdc)

Huggy the Dog is an environment created by [Thomas Simonini](https://github.com/simoninithomas) based on [Puppo The Corgi](https://blog.unity.com/engine-platform/puppo-the-corgi-cuteness-overload-with-the-unity-ml-agents-toolkit)

