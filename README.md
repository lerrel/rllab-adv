>Under Development
# Robust Adversarial Reinforcement Learning

This repo contains code for training RL agents with adversarial disturbance agents in our work on Robust Adversarial Reinforcement Learning ([RARL](https://arxiv.org/abs/1703.02702)). We build heavily build on the OpenAI rllab repo.

## Installation instructions

Since we build upon the [rllab](https://github.com/openai/rllab) package for the optimizers, the installation process is similar to `rllab's` manual installation. Most of the packages are virtually installated in the anaconda `rllab3-adv` enivronment.

- Dependencies for scipy:

```
sudo apt-get build-dep python-scipy
```

- Install python modules:

```
conda env create -f environment.yml
```

- [Install MuJoCo](https://github.com/openai/mujoco-py)

- Add `rllab-adv` to your `PYTHONPATH`.

```
export PYTHONPATH=<PATH_TO_RLLAB_ADV>:$PYTHONPATH
```

## Example

```python
# Enter the anaconda virtual environment
source activate rllab3-adv
# Train on InvertedPendulum
python adversarial/scripts/train_adversary.py --env InvertedPendulumAdv-v1 --folder ~/rllab-adv/results
```

## Contact
Lerrel Pinto -- lerrelpATcsDOTcmuDOTedu.
