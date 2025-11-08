# DurakZero: Self-Play Deep Reinforcement Learning for Durak

DurakZero is an adaptation of the original [DouZero](https://arxiv.org/abs/2106.06135) framework for the two-player 36-card variant of the Durak card game. It replaces the DouDizhu-specific environment, feature representation, and evaluation pipeline with an implementation of Durak that follows the rules in this repository. The project trains agents via Deep Monte-Carlo reinforcement learning using high-throughput self-play.

## Installation

DurakZero requires Python 3.8+ and PyTorch. To install the dependencies, run:

```bash
pip install -r requirements.txt
pip install -e .
```

## Training

Launch self-play training with:

```bash
python train.py
```

Important command line flags include:

* `--gpu_devices`: IDs of GPUs visible to the process.
* `--num_actor_devices`: number of devices used for self-play simulation.
* `--num_actors`: number of actors spawned per simulation device.
* `--training_device`: device ID (or `cpu`) for the learner.
* `--actor_device_cpu`: force actors to run on CPU.
* `--batch_size`, `--unroll_length`, `--learning_rate`, etc.: optimizer and update hyperparameters (see `douzero/dmc/arguments.py`).

Example: run actors on two GPUs and train on a third GPU.

```bash
python train.py --gpu_devices 0,1,2 --num_actor_devices 2 --num_actors 10 --training_device 2
```

To run entirely on CPU:

```bash
python train.py --actor_device_cpu --training_device cpu
```

Checkpoints are stored in the directory specified by `--savedir` (default `durakzero_checkpoints`).

### Notebook workflow

For an interactive pipeline that mirrors these steps, open `notebooks/durakzero_training.ipynb`. The notebook walks through environment setup, flag configuration, launching self-play, and evaluating saved checkpoints on both local machines and large accelerators such as H100 GPUs.

## Evaluation

Evaluate a checkpoint via Durak self-play:

```bash
python evaluate.py --checkpoint path/to/model.tar --num_games 1000
```

The script reports win counts and win rates for both players.

## Repository Structure

* `douzero/env/env.py` – full Durak rules implementation and feature encoder.
* `douzero/dmc/` – Deep Monte-Carlo training components (models, buffers, actor/learner loops).
* `douzero/evaluation/` – lightweight checkpoint loader and self-play evaluator for Durak.
* `train.py` – entry point for training.
* `evaluate.py` – entry point for evaluation.

## License

DurakZero inherits the Apache 2.0 license from the original DouZero project.
