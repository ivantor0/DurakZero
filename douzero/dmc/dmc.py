import os
import threading
import time
import timeit
from collections import deque

import torch
from torch import multiprocessing as mp
from torch import nn

from .file_writer import FileWriter
from .models import Model
from .utils import get_batch, log, create_env, create_buffers, create_optimizers, act

positions = ['player_0', 'player_1']
mean_episode_return_buf = {p: deque(maxlen=100) for p in positions}


def compute_loss(predictions, targets):
    return ((predictions.squeeze(-1) - targets) ** 2).mean()


def learn(position,
          actor_models,
          model,
          batch,
          optimizer,
          flags,
          lock):
    if flags.training_device != "cpu":
        device = torch.device('cuda:' + str(flags.training_device))
    else:
        device = torch.device('cpu')

    states = torch.flatten(batch['state'].to(device), 0, 1)
    actions = torch.flatten(batch['action'].to(device), 0, 1)
    targets = torch.flatten(batch['target'].to(device), 0, 1)

    episode_returns = batch['episode_return'][batch['done']]
    if len(episode_returns) > 0:
        mean_episode_return_buf[position].append(torch.mean(episode_returns).to(device))

    with lock:
        predictions = model.evaluate(position, states, actions)
        loss = compute_loss(predictions, targets)
        stats = {
            f'mean_episode_return_{position}': torch.mean(torch.stack(list(mean_episode_return_buf[position]))).item() if mean_episode_return_buf[position] else 0.0,
            f'loss_{position}': loss.item(),
        }

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(position), flags.max_grad_norm)
        optimizer.step()

        for actor_model in actor_models.values():
            actor_model.get_model(position).load_state_dict(model.get_model(position).state_dict())
        return stats


def train(flags):
    if not flags.actor_device_cpu or flags.training_device != 'cpu':
        if not torch.cuda.is_available():
            raise AssertionError("CUDA not available. If you have GPUs, please specify the ID after `--gpu_devices`. Otherwise, please train with CPU with `python3 train.py --actor_device_cpu --training_device cpu`")
    plogger = FileWriter(
        xpid=flags.xpid,
        xp_args=flags.__dict__,
        rootdir=flags.savedir,
    )
    checkpointpath = os.path.expandvars(
        os.path.expanduser('%s/%s/%s' % (flags.savedir, flags.xpid, 'model.tar')))

    T = flags.unroll_length
    B = flags.batch_size

    if flags.actor_device_cpu:
        device_iterator = ['cpu']
    else:
        device_iterator = range(flags.num_actor_devices)
        assert flags.num_actor_devices <= len(flags.gpu_devices.split(',')), 'The number of actor devices can not exceed the number of available devices'

    models = {}
    for device in device_iterator:
        model = Model(device=device)
        model.share_memory()
        model.eval()
        models[device] = model

    buffers = create_buffers(flags, device_iterator)

    actor_processes = []
    ctx = mp.get_context('spawn')
    free_queue = {}
    full_queue = {}

    for device in device_iterator:
        _free_queue = {p: ctx.SimpleQueue() for p in positions}
        _full_queue = {p: ctx.SimpleQueue() for p in positions}
        free_queue[device] = _free_queue
        full_queue[device] = _full_queue

    learner_model = Model(device=flags.training_device)
    optimizers = create_optimizers(flags, learner_model)

    stat_keys = []
    for p in positions:
        stat_keys.extend([f'mean_episode_return_{p}', f'loss_{p}'])
    frames, stats = 0, {k: 0 for k in stat_keys}
    position_frames = {p: 0 for p in positions}

    if flags.load_model and os.path.exists(checkpointpath):
        checkpoint_states = torch.load(
            checkpointpath, map_location=("cuda:" + str(flags.training_device) if flags.training_device != "cpu" else "cpu")
        )
        for p in positions:
            learner_model.get_model(p).load_state_dict(checkpoint_states["model_state_dict"][p])
            optimizers[p].load_state_dict(checkpoint_states["optimizer_state_dict"][p])
            for device in device_iterator:
                models[device].get_model(p).load_state_dict(learner_model.get_model(p).state_dict())
        stats = checkpoint_states["stats"]
        frames = checkpoint_states["frames"]
        position_frames = checkpoint_states["position_frames"]
        log.info(f"Resuming preempted job, current stats:\n{stats}")

    for device in device_iterator:
        for i in range(flags.num_actors):
            actor = ctx.Process(
                target=act,
                args=(i, device, free_queue[device], full_queue[device], models[device], buffers[device], flags))
            actor.start()
            actor_processes.append(actor)

    def batch_and_learn(i, device, position, local_lock, position_lock, lock=threading.Lock()):
        nonlocal frames, position_frames, stats
        while frames < flags.total_frames:
            batch = get_batch(free_queue[device][position], full_queue[device][position], buffers[device][position], flags, local_lock)
            _stats = learn(position, models, learner_model, batch, optimizers[position], flags, position_lock)

            with lock:
                for k in _stats:
                    stats[k] = _stats[k]
                to_log = dict(frames=frames)
                to_log.update({k: stats[k] for k in stat_keys})
                plogger.log(to_log)
                frames += T * B
                position_frames[position] += T * B

    for device in device_iterator:
        for m in range(flags.num_buffers):
            for p in positions:
                free_queue[device][p].put(m)

    threads = []
    locks = {}
    for device in device_iterator:
        locks[device] = {p: threading.Lock() for p in positions}
    position_locks = {p: threading.Lock() for p in positions}

    for device in device_iterator:
        for i in range(flags.num_threads):
            for position in positions:
                thread = threading.Thread(
                    target=batch_and_learn, name='batch-and-learn-%d' % i, args=(i, device, position, locks[device][position], position_locks[position]))
                thread.start()
                threads.append(thread)

    def checkpoint(frames):
        if flags.disable_checkpoint:
            return
        log.info('Saving checkpoint to %s', checkpointpath)
        _models = learner_model.get_models()
        torch.save({
            'model_state_dict': {k: _models[k].state_dict() for k in _models},
            'optimizer_state_dict': {k: optimizers[k].state_dict() for k in optimizers},
            "stats": stats,
            'flags': vars(flags),
            'frames': frames,
            'position_frames': position_frames,
        }, checkpointpath)

    timer = timeit.default_timer
    try:
        last_checkpoint_time = timer()
        while frames < flags.total_frames:
            time.sleep(5)
            now = timer()
            if now - last_checkpoint_time > flags.save_interval * 60:
                checkpoint(frames)
                last_checkpoint_time = now
    except KeyboardInterrupt:
        log.info('Caught KeyboardInterrupt, saving checkpoint...')
        checkpoint(frames)
    else:
        log.info('Training completed, saving final checkpoint...')
        checkpoint(frames)

    for actor in actor_processes:
        actor.join()

    for thread in threads:
        thread.join()

    plogger.close()
