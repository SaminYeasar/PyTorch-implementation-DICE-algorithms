import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import time
import os
import random
def random_seed(seed=None):
    np.random.seed(seed)
    torch.manual_seed(seed)

def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False

# class eval_mode(object):
#     def __init__(self, model):
#         self.model = model
#
#     def __enter__(self):
#         self.prev = self.model.training
#         self.model.train_(False)
#
#     def __exit__(self, *args):
#         self.model.train_(self.prev)
#         return False


def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)

class ReplayBuffer(Dataset):
    """Buffer to store environment transitions."""

    def __init__(self, obs_shape, action_shape, capacity, batch_size, device):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device



        # the proprioceptive obs is stored as float32, pixels obs as uint8
        self.obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8
        #obs_dtype = np.float32
        self.obses = np.empty((capacity, *obs_shape), dtype=self.obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=self.obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def add(self, obs, action, reward, next_obs, done=None):

        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], 1.- done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self):

        idxs = np.random.randint(0, self.capacity if self.full else self.idx, size=self.batch_size)

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]

        obses = torch.as_tensor(obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)



        # scale if image (default obs type uint8), don't worry it's already been converted to float
        if self.obs_dtype == np.uint8:
            obses = obses / 255.
            next_obses = next_obses / 255.

        return obses, actions, rewards, next_obses, not_dones



    def save(self, save_dir, filename=None):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save:self.idx],
            self.next_obses[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.not_dones[self.last_save:self.idx]
        ]
        # payload = [
        #     self.obses[self.last_save:self.idx],
        #     self.next_obses[self.last_save:self.idx],
        #     self.actions[self.last_save:self.idx],
        #     self.rewards[self.last_save:self.idx]
        # ]
        self.last_save = self.idx
        if filename != None:
            torch.save(payload, filename)
        else:
            torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.next_obses[start:end] = payload[1]
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.not_dones[start:end] = payload[4]
            self.idx = end   # very important to update the idx

    def __getitem__(self, idx):
        idx = np.random.randint(
            0, self.capacity if self.full else self.idx, size=1
        )
        idx = idx[0]
        obs = self.obses[idx]
        action = self.actions[idx]
        reward = self.rewards[idx]
        next_obs = self.next_obses[idx]
        not_done = self.not_dones[idx]

        #return obs, action, reward, next_obs
        return obs, action, reward, next_obs, not_done

    def __len__(self):
        return self.capacity if self.full else self.idx