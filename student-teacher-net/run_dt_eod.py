import logging
from mingpt.utils import set_seed
import numpy as np
from torch.utils.data import Dataset
from mingpt.model_eod import GPT, GPTConfig
from mingpt.trainer_eod import Trainer, TrainerConfig
import torch
import argparse
from create_eod_dataset import create_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--context_length', type=int, default=3)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--model_type', type=str, default='reward_conditioned')
parser.add_argument('--num_steps', type=int, default=110000)
parser.add_argument('--game', type=str, default='eod')
parser.add_argument('--is_train', type=bool, default = True)
parser.add_argument('--model_path', type=str, default = "")
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--data_dir_prefix', type=str, default='../../dteod/data/car_features2/')
parser.add_argument('--log_dir_prefix', type=str, default='../../dteod/data/tran/car_transition/')
args = parser.parse_args()
set_seed(args.seed)

class StateActionReturnDataset(Dataset):

    def __init__(self, data, block_size, actions, done_idxs, rtgs, timesteps):
        self.block_size = block_size
        self.vocab_size = max(actions)+1
        self.data = data
        self.actions = actions
        self.done_idxs = done_idxs
        self.rtgs = rtgs
        self.timesteps = timesteps
    
    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        block_size = self.block_size // 3
        done_idx = idx + block_size
        for i in self.done_idxs:
            if i > idx: # first done_idx greater than idx
                done_idx = min(int(i), done_idx)
                break

        idx = done_idx - block_size
        states = torch.tensor(np.array(self.data[idx:done_idx]), dtype=torch.float32).reshape(block_size, -1)
        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long).unsqueeze(1)
        rtgs = torch.tensor(self.rtgs[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(self.timesteps[idx:idx+1], dtype=torch.int64).unsqueeze(1)
        return states, actions, rtgs, timesteps

obss, actions, returns, done_idxs, rtgs, timesteps = create_dataset(args.num_steps, args.data_dir_prefix, args.log_dir_prefix)

logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

train_dataset = StateActionReturnDataset(obss, args.context_length*3, actions, done_idxs, rtgs, timesteps)

mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                  n_layer=4, n_head=8, n_embd=512, model_type=args.model_type, max_timestep=max(timesteps))
model = GPT(mconf)
epochs = args.epochs
tconf = TrainerConfig(max_epochs=epochs, batch_size=args.batch_size, learning_rate=6e-7,
                      lr_decay=True, warmup_tokens=1000, final_tokens=2000,
                      num_workers=4, seed=args.seed, model_type=args.model_type, game=args.game, max_timestep=max(timesteps))
trainer = Trainer(model, train_dataset, "./data/train/", tconf)

trainer.train()
