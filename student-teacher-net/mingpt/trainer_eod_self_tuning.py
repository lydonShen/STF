"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import math
import logging
import pickle
from tqdm import tqdm
import numpy as np
import os
from mingpt.eod_env_vsb import ViewScaleBrightnessSearchEnv
from torch.utils.data.dataloader import DataLoader
logger = logging.getLogger(__name__)
from mingpt.utils import sample
import torch


class TrainerConfig:
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1
    lr_decay = False
    warmup_tokens = 5000
    final_tokens = 10000
    ckpt_path = None
    num_workers = 0

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)
            print(self.device)

    def save_checkpoint(self,ephoc):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)
        torch.save(raw_model.state_dict(), "./model"+str(ephoc)+".pth")

    def train(self):
        model, config = self.model, self.config
        for param in model.ret_emb.parameters():
            param.requires_grad = False

        for param in model.action_embeddings.parameters():
            param.requires_grad = False

        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        def run_epoch(split, epoch_num=0):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)

            for it, (x, y, r, t) in pbar:
                x = x.to(self.device)
                y = y.to(self.device)
                r = r.to(self.device)
                t = t.to(self.device)

                with torch.set_grad_enabled(is_train):
                    logits, loss = model(x, y, y, r, t)
                    loss = loss.mean()
                    losses.append(loss.item())

                if is_train:
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    if config.lr_decay:
                        self.tokens += (y >= 0).sum()
                        if self.tokens < config.warmup_tokens:
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

            if not is_train:
                test_loss = float(np.mean(losses))
                logger.info("test loss: %f", test_loss)
                return test_loss

        best_return = -float('inf')

        self.tokens = 0

        for epoch in range(config.max_epochs):
            run_epoch('train', epoch_num=epoch)
            if epoch % 10 == 0:
                self.save_checkpoint(epoch)
            if self.config.model_type == 'naive':
                eval_return = self.get_returns(0,epoch)
            elif self.config.model_type == 'reward_conditioned':
                if self.config.game == 'Breakout':
                    eval_return = self.get_returns(90,epoch)
                elif self.config.game == 'Seaquest':
                    eval_return = self.get_returns(1150,epoch)
                elif self.config.game == 'Qbert':
                    eval_return = self.get_returns(14000,epoch)
                elif self.config.game == 'Pong':
                    eval_return = self.get_returns(20,epoch)
                elif self.config.game == 'SA':
                    eval_return = self.get_returns(3,epoch)
                elif self.config.game == 'VP':
                    eval_return = self.get_returns(3,epoch)
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError()

    def get_returns(self, ret,epoch):
        self.model.train(False)
        args= Args(self.config.game.lower(), self.config.seed)
        env = ViewScaleBrightnessSearchEnv()
        m=0
        T_rewards, T_Qs = [], []
        done = True
        search_dict = {}
        for i in os.listdir(self.test_dataset):
            N = []
            AN = []
            RN = []
            terminals = []
            terminals.append(False)
            m=m+1
            state = env.reset(self.test_dataset + str(i)).type(torch.float32).to(self.device).unsqueeze(0).unsqueeze(0)
            IN = []
            I0 = i.split(".")[0]
            N.append(I0)
            state = state.permute(0, 1, 4, 2, 3)
            rtgs = [ret]
            sampled_action = sample(self.model.module, state, 1, temperature=1.0, sample=True, actions=None, 
                rtgs=torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(-1), 
                timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(self.device))

            j = 0
            all_states = state
            actions = []
            while True:
                if done:
                    reward_sum, done = 0, False
                action = sampled_action.cpu().numpy()[0,-1]

                if j == 0 and action==0:
                    action = env.step_greedy()
                AN.append(action)
                new_value = torch.tensor(action, device='cuda:0')
                sampled_action=new_value
                actions += [sampled_action]
                print("action",action)
                state, reward, done, statename = env.step(action)
                IN.append(statename)
                RN.append(reward)
                terminals.append(done)
                search_dict[I0] = IN
                reward_sum += reward
                j += 1

                if done:
                    T_rewards.append(reward_sum)
                    print("reward_sum", reward_sum)
                    break

                state = state.unsqueeze(0).unsqueeze(0).to(self.device)
                state = state.permute(0, 1, 4, 2, 3)
                all_states = torch.cat([all_states, state], dim=0)
                rtgs += [rtgs[-1] - reward]
                sampled_action = sample(self.model.module, all_states.unsqueeze(0), 1, temperature=1.0, sample=True, 
                    actions=torch.tensor(actions, dtype=torch.long).to(self.device).unsqueeze(1).unsqueeze(0), 
                    rtgs=torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(-1), 
                    timesteps=(min(j, self.config.max_timestep) * torch.ones((1, 1, 1), dtype=torch.int64).to(self.device)))
            if m > 750:
                break

        with open("../../search"+str(epoch)+".pickle", "wb") as fp:
            pickle.dump(search_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
        env.close()
        eval_return = float(sum(T_rewards)/751)
        print("target return: %d, eval return: %f" % (ret, eval_return))
        self.model.train(True)
        return eval_return

class Args:
    def __init__(self, game, seed):
        self.device = torch.device('cuda')
        self.seed = seed
        self.max_episode_length = 108e3
        self.game = game
        self.history_length = 4
