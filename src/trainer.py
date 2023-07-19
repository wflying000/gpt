import math
import torch
import logging
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader

class TrainingArguments:
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 
    lr_decay = False
    warmup_tokens = 375e6 
    final_tokens = 260e9 
    ckpt_path = None
    num_workers = 0
    writer = None
    device = "cpu"
    
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(
        self, 
        model, 
        train_dataset, 
        eval_dataset, 
        training_args,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.training_args = training_args


    def train(self):
        model, config = self.model, self.training_args

        # create the optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        params_decay = [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]
        params_nodecay = [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]
        optim_groups = [
            {"params": params_decay, "weight_decay": config.weight_decay},
            {"params": params_nodecay, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=config.learning_rate, betas=config.betas)
        self.tokens = 0 # counter used for learning rate decay

        for epoch in tqdm(range(config.max_epochs), total=config.max_epochs):

            self.run_epoch(model, optimizer, 'train', config, epoch)

            if self.eval_dataset is not None:
                eval_loss = self.run_epoch(model, None, 'test', config, epoch)
                config.writer.add_scalar("loss/eval", eval_loss, epoch)

            if config.ckpt_path is not None:
                model.save(config.ckpt_path)

    def run_epoch(self, model, optimizer, split, config, epoch):
        is_train = split == 'train'
        model.train(is_train)
        data = self.train_dataset if is_train else self.test_dataset
        loader = DataLoader(
            data, batch_size=config.batch_size, num_workers=config.num_workers,
            collate_fn=data.collate_fn,
        )

        losses = []
        pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
        
        for it, (x, y, mask) in pbar:

            # place data on the correct device
            x = x.to(config.device)
            y = y.to(config.device)
            mask = mask.to(config.device)

            # forward the model
            with torch.set_grad_enabled(is_train):
                logits, loss = self.model(x, mask, y)
                loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                losses.append(loss.item())

            if is_train:

                # backprop and update the parameters
                model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                optimizer.step()

                # decay the learning rate based on our progress
                if config.lr_decay:
                    self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                    if self.tokens < config.warmup_tokens:
                        # linear warmup
                        lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                    else:
                        # cosine learning rate decay
                        progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                        lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                    lr = config.learning_rate * lr_mult
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                else:
                    lr = config.learning_rate

                # report progress
                pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")
                
                step = epoch * len(loader) + it
                if config.writer is not None:
                    config.writer.add_scalar('loss/train',  loss.item(), step)
                    config.writer.add_scalar('lr', lr, step)
        
        if not is_train:
            return np.mean(losses)