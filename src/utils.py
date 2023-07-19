import torch
import random
import numpy as np
import torch.nn.functional as F


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out


@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample=False, top_k=None, device="cpu"):
    block_size = model.get_block_size()
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:]
        seq_len = x_cond.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len)).to(device)
        logits, _ = model(x_cond, mask)
        
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        probs = F.softmax(logits, dim=-1)
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        x = torch.cat((x, ix), dim=1)

    return x


def evaluate_places(filepath, predicted_places):
  
    with open(filepath, encoding='utf-8') as fin:
        lines = [x.strip().split('\t') for x in fin]
        if len(lines[0]) == 1:
            print('No gold birth places provided; returning (0,0)')
            return (0,0)
        true_places = [x[1] for x in lines]
        total = len(true_places)
        assert total == len(predicted_places)
        correct = len(list(filter(lambda x: x[0] == x[1],
        zip(true_places, predicted_places))))
        return (float(total),float(correct))

        