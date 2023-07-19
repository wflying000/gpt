import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelConfig:
    hidden_size = 256
    num_heads = 8
    intermediate_size = 1024
    embed_dropout_prob = 0.1
    attn_dropout_prob = 0.1
    resid_dropout_prob = 0.1
    num_layers = 4
    max_length = 128

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)



def process_attention_mask(mask):
    if mask.dim() == 2:
        mask = mask[None, None, :, :]
    elif mask.dim() == 3:
        mask = mask[:, None, :, :]
    mask = (1.0 - mask) * (-1e8)
    return mask



class GPTAttention(nn.Module):
    def __init__(self, config):
        super(GPTAttention, self).__init__()
        assert config.hidden_size % config.num_heads == 0
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = nn.Dropout(config.attn_dropout_prob)
        self.resid_dropout = nn.Dropout(config.resid_dropout_prob)
        self.num_heads = config.num_heads
        self.head_size = config.hidden_size // config.num_heads
        self.attn_weights = None        
    
    def forward(self, hidden_state, mask):
        bsz, seq_len, hidden_size = hidden_state.size()

        query = self.q_proj(hidden_state).view(bsz, seq_len, self.num_heads, self.head_size).transpose(1, 2) # (bsz, num_heads, seq_len, head_size)
        key = self.k_proj(hidden_state).view(bsz, seq_len, self.num_heads, self.head_size).transpose(1, 2) # (bsz, num_heads, seq_len, head_size)
        value = self.v_proj(hidden_state).view(bsz, seq_len, self.num_heads, self.head_size).transpose(1, 2) # (bsz, num_heads, seq_len, head_size)

        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_size) # (bsz, num_heads, seq_len, seq_len)

        if mask is not None:
            attn_weights += mask
        
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_outputs = torch.matmul(attn_weights, value) # (bsz, num_heads, seq_len, head_size)
        attn_outputs = attn_outputs.transpose(1, 2).contiguous().view(bsz, seq_len, hidden_size)
        attn_outputs = self.out_proj(attn_outputs)
        attn_outputs = self.resid_dropout(attn_outputs)

        self.attn_weights = attn_weights

        return attn_outputs
    

class GPTMLP(nn.Module):
    def __init__(self, config):
        super(GPTMLP, self).__init__()
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.resid_dropout_prob)

    def forward(self, hidden_state):
        hidden_state = self.linear1(hidden_state)
        hidden_state = self.act(hidden_state)
        hidden_state = self.linear2(hidden_state)
        hidden_state = self.dropout(hidden_state)
        return hidden_state
    
    
class GPTBlock(nn.Module):
    def __init__(self, config):
        super(GPTBlock, self).__init__()
        self.attn = GPTAttention(config)
        self.mlp = GPTMLP(config)
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
    
    def forward(self, hidden_state, mask):
        hidden_state = hidden_state + self.attn(self.norm1(hidden_state), mask)
        hidden_state = hidden_state + self.mlp(self.norm2(hidden_state))
        return hidden_state
    

class GPTEmbeddings(nn.Module):
    def __init__(self, config):
        super(GPTEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_length, config.hidden_size)
        self.dropout = nn.Dropout(config.embed_dropout_prob)
        position_ids = torch.arange(config.max_length).unsqueeze(0)
        self.register_buffer("position_ids", position_ids)
        self.max_length = config.max_length

    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        assert seq_len <= self.max_length, f"seq_len: {seq_len}, max_length: {self.max_length}"

        position_ids = self.position_ids[:, :seq_len]
        position_embeddings = self.position_embeddings(position_ids)
        word_embeddings = self.word_embeddings(input_ids)

        embeddings = word_embeddings + position_embeddings
        embeddings = self.dropout(embeddings)

        return embeddings


class GPTModel(nn.Module):
    def __init__(self, config):
        super(GPTModel, self).__init__()
        self.embeddings = GPTEmbeddings(config)
        self.layers = nn.ModuleList([GPTBlock(config) for _ in range(config.num_layers)])
    
    def forward(self, input_ids, mask):
        embeddings = self.embeddings(input_ids)

        mask = process_attention_mask(mask)

        hidden_state = embeddings

        for layer in self.layers:
            hidden_state = layer(hidden_state, mask)

        return hidden_state
    

class BirthPlacePredictionModel(nn.Module):
    def __init__(self, config):
        super(BirthPlacePredictionModel, self).__init__()
        self.gpt = GPTModel(config)
        self.head = nn.Linear(config.hidden_size, config.vocab_size)
        self.norm = nn.Dropout(config.embed_dropout_prob)
        self.config = config

    def forward(self, input_ids, mask, labels=None):
        hidden_state = self.gpt(input_ids, mask)
        hidden_state = self.norm(hidden_state)
        logits = self.head(hidden_state)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=0)
        
        return logits, loss
    
    def get_block_size(self):
        return self.config.max_length
    
    def save(self, path):
        if hasattr(self, "module"):
            state_dict = self.module.state_dict()
        else:
            state_dict = self.state_dict()
        params = {
            "config": self.config,
            "state_dict": state_dict,
        }
        torch.save(params, path)
    
    @staticmethod
    def load(path):
        params = torch.load(path, map_location="cpu")
        model = BirthPlacePredictionModel(params["config"])
        model.load_state_dict(params["state_dict"])
        return model


    

        

    


