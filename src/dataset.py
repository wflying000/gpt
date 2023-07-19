import torch
import random
import numpy as np
from torch.utils.data import Dataset


class NameDataset(Dataset):
    def __init__(self, pretraining_dataset, data):
        self.MASK_CHAR = u"\u2047" 
        self.PAD_CHAR = u"\u25A1" 
        self.itos = pretraining_dataset.itos 
        self.stoi = pretraining_dataset.stoi 
        self.block_size = pretraining_dataset.block_size
        self.data = list(data.encode('utf-8').decode('ascii', errors='ignore').split('\n'))

    def __len__(self):
        return len(self.data) - 1

    def __getitem__(self, idx):
        inp, oup = self.data[idx].split('\t')
        x = inp + self.MASK_CHAR + oup + self.MASK_CHAR
        x = x + self.PAD_CHAR*(self.block_size - len(x))
        y = self.PAD_CHAR*(len(inp)-1) + x[len(inp):]
        
        x = x[:-1]
        x = [self.stoi[c] for c in x]
        y = [self.stoi[c] for c in y]

        item = {"x": x, "y": y}

        return item
    
    def collate_fn(self, item_list):
        x_list = [item["x"] for item in item_list]
        y_list = [item["y"] for item in item_list]

        x = torch.LongTensor(x_list)
        y = torch.LongTensor(y_list)

        seq_len = x.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len))

        return x, y, mask

    

class CharCorruptionDataset(Dataset):
    def __init__(self, data, block_size):
        self.MASK_CHAR = u"\u2047" # the doublequestionmark character, for mask
        self.PAD_CHAR = u"\u25A1" # the empty square character, for pad

        chars = list(sorted(list(set(data))))
        assert self.MASK_CHAR not in chars 
        assert self.PAD_CHAR not in chars
        chars.insert(0, self.MASK_CHAR)
        chars.insert(0, self.PAD_CHAR)

        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }

        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data.split('\n')
        self.data = [x for x in self.data if len(x) > 4]

        self.truncated_min_len = 4
        self.truncated_max_len = int(self.block_size * 7 / 8)

    def __len__(self):
        # returns the length of the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # TODO [part e]: see spec above
        data = self.data[idx]
        total = len(data)
        min_len = min(self.truncated_min_len, total)
        max_len = min(self.truncated_max_len, total)
        truncated_len = random.randint(min_len, max_len)
        truncated_data = data[:truncated_len]

        mask_content_len = np.random.binomial(n=truncated_len-3, p=(truncated_len-4)/(4*(truncated_len-3))) + 1
        start = random.randint(0, truncated_len - mask_content_len)
        end = start + mask_content_len
        if start > 0:
            prefix = truncated_data[:start]
        else:
            prefix = ""
        
        if end < truncated_len:
            suffix = truncated_data[end:]
        else:
            suffix = ""
        
        mask_content = truncated_data[start : end]
        
        masked_string = prefix + self.MASK_CHAR + suffix + self.MASK_CHAR + mask_content
        masked_string = masked_string + (self.PAD_CHAR) * (self.block_size - len(masked_string))

        x = masked_string[:-1]
        y = masked_string[1:]

        x = [self.stoi[c] for c in x]
        y = [self.stoi[c] for c in y]

        item = {"x": x, "y": y}

        return item
    
    def collate_fn(self, item_list):
        x_list = [item["x"] for item in item_list]
        y_list = [item["y"] for item in item_list]

        x = torch.LongTensor(x_list)
        y = torch.LongTensor(y_list)

        seq_len = x.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len))

        return x, y, mask

if __name__ == '__main__':
    import os, sys
    os.chdir(sys.path[0])
    # argp = argparse.ArgumentParser()
    # argp.add_argument('dataset_type', help="Type of dataset to sample from."
    #         "Options: namedata, charcorruption.",
    #         choices=["namedata", "charcorruption"])
    # args = argp.parse_args()
    dataset_type = "namedata"
    if dataset_type == 'namedata':
        # Even if it hasn't been implemented, we use it to define the vocab
        corruption_dataset = CharCorruptionDataset(open('../data/wiki.txt', encoding='utf-8').read(), 128)
        # Make the name dataset
        name_dataset = NameDataset(corruption_dataset,
            open('../data/birth_places_train.tsv', encoding='utf-8').read())
        for _, example in zip(range(4), name_dataset):
            x, y = example
            print('x:', ''.join([name_dataset.itos[int(c)] for c in x]))
            print('y:', ''.join([name_dataset.itos[int(c)] for c in y]))
        pass
    elif dataset_type == 'charcorruption':
        corruption_dataset = CharCorruptionDataset(open('../data/wiki.txt', encoding='utf-8').read(), 128)
        for _, example in zip(range(len(corruption_dataset)), corruption_dataset):
            x, y = example
            print('x:', ''.join([corruption_dataset.itos[int(c)] for c in x]))
            print('y:', ''.join([corruption_dataset.itos[int(c)] for c in y]))
    else:
        raise ValueError("Unknown dataset type in command line args: {}"
                .format(dataset_type))