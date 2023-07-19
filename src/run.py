import os
import random
import argparse
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

random.seed(0)

from dataset import CharCorruptionDataset, NameDataset
from model import ModelConfig, BirthPlacePredictionModel
from trainer import TrainingArguments, Trainer
import utils


def get_args():
    argp = argparse.ArgumentParser()
    argp.add_argument('function', help="Choose pretrain, finetune, or evaluate")
    argp.add_argument('pretrain_corpus_path', default=None)
    argp.add_argument('--reading_params_path',default=None)
    argp.add_argument('--writing_params_path',default=None)
    argp.add_argument('--finetune_corpus_path', default=None)
    argp.add_argument('--eval_corpus_path', default=None)
    argp.add_argument('--outputs_path', default=None)
    argp.add_argument('--pretrain_lr', default=6e-3, type=float)
    argp.add_argument('--finetune_lr', default=6e-4, type=float)
    argp.add_argument('--tb_expt_name', help='debug string for tb log.',
                    default='run')
    args = argp.parse_args()
    return args


def run(args):


    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

    output_dir = 'outputs/%s/%s_pt_lr_%f_ft_lr_%f' % (
        args.function,
        args.tb_expt_name,
        args.pretrain_lr,
        args.finetune_lr
    )
    writer = SummaryWriter(output_dir)
    
    block_size = 128
    text = open(args.pretrain_corpus_path, encoding='utf-8').read()
    pretrain_dataset = CharCorruptionDataset(text, block_size)

    model_config = ModelConfig(vocab_size=pretrain_dataset.vocab_size,
                               max_length=block_size)
    model = BirthPlacePredictionModel(model_config)
    model = model.to(device)

    if args.function == 'pretrain':
        assert args.writing_params_path is not None

        training_args = TrainingArguments(
            max_epochs=650, batch_size=128, learning_rate=args.pretrain_lr,
            lr_decay=True, warmup_tokens=512*20, final_tokens=200*len(pretrain_dataset)*block_size,
            num_workers=0, writer=writer, device=device,
        )
        pretrain_trainer = Trainer(model, pretrain_dataset, None, training_args)

        pretrain_trainer.train()

        model.save(args.writing_params_path)

    elif args.function == 'finetune':
        assert args.writing_params_path is not None
        assert args.finetune_corpus_path is not None

        train_data = open(args.finetune_corpus_path, encoding='utf-8').read()
        train_dataset = NameDataset(pretrain_dataset, train_data)
        
        max_epochs = 75
        if args.reading_params_path is not None:
            model = BirthPlacePredictionModel.load(args.reading_params_path)
            model = model.to(device)
            max_epochs = 10

        training_args = TrainingArguments(
            max_epochs=max_epochs, batch_size=256, learning_rate=args.finetune_lr, lr_decay=True,
            warmup_tokens=512*20, final_tokens=200*len(pretrain_dataset)*block_size,
            num_workers=0, writer=writer, device=device,
        )

        finetune_trainer = Trainer(model, train_dataset, None, training_args)
        finetune_trainer.train()

        model.save(args.writing_params_path)
        
    elif args.function == 'evaluate':
        assert args.outputs_path is not None
        assert args.reading_params_path is not None
        assert args.eval_corpus_path is not None
        model = BirthPlacePredictionModel.load(args.reading_params_path)
        model = model.to(device)
        correct = 0
        total = 0
        with open(args.outputs_path, 'w', encoding='utf-8') as fout:
            predictions = []
            for line in tqdm(open(args.eval_corpus_path, encoding='utf-8')):
                x = line.split('\t')[0]
                x = x + '⁇'
                x = torch.tensor([pretrain_dataset.stoi[s] for s in x], dtype=torch.long)[None,...].to(device)
                pred = utils.sample(model, x, 32, sample=False, device=device)[0]
                completion = ''.join([pretrain_dataset.itos[int(i)] for i in pred])
                pred = completion.split('⁇')[1]
                predictions.append(pred)
                fout.write(pred + '\n')
            total, correct = utils.evaluate_places(args.eval_corpus_path, predictions)
        if total > 0:
            print('Correct: {} out of {}: {}%'.format(correct, total, correct/total*100))
        else:
            print('Predictions written to {}; no targets provided'
                    .format(args.outputs_path))
    
    """
        1. fintune without pretrain
            train:
                python src/run.py finetune data/wiki.txt --writing_params_path vanilla.model.params --finetune_corpus_path data/birth_places_train.tsv
            eval:
                python src/run.py evaluate data/wiki.txt --reading_params_path vanilla.model.params --eval_corpus_path data/birth_dev.tsv --outputs_path vanilla.nopretrain.dev.predictions
            test:
                python src/run.py evaluate data/wiki.txt --reading_params_path vanilla.model.params --eval_corpus_path data/birth_test_inputs.tsv --outputs_path vanilla.nopretrain.test.predictions
    
        2. pretraine and finetune
            pretrain:
                python src/run.py pretrain data/wiki.txt --writing_params_path vanilla.pretrain.params
            finetune:
                python src/run.py finetune data/wiki.txt --reading_params_path vanilla.pretrain.params --writing_params_path vanilla.finetune.params --finetune_corpus_path data/birth_places_train.tsv
            eval:
                python src/run.py evaluate data/wiki.txt --reading_params_path vanilla.finetune.params --eval_corpus_path data/birth_dev.tsv --outputs_path vanilla.pretrain.dev.predictions
            test:
                python src/run.py evaluate data/wiki.txt --reading_params_path vanilla.finetune.params --eval_corpus_path data/birth_test_inputs.tsv --outputs_path vanilla.pretrain.test.predictions

    """

if __name__ == "__main__":
    args = get_args()
    run(args)
    

    

