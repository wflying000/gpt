# gpt

通过预训练实现问答任务

## 1. 不进行预训练直接训练问答任务

train
```
python src/run.py finetune data/wiki.txt \
    --writing_params_path vanilla.model.params \
    --finetune_corpus_path data/birth_places_train.tsv

```

evaluate
```
python src/run.py evaluate data/wiki.txt \
    --reading_params_path vanilla.model.params \
    --eval_corpus_path data/birth_dev.tsv \
    --outputs_path vanilla.nopretrain.dev.predictions
```

## 2. 先预训练再在问答任务上进行finetune

pretrain
```
python src/run.py pretrain data/wiki.txt --writing_params_path vanilla.pretrain.params
```

finetune
```
python src/run.py finetune data/wiki.txt \
    --reading_params_path vanilla.pretrain.params \
    --writing_params_path vanilla.finetune.params \
    --finetune_corpus_path data/birth_places_train.tsv
```

evaluate
```
python src/run.py evaluate data/wiki.txt \
    --reading_params_path vanilla.finetune.params \
    --eval_corpus_path data/birth_dev.tsv \
    --outputs_path vanilla.pretrain.dev.predictions
```