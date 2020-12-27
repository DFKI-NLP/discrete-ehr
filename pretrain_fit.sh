#!/usr/bin/env bash
python -m train "$@" -e 10 --patience 10 --tasks contr --prediction_steps=8 --finetune_emb --modelcls=models.ContrastiveModel --step_dropout=0.0 --event_dropout=0.5 --batch_size=4
python -m train "$@" -e 30 --patience 8 --finetune='*' --emb_suffix='*' --modelcls=models.MultitaskFinetune --batch_size=1
