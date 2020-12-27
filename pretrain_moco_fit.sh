#!/usr/bin/env bash
python -m train "$@" -e 10 --patience 5 --tasks contr --prediction_steps=3 --modelcls=models.MomentumContrastiveModel --step_dropout=0.0 --event_dropout=0.15 --batch_size=1 --K=50000
python -m train "$@" -e 20 --patience 8 --finetune='*' --modelcls=models.MultitaskFinetune --batch_size=1
