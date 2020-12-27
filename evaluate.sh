#!/bin/bash
cd mimic3-benchmarks
WANDB_FOLDER=$(echo ../wandb/*$1)

RESULT_FILE=$(echo $WANDB_FOLDER/**/test_listfile_predictions/decompensation-*$2.csv)
python -m mimic3benchmark.evaluation.evaluate_decomp $RESULT_FILE --save_file=$RESULT_FILE.json --test_listfile=data/decompensation/test_listfile.csv
RESULT_FILE=$(echo $WANDB_FOLDER/**/test_listfile_predictions/in_hospital_mortality-*$2.csv)
python -m mimic3benchmark.evaluation.evaluate_ihm $RESULT_FILE --save_file=$RESULT_FILE.json --test_listfile=data/in-hospital-mortality/test_listfile.csv
RESULT_FILE=$(echo $WANDB_FOLDER/**/test_listfile_predictions/length_of_stay_regression*$2.csv)
python -m mimic3benchmark.evaluation.evaluate_los $RESULT_FILE --save_file=$RESULT_FILE.json --test_listfile=data/length-of-stay/test_listfile.csv
RESULT_FILE=$(echo $WANDB_FOLDER/**/test_listfile_predictions/phenotyping-*$2.csv)
python -m mimic3benchmark.evaluation.evaluate_pheno $RESULT_FILE --save_file=$RESULT_FILE.json --test_listfile=data/phenotyping/test_listfile.csv
