# Self-supervised pretraining for learning distributed representations for electronic health records

ICU predictions on MIMIC-III using discrete event input with distributed representations, resulting in state-of-the-art results for [MIMIC-III Benchmark (Harutyunyan et al.)](https://github.com/YerevaNN/mimic3-benchmarks):

|                 | ('in_hospital_mortality', 'AUC of PRC')   | ('in_hospital_mortality', 'AUC of ROC')   | ('decompensation', 'AUC of PRC')   | ('decompensation', 'AUC of ROC')   | ('length_of_stay_regression', 'Kappa')   | ('length_of_stay_regression', 'MAD')   | ('phenotyping', 'Macro ROC AUC')   | ('phenotyping', 'Micro ROC AUC')   | ('phenotyping', 'Macro AUPRC')   | ('phenotyping', 'Micro AUPRC')   |
|:----------------|:------------------------------------------|:------------------------------------------|:-----------------------------------|:-----------------------------------|:-----------------------------------------|:---------------------------------------|:-----------------------------------|:-----------------------------------|:---------------------------------|:---------------------------------|
| ST              | .64 (.59, .688)                           | .918 (.905, .93)                          | .537 (.526, .548)                  | .962 (.96, .963)                   | .637 (.636, .638)                        | 73.394 (73.013, 73.762)                | .868 (.865, .871)                  | .897 (.895, .899)                  | .62 (.612, .627)                 | .684 (.678, .69)                 |
| MT              | .677 (.631, .721)                         | .921 (.907, .934)                         | .649 (.639, .659)                  | .973 (.972, .975)                  | .624 (.622, .625)                        | 76.485 (76.074, 76.891)                | .837 (.834, .841)                  | .871 (.868, .873)                  | .553 (.546, .561)                | .612 (.605, .618)                |
| Contrastive P=4 | .667 (.619, .712)                         | .923 (.911, .935)                         | .616 (.606, .627)                  | .968 (.966, .97)                   | .619 (.618, .62)                         | 77.607 (77.217, 77.981)                | .836 (.832, .839)                  | .869 (.867, .872)                  | .548 (.541, .556)                | .605 (.598, .612)                |


## How-to generate input data

> You need [MIMIC-III](https://mimic.physionet.org/) access to run the code.

1. Setup the benchmark and FastText submodule:
```sh
git submodule update
```

Setup the conda environment:
``` sh
conda env create -f environment.yml
```

2. Generate csv files for the benchmark tasks as explained in `mimic3-benchmarks/README.md` under _Building a benchmark_.
You need to generate for all tasks not only multitask as the evaluation scripts in the benchmark depend on label files generated for each task.

3. Install FastText:
```sh
cd fastText
make
```

4. Extract demographic information:
``` sh
python -m dataloader.generate_demographic_csv.py

```

## How-to train on MIMIC-III

1. Setup a virtual environment with conda for easy CUDA support:
``` sh
conda env create --name ehr --file=environments.yml
conda activate ehr
```

2. Extract bin-edges and patient sentences using the notebook `dataloader/extract.ipynb`:

This generates following files:
``` sh
med_values.<table>*.txt
med_bin_edges.<table>*.txt
dem.*.params
data/sentences.mimic3.txt
embeddings/sentences.mimic3.counts
```

3. Train fasttext embeddings. You can skip this step as pretrained vectors are included in `embeddings/`:
``` sh
./fastText/fasttext skipgram -input embeddings/sentences.mimic3.txt -output embeddings/sentences.mimic3.txt.100d.Fasttext.15ws
```

4. (Optional step) Pretrain encoders with contrastive predictive coding:

``` sh
python -m pretrain cpc
<logs wandb run id>
```

> Default parameters are in `pretrain.py` file.

5. Finetune on benchmark tasks with a multitask model as explained in the paper:

Without contrastive pretraining:
```sh
python -m finetune base -e 20
```

Using contrastive pretraining step, with its wandb run id:
```sh
python -m finetune base --finetune=<WANDB_ID>
```

> Default parameters are in `finetune.py` file.


## Evaluation

First we generate validation and test predictions as csv, and then use the evaluation scripts provided by Harutyunyan et al.
``` sh
python -m collect <wandb_id>; sh evaluate.sh <wandb_id>
```

This results in files for predictions and the evaluation result for each task:
```sh
wandb/*-<wandb_id>/files/test_listfile_predictions/<task>-*.csv
wandb/*-<wandb_id>/files/test_listfile_predictions/<task>-*.csv.json
```
