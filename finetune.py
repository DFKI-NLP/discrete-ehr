import argparse

import wandb
from git import Repo

import utils
from run import run


def parse_arguments():
    parser = argparse.ArgumentParser(description='train multitask model for MIMIC-III')
    parser.add_argument('name', default=None, help='experiment name')
    parser.add_argument('--finetune', default=False,
                        help='resume training with id')

    parser.add_argument('--datalimit', default=None, type=float,
                        help='Ratio of training data to use.')

    parser.add_argument('--dev', action='store_true',
                        help='path to data as outputted by mimic3-benchmarks')
    parser.add_argument('--test', default=False,
                        help='finally test on test set')
    parser.add_argument('--resume', default=False,
                        help='resume training with id')
    parser.add_argument('--config', default=False,
                        help='run id to load a config from')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--use_cache', default=True,
                        type=eval, choices=[True, False])
    parser.add_argument('--CV', default=0, type=int)
    parser.add_argument('-d', '--data_path', default='mimic3-benchmarks/data/multitask',
                        help='path to data as outputted by mimic3-benchmarks')

    # Data
    parser.add_argument('--input_tables', nargs='+', default=['CHARTEVENTS', 'LABEVENTS', 'OUTPUTEVENTS', 'dem', 'PRESCRIPTIONS', 'INPUTEVENTS_*'],
                        help='list of input tables from CHARTEVENTS, LABEVENTS, OUTPUTEVENTS, INPUTEVENTS_*, PRESCRIPTIONS, dem')
    parser.add_argument('--joint_tables', default=False,
                        type=eval, choices=[True, False])
    parser.add_argument('--event_limit', default=300, type=int)
    parser.add_argument('--step_limit', default=720, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--random_split', default=False, action='store_true')
    parser.add_argument('--sampler', default='torch.utils.data.RandomSampler')

    # Training
    parser.add_argument('--init_eval', default=False, action='store_true')
    parser.add_argument('-e', '--n_epochs', default=20, type=int,
                        help='max number of epochs to train')
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--clip_grad', default=10, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--cyclical_lr', nargs=2, default=False, type=float,
                        help='tuple of lr range low high.')
    parser.add_argument('--batch_update', default=32, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--patience', default=8, type=int)
    parser.add_argument('--tasks', nargs='+', default=['ihm', 'decomp', 'los_reg', 'los_cl', 'phen'],
                        help='list of tasks from ihm decomp los_reg los_cl phen')
    parser.add_argument('--loss_weight_ihm', default=.2, type=float,
                        help='loss weigt for multitask learning.')
    parser.add_argument('--loss_weight_decomp', default=2., type=float,
                        help='loss weigt for multitask learning.')
    parser.add_argument('--loss_weight_los_reg', default=.1, type=float,
                        help='loss weigt for multitask learning.')
    parser.add_argument('--loss_weight_los_cl', default=1., type=float,
                        help='loss weigt for multitask learning.')
    parser.add_argument('--loss_weight_phen', default=1., type=float,
                        help='loss weigt for multitask learning.')

    # Model
    parser.add_argument('--modelcls', default='models.MultitaskFinetune')
    parser.add_argument('--patient_modelcls', default='models.PatientChannelRNNMaxPoolEncoder')
    parser.add_argument('--timestep_agg_modelcls', default='models.SeperateTimestepEncoder')
    parser.add_argument('--timestep_modelcls', default='models.LinearMaxSumPool')
    parser.add_argument('--event_modelcls', default='models.EventEncoder')
    parser.add_argument('--eventcls', default='dataloader.utils.BinnedEvent')
    parser.add_argument('--hidden_size', default=50, type=int)
    parser.add_argument('--pat_hidden_size', default=128, type=int)
    parser.add_argument('--pat_hidden_size0', default=128, type=int)
    parser.add_argument('--pat_layers', default=1, type=int)
    parser.add_argument('--dem_size', default=8, type=int)

    parser.add_argument('--freeze_encoders', default=False,
                        type=eval, choices=[True, False])

    # Input / Embeddings
    parser.add_argument('--min_word_count', default=10, type=int)
    parser.add_argument('--padaware', default=True,
                        type=eval, choices=[True, False])
    parser.add_argument('--pat_padaware', default=True,
                        type=eval, choices=[True, False])

    parser.add_argument('--rand_emb', default=False,
                        type=eval, choices=[True, False])
    parser.add_argument('--freeze_emb', default=True,
                        help='Whether to freeze or finetune embedding layer. Defaults to freezing.',
                        type=eval, choices=[True, False])
    parser.add_argument('--drop_checkpoint_embeddings', default=False,
                        type=eval, choices=[True, False])
    parser.add_argument('--vocab_file', default='embeddings/sentences.mimic3.train.kmeans.10bins.cloip.patient.train.counts')
    parser.add_argument('--emb_path', default='embeddings/sentences.mimic3.train.kmeans.10bins.cloip.patient.train.txt.Fasttext.vec')
    parser.add_argument('--emb_dim', default=0, type=int,
                        help='Required only when using random embeddings.')
    parser.add_argument('--suffix', default='.mimic3.train')
    parser.add_argument('--n_bins', default=10, type=int)
    parser.add_argument('--strategy', default='kmeans',
                        type=str, choices=['uniform', 'kmeans', 'quantile'])

    parser.add_argument('--normalize', default='',
                        choices=['', 'models.LNNormalizer', 'models.Normalizer'])
    parser.add_argument('--normalize_mlp', default=False,
                        type=eval, choices=[True, False])
    parser.add_argument('--sep_bin', default=False,
                        type=eval, choices=[True, False])
    parser.add_argument('--append_bin', default=False,
                        type=eval, choices=[True, False])
    parser.add_argument('--include_time', default=True,
                        type=eval, choices=[True, False])
    parser.add_argument('--include_float', default=False, action='store_true')
    parser.add_argument('--include_demfc', default=True,
                        type=eval, choices=[True, False])
    parser.add_argument('--include_source', default=False,
                        type=eval, choices=[True, False])
    parser.add_argument('--include_timestep', default=True, action='store_true')
    parser.add_argument('--include_eventd', default=True, action='store_true')
    parser.add_argument('--include_dem', default=True,
                        type=eval, choices=[True, False])

    # Regularization
    parser.add_argument('--step_dropout', default=.1, type=float)
    parser.add_argument('--event_dropout', default=.15, type=float)
    parser.add_argument('--input_dropout', default=0., type=float)
    parser.add_argument('--visit_dropout', default=0., type=float)
    parser.add_argument('--decision_dropout', default=0., type=float)
    parser.add_argument('--dem_dropout', default=0.3, type=float)

    return parser.parse_args()


repo = Repo('')
commit_hash = str(repo.commit('HEAD'))

args = parse_arguments()

config = vars(args)
if args.config or args.resume or args.test:
    config = utils.load_config(args.config or args.resume or args.test)
    config.update({'dev': args.dev})
    if args.resume or args.test:
        config.update({'resume': args.resume,
                       'test': args.test})
    if args.resume:
        config.update({'n_epochs': args.n_epochs})

wandb.init(
    name=args.name if not args.dev else args.name + '_dev',
    id=args.resume or args.test,
    project="thesis",
    notes=f'commit:{commit_hash}',
    resume=args.resume,
    config=config)

run(**config)
