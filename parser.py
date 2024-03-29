from argparse import ArgumentParser

parser = ArgumentParser()

# todo: Model Hyperparameters
parser.add_argument('--model_name', default='dinov2_finetune', type=str)
parser.add_argument('--pretrained_model_name', default='dinov2_vitl14', type=str)
parser.add_argument('--num_trainable_blocks', default=2, type=int)
parser.add_argument('--norm_layer', default=True, type=bool)

parser.add_argument('--mix_in_channels', default=1024, type=int)
parser.add_argument('--mix_token_num', default=529, type=int)
parser.add_argument('--mix_out_channels', default=1024, type=int)
parser.add_argument('--mix_mix_depth', default=5, type=int)
parser.add_argument('--mix_mlp_ratio', default=2, type=int)
parser.add_argument('--mix_out_rows', default=4, type=int)

parser.add_argument('--rerank', default=True, type=bool)
parser.add_argument('--triplet_loss_ratio', default=0.5, type=float)
parser.add_argument('--num_learned_tokens', default=8, type=int)
parser.add_argument('--channels_reduced', default=0, type=int)

parser.add_argument('--trans_heads', default=8, type=int)
parser.add_argument('--trans_dropout', default=0.3, type=int)
parser.add_argument('--trans_layers', default=4, type=int)

parser.add_argument('--num_classifier', default=2, type=int)

# todo: Datasets information
# Typically, we need to verify the performance of our model on multiple validation datasets.
# Here, we can assign train/eval/test datasets. Here, we use standard_data for
parser.add_argument('--train_dataset', default='gsvcities_dataset', type=str)
# args for training dataset GSVCities
parser.add_argument('--image_size', default=(322, 322), type=tuple)
parser.add_argument('--shuffle_all', default=False, type=bool)
parser.add_argument('--img_per_place', default=4, type=int)
parser.add_argument('--min_img_per_place', default=4, type=int)
parser.add_argument('--random_sample_from_each_place', default=True, type=bool)
parser.add_argument('--persistent_workers', default=False, type=bool)
# args for eval dataset
parser.add_argument(
    '--eval_datasets',
    default=[
        'mapillary_dataset',
        'spedtest_dataset',
        'tokyo247_dataset',
        'nordland_dataset',
        'pittsburg30k_dataset',

        # 'pittsburg250k_dataset',
        # 'essex3in1_dataset',
    ], type=list)
# set monitor dataset
parser.add_argument('--monitor_metric', default='mapillary_dataset_Rank', type=str)
parser.add_argument('--recall_top_k', default=[1, 5, 10, 100], type=list)

# todo: Basic Training Control for global trainer
# set random seed
parser.add_argument('--seed', default=1234, type=int)
# use GPU or CPU
parser.add_argument('--accelerator', default='gpu', type=str)
# select GPU device
parser.add_argument('--devices', default=[0], type=list)
# set training epochs
parser.add_argument('--epochs', default=100, type=int)
# set batch size
parser.add_argument('--batch_size', default=32, type=int)
# set number of process worker in dataloader
parser.add_argument('--num_workers', default=15, type=int)
# set init learning rate for global trainer
parser.add_argument('--lr', default=1e-5, type=float)
# select optimizer. We have defined multiple optimizers in model_interface.py, we can select one for our study here.
parser.add_argument('--optimizer', choices=['sgd', 'adamw', 'adam'], default='adamw', type=str)
# set momentum of optimizer. It should set for sgd. When we use adam or adamw optimizer, no need to set it
parser.add_argument('--momentum', default=0.9, type=float)
# set weight_decay rate for optimizer
parser.add_argument('--weight_decay', default=9.5e-09, type=float)

# todo: LR Scheduler. Used for dynamically adjusting learning rates
# Here, we can use gradual warmup to , i.e., start with an initially small learning rate,
# and increase a little bit for each STEP until the initially set relatively large learning rate is reached,
# and then use the initially set learning rate for training.
parser.add_argument('--warmup_steps', default=100, type=int)

# select lr_scheduler. We have defined multiple lr_scheduler in model_interface.py, we can select one for our study here.
parser.add_argument('--lr_scheduler', choices=['step', 'multi_step', 'cosine', 'linear', 'exp'], default='cosine',
                    type=str)

# Set args for Different Scheduler
# For CosineAnnealingLR
parser.add_argument('--T_max', default=100, type=int)
parser.add_argument('--eta_min', default=6e-8, type=float)

# For StepLR
# parser.add_argument('--lr_decay_steps', default=20, type=int)
# parser.add_argument('--lr_decay_rate', default=0.5, type=float)

# For MultiStepLR
# parser.add_argument('--milestones', default=[30, 35, 40], type=list)
# # lr_decay_rate controls the change rate of learning rate
# parser.add_argument('--lr_decay_rate', default=0.5, type=float)

# For LinearLR
# parser.add_argument('--start_factor', default=1, type=float)
# parser.add_argument('--end_factor', default=0.2, type=float)
# parser.add_argument('--total_iters', default=1000 * 100, type=int)

# For ExponentialLR
# parser.add_argument('--gamma', default=0.99, type=float)

# todo: loss function
# select loss function. We have defined multiple loss function in model_interface.py,
# set args for loss function and triplet miner
parser.add_argument('--triplet_loss_function', choices=['MultiSimilarityLoss', 'TripletMarginLoss'],
                    default='MultiSimilarityLoss', type=str)
parser.add_argument('--miner_name', choices=['MultiSimilarityMiner', 'TripletMarginMiner'],
                    default='MultiSimilarityMiner', type=str)
parser.add_argument('--miner_margin', default=0.1, type=float)

parser.add_argument('--match_miner_name', choices=['BatchEasyHardMiner', 'BatchHardMiner'],
                    default='BatchHardMiner', type=str)

parser.add_argument('--faiss_gpu', default=False, type=bool)

parser.add_argument('--class_loss_name',
                    choices=['CrossEntropy', 'BCEWithLogitsLoss', 'FocalLoss', 'LocalLoss', 'CosineEmbeddingLoss'],
                    default='CrossEntropy', type=str)
parser.add_argument('--l_softmax_linear_margin', default=0, type=int)

# whether to use early stopping
parser.add_argument('--use_early_stopping', default=True, type=bool)
parser.add_argument('--patience', default=3, type=int)

# if use gradient accumulate, set arg for in
parser.add_argument('--gradient_accumulate', default=False, type=bool)
parser.add_argument('--gradient_accumulate_start_epoch', default=0, type=int)
parser.add_argument('--gradient_accumulate_factor', default=4, type=int)

# args for Exponential Moving Averaging (EMA)
parser.add_argument('--EMA', default=False, type=bool)
parser.add_argument('--EMA_decay', default=0.9999, type=float)
parser.add_argument('--EMA_validate_original_weights', default=True, type=bool)
parser.add_argument('--EMA_every_n_steps', default=32, type=int)  # use for each epoch
parser.add_argument('--EMA_cpu_offload', default=False, type=bool)

# set if StochasticWeightAveraging need to be used
parser.add_argument('--StochasticWeightAveraging', default=False, type=bool)
parser.add_argument('--swa_lrs', default=1e-5, type=float)
parser.add_argument('--swa_epoch_start', default=0.75, type=float)
parser.add_argument('--annealing_epochs', default=10, type=int)
parser.add_argument('--annealing_strategy', default='cos', type=str)

# # args for cross vit matcher
# parser.add_argument('--mix_depth', default=4, type=int)
# parser.add_argument('--mlp_ratio', default=2, type=int)
# parser.add_argument('--out_rows', default=1, type=int)
#
# parser.add_argument('--sm_dim', default=128, type=int)
# parser.add_argument('--lg_dim', default=128, type=int)
#
# parser.add_argument('--sm_enc_depth', default=1, type=int)
# parser.add_argument('--sm_enc_heads', default=8, type=int)
# parser.add_argument('--sm_enc_mlp_dim', default=768, type=int)
# parser.add_argument('--sm_enc_dim_head', default=64, type=int)
#
# parser.add_argument('--lg_enc_depth', default=1, type=int)
# parser.add_argument('--lg_enc_heads', default=8, type=int)
# parser.add_argument('--lg_enc_mlp_dim', default=768, type=int)
# parser.add_argument('--lg_enc_dim_head', default=64, type=int)
#
# parser.add_argument('--cross_attn_depth', default=1, type=int)
# parser.add_argument('--cross_attn_heads', default=8, type=int)
# parser.add_argument('--cross_attn_dim_head', default=64, type=int)
#
# parser.add_argument('--dropout', default=0.1, type=float)
# parser.add_argument('--emb_dropout', default=0.1, type=float)
#
# parser.add_argument('--num_classes', default=2, type=int)
# parser.add_argument('--depth', default=4, type=int)
