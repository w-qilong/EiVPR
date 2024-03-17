""" This main entrance of the whole project.

    Most of the code should not be changed, please directly
    add all the input arguments of your model's constructor
    and the dataset file's constructor. The MInterface and
    DInterface can be seen as transparent to all your args.
"""
import os
import warnings

import pytorch_lightning as pl
from pytorch_lightning import Trainer

from data import DInterface
from model import MatchMInterface, AggMInterface, MInterface
# import call callbacks functions and parser for args
from utils.call_backs import load_callbacks
from parser import parser
from collections import OrderedDict

warnings.filterwarnings("ignore")

import torch
import sys

sys.path.append('/')
sys.path.append('utils/')

torch.set_float32_matmul_precision('high')

model_version = 1


def main(args):
    # set random seed
    pl.seed_everything(args.seed)

    # init pytorch_lighting data and model module
    # vars(args) transformer property and value of a python object into a dict
    data_module = DInterface(**vars(args))


    model_1 = AggMInterface(**vars(args))
    model_2 = AggMInterface(**vars(args))
    model_3 = AggMInterface(**vars(args))
    fl_model = AggMInterface(**vars(args))
    checkpoint_path = os.path.join(f'logs/{args.model_name}/lightning_logs/version_{model_version}/checkpoints')
    print(checkpoint_path)

    checkpoints = os.listdir(checkpoint_path)
    checkpoints = [i for i in checkpoints if i != 'last.ckpt']
    for model, checkpoint in zip([model_1, model_2, model_3], checkpoints):
        model.load_state_dict(torch.load(os.path.join(checkpoint_path, checkpoint))['state_dict'])

    models = [model_1, model_2, model_3]
    worker_state_dict = [x.state_dict() for x in models]

    weight_keys = list(worker_state_dict[0].keys())
    fed_state_dict = OrderedDict()
    for key in weight_keys:
        key_sum = 0
        for i in range(len(models)):
            key_sum = key_sum + worker_state_dict[i][key]
        fed_state_dict[key] = key_sum / len(models)

    fl_model.load_state_dict(fed_state_dict)

    # add callbacks to args and send it to Trainer
    args.callbacks = load_callbacks(args)
    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        default_root_dir=f'./logs/{args.model_name}',  # we use current model for log folder name
        max_epochs=args.epochs,
        callbacks=args.callbacks,  # we only run the checkpointing callback (you can add more)
        reload_dataloaders_every_n_epochs=1,  # we reload the dataset to shuffle the order
        check_val_every_n_epoch=1,  # run validation every epoch
        log_every_n_steps=20,
        enable_model_summary=True,
        benchmark=True,
        num_sanity_val_steps=0,  # runs a validation step before starting training
        precision='16-mixed',  # we use half precision to reduce  memory usage

        # todo: used for debug
        # profiler=profiler,
        # fast_dev_run=True,  # uncomment or dev mode (only runs a one iteration train and validation, no checkpointing).
        # limit_train_batches=1,
        # limit_val_batches=1
    )

    # validate model using defined test_dataloader, you have to set the ckpt_path
    trainer.validate(model=fl_model, datamodule=data_module)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
