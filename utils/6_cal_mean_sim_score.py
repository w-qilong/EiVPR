""" This main entrance of the whole project.

    Most of the code should not be changed, please directly
    add all the input arguments of your model's constructor
    and the dataset file's constructor. The MInterface and
    DInterface can be seen as transparent to all your args.
"""

import warnings

import pytorch_lightning as pl

from data import DInterface
# import call callbacks functions and parser for args
from parser import parser

warnings.filterwarnings("ignore")
import torch

torch.set_float32_matmul_precision('high')

from model.dino_backbone_ver1 import Dinov2Backbone
import torch.nn.functional as F
import pandas as pd


def main(args):
    # set random seed
    pl.seed_everything(args.seed)

    # add profiler
    # profiler = AdvancedProfiler(dirpath="./profiler", filename="perf_logs")

    # init pytorch_lighting data and model module
    # vars(args) transformer property and value of a python object into a dict
    data_module = DInterface(**vars(args))

    ###############################
    data_module.setup('fit')

    eval_datasets = data_module.eval_dataset
    eval_loader = data_module.val_dataloader()
    loader = zip(eval_datasets, eval_loader)

    layer_num = 8
    model = Dinov2Backbone(
        num_blocks=layer_num,
        return_token=False
    ).cuda()

    sim_score = dict()
    for name, dataloader in loader:
        sim_score[name] = []
        for data, index in dataloader:
            local_ = model(data.cuda())
            sim = F.cosine_similarity(local_.unsqueeze(1), local_.unsqueeze(2), dim=-1)
            mean_sim = sim.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).round(
                decimals=6).detach().cpu().squeeze()
            sim_score[name].extend(mean_sim.tolist())

    out = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in sim_score.items()])).round(decimals=6)
    print(out.head())
    out.to_csv(f'results/{layer_num}_layer_feats_mean.csv', index=False)

    # print(mean_sim.shape)
    # plt.imshow(sim[0].numpy())
    # plt.show()
    # break

    ###############

    # # todo: used for training dino+aggregator
    # if args.model_name == 'dinov2_aggregator':
    #     model = AggMInterface(**vars(args))
    #
    # elif args.model_name == 'dinov2_matcher':
    #     model = MatchMInterface(**vars(args))
    #
    # # add callbacks to args and send it to Trainer
    # args.callbacks = load_callbacks(args)
    #
    # trainer = Trainer(
    #     accelerator=args.accelerator,
    #     devices=args.devices,
    #     default_root_dir=f'./logs/{args.model_name}',  # we use current model for log folder name
    #     max_epochs=args.epochs,
    #     callbacks=args.callbacks,  # we only run the checkpointing callback (you can add more)
    #     reload_dataloaders_every_n_epochs=1,  # we reload the dataset to shuffle the order
    #     check_val_every_n_epoch=2,  # run validation every epoch
    #     log_every_n_steps=20,
    #     enable_model_summary=True,
    #     benchmark=True,
    #     num_sanity_val_steps=0,  # runs a validation step before starting training
    #     precision='16-mixed',  # we use half precision to reduce  memory usage
    #
    #     # todo: used for debug
    #     # profiler=profiler,
    #     # fast_dev_run=True,  # uncomment or dev mode (only runs a one iteration train and validation, no checkpointing).
    #     # limit_train_batches=1,
    #     # limit_val_batches=1
    # )
    #
    # # train and eval model using train_dataloader and eval_dataloader
    # trainer.fit(model, data_module)
    #
    # # validate model using defined test_dataloader, you have to set the ckpt_path
    # # trainer.validate(model=model, datamodule=data_module,
    # #                  ckpt_path=r'/media/cartolab/DataDisk/wuqilong_file/VPR_project/logs/dinov2_matcher/lightning_logs/version_4/checkpoints/dinov2_matcher_epoch(94)_step(92815)_R1[0.9029]_R5[0.9583]_R10[0.9711].ckpt')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
