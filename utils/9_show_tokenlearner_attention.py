import warnings
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from data import DInterface
from model import AggMInterface
# import call callbacks functions and parser for args
from utils.call_backs import load_callbacks
from parser import parser
from torchvision import transforms as T
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import torch

import os

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision('high')

# define image mean and std
mean_std = {'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]}

# define transform for training dataset
valid_transform = T.Compose([
    T.Resize((322, 322), interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize(mean=mean_std['mean'], std=mean_std['std'])
])

args = parser.parse_args()
data_module = DInterface(**vars(args))
if args.model_name == 'dinov2_finetune':
    model = AggMInterface.load_from_checkpoint(
        '/media/cartolab/DataDisk/wuqilong_file/VPR_project_v1/logs/dinov2_finetune/lightning_logs/version_16/checkpoints/dinov2_finetune_epoch(38)_step(76206)_R1[0.9135]_R5[0.9581]_R10[0.9649].ckpt').model.backbone
    model.eval()
    print(model)


    image_path = '/media/cartolab/DataDisk/wuqilong_file/VPR_datasets/MSLS_Dataset/MSLS/train_val/cph/query/images/0vlzaZELuI_YJyLtcIQgRw.jpg'
    # load image
    image_query = Image.open(image_path).resize((322, 322))
    a = ImageDraw.ImageDraw(image_query)
    width_num = 23

    for i in range(width_num):
        for j in range(width_num):
            a.rectangle(((14 * i, 14 * j), (14 * (i + 1), 14 * (j + 1))), fill=None, outline='green', width=1)

    fig, ax = plt.subplots(1, 2, figsize=(6, 4))

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].imshow(image_query)

    image_query_ = valid_transform(image_query)
    image_query_ = image_query_.unsqueeze(0)

    # define hook function
    features_in_hook = []
    features_out_hook = []


    def hook(module, fea_in, fea_out):
        features_in_hook.append(fea_in)
        features_out_hook.append(fea_out)


    layer_name = f'token_learner.attention_maps'
    for (name, module) in model.named_modules():
        print(name)
        if name == layer_name:
            module.register_forward_hook(hook=hook)

    model(image_query_.cuda())
    maps = features_out_hook[0][0]
    print(maps[0])

    maps = maps.cpu().detach().numpy()
    first_row_maps = maps[0]
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].imshow(maps[0])

    fig.tight_layout()
    output_filepath = os.path.join('../attention_visual_result/show_learner_attentions/', os.path.basename(image_path),
                                  )
    plt.savefig(output_filepath, dpi=600, bbox_inches='tight')
    plt.show()

    # fig, ax = plt.subplots(1, 2, figsize=(9, 9))
    # ax[0, 0].imshow(image_query)
    # # plt.show()
    #
    # image_query_ = valid_transform(image_query)
    # image_query_ = image_query_.unsqueeze(0)
    #
    # # define hook function
    # features_in_hook = []
    # features_out_hook = []
    #
    # def hook(module, fea_in, fea_out):
    #     features_in_hook.append(fea_in)
    #     features_out_hook.append(fea_out)
    #
    # layer_name = f'token_learner.attention_maps'
    # for (name, module) in model.named_modules():
    #     print(name)
    #     if name == layer_name:
    #         module.register_forward_hook(hook=hook)
    #
    # model(image_query_.cuda())
    # maps=features_out_hook[0][0]
    # print(maps[0])
    #
    # # maps =torch.rot90(maps,k=3,dims=(1,2))
    # # print(maps[0])
    # maps=maps.cpu().detach().numpy()
    # first_row_maps = maps[:2]
    # other_row_maps = maps[2:]
    #
    # ax[0, 1].imshow(maps[0])
    # ax[0, 2].imshow(maps[1])
    #
    # for i in range(len(other_row_maps)):
    #     row = i // 3
    #     column = i % 3
    #     ax[row+1, column].imshow(other_row_maps[i])
    #
    # plt.show()
