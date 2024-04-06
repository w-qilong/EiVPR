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
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    model = AggMInterface(**vars(args))
    state_dict = torch.load(
        '/media/cartolab/DataDisk/wuqilong_file/VPR_project_v1/logs/dinov2_finetune/lightning_logs/version_17/checkpoints/dinov2_finetune_epoch(15)_step(31264)_R1[0.9095]_R5[0.9554]_R10[0.9622].ckpt')

    for k, v in state_dict['state_dict'].items():
        if 'learner' in k:
            name=k.replace('learner', 'reducer')
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v
    # load params
    model.load_state_dict(new_state_dict)  # 从新加载这个模型。
    model=model.model.backbone
    model.cuda()
    model.eval()


    # model = AggMInterface.load_from_checkpoint(
    #     '/media/cartolab/DataDisk/wuqilong_file/VPR_project_v1/logs/dinov2_finetune/lightning_logs/version_16/checkpoints/dinov2_finetune_epoch(38)_step(76206)_R1[0.9135]_R5[0.9581]_R10[0.9649].ckpt').model.backbone
    # model.eval()
    # print(model)

    image_path = '/media/cartolab/DataDisk/wuqilong_file/VPR_datasets/MSLS_Dataset/MSLS/test/athens/query/images/9tfP0v69U0Gmyp_vZ0AzSg.jpg'
    # load image
    image_query = Image.open(image_path).resize((322, 322))

    # 为图像添加网格线
    # a = ImageDraw.ImageDraw(image_query)
    # width_num = 23
    # for i in range(width_num):
    #     for j in range(width_num):
    #         a.rectangle(((14 * i, 14 * j), (14 * (i + 1), 14 * (j + 1))), fill=None, outline='green', width=1)

    # 可视化结果
    fig, ax = plt.subplots(2, 9, figsize=(18, 4))
    ax[0][0].set_xticks([])
    ax[0][0].set_yticks([])
    ax[0][0].imshow(image_query)
    ax[1][0].set_xticks([])
    ax[1][0].set_yticks([])
    ax[1][0].imshow(image_query)
    image_query_ = valid_transform(image_query)
    image_query_ = image_query_.unsqueeze(0)
    # define hook function
    features_in_hook = []
    features_out_hook = []
    def hook(module, fea_in, fea_out):
        features_in_hook.append(fea_in)
        features_out_hook.append(fea_out)
    layer_name = f'token_reducer.attention_maps'
    for (name, module) in model.named_modules():
        print(name)
        if name == layer_name:
            module.register_forward_hook(hook=hook)

    model(image_query_.cuda())
    maps = features_out_hook[0][0]
    print(maps[0])

    maps = maps.cpu().detach().numpy()

    for i in range(len(maps)):
        if i<8:
            ax[0][i + 1].set_xticks([])
            ax[0][i + 1].set_yticks([])
            ax[0][i + 1].imshow(maps[i])
        else:
            ax[1][i-8 + 1].set_xticks([])
            ax[1][i-8 + 1].set_yticks([])
            ax[1][i-8 + 1].imshow(maps[i])

    # first_row_maps = maps[0]
    # ax[1].set_xticks([])
    # ax[1].set_yticks([])
    # ax[1].imshow(maps[0])

    fig.tight_layout()
    fig.tight_layout()
    output_filepath = os.path.join('../attention_visual_result/show_learner_attentions_v4/', os.path.basename(image_path),
                                   )
    plt.savefig(output_filepath, dpi=600, bbox_inches='tight')
    plt.show()
