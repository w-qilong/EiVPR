""" This main entrance of the whole project.

    Most of the code should not be changed, please directly
    add all the input arguments of your model's constructor
    and the dataset file's constructor. The MInterface and
    DInterface can be seen as transparent to all your args.
"""

import warnings

import pytorch_lightning as pl

from model import MInterface
# import call callbacks functions and parser for args
from parser import parser

warnings.filterwarnings("ignore")

import torch

torch.set_float32_matmul_precision('high')
import torchvision.transforms as T
from PIL import Image, ImageDraw

# define image mean and std
mean_std = {'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]}


def main(args):
    # set random seed
    pl.seed_everything(args.seed)

    # load checkpoint
    model = MInterface.load_from_checkpoint(
        '/media/cartolab/DataDisk/wuqilong_file/VPR_project/logs/dinov2_backbone/lightning_logs/version_3/checkpoints/dinov2_backbone_epoch(55)_step(54712)_R1[0.9189]_R5[0.9619]_R10[0.9742].ckpt')
    print(model)
    model.eval()

    # define transform for training dataset
    valid_transform = T.Compose([
        T.Resize((322, 322), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=mean_std['mean'], std=mean_std['std'])
    ])

    # load image
    image_query_path = '/media/cartolab/DataDisk/wuqilong_file/VPR_datasets/Pittsburgh_250k/queries_real/005/005001_pitch1_yaw7.jpg'
    image_query = Image.open(image_query_path)
    image_query_ = valid_transform(image_query).unsqueeze(0).cuda()

    # define hook function
    features_in_hook = []
    features_out_hook = []

    def hook(module, fea_in, fea_out):
        features_in_hook.append(fea_in)
        features_out_hook.append(fea_out)
        return None

    layer_name = 'model.model.norm'
    # layer_name = 'model.model.blocks.11.norm1'

    for (name, module) in model.named_modules():
        print(name)
        if name == layer_name:
            module.register_forward_hook(hook=hook)

    cls_feat, mix_feats = model(image_query_)
    local_feats = features_out_hook[0][:, 1:, :]

    top_k=128

    correlation = torch.matmul(mix_feats, local_feats.permute((0, 2, 1)))
    order_f = torch.argsort(correlation, dim=2, descending=True)[0][0][:top_k]

    img = Image.open(image_query_path).resize((322, 322))
    a = ImageDraw.ImageDraw(img)

    width_num = 23
    height_num = 23

    column = order_f // width_num
    height = order_f % width_num

    for co, he in zip(column, height):
        a.rectangle(((14 * co, 14 * he), (14 * (co + 1), 14 * (he + 1))), fill=None, outline='green', width=2)

    img.show()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
