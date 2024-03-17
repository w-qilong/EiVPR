import torch
from PIL import Image, ImageDraw
import numpy as np
import sys
from torchvision import transforms as T
import torch.nn as nn
import matplotlib.pyplot as plt
import os

sys.path.append('../')
from model import MInterface

'''
根据查询图像的局部特征，计算查询与参考图像局部特征之间的相似性。以验证Dino各层对光照、季节的变化环境下的鲁棒性。
'''

# define image mean and std
mean_std = {'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]}


def highlight_grid(image, grid_indexes, grid_size=16):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    W, H = image.size
    h = H / grid_size[0]
    w = W / grid_size[1]
    image = image.copy()
    for grid_index in grid_indexes:
        x, y = np.unravel_index(grid_index, (grid_size[0], grid_size[1]))
        a = ImageDraw.ImageDraw(image)
        a.rectangle([(y * w, x * h), (y * w + w, x * h + h)], fill=None, outline='red', width=2)
    return image


if __name__ == '__main__':
    with torch.no_grad():
        # load checkpoint
        model = MInterface.load_from_checkpoint(
            '/media/cartolab/DataDisk/wuqilong_file/VPR_project/logs/dinov2_aggregator_mixer/lightning_logs/version_0/checkpoints/HMNet_epoch(29)_step(18780)_R1[0.9206]_R5[0.9626]_R10[0.9732].ckpt')
        model.eval()

        # define input image path
        image_query_path = '/media/cartolab/DataDisk/wuqilong_file/VPR_datasets/Pittsburgh_250k/000/000002_pitch1_yaw10.jpg'
        image_ref_path = '/media/cartolab/DataDisk/wuqilong_file/VPR_datasets/Pittsburgh_250k/000/000002_pitch1_yaw9.jpg'

        if not os.path.exists(os.path.join('../point_correspondences', os.path.basename(image_query_path))):
            os.mkdir(os.path.join('../point_correspondences', os.path.basename(image_query_path)))

        # define transform for training dataset
        valid_transform = T.Compose([
            T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=mean_std['mean'], std=mean_std['std'])
        ])

        # load image
        image_query = Image.open(image_query_path).resize((224, 224))
        image_query_ = valid_transform(image_query)

        image_ref = Image.open(image_ref_path).resize((224, 224))
        image_ref_ = valid_transform(image_ref)

        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        plt.ion()

        # define hook function
        features_in_hook = []
        features_out_hook = []


        def hook(module, fea_in, fea_out):
            features_in_hook.append(fea_in)
            features_out_hook.append(fea_out)
            return None


        for i in range(0, 11):
            # define target layer name
            layer_name = f'model.backbone.model.blocks.{i}'
            for (name, module) in model.named_modules():
                if name == layer_name:
                    module.register_forward_hook(hook=hook)

            model(image_query_.unsqueeze(0).cuda())
            model(image_ref_.unsqueeze(0).cuda())

            query_local_features = features_out_hook[0][:, 1:].squeeze()
            ref_local_features = features_out_hook[1][:, 1:].squeeze()
            print(query_local_features.shape)
            print(ref_local_features.shape)

            # calculate similarity of selected query token and ref tokens
            selected_token_index = 17
            select_token = query_local_features[selected_token_index]
            cos_sim = nn.CosineSimilarity(dim=1)
            sim_score = cos_sim(select_token, ref_local_features).reshape(16, 16).cpu().numpy()
            print(sim_score.min())
            print(sim_score.max())

            plt.title(f'layer {i}')
            image_query = highlight_grid(image_query, [selected_token_index], grid_size=14)
            ax[0].imshow(image_query)
            ax[0].axis('off')

            ax[1].imshow(image_ref)
            ax[1].axis('off')

            im = ax[2].imshow(sim_score, cmap='OrRd', vmin=0.001, vmax=1.0)
            ax[2].axis('off')

            fig.subplots_adjust(wspace=0.05)
            # plt.savefig(f'../point_correspondences/{os.path.basename(image_query_path)}/layer{i}.tif', dpi=600,
            #             bbox_inches='tight')

            # plt.colorbar(im)
            # plt.show()

            # clear cache
            features_in_hook = []
            features_out_hook = []
            # remove hook
            h.remove()
            plt.pause(2)
            # plt.show()
            # break
