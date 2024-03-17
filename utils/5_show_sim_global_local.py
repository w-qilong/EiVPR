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
            '/media/cartolab/DataDisk/wuqilong_file/VPR_project/logs/dinov2_aggregator/lightning_logs/version_1/checkpoints/dinov2_aggregator_epoch(26)_step(13203)_R1[0.8878]_R5[0.9500]_R10[0.9608].ckpt')
        model.eval()
        print(model)

        # define input image path
        image_path = '/media/cartolab/DataDisk/wuqilong_file/VPR_project/attention_visual_result/results/NN-8TjLjIJVTwHBcAYdnbA_ori.jpg'

        # define transform for training dataset
        valid_transform = T.Compose([
            T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=mean_std['mean'], std=mean_std['std'])
        ])

        # load image
        image_query = Image.open(image_path).resize((224, 224))
        image_query_ = valid_transform(image_query).unsqueeze(0).cuda()

        fig, ax = plt.subplots(1, 2, figsize=(8, 4))

        # define hook function
        features_out_hook = []


        def hook(module, fea_in, fea_out):
            features_out_hook.append(fea_out)


        hook_layer_name = ['model.aggregator.mix.3', 'model.aggregator']
        for (name, module) in model.named_modules():
            if name in hook_layer_name:
                module.register_forward_hook(hook=hook)

        model(image_query_)

        local_feats, global_feats = features_out_hook[0].permute(0, 2, 1) , features_out_hook[1]

        rank_score = torch.cosine_similarity(global_feats.unsqueeze(dim=1), local_feats,
                                             dim=2)

        rank_score = torch.reshape(rank_score, (16, 16))

        ax0 = ax[0].imshow(image_query)
        ax[0].axis('off')

        ax1 = ax[1].imshow(rank_score.cpu().rot90(1,(0,1)).numpy())
        # ax1 = ax[1].imshow(rank_score.cpu().numpy())
        ax[1].axis('off')
        # fig.colorbar(ax1,ax=ax[1])

        plt.show()
        #
        # ax[1].imshow(rank_score.reshape())
        # ax[1].axis('off')
        #
        # im = ax[2].imshow(sim_score, cmap='OrRd', vmin=0.001, vmax=1.0)
        # ax[2].axis('off')
        #
        # fig.subplots_adjust(wspace=0.05)
        # # plt.savefig(f'../point_correspondences/{os.path.basename(image_query_path)}/layer{i}.tif', dpi=600,
        # #             bbox_inches='tight')
        #
        # # plt.colorbar(im)
        # # plt.show()
        #
        # # clear cache
        # features_in_hook = []
        # features_out_hook = []
        # # remove hook
        # h.remove()
        # plt.pause(2)
        # # plt.show()
        # # break
