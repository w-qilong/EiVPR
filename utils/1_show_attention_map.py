import os

import PIL
import cv2
import numpy as np
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
from PIL import Image
import sys

sys.path.append('../')

from model import  AggMInterface

# define image mean and std
mean_std = {'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]}


def reshape_transform(tensor, height=16, width=16):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


if __name__ == '__main__':
    # load checkpoint
    model = AggMInterface.load_from_checkpoint(
        '/media/cartolab/DataDisk/wuqilong_file/VPR_project_v1/logs/dinov2_finetune/lightning_logs/version_22/checkpoints/dinov2_finetune_epoch(19)_step(39080)_R1[0.8865]_R5[0.9459]_R10[0.9595].ckpt').model.backbone.model
    print(model)
    model.eval()
    # for (name, module) in model.named_modules():
    #     print(name)

    # define input image path
    image_path = '/media/cartolab/DataDisk/wuqilong_file/VPR_project_v1/attention_visual_result/results/2H71VedR_-mlf3RYjkSKrA_ori.jpg'
    output_folder = '../attention_visual_result/天气'
    ori_path = os.path.basename(image_path)
    print(ori_path)
    output_path = os.path.join(output_folder, ori_path)
    print(output_path)

    # load image
    rgb_img = Image.open(image_path)
    rgb_img = rgb_img.resize((224, 224))
    # rgb_img.save(output_path, quality=95, dpi=(600.0, 600.0))

    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img, mean=mean_std['mean'],
                                    std=mean_std['std'])

    # we can use follow code get middle feature map from model
    # get middle feature map
    '''
    features_in_hook = []
    features_out_hook = []

    def hook(module, fea_in, fea_out):
        features_in_hook.append(fea_in)
        features_out_hook.append(fea_out)
        return None

    # define target layer name
    layer_name = 'model.backbone.model.norm'
    for (name, module) in model.named_modules():
        if name == layer_name:
            module.register_forward_hook(hook=hook)
            
    out = model(input_tensor)
    print(features_in_hook[0][0].shape)  # 勾的是指定层的输入
    print(features_out_hook[0][0].shape)  # 勾的是指定层的输出
    '''

    # set target layer for grad cam
    target_layers = [model.blocks[-1].norm1]
    # target_layers = [model.norm]
    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform, use_cuda=True)

    #  visual
    grayscale_cam = cam(input_tensor=input_tensor,
                        targets=None,
                        eigen_smooth=True,
                        # aug_smooth=True
                        )
    print(grayscale_cam.shape)

    # Here grayscale_cam has only one image in the batch
    grayscale_cam = grayscale_cam[0, :]
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, colormap=cv2.COLORMAP_JET, image_weight=0.75, )
    # cam_image = show_cam_on_image(rgb_img, grayscale_cam)
    cam_image = Image.fromarray(cv2.cvtColor(cam_image, cv2.COLOR_BGR2RGB))
    output_result_path = os.path.join(output_folder, ori_path.split('.')[0] + '.tif')

    cam_image.show('heat map')
    # cam_image.save(output_result_path, dpi=(600, 600))
