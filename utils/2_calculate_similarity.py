import torch
import torch.nn as nn
from skimage.io import imshow
import matplotlib.pyplot as plt

global_feature = torch.randn((2, 256)).repeat(2, 128, 1)
print(global_feature.shape)
local_features = torch.randn((2, 256, 256))
print(local_features.shape)

cos = nn.CosineSimilarity(dim=1)
score = cos(global_feature, local_features).reshape(2,16,16)[0].numpy()
print(score.shape)
# print(score)
# # print(score.shape)
# # print(score[1])
# imshow(score[1])
plt.imshow(score)
plt.show()
