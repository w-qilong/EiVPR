import numpy as np

all_features = np.empty((100, 768),dtype="float32")

features = np.ones((10, 768),dtype="float32")

all_features[:10,:]=features
all_features[10:20,:]=features

print(all_features[10:20])
print(all_features.shape)