import torch
from torch.utils.data import DataLoader, Subset
from model import RerankMInterface
from parser import parser

from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import faiss
from prettytable import PrettyTable
import torch.nn.functional as F

DATASET_ROOT = r'/media/cartolab/DataDisk/wuqilong_file/VPR_datasets/VPR_Bench_Datasets/SPEDTEST/'
path_obj = Path(DATASET_ROOT)
if not path_obj.exists():
    raise Exception('Please make sure the path to Nordland dataset is correct')

if not path_obj.joinpath('query'):
    raise Exception(
        f'Please make sure the directory train_val from Nordland dataset is situated in the directory {DATASET_ROOT}')

if not path_obj.joinpath('ref'):
    raise Exception(
        f'Please make sure the directory train_val from Nordland dataset is situated in the directory {DATASET_ROOT}')


class SpedtestDataset(Dataset):
    def __init__(self, input_transform=None):
        self.input_transform = input_transform

        # hard coded reference image names, this avoids the hassle of listing them at each epoch.
        self.dbImages = np.load(
            '../datasets/sped/sped_val_dbImages.npy')

        # hard coded query image names.
        self.qImages = np.load('../datasets/sped/sped_val_qImages.npy')

        # hard coded index of query images
        self.qIdx = np.load('../datasets/sped/sped_val_qIdx.npy')

        # hard coded groundtruth (correspondence between each query and its matches)
        self.pIdx = np.load('../datasets/sped/sped_val_pIdx.npy',
                            allow_pickle=True)

        # concatenate reference images then query images so that we can use only one dataloader
        self.images = np.concatenate((self.dbImages, self.qImages[self.qIdx]))

        # we need to keep the number of references so that we can split references-queries
        # when calculating recall@K
        self.num_references = len(self.dbImages)

        # you can use follow code to show some sample for query and correspond ref
        #
        # fig,axes=plt.subplots(1,2)
        # index=0
        # query=self.qImages[self.qIdx[index]]
        # img=Image.open(DATASET_ROOT+ query)
        # axes[0].imshow(img)
        # ref=self.dbImages[self.pIdx[index]]
        # print(ref)
        # axes[1].imshow(Image.open(DATASET_ROOT+ref[0]))
        # axes[2].imshow(Image.open(DATASET_ROOT+ref[1]))
        # axes[3].imshow(Image.open(DATASET_ROOT+ref[2]))
        # axes[1].imshow(Image.open(DATASET_ROOT+ref[3]))
        # axes[1].imshow(Image.open(DATASET_ROOT+ref[4]))
        # plt.show()

    def __getitem__(self, index):
        img = Image.open(DATASET_ROOT + self.images[index])

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)


# define config for valid dataloader
valid_loader_config = {
    'batch_size': 60,
    'num_workers': 8,
    'drop_last': False,
    'pin_memory': True,
    'shuffle': False,
    'persistent_workers': False}

# define image mean and std
mean_std = {'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]}

# define transform for training dataset
valid_transform = T.Compose([
    T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize(mean=mean_std['mean'], std=mean_std['std'])
])

if __name__ == '__main__':
    # init model
    args = parser.parse_args()
    # init data
    val_dataset = SpedtestDataset(input_transform=valid_transform)
    dataloader = DataLoader(dataset=val_dataset, **valid_loader_config)
    # init model
    model = RerankMInterface.load_from_checkpoint(
        checkpoint_path='/media/cartolab/DataDisk/wuqilong_file/VPR_project/logs/dinov2_ranker/lightning_logs/version_0/checkpoints/last.ckpt'
    ).model
    model.eval()

    feats = []
    with torch.no_grad():
        for index, (data, _) in enumerate(dataloader):
            global_feature, local_feature = model.backbone_forward(data.cuda())
            feats.append(global_feature)

        feats = torch.cat(feats, dim=0).cpu()
        num_references = val_dataset.num_references
        r_list = feats[: num_references]  # list of ref images descriptors
        q_list = feats[num_references:]  # list of query images descriptors
        gt = val_dataset.pIdx

        embed_size = r_list.shape[1]
        faiss_index = faiss.IndexFlatL2(embed_size)

        # add references
        faiss_index.add(r_list)

        k_values = [1, 5, 10, 15, 20, 50, 100]
        # search for queries in the index
        _, predictions = faiss_index.search(q_list, max(k_values))  # predictions为q_list中每个查询对应的topk参考结果索引

        # calculate rerank score for results from global retrieval
        q_data = Subset(val_dataset, indices=range(num_references, len(val_dataset)))
        ref_data = Subset(val_dataset, indices=range(0, num_references))

        # fig, ax = plt.subplots(1, 2)

        rerank_predictions = []
        for index, (query, _) in enumerate(q_data):
            # ax[0].imshow(query.permute(1, 2, 0).numpy())
            top_k_ref = predictions[index]
            ref_images = Subset(ref_data, indices=top_k_ref)
            ref_loader = DataLoader(ref_images, batch_size=len(ref_images))
            for i, (data, _) in enumerate(ref_loader):
                ref_images = data

            # ax[1].imshow(ref_images[2][0].permute(1, 2, 0).numpy())
            # plt.show()
            query_data = query.repeat(len(top_k_ref), 1, 1, 1)
            query_global, query_local = model.backbone_forward(query_data.cuda())
            ref_global, ref_local = model.backbone_forward(ref_images.cuda())

            match_score = model(query_local, ref_local)
            match_score = F.sigmoid(match_score).cpu().numpy().tolist()

            rerank = zip(match_score, top_k_ref.numpy().tolist())
            rerank = sorted(rerank, key=lambda t: t[0], reverse=True)
            rerank = [i[1] for i in rerank]
            print(top_k_ref)
            print(rerank)
            rerank_predictions.append(rerank)


        # start calculating recall_at_k
        correct_at_k = np.zeros(len(k_values))
        for q_idx, pred in enumerate(rerank_predictions):
            for i, n in enumerate(k_values):
                # if in top N then also in top NN, where NN > N
                if np.any(np.in1d(pred[:n], gt[q_idx])):
                    correct_at_k[i:] += 1
                    break

        correct_at_k = correct_at_k / len(rerank_predictions)
        d = {k: v for (k, v) in zip(k_values, correct_at_k)}

        print()  # print a new line
        table = PrettyTable()
        table.field_names = ['K'] + [str(k) for k in k_values]
        table.add_row(['Recall@K'] + [f'{100 * v:.2f}' for v in correct_at_k])
        print(table.get_string(title="Performances on spet"))
