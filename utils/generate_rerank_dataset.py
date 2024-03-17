import torch
from model import AggMInterface
from data import DInterface
from parser import parser
from tqdm import tqdm

args = parser.parse_args()
data_module = DInterface(**vars(args))

checkpoint_path = '/logs/dinov2_finetune/lightning_logs/version_1/checkpoints/dinov2_finetune_epoch(28)_step(28333)_R1[0.9041]_R5[0.9486]_R10[0.9595].ckpt'
model = AggMInterface.load_from_checkpoint(checkpoint_path).model
# print(model)

all_global_feats = []
all_local_feats = []
all_labels = []

train_dataloader = data_module.train_dataloader()

for idx, (places, labels) in tqdm(enumerate(train_dataloader)):
    BS, N, ch, h, w = places.shape

    # reshape places and labels
    images = places.view(BS * N, ch, h, w)
    labels = labels.view(-1)
    print(labels)
    break
    # global_feats, local_feats = model.dino_forward(images.cuda())
    # all_global_feats.append(global_feats.detach().cpu())
    # all_local_feats.append(local_feats.detach().cpu())
    # all_labels.append(labels)

# all_global_feats = torch.cat(all_global_feats)
# all_local_feats = torch.cat(all_local_feats)
# all_labels = torch.cat(all_labels)
# torch.save(all_global_feats, './global_feats.pt')
# torch.save(all_local_feats, './local_feats')
# torch.save(all_labels, './labels.pt')
