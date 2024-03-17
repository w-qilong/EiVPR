import importlib
import inspect

import numpy as np
import pytorch_lightning as pl
import torch
import torch.optim.lr_scheduler as lrs
import torchmetrics
from prettytable import PrettyTable
from utils.local_match import rerank_function

from utils import losses, validation


class AggMInterface(pl.LightningModule):
    def __init__(self, **kargs):
        super().__init__()
        # self.save_hyperparameters() Equivalent to self.hparams = hparams,
        # this line is equivalent to assigning a value to the self.hparams parameter
        self.kargs = kargs
        self.save_hyperparameters()

        self.load_model()
        self.configure_loss()
        self.save_hyperparameters()

        if self.hparams.rerank:
            # we use torchmetrics for calculate accuracy
            self.metric = torchmetrics.Accuracy(task='multiclass', num_classes=2)
            if not self.hparams.triplet_loss_ratio:
                # # init AutomaticWeightedLoss module
                self.automaticWeightedLoss = losses.AutomaticWeightedLoss(2)

    # load and init model by model file name and Class name.
    def load_model(self):
        name = self.hparams.model_name
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            Model = getattr(importlib.import_module(
                '.' + name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')

        self.model = self.instancialize(Model)

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getfullargspec(Model.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)

    def forward(self, x):
        # return global and ranked local features
        return self.model.dino_forward(x)

    def configure_loss(self):
        # define loss function
        self.triplet_loss_name = self.hparams.triplet_loss_function
        if self.triplet_loss_name in ['MultiSimilarityLoss', 'HardTripletLoss', 'TripletMarginLoss',
                                      'CentroidTripletLoss',
                                      'NTXentLoss', 'FastAPLoss', 'Lifted', 'ContrastiveLoss', 'CircleLoss',
                                      'SupConLoss']:
            self.triplet_loss_function = losses.get_loss(self.triplet_loss_name)
        else:
            raise ValueError(f'Optimizer {self.triplet_loss_name} has not been added to "configure_loss()"')

        # define triplet miner
        self.miner_name = self.hparams.miner_name
        if self.miner_name in ['TripletMarginMiner', 'MultiSimilarityMiner', 'PairMarginMiner']:
            self.miner = losses.get_miner(self.miner_name, self.hparams.miner_margin)
        else:
            raise ValueError(f'Optimizer {self.miner_name} has not been added to "configure_loss()"')

        # define match loss function
        self.class_loss_name = self.hparams.class_loss_name
        if self.class_loss_name in ['CrossEntropy', 'BCEWithLogitsLoss', 'FocalLoss', 'LocalLoss']:
            self.class_loss_function = losses.get_loss(self.class_loss_name)
        else:
            raise ValueError(f'Optimizer {self.class_loss_name} has not been added to "configure_loss()"')

        # define hard miner
        self.hardminer = losses.get_miner(self.hparams.match_miner_name)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # manually warm up lr without a scheduler
        if self.hparams.warmup_steps and self.trainer.global_step < self.hparams.warmup_steps:
            lr_scale = min(1., float(self.trainer.global_step + 1) / self.hparams.warmup_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.hparams.lr

        # update params
        optimizer.step(closure=optimizer_closure)

    def on_train_epoch_start(self):
        # we will keep track of the % of trivial pairs/triplets at the loss level
        self.triplet_batch_acc = []

    def training_step(self, batch, batch_idx):
        places, labels = batch

        # Note that GSVCities yields places (each containing N images)
        # which means the dataloader will return a batch containing BS places
        BS, N, ch, h, w = places.shape

        # reshape places and labels
        images = places.view(BS * N, ch, h, w)
        labels = labels.view(-1)

        # Feed forward the batch to the model
        global_feature, local_features = self.forward(images)

        if self.triplet_loss_name == 'MultiSimilarityLoss' and self.miner_name == 'MultiSimilarityMiner':
            miner_outputs = self.miner(global_feature, labels)
            triplet_loss = self.triplet_loss_function(global_feature, labels, miner_outputs)
            self.log('triplet_loss', triplet_loss, prog_bar=False, logger=True)

            # calculate the % of trivial pairs/triplets
            # which do not contribute in the loss value
            nb_samples = global_feature.shape[0]
            nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
            triplet_batch_acc = 1.0 - (nb_mined / nb_samples)

            # get mean accuracy
            self.triplet_batch_acc.append(triplet_batch_acc)
            self.log('triplet_mean_acc', sum(self.triplet_batch_acc) / len(self.triplet_batch_acc), prog_bar=True,
                     logger=True)

            # For MS loss, see https://blog.csdn.net/m0_46204224/article/details/117997854
            # MS-Loss包含两部分，前一部分是所有Positive Part对应的loss, 后分一部是所有Negative Part对应的loss。
            # anchor_index和positive_index记录Positive pair在global_feature的index。
            # 在此处，一个batch包含60个位置，一个位置包含四张图像。则anchor_index_positive, positive_index的长度为240*3，因为一张图像包含三张对应的positive图像。
            # anchor_index_negative, negative_index 的最大长度为240 *（240-4）=56640。但由于设置了对应的margin对负对进行筛选,
            # 所以anchor_index_negative, negative_index的长度小于56640

            if self.hparams.rerank:
                # use all (anchor, positive), (anchor, positive) pairs
                # anchor_index_positive, positive_index, anchor_index_negative, negative_index = miner_outputs
                # there can be no (anchor, positive), (anchor, positive) pairs, so we need to set some relus
                # if anchor_index_positive.shape[0] > 0 and anchor_index_negative.shape[0] > 0:
                # 1. random generate (anchor negative) pairs
                # anchor_local_features = torch.index_select(local_features, dim=0, index=anchor_index_positive)
                # positive_local_features = torch.index_select(local_features, dim=0, index=positive_index)
                # anchor_negative_local_features = torch.index_select(local_features, dim=0,
                #                                                     index=anchor_index_negative)
                # negative_local_features = torch.index_select(local_features, dim=0, index=negative_index)

                # 2. use hardest miner
                anchor_index_, anchor_index_positive_, negative_index_ = self.hardminer(global_feature, labels)
                anchor_local_features = torch.index_select(local_features, dim=0, index=anchor_index_)
                positive_local_features = torch.index_select(local_features, dim=0, index=anchor_index_positive_)
                negative_local_features = torch.index_select(local_features, dim=0, index=negative_index_)

                # calculate rerank score
                anchor_positive_match_score = self.model(
                    anchor_local_features, positive_local_features
                )
                anchor_negative_match_score = self.model(
                    anchor_local_features, negative_local_features
                )
                # cat outputs
                logits = torch.cat([anchor_positive_match_score, anchor_negative_match_score], dim=0)
                # set class target
                labels = torch.zeros(anchor_positive_match_score.shape[0] + anchor_negative_match_score.shape[0],
                                     dtype=torch.long).cuda()
                labels[:anchor_positive_match_score.shape[0]] = 1

                # calculate match loss via CrossEntropyLoss
                match_loss = self.class_loss_function(logits, labels)
                # calculate match accuracy
                batch_match_acc = self.metric(torch.argmax(logits.softmax(dim=1), dim=1), labels)
                self.log('match_loss', match_loss, prog_bar=False, logger=True)
                self.log('match_acc', batch_match_acc, prog_bar=True, logger=True)

                # calculate total loss
                if self.hparams.triplet_loss_ratio:
                    # weighted triplet loss and match loss
                    total_loss = self.hparams.triplet_loss_ratio * triplet_loss + (
                            1 - self.hparams.triplet_loss_ratio) * match_loss
                    # total_loss = triplet_loss + match_loss
                else:
                    total_loss = self.automaticWeightedLoss(triplet_loss, match_loss)
                    # get loss rate
                    triple_rate, match_rate = self.automaticWeightedLoss.params.data[0], \
                        self.automaticWeightedLoss.params.data[
                            1]
                    self.log('triple_rate', triple_rate, prog_bar=False, logger=True)
                    self.log('match_rate', match_rate, prog_bar=False, logger=True)
                self.log('total_loss', total_loss, prog_bar=False, logger=True)
                return {'loss': total_loss}
            else:
                return {'loss': triplet_loss}

        if self.triplet_loss_name == 'TripletMarginLoss' and self.miner_name == 'TripletMarginMiner':
            miner_outputs = self.miner(global_feature, labels)
            # 一个batch中可能没有满足条件的三元组，需要排除这种可能性
            if miner_outputs[0].shape[0] > 0:
                triplet_loss = self.triplet_loss_function(global_feature, labels, miner_outputs)

                # triplet_loss = self.triplet_loss_function(global_feature, labels, miner_outputs)
                self.log('triplet_loss', triplet_loss, prog_bar=False, logger=True)

                # calculate the % of trivial pairs/triplets
                # which do not contribute in the loss value
                nb_samples = global_feature.shape[0]
                nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
                triplet_batch_acc = 1.0 - (nb_mined / nb_samples)
                # get mean accuracy
                self.triplet_batch_acc.append(triplet_batch_acc)
                self.log('triplet_mean_acc', sum(self.triplet_batch_acc) / len(self.triplet_batch_acc), prog_bar=True,
                         logger=True)

                # if we need rerank, need to calculate focal feature match loss
                if self.hparams.rerank:
                    # generate hardest triplet for each element in a batch
                    # anchor_index_, anchor_index_positive_, negative_index_ = self.hardminer(global_feature, labels)
                    # anchor_local_features = torch.index_select(local_features, dim=0, index=anchor_index_)
                    # positive_local_features = torch.index_select(local_features, dim=0, index=anchor_index_positive_)
                    # negative_local_features = torch.index_select(local_features, dim=0, index=negative_index_)

                    anchor_index, positive_index, negative_index = miner_outputs
                    anchor_local_features = torch.index_select(local_features, dim=0, index=anchor_index)
                    positive_local_features = torch.index_select(local_features, dim=0, index=positive_index)
                    negative_local_features = torch.index_select(local_features, dim=0, index=negative_index)

                    # calculate rerank score
                    anchor_positive_match_score = self.model(
                        anchor_local_features, positive_local_features
                    )
                    anchor_negative_match_score = self.model(
                        anchor_local_features, negative_local_features
                    )
                    # cat outputs
                    logits = torch.cat([anchor_positive_match_score, anchor_negative_match_score], dim=0)
                    # set class target
                    labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
                    labels[:anchor_positive_match_score.shape[0]] = 1
                    # calculate match acc
                    batch_match_acc = self.metric(torch.argmax(logits.softmax(dim=1), dim=1), labels)
                    self.log('batch_match_acc', batch_match_acc, prog_bar=True, logger=True)
                    match_loss = self.class_loss_function(logits, labels)
                    self.log('match_loss', match_loss, prog_bar=False, logger=True)

                    # # calculate total loss
                    # match_loss = self.class_loss_function(anchor_local_features, positive_local_features,
                    #                                       negative_local_features)
                    # self.log('match_loss', match_loss, prog_bar=False, logger=True)

                    if self.hparams.triplet_loss_ratio:
                        # total_loss = triplet_loss + match_loss
                        # weighted triplet loss and match loss
                        total_loss = self.hparams.triplet_loss_ratio * triplet_loss + (
                                1 - self.hparams.triplet_loss_ratio) * match_loss
                    else:
                        # if weighted value is not given, use method for adaption values for two loss
                        total_loss = self.automaticWeightedLoss(triplet_loss, match_loss)
                        # get loss rate
                        triple_rate, match_rate = self.automaticWeightedLoss.params.data[0], \
                            self.automaticWeightedLoss.params.data[
                                1]
                        self.log('triple_rate', triple_rate, prog_bar=False, logger=True)
                        self.log('match_rate', match_rate, prog_bar=False, logger=True)

                    self.log('total_loss', total_loss, prog_bar=False, logger=True)
                    return {'loss': total_loss}
                else:
                    return {'loss': triplet_loss}

    def on_train_epoch_end(self):
        # we empty the batch_acc list for next epoch
        self.triplet_batch_acc = []

        # metric on all batches using custom accumulation
        if self.hparams.rerank:
            epoch_acc = self.metric.compute()
            self.log('match_epoch_acc', epoch_acc, prog_bar=False, logger=True)
            self.metric.reset()

    def on_validation_epoch_start(self):
        self.val_outputs = [[] for _ in range(len(self.trainer.datamodule.eval_set))]
        if self.hparams.rerank:
            # if rerank, we need to store local features
            self.val_local_outputs = [[] for _ in range(len(self.trainer.datamodule.eval_set))]

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        places, _ = batch
        # calculate descriptors
        global_feature, local_features = self(places)
        # save each batch outputs for each dataloader
        self.val_outputs[dataloader_idx].append(global_feature.detach().cpu())
        if self.hparams.rerank:
            self.val_local_outputs[dataloader_idx].append(local_features.detach().cpu())

    def on_validation_epoch_end(self):
        """this return descriptors in their order
                depending on how the validation dataset is implemented
                for this project (MSLS val, Pittburg val), it is always references then queries
                [R1, R2, ..., Rn, Q1, Q2, ...]
                """
        dm = self.trainer.datamodule
        k_values = self.hparams.recall_top_k  # recall K (1,5,10)
        val_step_outputs = self.val_outputs

        for i, (val_set_name, val_dataset) in enumerate(zip(dm.eval_dataset, dm.eval_set)):
            if 'pitts' in val_set_name:
                # split to ref and queries
                num_references = val_dataset.dbStruct.numDb
                positives = val_dataset.getPositives()
            elif 'mapillary' in val_set_name:
                # split to ref and queries
                num_references = val_dataset.num_references
                positives = val_dataset.pIdx
            elif 'nordland' in val_set_name:
                # split to ref and queries
                num_references = val_dataset.num_references
                positives = val_dataset.pIdx
            elif 'spedtest' in val_set_name:
                # split to ref and queries
                num_references = val_dataset.num_references
                positives = val_dataset.pIdx
            elif 'essex3in1' in val_set_name:
                # split to ref and queries
                num_references = val_dataset.num_references
                positives = val_dataset.pIdx
            elif 'tokyo' in val_set_name:
                num_references = val_dataset.dbStruct.numDb
                positives = val_dataset.getPositives()

            else:
                print(f'Please implement validation_epoch_end for {val_set_name}')
                raise NotImplemented

            # get and concat all global features
            global_feats = torch.concat(val_step_outputs[i], dim=0)
            r_global_list = global_feats[: num_references]  # list of ref images descriptors
            q_global_list = global_feats[num_references:]  # list of query images descriptors
            # get the results of first ranking
            pitts_dict, predictions = validation.get_validation_recalls(r_list=r_global_list,
                                                                        q_list=q_global_list,
                                                                        k_values=k_values,
                                                                        gt=positives,
                                                                        print_results=True,
                                                                        dataset_name=val_set_name,
                                                                        faiss_gpu=self.hparams.faiss_gpu
                                                                        )
            for k in k_values[:3]:
                self.log(f'{val_set_name}_NoRank/R{k}', pitts_dict[k], prog_bar=False, logger=True)

            if self.hparams.rerank:
                val_step_local_outputs = self.val_local_outputs
                local_feats = torch.concat(val_step_local_outputs[i], dim=0)
                # get local features
                r_local_list = local_feats[: num_references]  # list of local ref images descriptors
                q_local_list = local_feats[num_references:]  # list of local query images descriptors
                # second rerank
                # rerank_predictions = rerank_function(predictions, q_local_list, r_local_list)

                rerank_predictions = []  # save rerank results
                self.model.eval()
                with torch.no_grad():
                    for index, pre in enumerate(predictions):
                        query_feat = q_local_list[index]
                        ref_feat = torch.index_select(r_local_list, dim=0, index=pre)
                        rank_score = self.model(query_feat.repeat(len(ref_feat), 1, 1).cuda(), ref_feat.cuda())
                        rank_score = rank_score.softmax(dim=1)[:, 1]
                        rerank = zip(rank_score, pre)

                        positive = []
                        negative = []
                        for item in rerank:
                            if item[0] < 0.5:
                                negative.append(item)
                            else:
                                positive.append(item)
                        rerank = positive + negative

                        # rerank = sorted(rerank, key=lambda t: t[0], reverse=True)
                        rerank = [i[1] for i in rerank]
                        rerank_predictions.append(rerank)

                correct_at_k = np.zeros(len(k_values))
                for q_idx, pred in enumerate(rerank_predictions):
                    for i, n in enumerate(k_values):
                        # if in top N then also in top NN, where NN > N
                        if np.any(np.in1d(pred[:n], positives[q_idx])):
                            correct_at_k[i:] += 1
                            break
                correct_at_k = correct_at_k / len(predictions)
                d = {k: v for (k, v) in zip(k_values, correct_at_k)}

                print()  # print a new line
                table = PrettyTable()
                table.field_names = ['K'] + [str(k) for k in k_values]
                table.add_row(['Recall@K'] + [f'{100 * v:.2f}' for v in correct_at_k])
                print(table.get_string(title=f"Rerank Performances on {val_set_name}"))

                for k in k_values[:3]:
                    self.log(f'{val_set_name}_Rank/R{k}', d[k], prog_bar=False, logger=True)

                del num_references, predictions, r_global_list, q_global_list
                del local_feats, r_local_list, q_local_list
        # delete
        del global_feats, val_step_outputs, val_step_local_outputs
        print('\n\n')

    def configure_optimizers(self):
        # If weight_decay is set, set its value to optimizer
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0

        # set optimizer and its hparams
        if self.hparams.optimizer.lower() == 'sgd':
            if self.hparams.rerank and not self.hparams.triplet_loss_ratio:
                optimizer = torch.optim.SGD(
                    [
                        {'params': self.model.parameters()},
                        {'params': self.automaticWeightedLoss.parameters(), 'weight_decay': 0}
                    ],
                    lr=self.hparams.lr,
                    weight_decay=weight_decay
                )
            else:
                optimizer = torch.optim.SGD(self.parameters(),
                                            lr=self.hparams.lr,
                                            weight_decay=weight_decay,
                                            momentum=self.hparams.momentum)

        elif self.hparams.optimizer.lower() == 'adamw':
            if self.hparams.rerank and not self.hparams.triplet_loss_ratio:
                optimizer = torch.optim.AdamW(
                    [
                        {'params': self.model.parameters()},
                        {'params': self.automaticWeightedLoss.parameters(), 'weight_decay': 0}
                    ],
                    lr=self.hparams.lr,
                    weight_decay=weight_decay
                )
            else:
                optimizer = torch.optim.AdamW(
                    self.parameters(),
                    lr=self.hparams.lr,
                    weight_decay=weight_decay)

        elif self.hparams.optimizer.lower() == 'adam':
            optimizer = torch.optim.Adam(self.parameters(),
                                         lr=self.hparams.lr,
                                         weight_decay=weight_decay)
        else:
            raise ValueError(f'Optimizer {self.optimizer} has not been added to "configure_optimizers()"')

        # Use lr_scheduler
        if not self.hparams.lr_scheduler:
            return optimizer
        else:
            if self.hparams.lr_scheduler == 'step':
                scheduler = lrs.StepLR(optimizer,
                                       step_size=self.hparams.lr_decay_steps,
                                       gamma=self.hparams.lr_decay_rate)

            elif self.hparams.lr_scheduler == 'multi_step':
                scheduler = lrs.MultiStepLR(optimizer,
                                            milestones=self.hparams.milestones,
                                            gamma=self.hparams.lr_decay_rate)

            elif self.hparams.lr_scheduler == 'cosine':
                scheduler = lrs.CosineAnnealingLR(optimizer,
                                                  T_max=self.hparams.T_max,
                                                  eta_min=self.hparams.eta_min)

            elif self.hparams.lr_scheduler == 'linear':
                scheduler = lrs.LinearLR(
                    optimizer,
                    start_factor=self.hparams.start_factor,
                    end_factor=self.hparams.end_factor,
                    total_iters=self.hparams.total_iters
                )

            elif self.hparams.lr_scheduler == 'exp':
                scheduler = lrs.ExponentialLR(
                    optimizer,
                    gamma=self.hparams.gamma
                )

            else:
                raise ValueError('Invalid lr_scheduler type!')
            return [optimizer], [scheduler]
