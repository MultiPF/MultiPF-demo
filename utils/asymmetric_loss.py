import torch
import torch.nn as nn
from collections import defaultdict
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.distributions import Categorical
from sklearn.metrics import precision_score,f1_score,recall_score, accuracy_score, average_precision_score
import json
import os

def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)  # x：(n_q,text_dim)-->(n_q,n_cls,text_dim)
    y = y.unsqueeze(0).expand(n, m, d)  # y：(n_cls,text_dim)-->(n_q,n_cls,text_dim)

    return torch.pow(x - y, 2).sum(2)  # (n_q,n_cls)

def triple_loss(dists, labels, device_id, tempthre=0.2):
    device = torch.device("cuda", device_id)
    loss = torch.tensor(0.0).to(device)
    #  add triple loss
    for index, logit in enumerate(dists):
        score = dists[index]
        true_scores, preindex = [], []
        label = labels[index]
        for i, item in enumerate(label):
            if item == 1:
                true_scores.append(score[i])
                preindex.append(i)
        maxscore = max(true_scores)
        size = score.shape[0]
        secondmin = 100000
        for j in range(size):
            if j not in preindex and score[j] < secondmin:
                secondmin = score[j]
        if maxscore - secondmin + tempthre > 0.0:
            loss += (maxscore - secondmin + tempthre).to(device)
    loss /= dists.shape[0]

    return loss

class ProtypicalLoss(nn.Module):

    def __init__(self, device_id, beta, gamma, learn_alpha, uncertainty_weight, use_pfa=None,
                 pfa_3d_key_smoothing=0, use_img_proto=1, use_text_proto=1, split=None):
        super(ProtypicalLoss, self).__init__()
        self.count_lossfn = CrossEntropyLoss()


        self.device_id = device_id
        self.beta = beta
        self.gamma = gamma
        self.learn_alpha = learn_alpha
        self.uncertainty_weight = uncertainty_weight
        self.use_pfa = use_pfa
        self.pfa_3d_key_smoothing = pfa_3d_key_smoothing
        self.use_img_proto = use_img_proto
        self.use_text_proto = use_text_proto
        self.split = split

    def forward(self, image_features, image_features_3d, text_features, count_outputs, weights, support_size, target,
                class_embs, classnames, alpha=0, loss_weight=None,
                pfa_tensor_u=None, target_a_U_bs=None, image_features_k_abs=None):
        use_pfa = self.use_pfa


        query_features = image_features[support_size:]


        query_ids = target[support_size:]


        if use_pfa:
            mse_loss = nn.MSELoss()
            a_U_b_emb = pfa_tensor_u[0]
            b_U_a_emb = pfa_tensor_u[1]
            pfa_loss = mse_loss(a_U_b_emb, b_U_a_emb)


            a_u_b_label = target_a_U_bs

            # consider support samples may have multiple labels
            support_dict = defaultdict(list)
            for i in range(support_size):
                label = target[i]
                assert support_size == label.shape[
                    0], f"support_size ({support_size}) must equal to label_size ({label.shape[0]})"
                for j in range(label.shape[0]):
                    if label[j] == 1:
                        # support_dict[j].append(image_features_3d[i])  # n_cls,n_img, n_dim
                        class_embs_j = class_embs[j].float()
                        class_embs_j = class_embs_j.unsqueeze(dim=0)
                        image_features_3d_i = image_features_3d[i]

                        attention_scores = torch.matmul(class_embs_j, image_features_3d_i.T)
                        attention_weights = F.softmax(attention_scores, dim=-1)
                        # print(attention_weights.size())
                        weighted_i = torch.matmul(attention_weights, image_features_3d_i)
                        weighted_i = weighted_i / weighted_i.norm(dim=-1, keepdim=True)
                        support_dict[j].append(weighted_i.squeeze())


            # print(len(a_u_b_label))
            for ii in range(int(len(a_u_b_label))):
                label = a_u_b_label[ii]
                a_U_b_emb_ii = a_U_b_emb[ii]
                for jj in range(label.shape[0]):
                    if label[jj] == 1:
                        class_embs_jj = class_embs[jj].float()
                        class_embs_jj = class_embs_jj.unsqueeze(dim=0)

                        # print(class_embs_i.size())
                        attention_scores = torch.matmul(class_embs_jj, a_U_b_emb_ii.T)
                        attention_weights = F.softmax(attention_scores, dim=-1)
                        # print(attention_weights.size())
                        weighted_a_U_b_emb_ii = torch.matmul(attention_weights, a_U_b_emb_ii)
                        weighted_a_U_b_emb_ii = weighted_a_U_b_emb_ii / weighted_a_U_b_emb_ii.norm(dim=-1, keepdim=True)
                        support_dict[jj].append(weighted_a_U_b_emb_ii.squeeze())

            support_prototypes = []



        else:
            # consider support samples may have multiple labels
            support_dict = defaultdict(list)
            for i in range(support_size):
                label = target[i]
                assert support_size == label.shape[
                    0], f"support_size ({support_size}) must equal to label_size ({label.shape[0]})"
                for j in range(label.shape[0]):
                    if label[j] == 1:
                        # support_dict[j].append(image_features_3d[i])  # n_cls,n_img, n_dim
                        class_embs_j = class_embs[j].float()
                        class_embs_j = class_embs_j.unsqueeze(dim=0)
                        image_features_3d_i = image_features_3d[i]

                        attention_scores = torch.matmul(class_embs_j, image_features_3d_i.T)
                        attention_weights = F.softmax(attention_scores, dim=-1)
                        # print(attention_weights.size())
                        weighted_i = torch.matmul(attention_weights, image_features_3d_i)
                        weighted_i = weighted_i / weighted_i.norm(dim=-1, keepdim=True)
                        support_dict[j].append(weighted_i.squeeze())
            support_prototypes = []


        text_features = text_features.unsqueeze(dim=1)  # (n_classes,1,text_dim)
        # print(text_features.size())
        t_type = type(text_features)


        for i in range(support_size):
            sample_embs = torch.stack(support_dict[i])
            support_prototypes.append(sample_embs.mean(dim=0))  # (n_classes,feature_dim)
        # support_prototypes = torch.stack(support_prototypes).unsqueeze(dim=1)  # (n_classes,1,feature_dim)
        support_prototypes = torch.stack(support_prototypes).unsqueeze(dim=1)  # (n_classes+n_generate,1,feature_dim)
        print('a')

        if self.use_img_proto and self.use_text_proto:
            if self.learn_alpha:
                prototypes = support_prototypes * alpha + text_features * (1-alpha)
                prototypes = prototypes.squeeze(dim=1)
            else:
                label_features = torch.cat((support_prototypes, text_features), dim=1)  # (n_classes,2,text_dim)
                prototypes = label_features.mean(dim=1)  # (n_classes,text_dim)
        elif self.use_text_proto:
            prototypes = text_features
            prototypes = prototypes.squeeze(dim=1)
            print('use_text_proto')
        elif self.use_img_proto:
            prototypes = support_prototypes
            prototypes = prototypes.squeeze(dim=1)
            print('use_img_proto')
        else:
            print('self.use_img_proto and self.use_text_proto should not both be 0')
            raise ValueError

        if use_pfa:
            if self.pfa_3d_key_smoothing:
                image_features_k_ab = image_features_k_abs  # n,50,4096
                image_features_k_ab = image_features_k_ab / image_features_k_ab.norm(dim=-1, keepdim=True)
                image_features_k_weight = image_features_k_ab @ image_features_k_ab.transpose(1, 2)  # n,50,50
                image_features_k_weight = torch.mean(image_features_k_weight, dim=-1).unsqueeze(dim=-1)  # n,50,1
                image_features_k_weight = F.softmax(image_features_k_weight, dim=-2)
                a_U_b_emb_pool = image_features_k_weight * a_U_b_emb
                a_U_b_emb_pool = torch.sum(a_U_b_emb_pool, dim=-2)
                a_U_b_emb_pool = a_U_b_emb_pool / a_U_b_emb_pool.norm(dim=-1, keepdim=True)
                dists_pfa = euclidean_dist(a_U_b_emb_pool, prototypes)  # (n_q,n_cls)
                log_p_y_pfa = F.log_softmax(-dists_pfa, dim=1)  # num_query x num_class
            else:
                a_U_b_emb_avgpool = F.avg_pool1d(a_U_b_emb.permute(0, 2, 1), kernel_size=a_U_b_emb.size(1)).permute(0, 2, 1).squeeze()
                a_U_b_emb_avgpool = a_U_b_emb_avgpool / a_U_b_emb_avgpool.norm(dim=-1, keepdim=True)
                # print(a_U_b_emb_avgpool.size())
                dists_pfa = euclidean_dist(a_U_b_emb_avgpool, prototypes)  # (n_q,n_cls)
                log_p_y_pfa = F.log_softmax(-dists_pfa, dim=1)  # num_query x num_class

            loss_pfa = - a_u_b_label * log_p_y_pfa
            loss_pfa = loss_pfa.mean()

        dists = euclidean_dist(query_features, prototypes)  # (n_q,n_cls)
        log_p_y = F.log_softmax(-dists, dim=1)  # num_query x num_class

        loss = - query_ids * log_p_y

        loss = loss.mean()



        if weights is not None:

            dist = Categorical(weights)
            entropy_loss = dist.entropy()
            entropy_loss = entropy_loss.mean()
        else:
            entropy_loss = 0

        labels_count = target.sum(dim=1)-1
        four = torch.ones_like(labels_count)*3
        labels_count = torch.where(labels_count > 3, four, labels_count)

        count_loss = self.count_lossfn(count_outputs, labels_count)


        if use_pfa:

            if self.uncertainty_weight:
                loss_list = [loss, count_loss, entropy_loss, loss_pfa, pfa_loss]
                final_loss = []
                for i in range(len(loss_list)):
                    final_loss.append(loss_list[i] / (2 * loss_weight[i].pow(2)) + torch.log(loss_weight[i]))

                all_loss = torch.sum(torch.stack(final_loss))


            else:
                all_loss = loss + count_loss + entropy_loss + loss_pfa + pfa_loss


        else:
            if self.uncertainty_weight:
                loss_list = [loss, count_loss, entropy_loss]
                final_loss = []
                for i in range(len(loss_list)):
                    final_loss.append(loss_list[i] / (2 * loss_weight[i].pow(2)) + torch.log(loss_weight[i]))

                all_loss = torch.sum(torch.stack(final_loss))


            else:
                all_loss = loss + self.beta * count_loss + self.gamma * entropy_loss

        # multi
        _, count_pred = torch.max(count_outputs, 1, keepdim=True)
        labels_count = labels_count.cpu().detach()
        count_pred = count_pred.cpu().detach()
        c_acc = accuracy_score(labels_count, count_pred)
        query_count = count_pred[support_size:]

        

        sorts, indices = torch.sort(log_p_y, descending=True)
        # Returns the sorted tensor and the index of the elements in the sorted tensor respectively.
        x = []
        for i, t in enumerate(query_count):
            x.append(log_p_y[i][indices[i][query_count[i][0]]])

        device = torch.device("cuda", self.device_id)
        x = torch.tensor(x).view(log_p_y.shape[0], 1).to(device)
        one = torch.ones_like(log_p_y)
        zero = torch.zeros_like(log_p_y)
        y_pred = torch.where(log_p_y >= x, one, log_p_y)
        y_pred = torch.where(y_pred < x, zero, y_pred)

        target_mode = 'macro'

        query_ids = query_ids.cpu().detach()
        y_pred = y_pred.cpu().detach()
        macro_p = precision_score(query_ids, y_pred, average=target_mode)
        macro_r = recall_score(query_ids, y_pred, average=target_mode)
        macro_f = f1_score(query_ids, y_pred, average=target_mode)
        acc = accuracy_score(query_ids, y_pred)

        micro_p = precision_score(query_ids, y_pred, average='micro')
        micro_r = recall_score(query_ids, y_pred, average='micro')
        micro_f = f1_score(query_ids, y_pred, average='micro')

        pred_logits = F.softmax(-dists, dim=1).cpu().detach()
        ma_ap = average_precision_score(query_ids.cpu().numpy(), pred_logits.cpu().numpy(), average='macro')
        mi_ap = average_precision_score(query_ids.cpu().numpy(), pred_logits.cpu().numpy(), average='micro')

        return all_loss, 100 * acc, 100 * c_acc, \
                100 * mi_ap, 100 * micro_p, 100 * micro_r, 100 * micro_f, \
                100 * ma_ap, 100 * macro_p, 100 * macro_r, 100 * macro_f


