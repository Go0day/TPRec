from __future__ import absolute_import, division, print_function

import sys
import os
import argparse
from math import log
from datetime import datetime
from tqdm import tqdm
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
import threading
from functools import reduce

from knowledge_graph import KnowledgeGraph
from kg_env_lstm import BatchKGEnvironment
from train_agent_LSTM import ActorCritic
from utils import *

def load_clus_weight(dataset, usage='train'):
    uc_file = TMP_DIR[dataset] + '/' + usage + '_user_clus_weight.pkl'
    user_clus_weight = pickle.load(open(uc_file, 'rb'))
    return user_clus_weight

def evaluate(topk_matches, test_user_products):
    """Compute metrics for predicted recommendations.
    Args:
        topk_matches: a list or dict of product ids in ascending order.
    """
    invalid_users = []
    # Compute metrics
    precisions, recalls, ndcgs, hits = [], [], [], []
    test_user_idxs = list(test_user_products.keys())
    for uid in test_user_idxs:
        if uid not in topk_matches or len(topk_matches[uid]) < 10:
            # 看predict的时候有没有算漏了
            invalid_users.append(uid)
            continue
        # pred_list为毛要翻转一下
        pred_list, rel_set = topk_matches[uid][::-1], test_user_products[uid] 
        if len(pred_list) == 0:
            continue

        dcg = 0.0
        hit_num = 0.0
        for i in range(len(pred_list)):
            if pred_list[i] in rel_set:
                dcg += 1. / (log(i + 2) / log(2))
                hit_num += 1
        # idcg
        idcg = 0.0
        for i in range(min(len(rel_set), len(pred_list))):
            idcg += 1. / (log(i + 2) / log(2))
        ndcg = dcg / idcg
        recall = hit_num / len(rel_set)
        precision = hit_num / len(pred_list)
        hit = 1.0 if hit_num > 0.0 else 0.0

        ndcgs.append(ndcg)
        recalls.append(recall)
        precisions.append(precision)
        hits.append(hit)

    avg_precision = np.mean(precisions) * 100
    avg_recall = np.mean(recalls) * 100
    avg_ndcg = np.mean(ndcgs) * 100
    avg_hit = np.mean(hits) * 100
    print('NDCG={:.3f} |  Recall={:.3f} | HR={:.3f} | Precision={:.3f} | Invalid users={}'.format(
            avg_ndcg, avg_recall, avg_hit, avg_precision, len(invalid_users)))


def batch_beam_search(env, model, uids, device, topk=[25, 5, 1]):
    def _batch_acts_to_masks(batch_acts):
        batch_masks = []
        for acts in batch_acts:
            num_acts = len(acts)
            act_mask = np.zeros(model.act_dim, dtype=np.uint8)
            act_mask[:num_acts] = 1
            batch_masks.append(act_mask)
        return np.vstack(batch_masks)
    # 为这一个 dataloader batch 的测试用户uids初始化
    # 当前的batch path, state, action以及reward -> 0.
    state_pool = env.reset(uids)  # numpy of [batch_size, dim], cur state.
    # 刚开始batch_path是(SELF_LOOP, USER, uid)
    path_pool = env._batch_path  # list of (relation, node_type, node_id), size=bs
    probs_pool = [[] for _ in uids]
    model.eval() # 模型评估时使用 model.eval对dropout等层进行freeze.
    for hop in range(3):
        state_tensor = torch.FloatTensor(state_pool).to(device)
        # act_pool中包含一个batch的动作空间, 最多为251.
        acts_pool = env._batch_get_actions(path_pool, False)  # list of list, size=bs
        '''batch_mask中每一个都只保留最多251的act, 不足补零, 多了截断.'''
        actmask_pool = _batch_acts_to_masks(acts_pool)  # numpy of [bs, dim]
        actmask_tensor = torch.ByteTensor(actmask_pool).to(device)
        # 进入model.forward 传播阶段
        probs, _ = model((state_tensor, actmask_tensor))  # Tensor of [bs, act_dim], ex. probs, value = self((state, act_mask)).
        # 我咋感觉是为了使mask和未mask的action差异更大点. mask掉的位置依然是0.
        probs = probs + actmask_tensor.float()  # In order to differ from masked actions
        # topk=[25, 5, 1], 每个hop在那个251的动作空间找概率分布.
        topk_probs, topk_idxs = torch.topk(probs, topk[hop], dim=1)  # LongTensor of [bs, k]
        topk_idxs = topk_idxs.detach().cpu().numpy()
        topk_probs = topk_probs.detach().cpu().numpy() # 得到top k的概率分布以及索引.

        new_path_pool, new_probs_pool = [], []
        for row in range(topk_idxs.shape[0]):
            # 遍历batch中每一行.
            path = path_pool[row]
            probs = probs_pool[row]
            for idx, p in zip(topk_idxs[row], topk_probs[row]):
                # 遍历top K中每一个idx以及对应的概率
                if idx >= len(acts_pool[row]):  # act idx is invalid
                    continue
                relation, next_node_id = acts_pool[row][idx]  # (relation, next_node_id)
                if relation == SELF_LOOP:
                    # self_loop 当然就是原地踏步了
                    next_node_type = path[-1][1]
                else:
                    # 向前走一步
                    next_node_type = KG_RELATION[path[-1][1]][relation]
                # 保留top K中对应的每一跳的信息
                new_path = path + [(relation, next_node_type, next_node_id)]
                new_path_pool.append(new_path)
                new_probs_pool.append(probs + [p])
        path_pool = new_path_pool # 1 hop 之后是16 * 25,  400维的动作空间.
        probs_pool = new_probs_pool # 并且对应每种 hop action的概率. 3 hop 就是3维[1.9496549, 1.9625425, 1.9999171].
        if hop < 2:
            # 3跳截至.
            state_pool = env._batch_get_state(path_pool)

    return path_pool, probs_pool


def predict_paths(policy_file, path_file, args):
    ''' 相当于按照batch分批, 给每个user分配三跳的action_path以及对应的probs. 
        每一个user三个hop的Top K - (25, 5, 1) * batch_size;
        一个user有125条path, 每条path对应概率[prob_hop_1, prob_hop_2, prob_hop_3]'''

        
    print('Predicting paths...')
    env = BatchKGEnvironment(args.dataset, args.max_acts, max_path_len=args.max_path_len, state_history=args.state_history, mode='test')
    pretrain_sd = torch.load(policy_file)
    model = ActorCritic(env.state_dim, env.act_dim, gamma=args.gamma, hidden_sizes=args.hidden).to(args.device)
    model_sd = model.state_dict() # 包含层的参数
    model_sd.update(pretrain_sd)
    model.load_state_dict(model_sd)
    # {uid: [pid1, pid2]...}
    test_labels = load_labels(args.dataset, 'test')
    test_uids = list(test_labels.keys())

    batch_size = 16
    start_idx = 0
    all_paths, all_probs = [], []
    pbar = tqdm(total=len(test_uids))
    while start_idx < len(test_uids):
        # 相当于一次dataloader
        end_idx = min(start_idx + batch_size, len(test_uids))
        batch_uids = test_uids[start_idx:end_idx]
        paths, probs = batch_beam_search(env, model, batch_uids, args.device, topk=args.topk)
        all_paths.extend(paths)
        all_probs.extend(probs)
        start_idx = end_idx
        pbar.update(batch_size)
    predicts = {'paths': all_paths, 'probs': all_probs}
    pickle.dump(predicts, open(path_file, 'wb'))


def evaluate_paths(path_file, train_labels, test_labels):
    embeds = load_embed(args.dataset)
    # ===================================================
    clus_weight = load_clus_weight(args.dataset, usage='test')
    purchase_matrix = []
    # gg = len(clus_weight)
    for userID in range(len(clus_weight)):
        # 如果embeds[USER]是dict的话就这样干, 主要现在测试不了
        ff = 1
        for clus_wt in clus_weight[userID]:
            if ff is 0:
                purchase_emd += embeds[PURCHASE[clus_wt]][0] * clus_weight[userID][clus_wt]
                # mention_emd += self.embeds[MENTION[clus_wt]][0] * self.clus_weight[path[0][-1]][clus_wt]
            else:
                ff = 0
                purchase_emd = embeds[PURCHASE[clus_wt]][0] * clus_weight[userID][clus_wt]
                # mention_emd = self.embeds[MENTION[clus_wt]][0] * self.clus_weight[path[0][-1]][clus_wt]
        purchase_matrix.append(purchase_emd)



    # ===================================================
    user_embeds = embeds[USER]
    # purchase_embeds = embeds[PURCHASE][0]
    purchase_embeds = np.array(purchase_matrix)
    product_embeds = embeds[PRODUCT]
    # 得到所有train的scores. 这个score有啥用??????????
    scores = np.dot(user_embeds + purchase_embeds, product_embeds.T) # 我猜是用来scale的.

    # 1) Get all valid paths for each user, compute path score and path probability.
    results = pickle.load(open(path_file, 'rb')) # 这是test的推理结果.
    pred_paths = {uid: {} for uid in test_labels}
    for path, probs in zip(results['paths'], results['probs']):
        # test里的推理路径和每一跳的概率
        if path[-1][1] != PRODUCT:
            # 只看结果是商品的.
            continue
        uid = path[0][2]
        if uid not in pred_paths:
            # 并且路径user在test_label里的.
            continue
        pid = path[-1][2]
        if pid not in pred_paths[uid]:
            pred_paths[uid][pid] = []
        path_score = scores[uid][pid]
        path_prob = reduce(lambda x, y: x * y, probs) # 路径概率累乘.
        # 存储推理预测得到的[uid][pid]对应的路径得分和概率以及三跳的路径记录.
        pred_paths[uid][pid].append((path_score, path_prob, path))

    # 2) Pick best path for each user-product pair, also remove pid if it is in train set.
    best_pred_paths = {}
    for uid in pred_paths:
        train_pids = set(train_labels[uid])
        best_pred_paths[uid] = []
        for pid in pred_paths[uid]:
            if pid in train_pids:
                # 不保留train里面的
                continue
            # Get the path with highest probability
            # 其中路径得分是第二位x[1]
            # 因为就算是最后推荐的商品一样, 但是其中所hop的路径以及其得分也不一样, 只保留最高的.
            sorted_path = sorted(pred_paths[uid][pid], key=lambda x: x[1], reverse=True)
            best_pred_paths[uid].append(sorted_path[0]) # 排序后保留得分最高的.

    # 3) Compute top 10 recommended products for each user.
    sort_by = 'score'
    pred_labels = {}
    for uid in best_pred_paths:
        # 先按得分, 或者先按路径概率, 进行排序.
        if sort_by == 'score':
            sorted_path = sorted(best_pred_paths[uid], key=lambda x: (x[0], x[1]), reverse=True)
        elif sort_by == 'prob':
            sorted_path = sorted(best_pred_paths[uid], key=lambda x: (x[1], x[0]), reverse=True)
        # 前十的商品pid
        top10_pids = [p[-1][2] for _, _, p in sorted_path[:10]]  # from largest to smallest
        # add up to 10 pids if not enough
        if args.add_products and len(top10_pids) < 10:
            # 不够10个, 添加得分降序且不在train&top 10 列表中的pid.
            train_pids = set(train_labels[uid])
            cand_pids = np.argsort(scores[uid])
            for cand_pid in cand_pids[::-1]:
                if cand_pid in train_pids or cand_pid in top10_pids:
                    continue
                top10_pids.append(cand_pid)
                if len(top10_pids) >= 10:
                    break
        # end of add
        pred_labels[uid] = top10_pids[::-1]  # change order to from smallest to largest!

    evaluate(pred_labels, test_labels)


def test(args):
    policy_file = args.log_dir + '/policy_model_epoch_{}.ckpt'.format(args.epochs)
    path_file = args.log_dir + '/policy_path  s_epoch{}.pkl'.format(args.epochs)

    train_labels = load_labels(args.dataset, 'train')
    test_labels = load_labels(args.dataset, 'test')

    if args.run_path:
        predict_paths(policy_file, path_file, args)
    if args.run_eval:
        evaluate_paths(path_file, train_labels, test_labels)


if __name__ == '__main__':
    boolean = lambda x: (str(x).lower() == 'true')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=BEAUTY, help='One of {cloth, beauty, cell, cd}')
    parser.add_argument('--name', type=str, default='train_agent', help='directory name.')
    parser.add_argument('--seed', type=int, default=123, help='random seed.')
    parser.add_argument('--gpu', type=str, default='1', help='gpu device.')
    parser.add_argument('--epochs', type=int, default=50, help='num of epochs.')
    parser.add_argument('--max_acts', type=int, default=250, help='Max number of actions.')
    parser.add_argument('--max_path_len', type=int, default=3, help='Max path length.')
    parser.add_argument('--gamma', type=float, default=0.99, help='reward discount factor.')
    parser.add_argument('--state_history', type=int, default=1, help='state history length')
    parser.add_argument('--hidden', type=int, nargs='*', default=[512, 256], help='number of samples')
    parser.add_argument('--add_products', type=boolean, default=False, help='Add predicted products up to 10')
    parser.add_argument('--topk', type=int, nargs='*', default=[25, 5, 1], help='number of samples')
    parser.add_argument('--run_path', type=boolean, default=True, help='Generate predicted path? (takes long time)')
    parser.add_argument('--run_eval', type=boolean, default=True, help='Run evaluation?')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

    args.log_dir = TMP_DIR[args.dataset] + '/' + args.name
    test(args)

