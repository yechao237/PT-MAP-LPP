import collections
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from scipy.io import savemat
import math
import torch.nn.functional as F
import torch.optim as optim
from numpy import linalg as LA
from tqdm.notebook import tqdm

use_gpu = torch.cuda.is_available()

import numpy as np
import scipy.sparse as sparse
from scipy.linalg import eigh


def My_L2Norm(x):
    y = x / np.tile(np.sqrt(np.sum(x ** 2, axis=1, keepdims=True)).reshape(-1, 1), (1, x.shape[1]))
    return y


def My_LPP(data, W, options):
    data = np.array(data, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    D = D.astype(np.float64)
    Sl = data.T @ L @ data
    Sd = data.T @ D @ data

    Sl = (Sl + Sl.T) / 2
    Sd = (Sd + Sd.T) / 2
    Sl = Sl + options["alpha"] * np.eye(Sl.shape[1])
    # 计算特征值和特征向量
    w, v = eigh(Sd, Sl)
    w, v = w[-options['d']:], v[:, -options['d']:]

    P = np.flip(v, axis=1)
    P = -P
    for i in range(P.shape[1]):
        if P[0, i] < 0:
            P[:, i] *= -1
    return P


def My_EuDist2(fea_a, fea_b, bSqrt=True):
    fea_a = np.array(fea_a)
    fea_b = np.array(fea_b)
    aa = np.sum(fea_a ** 2, axis=1)
    bb = np.sum(fea_b ** 2, axis=1)
    ab = fea_a @ fea_b.T

    aa = aa.reshape(-1, 1)
    bb = bb.reshape(-1, 1)

    D = np.add(aa, bb.T) - 2 * ab
    D[D < 0] = 0
    if bSqrt:
        D = np.sqrt(D)
    return D


def My_constructW(fea, options):
    nSmp = fea.shape[0]
    G = np.zeros((nSmp * (options['k'] + 1), 3))
    selected_fea = fea[0:options['WDim'], :]
    dist = My_EuDist2(selected_fea, fea, 0)
    nSmpNow = len(selected_fea)
    dump = np.zeros((nSmpNow, options['k'] + 1))
    idx = dump.copy()
    for j in range(options['k'] + 1):
        dump[:, j], idx[:, j] = np.min(dist, axis=1), np.argmin(dist, axis=1)
        temp = idx[:, j] * nSmpNow + np.arange(nSmpNow)  # python按行存储
        temp = temp.astype(int)
        temp = np.clip(temp, 0, dist.size - 1)
        for k in range(options['WDim']):
            index = temp[k]
            row, col = np.unravel_index(index,
                                        (options['WDim'], options['WDim']))  # 将索引i转换为对应的行、列下标，注意在Python中，行、列下标从0开始
            dist[col, row] = 1e100  # 将dist(row, col)赋值为1e100
    dump = np.exp(-dump / (2 * options['t'] ** 2))
    G[:, 0] = np.tile(np.arange(nSmp), options['k'] + 1)
    G[:, 1] = np.ravel(idx, order='F').flatten()
    G[:, 2] = np.ravel(dump, order='F').flatten()
    W = sparse.csr_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(nSmp, nSmp))
    W = W.toarray()
    W = W - np.diag(np.diag(W))
    W = np.maximum(W, W.transpose())
    return W


# def LPP(ndatas, n_lsamples, options, W):
#     ndatas = ndatas.cpu().numpy()
#     supportX = ndatas[:, :n_lsamples, :].squeeze()
#     queryX = ndatas[:, n_lsamples:, :].squeeze()
#     n_runs = len(ndatas)
#     P = np.zeros((n_runs, ndatas.shape[2], options['d']))
#     supportX_2 = np.zeros((n_runs, n_lsamples, options['d']))
#     queryX_2 = np.zeros((n_runs, ndatas.shape[1] - n_lsamples, options['d']))
#     for i in range(n_runs):
#         P[i] = My_LPP(np.concatenate((supportX[i], queryX[i])), W[i], options)
#         # domainS_proj和domainT_proj，每轮将domainS_features和domainT_features降至5维
#         domainS_proj = np.dot(supportX[i], P[i])
#         domainT_proj = np.dot(queryX[i], P[i])
#         proj_mean = np.mean(np.concatenate((domainS_proj, domainT_proj)), axis=0)
#         domainS_proj = domainS_proj - np.tile(proj_mean, (domainS_proj.shape[0], 1))
#         domainT_proj = domainT_proj - np.tile(proj_mean, (domainT_proj.shape[0], 1))
#         domainS_proj = My_L2Norm(domainS_proj)
#         domainT_proj = My_L2Norm(domainT_proj)
#         supportX_2[i] = domainS_proj
#         queryX_2[i] = domainT_proj
#     ndatas = np.concatenate((supportX_2, queryX_2), axis=1)  # 在第一维上进行拼接
#     ndatas = torch.from_numpy(ndatas)
#     return ndatas


def LPP(ndatas, options, W):
    ndatas = ndatas.cpu().numpy()
    n_runs = len(ndatas)
    P = np.zeros((n_runs, ndatas.shape[2], options['d']))
    ndatas_2 = np.zeros((n_runs, ndatas.shape[1], options['d']))

    for i in range(n_runs):
        P[i] = My_LPP(ndatas[i], W[i], options)
        ndatas_proj = np.dot(ndatas[i], P[i])
        proj_mean = np.mean(ndatas_proj, axis=0)
        ndatas_proj = ndatas_proj - np.tile(proj_mean, (ndatas_proj.shape[0], 1))
        ndatas_proj = My_L2Norm(ndatas_proj)
        ndatas_2[i] = ndatas_proj

    ndatas = np.array(ndatas_2)
    ndatas = torch.from_numpy(ndatas)
    return ndatas, P


def get_LPP_datas(ndatas, n_runs, options):
    ndatas = ndatas.cpu().numpy()  # 全部数据
    n_sum = ndatas.shape[1]  # 一个task中的数据个数
    W = np.zeros((n_runs, n_sum, n_sum))
    # 1.无监督k近邻获取数据特征矩阵W(80, 80)  My_constructW、My_EuDist2
    for i in range(n_runs):
        # W[i] = My_constructW(np.concatenate((supportX[i], queryX[i])), options)
        W[i] = My_constructW(ndatas[i, :, :].squeeze(), options)
    ndatas = torch.from_numpy(ndatas)
    ndatas, P = LPP(ndatas, options, W)  # 执行降维
    ndatas = ndatas.cuda()
    return ndatas, P


# ========================================
#      loading datas


def centerDatas(datas):
    datas[:, :n_lsamples] = datas[:, :n_lsamples, :] - datas[:, :n_lsamples].mean(1, keepdim=True)
    datas[:, :n_lsamples] = datas[:, :n_lsamples, :] / torch.norm(datas[:, :n_lsamples, :], 2, 2)[:, :, None]
    datas[:, n_lsamples:] = datas[:, n_lsamples:, :] - datas[:, n_lsamples:].mean(1, keepdim=True)
    datas[:, n_lsamples:] = datas[:, n_lsamples:, :] / torch.norm(datas[:, n_lsamples:, :], 2, 2)[:, :, None]
    return datas


# def centerDatas(datas):
#     # centre of mass of all data support + querries
#     # mean = datas[:, :].mean(1, keepdim=True)
#     datas[:, :] = datas[:, :, :] - datas[:, :].mean(1, keepdim=True)
#     datas[:, :] = datas[:, :, :] / torch.norm(datas[:, :, :], 2, 2)[:, :, None]
#     return datas


def scaleEachUnitaryDatas(datas):
    norms = datas.norm(dim=2, keepdim=True)
    return datas / norms


def QRreduction(datas):
    ndatas = torch.linalg.qr(datas.permute(0, 2, 1)).R
    ndatas = ndatas.permute(0, 2, 1)
    return ndatas


class Model:
    def __init__(self, n_ways):
        self.n_ways = n_ways


# ---------  GaussianModel
class GaussianModel(Model):
    def __init__(self, n_ways, lam):
        super(GaussianModel, self).__init__(n_ways)
        self.mus = None  # shape [n_runs][n_ways][n_nfeat]
        self.lam = lam

    def clone(self):
        other = GaussianModel(self.n_ways)
        other.mus = self.mus.clone()
        return self

    def cuda(self):
        self.mus = self.mus.cuda()

    def initFromLabelledDatas(self):
        self.mus = ndatas.reshape(n_runs, n_shot + n_unlabelled, n_ways, n_nfeat)[:, :n_shot, ].mean(1)

    def updateFromEstimate(self, estimate, alpha):

        Dmus = estimate - self.mus
        self.mus = self.mus + alpha * (Dmus)

    def compute_optimal_transport(self, M, r, c, epsilon=1e-6):

        r = r.cuda()
        c = c.cuda()
        n_runs, n, m = M.shape
        P = torch.exp(- self.lam * M)
        P /= P.view((n_runs, -1)).sum(1).unsqueeze(1).unsqueeze(1)

        u = torch.zeros(n_runs, n).cuda()
        maxiters = 1000
        iters = 1
        # normalize this matrix
        while torch.max(torch.abs(u - P.sum(2))) > epsilon:
            u = P.sum(2)
            P *= (r / u).view((n_runs, -1, 1))
            P *= (c / P.sum(1)).view((n_runs, 1, -1))
            if iters == maxiters:
                break
            iters = iters + 1
        return P, torch.sum(P * M)

    def getProbas(self):
        # compute squared dist to centroids [n_runs][n_samples][n_ways]
        dist = (ndatas.unsqueeze(2) - self.mus.unsqueeze(1)).norm(dim=3).pow(2)

        p_xj = torch.zeros_like(dist)
        r = torch.ones(n_runs, n_usamples)
        c = torch.ones(n_runs, n_ways) * n_queries

        p_xj_test, _ = self.compute_optimal_transport(dist[:, n_lsamples:], r, c, epsilon=1e-6)
        p_xj[:, n_lsamples:] = p_xj_test

        p_xj[:, :n_lsamples].fill_(0)
        p_xj[:, :n_lsamples].scatter_(2, labels[:, :n_lsamples].unsqueeze(2), 1)

        return p_xj

    def estimateFromMask(self, mask):

        emus = mask.permute(0, 2, 1).matmul(ndatas).div(mask.sum(dim=1).unsqueeze(2))

        return emus


# =========================================
#    MAP
# =========================================

class MAP:
    def __init__(self, alpha=None):

        self.verbose = False
        self.progressBar = False
        self.alpha = alpha

    def getAccuracy(self, probas):
        olabels = probas.argmax(dim=2)  # 这里前n_lsamples个应该替换
        matches = labels.eq(olabels).float()
        acc_test = matches[:, n_lsamples:].mean(1)

        m = acc_test.mean().item()
        pm = acc_test.std().item() * 1.96 / math.sqrt(n_runs)
        return m, pm, olabels

    def performEpoch(self, model, epochInfo=None):

        p_xj = model.getProbas()
        self.probas = p_xj

        m_estimates = model.estimateFromMask(self.probas)

        # update centroids
        model.updateFromEstimate(m_estimates, self.alpha)

        if self.verbose:
            op_xj = model.getProbas()
            acc1, acc2, olabels = self.getAccuracy(op_xj)
            print("output model accuracy", acc1, " ", acc2)

    def loop(self, model, n_epochs=20):

        self.probas = model.getProbas()

        if self.progressBar:
            if type(self.progressBar) == bool:
                pb = tqdm(total=n_epochs)
            else:
                pb = self.progressBar

        for epoch in range(1, n_epochs + 1):
            if self.verbose:
                print(f"----- epoch[{epoch:3d}]  lr_p: {self.alpha:.3f}")
                # print("----- epoch[{:3d}]  lr_p: {:0.3f}  lr_m: {:0.3f}".format(epoch, self.alpha))
            self.performEpoch(model, epochInfo=(epoch, n_epochs))
            if (self.progressBar): pb.update()

        # get final accuracy and return it
        op_xj = model.getProbas()
        acc1, acc2, olabels = self.getAccuracy(op_xj)
        acc = [acc1, acc2]

        # 保存mat
        # ptmappromat = dataset + "_" + datasettype + "promat"
        # savemat(f'{ptmappromat}_wrn_{n_shot}shots.mat', mdict={'probs': op_xj.cpu().detach().numpy()})

        return acc, olabels


def LPP_from_P(query_data, P, options):
    n_runs = query_data.shape[0]
    query_data_2 = np.zeros((n_runs, query_data.shape[1], options['d']))
    for i in range(n_runs):
        query_data_proj = np.dot(query_data[i], P[i])
        proj_mean = np.mean(query_data_proj, axis=0)
        query_data_proj = query_data_proj - np.tile(proj_mean, (query_data_proj.shape[0], 1))
        query_data_proj = My_L2Norm(query_data_proj)
        query_data_2[i] = query_data_proj
    query_data = np.array(query_data_2)
    query_data = torch.from_numpy(query_data)
    query_data = query_data.cuda()
    return query_data


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import scipy as sp
from scipy.stats import t


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), sp.stats.sem(a)
    h = se * t._ppf((1 + confidence) / 2., n - 1)
    return m, h


def semi_pt(xs, ys, xq, yq):  # xs 支持集 xq 查询集
    # print(xs.device.type, ys.device.type, xq.device.type, yq.device.type)
    print(xs.shape, ys.shape, xq.shape, yq.shape)
    acc = []
    # for i in range(100):
    for i in range(xs.shape[0]):
        if i % 500 == 0:
            print(i)
        classifier = LogisticRegression(C=10, multi_class='auto', solver='lbfgs', max_iter=1000)  # 逻辑回归
        support_features, query_features = xs[i, :], xq[i, :]
        support_ys, query_ys = ys[i, :], yq[i, :]

        classifier.fit(support_features, support_ys)
        query_ys_pred = classifier.predict(query_features)
        acc.append(metrics.accuracy_score(query_ys, query_ys_pred))
    return mean_confidence_interval(acc)

import time
def semi_ptmap(xs, ys, xq, yq):
    # print(xs.device.type, ys.device.type, xq.device.type, yq.device.type)

    n_shot = 31
    n_ways = 5
    n_lsamples = n_ways * n_shot

    ndatas = torch.cat((xs, xq), dim=1)
    labels = torch.cat((ys, yq), dim=1)

    n_runs = ndatas.shape[0]
    # Power transform
    beta = 0.5
    ndatas[:, ] = torch.pow(ndatas[:, ] + 1e-6, beta)
    ndatas = QRreduction(ndatas)
    ndatas = scaleEachUnitaryDatas(ndatas)

    # trans-mean-sub

    ndatas = centerDatas(ndatas)

    print("size of the datas...", ndatas.size())

    # switch to cuda
    ndatas = ndatas.cuda()
    labels = labels.cuda()

    start_time = time.time()  # 记录开始时间

    # LPP
    # ndatas = get_LPP_datas(ndatas, n_lsamples, n_runs)
    n_nfeat = ndatas.size(2)

    # MAP
    lam = 10
    model = GaussianModel(n_ways, lam)
    model.initFromLabelledDatas()

    # alpha = 0.2
    alpha = 0.3
    optim = MAP(alpha)

    optim.verbose = True
    optim.progressBar = True

    acc_test, olabels = optim.loop(model, n_epochs=20)
    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算经过的时间
    print(f"函数执行时间为: {elapsed_time} 秒")
    print("final accuracy found {:0.2f}±{:0.2f}".format(*(100*x for x in acc_test)))


if __name__ == '__main__':
    # ---- data loading
    n_shot = 5
    n_ways = 5
    n_unlabelled = 100  # semi-supervised learning
    dataset = r"cifar"
    datasettype = r"semi"
    backbone = r"wrn"

    n_queries = 15  # transductive learning
    n_lsamples = n_ways * n_shot  # 已标记样本
    n_usamples = n_ways * n_unlabelled  # 半监督样本
    total_q = n_ways * n_queries  # 查询集
    n_samples = n_lsamples + n_usamples + total_q  # 全部样本
    semi_train_samples = n_lsamples + n_usamples  # semi_train_samples支持集+半监督样本 semi-train

    import FSLTask
    n_runs = FSLTask._maxRuns

    cfg = {'shot': n_shot, 'ways': n_ways, 'queries': n_queries + n_unlabelled}
    FSLTask.loadDataSet(dataset)
    FSLTask.setRandomStates(cfg)
    or_ndatas = FSLTask.GenerateRunSet(cfg=cfg)

    or_ndatas = or_ndatas.permute(0, 2, 1, 3).reshape(n_runs, n_samples, -1)
    labels = torch.arange(n_ways).view(1, 1, n_ways).expand(n_runs, n_shot + n_queries + n_unlabelled, 5).clone().view(n_runs,
                                                                                                        n_samples)
    or_labels = labels.cuda()
    # partition for semi-supervised learning
    ndatas, query_data = or_ndatas[:, :semi_train_samples, :], or_ndatas[:, semi_train_samples:,
                                                               :]  # 训练的数据和查询集数据。因此不能进行QR分解
    labels, q_labels = labels[:, :semi_train_samples], labels[:, semi_train_samples:]  # 训练的标签和查询集标签
    print(ndatas.shape, query_data.shape, labels.shape, q_labels.shape)

    # 半监督样本 和 transductive的查询集 不能一起，但 支持集+半监督样本 的结果可以用于优化 支持集+transductive的查询集
    # ilpc半监督:未标记支持集训练得到伪标签，然后逻辑回归
    # 一种新的方式的实现

    # Power transform
    beta = 0.5
    ndatas[:, ] = torch.pow(ndatas[:, ] + 1e-6, beta)  # 幂转换
    ndatas = scaleEachUnitaryDatas(ndatas)  # 标准化
    mean_support = ndatas[:, :].mean(1, keepdim=True)
    ndatas = centerDatas(ndatas)  # 中心化

    # switch to cuda
    ndatas = ndatas.cuda()
    labels = labels.cuda()

    ndatas_save = ndatas

    # LPP1：ndatas降维至35
    options = {'WDim': ndatas.shape[1], 'NeighborMode': 'KNN', 'WeightMode': 'HeatKernel', 'k': 7, 't': 1, 'd': 35,
               'alpha': 10}
    ndatas, P = get_LPP_datas(ndatas, n_runs, options)
    n_nfeat = ndatas.size(2)  # 维度信息

    query_data[:, ] = torch.pow(query_data[:, ] + 1e-6, beta)  # 查询集幂转换
    query_data = scaleEachUnitaryDatas(query_data)  # 查询集标准化
    query_data = query_data - mean_support  # 查询集中心化

    query_data_save = query_data.cuda()
    # 保存mat
    or_ndatas = torch.cat((ndatas_save, query_data_save), dim=1)
    # savemat(f'{dataset}_{datasettype}_{backbone}_{n_shot}shots.mat', mdict={'features': or_ndatas.cpu().detach().numpy(), 'labels':or_labels.cpu().detach().numpy()})

    # LPP2：query_data降维至35
    query_data = LPP_from_P(query_data, P, options)  # 使用P进行降维

    # trans-mean-sub
    print("size of the datas...", ndatas.size())


    # MAP
    lam = 10
    model = GaussianModel(n_ways, lam)
    model.initFromLabelledDatas()

    alpha = 0.3 if n_shot == 1 else 0.2

    optim = MAP(alpha)

    optim.verbose = True
    optim.progressBar = True

    acc_test, olabels = optim.loop(model, n_epochs=20)

    print("final accuracy PT-MAP: {:0.2f}±{:0.2f}".format(*(100 * x for x in acc_test)))
    ndatas, olabels, query_data, q_labels = ndatas.cpu(), olabels.cpu(), query_data.cpu(), q_labels.cpu()

    acc_pt, acc_std_pt = semi_pt(ndatas, olabels, query_data,
                                 q_labels)  # ndatas全部数据, olabels全部伪标签, query_data查询集数据, q_labels查询集标签
    print('final accuracy semi_pt: {:0.2f}±{:0.2f}, shots: {}, unlabelled: {}'.format(acc_pt * 100, acc_std_pt * 100,
                                                                                         n_shot, n_unlabelled))
