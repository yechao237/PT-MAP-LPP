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
import math
import scipy.sparse as sparse
import torch.nn.functional as F
from numpy import linalg as LA
from tqdm.notebook import tqdm
from scipy.linalg import eigh

use_gpu = torch.cuda.is_available()


# ========================================
#      loading datas


def My_constructW1(label):
    n = len(label)
    W = np.zeros((n, n))
    num_class = int(np.amax(label))
    for i in range(num_class + 1):
        W += (label == i).astype(int).reshape(-1, 1) @ (label == i).astype(int).reshape(1, -1)
    return W


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
    w, v = w[-options['ReducedDim']:], v[:, -options['ReducedDim']:]

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
    selected_fea = fea[0:80, :]
    dist = My_EuDist2(selected_fea, fea, 0)
    nSmpNow = len(selected_fea)
    dump = np.zeros((nSmpNow, options['k'] + 1))
    idx = dump.copy()
    for j in range(options['k'] + 1):
        dump[:, j], idx[:, j] = np.min(dist, axis=1), np.argmin(dist, axis=1)
        temp = idx[:, j] * nSmpNow + np.arange(nSmpNow)  # python按行存储
        temp = temp.astype(int)
        temp = np.clip(temp, 0, dist.size - 1)
        for k in range(80):
            index = temp[k]
            row, col = np.unravel_index(index, (80, 80))  # 将索引i转换为对应的行、列下标，注意在Python中，行、列下标从0开始
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


def centerDatas(datas):
    datas[:, :n_lsamples] = datas[:, :n_lsamples, :] - datas[:, :n_lsamples].mean(1, keepdim=True)
    datas[:, :n_lsamples] = datas[:, :n_lsamples, :] / torch.norm(datas[:, :n_lsamples, :], 2, 2)[:, :, None]
    datas[:, n_lsamples:] = datas[:, n_lsamples:, :] - datas[:, n_lsamples:].mean(1, keepdim=True)
    datas[:, n_lsamples:] = datas[:, n_lsamples:, :] / torch.norm(datas[:, n_lsamples:, :], 2, 2)[:, :, None]
    return datas


def scaleEachUnitaryDatas(datas):
    norms = datas.norm(dim=2, keepdim=True)
    return datas / norms


def QRreduction(datas):
    ndatas = torch.qr(datas.permute(0, 2, 1)).R
    ndatas = ndatas.permute(0, 2, 1)
    return ndatas


def initGaussianModel(n_ways, lam, n_runs, n_shot, n_queries, n_nfeat, ndatas):
    mus = ndatas.reshape(n_runs, n_shot + n_queries, n_ways, n_nfeat)[:, :n_shot, ].mean(1)
    model = {"n_ways": n_ways, "mus": mus, "lam": lam}
    return model


def cloneGaussianModel(model):
    other = {"n_ways": model["n_ways"], "mus": model["mus"].clone(), "lam": model["lam"]}
    return other


def cudaGaussianModel(model):
    model["mus"] = model["mus"].cuda()
    return model


def updateGaussianModel(model, estimate, alpha):
    Dmus = estimate - model["mus"]
    model["mus"] = model["mus"] + alpha * (Dmus)
    return model


def compute_optimal_transport(M, r, c, lam, epsilon=1e-6):
    r = r.cuda()
    c = c.cuda()
    n_runs, n, m = M.shape
    P = torch.exp(- lam * M)
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


def LPP(ndatas, n_lsamples, options, W):
    ndatas = ndatas.cpu().numpy()
    supportX = ndatas[:, :n_lsamples, :].squeeze()
    queryX = ndatas[:, n_lsamples:, :].squeeze()
    n_runs = len(ndatas)
    P = np.zeros((n_runs, ndatas.shape[2], options['ReducedDim']))
    supportX_2 = np.zeros((n_runs, n_lsamples, options['ReducedDim']))
    queryX_2 = np.zeros((n_runs, ndatas.shape[1] - n_lsamples, options['ReducedDim']))
    for i in range(n_runs):
        P[i] = My_LPP(np.concatenate((supportX[i], queryX[i])), W[i], options)
        # domainS_proj和domainT_proj，每轮将domainS_features和domainT_features降至5维
        domainS_proj = np.dot(supportX[i], P[i])
        domainT_proj = np.dot(queryX[i], P[i])
        proj_mean = np.mean(np.concatenate((domainS_proj, domainT_proj)), axis=0)
        domainS_proj = domainS_proj - np.tile(proj_mean, (domainS_proj.shape[0], 1))
        domainT_proj = domainT_proj - np.tile(proj_mean, (domainT_proj.shape[0], 1))
        domainS_proj = My_L2Norm(domainS_proj)
        domainT_proj = My_L2Norm(domainT_proj)
        supportX_2[i] = domainS_proj
        queryX_2[i] = domainT_proj
    ndatas = np.concatenate((supportX_2, queryX_2), axis=1)  # 在第一维上进行拼接
    ndatas = torch.from_numpy(ndatas)
    return ndatas


def construct_labelW1(supportY, p_xj, n_lsamples, n_runs, W):
    supportY = supportY.cpu().numpy()
    olabels = p_xj.argmax(dim=2)
    olabels = olabels.cpu().numpy()
    olabels = olabels[:, n_lsamples:]
    for i in range(n_runs):
        W1[i] = My_constructW1(np.hstack((supportY[i], olabels[i])))
    return W1


def get_W1(W1, supportY, p_xj, n_lsamples,  n_runs, epoch, n_epochs):
    supportY = supportY.cpu().numpy()
    oprob = p_xj.amax(dim=2)
    olabels = p_xj.argmax(dim=2)
    olabels = olabels.cpu().numpy()
    oprob = oprob.cpu().numpy()
    olabels = olabels[:, n_lsamples:]
    oprob = oprob[:, n_lsamples:]
    # 根据prob选择trustable的伪标签，构建标签特征矩阵，进入下一次迭代
    for i in range(n_runs):
        # 根据概率来选择标签，选择1/2, 1/2, 3/4, 1, 1，这部分如果替换，注意将未选择的pseudoLabels，标签设置为-1
        p = 1 - (epoch + 1) / (n_epochs - 1)
        p = max(p, 0)
        p = min(p, 0.2)
        sortedProb = np.sort(oprob[i])
        index = np.argsort(oprob[i])
        sortedPredLabels = olabels[i][index]
        trustable = np.zeros(len(oprob[i]))
        num_class = len(np.unique(supportY[i]))
        for j in range(num_class):
            thisClassProb = sortedProb[sortedPredLabels == j]
            if len(thisClassProb) > 0:
                trustable += (oprob[i] > thisClassProb[np.floor(len(thisClassProb) * p).astype(int)]) * (
                        olabels[i] == j)  # trustable为1则选择
        pseudoLabels = olabels[i].copy()
        pseudoLabels[trustable == 0] = -1  # 标签设置为-1
        W1[i] = My_constructW1(np.hstack((supportY[i], olabels[i])))
    return W1


def getProbasGaussianModel(model, ndatas, labels, n_lsamples, n_queries, W, supportY, options, epoch=0, n_epochs=20):
    # LPP降维
    ndatas = LPP(ndatas, n_lsamples, options, W)

    n_runs, n_samples, n_nfeat = ndatas.size()
    n_ways = model["n_ways"]
    # compute squared dist to centroids [n_runs][n_samples][n_ways]
    ndatas = ndatas.cuda()
    model["mus"] = model["mus"].cuda()
    dist = (ndatas.unsqueeze(2)-model["mus"].unsqueeze(1)).norm(dim=3).pow(2)
    p_xj = torch.zeros_like(dist)
    r = torch.ones(n_runs, n_samples - n_lsamples)
    c = torch.ones(n_runs, n_ways) * n_queries

    p_xj_test, _ = compute_optimal_transport(dist[:, n_lsamples:], r, c, model["lam"], epsilon=1e-6)
    p_xj[:, n_lsamples:] = p_xj_test

    p_xj[:, :n_lsamples].fill_(0)
    p_xj[:, :n_lsamples].scatter_(2, labels[:, :n_lsamples].unsqueeze(2), 1)

    # 构建标签特征矩阵
    # W1 = get_W1(W1, supportY, p_xj, n_lsamples, n_runs, epoch, n_epochs)
    # 每轮都对全部数据建立标签矩阵,但不更新
    W1 = construct_labelW1(supportY, p_xj, n_lsamples, n_runs, W)

    # 每轮输出准确率
    if epoch != 0 :
        acc_test = getAccuracyGaussianModel(p_xj, labels, n_lsamples, n_queries)
        print(f"accuracy in epoch{epoch}: {acc_test[0] * 100:.2f} +- {acc_test[1] * 100:.2f}")

    return p_xj, W, W1


def estimateFromMaskGaussianModel(model, mask, ndatas):
    mask = mask.double()  # 将 mask 转换为 Double 类型
    ndatas = ndatas.double()  # 将 ndatas 转换为 Double 类型
    mask = mask.cuda()
    ndatas = ndatas.cuda()
    emus = mask.permute(0,2,1).matmul(ndatas).div(mask.sum(dim=1).unsqueeze(2))
    return emus


def getAccuracyGaussianModel(probas, labels, n_lsamples, n_queries):
    olabels = probas.argmax(dim=2)
    matches = labels.eq(olabels).float()
    acc_test = matches[:,n_lsamples:].mean(1)
    m = acc_test.mean().item()
    pm = acc_test.std().item() *1.96 / math.sqrt(n_runs)
    return m, pm


def performEpochGaussianModel(model, ndatas, labels, n_lsamples, n_queries, alpha, W, epoch=0):
    p_xj, W, W1 = getProbasGaussianModel(model, ndatas, labels, n_lsamples, n_queries, W, supportY, options, epoch)
    ndatas_2 = LPP(ndatas, n_lsamples, options, W)
    emus = estimateFromMaskGaussianModel(model, p_xj, ndatas_2)
    model = updateGaussianModel(model, emus, alpha)
    return model, p_xj, W, W1


def loopGaussianModel(model, ndatas, labels, n_lsamples, n_queries, alpha, W, W1, supportY, options,
                      n_epochs=20, verbose=False, progressBar=False):

    for epoch in range(1, n_epochs+1):
        if verbose:
            print(f"----- epoch[{epoch:3d}]  lr_p: {alpha:.3f}")
        model, p_xj, W, W1 = performEpochGaussianModel(model, ndatas, labels, n_lsamples, n_queries, alpha, W, epoch)

    # 计算数组中0的数量
    num_zeros1 = W.size - np.count_nonzero(W)
    print("Number of zeros in the array1:", num_zeros1)

    # 标签特征矩阵
    W = W1 * 1 * W

    # 计算数组中0的数量
    num_zeros2 = W.size - np.count_nonzero(W)
    print("Number of zeros in the array2:", num_zeros2)

    for epoch in range(1, n_epochs+1):
        if verbose:
            print(f"----- epoch[{epoch:3d}]  lr_p: {alpha:.3f}")
        model, p_xj, W, W1 = performEpochGaussianModel(model, ndatas, labels, n_lsamples, n_queries, alpha, W, epoch)

    # get final accuracy and return it
    p_xj, W, W1 = getProbasGaussianModel(model, ndatas, labels, n_lsamples, n_queries, W, supportY, options)
    acc_test = getAccuracyGaussianModel(p_xj, labels, n_lsamples, n_queries)
    return acc_test


if __name__ == '__main__':
    # ---- data loading
    n_shot = 1
    n_ways = 5
    n_queries = 15
    n_runs = 100  # 原本为10000
    n_lsamples = n_ways * n_shot
    n_usamples = n_ways * n_queries
    n_samples = n_lsamples + n_usamples

    import FSLTask

    cfg = {'shot': n_shot, 'ways': n_ways, 'queries': n_queries}
    FSLTask.loadDataSet("miniimagenet")
    FSLTask.setRandomStates(cfg)
    ndatas = FSLTask.GenerateRunSet(cfg=cfg)
    ndatas = ndatas.permute(0, 2, 1, 3).reshape(n_runs, n_samples, -1)
    labels = torch.arange(n_ways).view(1, 1, n_ways).expand(n_runs, n_shot + n_queries, 5).clone().view(n_runs,
                                                                                                        n_samples)
    # Power transform
    beta = 0.5
    ndatas[:, ] = torch.pow(ndatas[:, ] + 1e-6, beta)

    ndatas = QRreduction(ndatas)
    n_nfeat = ndatas.size(2)

    ndatas = scaleEachUnitaryDatas(ndatas)
    # trans-mean-sub
    ndatas = centerDatas(ndatas)

    print("size of the datas...", ndatas.size())

    # switch to cuda
    ndatas = ndatas.cuda()
    labels = labels.cuda()


    # LPP
    ndatas = ndatas.cpu().numpy()
    supportX = ndatas[:, :n_lsamples, :].squeeze()
    supportY = labels[:, :n_lsamples].squeeze()
    queryX = ndatas[:, n_lsamples:, :].squeeze()
    options = {'NeighborMode': 'KNN', 'WeightMode': 'HeatKernel', 'k': 7, 't': 1, 'ReducedDim': 35, 'alpha': 10}
    n_runs = len(ndatas)
    n_sum = ndatas.shape[1]
    W = np.zeros((n_runs, n_sum, n_sum))  # 无监督
    W1 = np.zeros((n_runs, n_sum, n_sum))  # 有监督
    # 1.无监督k近邻获取数据特征矩阵W(80, 80)  My_constructW、My_EuDist2
    for i in range(n_runs):
        W[i] = My_constructW(np.concatenate((supportX[i], queryX[i])), options)
    ndatas = torch.from_numpy(ndatas)
    ndatas_2 = LPP(ndatas, n_lsamples, options, W)
    n_nfeat = ndatas_2.size(2)
    ndatas = ndatas.cuda()


    # MAP
    lam = 10
    model = {"n_ways": n_ways, "mus": None, "lam": lam}
    model["mus"] = ndatas_2.reshape(n_runs, n_shot + n_queries, n_ways, n_nfeat)[:, :n_shot, ].mean(1)

    # alpha = 0.2
    alpha = 0.3
    acc_test = loopGaussianModel(model, ndatas, labels, n_lsamples, n_queries, alpha, W, W1, supportY, options,
                                 n_epochs=10, verbose=True, progressBar=True)

    print(f"final accuracy found: {acc_test[0] * 100:.2f} +- {acc_test[1] * 100:.2f}")
