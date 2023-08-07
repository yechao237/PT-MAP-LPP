import torch
import math
from tqdm.notebook import tqdm
import os

os.environ['OMP_NUM_THREADS'] = '1'

use_gpu = torch.cuda.is_available()

import numpy as np
import scipy.sparse as sparse
from scipy.linalg import eigh
from sklearn.preprocessing import normalize


def My_L2Norm(x):
    y = x / np.tile(np.sqrt(np.sum(x ** 2, axis=1, keepdims=True)).reshape(-1, 1), (1, x.shape[1]))
    return y


def My_LPP(data, W, options):
    data = np.array(data, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    D = np.diag(np.sum(W, axis=1))  # formula 3
    L = D - W  # formula 4
    D = D.astype(np.float64)
    Sl = data.T @ L @ data  # formula 5
    Sd = data.T @ D @ data  # formula 6

    Sl = (Sl + Sl.T) / 2
    Sd = (Sd + Sd.T) / 2
    Sl = Sl + options["alpha"] * np.eye(Sl.shape[1])
    # 计算特征值和特征向量
    w, v = eigh(Sd, Sl)  # formula 7
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

    D = np.add(aa, bb.T) - 2 * ab  # formula 1
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
    dump = np.exp(-dump / (2 * options['t'] ** 2))  # formula 2
    G[:, 0] = np.tile(np.arange(nSmp), options['k'] + 1)
    G[:, 1] = np.ravel(idx, order='F').flatten()
    G[:, 2] = np.ravel(dump, order='F').flatten()
    W = sparse.csr_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(nSmp, nSmp))
    W = W.toarray()
    W = W - np.diag(np.diag(W))
    W = np.maximum(W, W.transpose())
    return W


def LPP(ndatas, options, W):
    ndatas = ndatas.cpu().numpy()
    n_runs = len(ndatas)
    P = np.zeros((n_runs, ndatas.shape[2], options['ReducedDim']))
    ndatas_2 = np.zeros((n_runs, ndatas.shape[1], options['ReducedDim']))

    for i in range(n_runs):
        P[i] = My_LPP(ndatas[i], W[i], options)
        ndatas_proj = np.dot(ndatas[i], P[i])  # formula 8
        proj_mean = np.mean(ndatas_proj, axis=0)
        ndatas_proj = ndatas_proj - np.tile(proj_mean, (ndatas_proj.shape[0], 1))
        ndatas_proj = My_L2Norm(ndatas_proj)
        ndatas_2[i] = ndatas_proj

    ndatas = np.array(ndatas_2)
    ndatas = torch.from_numpy(ndatas)
    return ndatas


def get_LPP_datas(ndatas, options):
    n_runs = ndatas.shape[0]
    ndatas = ndatas.cpu().numpy()  # 全部数据
    n_sum = ndatas.shape[1]  # 一个task中的数据个数
    W = np.zeros((n_runs, n_sum, n_sum))
    # 1.无监督k近邻获取数据特征矩阵W(80, 80)  My_constructW、My_EuDist2
    for i in range(n_runs):
        # W[i] = My_constructW(np.concatenate((supportX[i], queryX[i])), options)
        W[i] = My_constructW(ndatas[i, :, :].squeeze(), options)
    ndatas = torch.from_numpy(ndatas)
    ndatas = LPP(ndatas, options, W)  # 执行降维
    ndatas = ndatas.cuda()
    return ndatas


# ========================================
#      loading datas


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
        self.mus = ndatas.reshape(n_runs, n_shot + n_queries, n_ways, n_nfeat)[:, :n_shot, ].mean(1)

    def updateFromEstimate(self, estimate, alpha):

        Dmus = estimate - self.mus
        self.mus = self.mus + alpha * (Dmus)  # formula 10

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
        olabels = probas.argmax(dim=2)
        matches = labels.eq(olabels).float()
        acc_test = matches[:, n_lsamples:].mean(1)

        m = acc_test.mean().item()
        pm = acc_test.std().item() * 1.96 / math.sqrt(n_runs)
        return m, pm

    def performEpoch(self, model, epochInfo=None):

        p_xj = model.getProbas()
        self.probas = p_xj

        if self.verbose:
            print("accuracy from filtered probas", self.getAccuracy(self.probas))

        m_estimates = model.estimateFromMask(self.probas)  # 期望值

        # update centroids
        model.updateFromEstimate(m_estimates, self.alpha)

        if self.verbose:
            op_xj = model.getProbas()
            acc = self.getAccuracy(op_xj)
            print("output model accuracy", acc)

    def loop(self, model, n_epochs=20):

        self.probas = model.getProbas()
        if self.verbose:
            print("initialisation model accuracy", self.getAccuracy(self.probas))

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
        acc = self.getAccuracy(op_xj)
        return acc


def dim_reduce(params, ndatas, n_lsamples=None, labels=None):
    ndatas = ndatas.cpu().numpy()
    X = normalize(ndatas)
    n_features = X.shape[1]
    if params["type"] == 'surpervised':
        labels = labels.cpu().numpy()
        n_classes = len(np.unique(labels))
        X_support = X[:n_lsamples, :]
    else:
        n_classes = None
        X_support = None
    if params["type"] == 'unsurpervised':
        if params["reduce"] == 'isomap':
            from sklearn.manifold import Isomap
            embed = Isomap(n_components=params["d"], n_neighbors=10)
        elif params["reduce"] == 'itsa':
            from sklearn.manifold import LocallyLinearEmbedding
            embed = LocallyLinearEmbedding(n_components=params["d"], n_neighbors=10, method='ltsa')
        elif params["reduce"] == 'lle':
            from sklearn.manifold import LocallyLinearEmbedding
            embed = LocallyLinearEmbedding(n_components=params["d"], n_neighbors=10, eigen_solver='dense')
        elif params["reduce"] == 'se':
            from sklearn.manifold import SpectralEmbedding
            embed = SpectralEmbedding(n_components=params["d"])
        elif params["reduce"] == 'pca':
            from sklearn.decomposition import PCA
            embed = PCA(n_components=params["d"])
        else:
            embed = None
            print("This is not a suitable unsurpervised dimensionality reduction algorithm!")
        if params["reduce"] == 'none':
            X = X
        else:
            X = embed.fit_transform(X)
    elif params["type"] == 'surpervised':
        if params["reduce"] == 'lda':
            if params["d"] > min(n_features, n_classes - 1):
                raise ValueError("n_components cannot be larger than min(n_features, n_classes - 1) for LDA.")
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            embed = LinearDiscriminantAnalysis(n_components=params["d"])
        elif params["reduce"] == 'pls':
            from sklearn.cross_decomposition import PLSRegression
            embed = PLSRegression(n_components=params["d"])
        elif params["reduce"] == 'cca':
            from sklearn.cross_decomposition import CCA
            from sklearn.preprocessing import OneHotEncoder
            y_support = OneHotEncoder().fit_transform(labels.reshape(-1, 1)).toarray()
            embed = CCA(n_components=params["d"])
            X_support_transformed, _ = embed.fit_transform(X_support, y_support)
            X_query_transformed = embed.transform(X[n_lsamples:, :])
            X = np.vstack([X_support_transformed, X_query_transformed])
        elif params["reduce"] == 'slle':
            from sklearn.manifold import LocallyLinearEmbedding
            embed = LocallyLinearEmbedding(n_components=params["d"], n_neighbors=params["d"], method='standard',
                                           reg=0.001)
        elif params["reduce"] == 'lmnn':
            from metric_learn import LMNN
            embed = LMNN(n_components=params["d"], k=5)
        else:
            embed = None
            print("This is not a suitable surpervised dimensionality reduction algorithm!")
        if params["reduce"] == 'none':
            X = X
        else:
            if params["reduce"] not in ['cca']:
                embed.fit(X_support, labels)
                X = embed.transform(X)
    else:
        print("This is not a suitable dimensionality reduction type!")
    return torch.from_numpy(X).cuda()


if __name__ == '__main__':
    # ---- data loading
    n_shot = 5
    n_ways = 5
    n_queries = 15
    n_lsamples = n_ways * n_shot
    n_usamples = n_ways * n_queries
    n_samples = n_lsamples + n_usamples

    import FSLTask

    n_runs = FSLTask._maxRuns
    cfg = {'shot': n_shot, 'ways': n_ways, 'queries': n_queries}
    FSLTask.loadDataSet("miniImagenet_both")
    FSLTask.setRandomStates(cfg)
    ndatas = FSLTask.GenerateRunSet(cfg=cfg)
    ndatas = ndatas.permute(0, 2, 1, 3).reshape(n_runs, n_samples, -1)
    labels = torch.arange(n_ways).view(1, 1, n_ways).expand(n_runs, n_shot + n_queries, 5).clone().view(n_runs,
                                                                                                        n_samples)

    # Power transform
    beta = 0.5
    ndatas[:, ] = torch.pow(ndatas[:, ] + 1e-6, beta)  # formula 9

    ndatas = QRreduction(ndatas)

    ndatas = scaleEachUnitaryDatas(ndatas)

    # trans-mean-sub

    ndatas = centerDatas(ndatas)

    print("size of the datas...", ndatas.size())

    # switch to cuda
    ndatas = ndatas.cuda()
    labels = labels.cuda()

    options = {'WDim': ndatas.shape[1], 'NeighborMode': 'KNN', 'WeightMode': 'HeatKernel', 'k': 7, 't': 1,
               'ReducedDim': 35,
               'alpha': 10}

    # LPP
    ndatas_1 = get_LPP_datas(ndatas, options)
    print(ndatas_1.shape)

    # Other reduce dim methods

    # "type": surpervised unsurpervised
    # "unsurpervised"-"d": isomap* 5  itsa 6   lle* 5   se* 4   pca* 5
    # "surpervised"-"d": lda 4   pls 1-shot:4 5-shot:5   cca 4
    # "surpervised"-"d": slle* 1-shot:4 5-shot:24   lmnn* 70(1-shot:k=1 5-shot:k=5)  # lmnn较慢

    options_2 = {'type': 'surpervised', 'reduce': 'lmnn', 'd': 70}
    ndatas_2 = torch.zeros(n_runs, n_samples, options_2["d"]).cuda()
    support_labels = labels[:, :n_lsamples]
    for i in range(n_runs):
        if options_2["type"] == 'unsurpervised':
            ndatas_2[i] = dim_reduce(options_2, ndatas[i])
        elif options_2["type"] == 'surpervised':
            ndatas_2[i] = dim_reduce(options_2, ndatas[i], n_lsamples, support_labels[i])
    print(ndatas_2.shape)
    ndatas = torch.cat((ndatas_1, ndatas_2), dim=2)

    # ndatas = ndatas_1
    n_nfeat = ndatas.size(2)

    # MAP
    lam = 10
    model = GaussianModel(n_ways, lam)
    model.initFromLabelledDatas()

    alpha = 0.2
    # alpha = 0.3

    optim = MAP(alpha)

    optim.verbose = False
    optim.progressBar = True

    acc_test = optim.loop(model, n_epochs=20)

    print("final accuracy found {:0.2f} +- {:0.2f}".format(*(100 * x for x in acc_test)))



