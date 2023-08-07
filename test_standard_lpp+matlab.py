import time
import torch
import math

use_gpu = torch.cuda.is_available()

# ========================================
#      loading datas
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


def My_CosDist(fea_a, fea_b):
    fea_a = np.array(fea_a)
    fea_b = np.array(fea_b)
    dot_product = np.dot(fea_a, fea_b.T)
    norm_a = np.linalg.norm(fea_a, axis=1)
    norm_b = np.linalg.norm(fea_b, axis=1)
    return 1 - dot_product / (norm_a * norm_b)


def My_constructW(fea, options):
    nSmp = fea.shape[0]
    G = np.zeros((nSmp * (options['k'] + 1), 3))
    selected_fea = fea[0:options['WDim'], :]
    dist = My_EuDist2(selected_fea, fea, 0)
    # dist = My_CosDist(selected_fea, fea)
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


def get_LPP_datas(ndatas, n_lsamples, n_runs):
    ndatas = ndatas.cpu().numpy()
    supportX = ndatas[:, :n_lsamples, :].squeeze()
    queryX = ndatas[:, n_lsamples:, :].squeeze()
    n_sum = ndatas.shape[1]
    options = {'WDim': n_sum, 'NeighborMode': 'KNN', 'WeightMode': 'HeatKernel', 'k': 7, 't': 1, 'ReducedDim': 35,
               'alpha': 10}
    W = np.zeros((n_runs, n_sum, n_sum))
    # 1.无监督k近邻获取数据特征矩阵W(80, 80)  My_constructW、My_EuDist2
    for i in range(n_runs):
        W[i] = My_constructW(np.concatenate((supportX[i], queryX[i])), options)
    ndatas = torch.from_numpy(ndatas)
    ndatas = LPP(ndatas, n_lsamples, options, W)  # 执行降维
    ndatas = ndatas.cuda()
    return ndatas


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

        m_estimates = model.estimateFromMask(self.probas)

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

        # if self.progressBar:
        #     if type(self.progressBar) == bool:
        #         pb = tqdm(total = n_epochs)
        #     else:
        #         pb = self.progressBar

        for epoch in range(1, n_epochs + 1):
            if self.verbose:
                print(f"----- epoch[{epoch:3d}]  lr_p: {self.alpha:.3f}")
            self.performEpoch(model, epochInfo=(epoch, n_epochs))

            # if (self.progressBar): pb.update()

        # get final accuracy and return it
        op_xj = model.getProbas()

        import h5py
        with h5py.File('miniimagenetpro_5_shot.mat', 'r') as f:
            data = np.array(f['probMatrix_all'])
        probMatrix_mat = data
        probMatrix_mat = probMatrix_mat.transpose(2, 1, 0)
        # probMatrix_all为(1000,75,5)的numpy数组，先转换为torch张量
        probMatrix_mat = torch.from_numpy(probMatrix_mat).cuda()
        # 计算op_xj和probMatrix_all在概率维度上的平均值
        average_probs = (op_xj[:, n_lsamples:, :] + probMatrix_mat) / 2
        op_xj[:, n_lsamples:, :] = average_probs

        acc = self.getAccuracy(op_xj)
        return acc


if __name__ == '__main__':
    # ---- data loading
    n_shot = 5
    n_ways = 5
    n_queries = 15
    n_lsamples = n_ways * n_shot
    n_usamples = n_ways * n_queries
    n_samples = n_lsamples + n_usamples

    import FSLTask

    cfg = {'shot': n_shot, 'ways': n_ways, 'queries': n_queries}
    FSLTask.loadDataSet("miniimagenet_both")  # iLPC的预训练模型和PT-MAP重复的是一样的
    FSLTask.setRandomStates(cfg)
    ndatas = FSLTask.GenerateRunSet(cfg=cfg)
    n_runs = FSLTask._maxRuns
    ndatas = ndatas.permute(0, 2, 1, 3).reshape(n_runs, n_samples, -1)
    labels = torch.arange(n_ways).view(1, 1, n_ways).expand(n_runs, n_shot + n_queries, 5).clone().view(n_runs,
                                                                                                        n_samples)

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

    # 保存mat
    # from scipy.io import savemat
    # dataset = r"miniimagenetsf"
    # savemat(f'{dataset}_wrn_{n_shot}shots.mat', mdict={'features': ndatas.cpu().detach().numpy(), 'labels':labels.cpu().detach().numpy()})
    # print(111)

    start_time = time.time()  # 记录开始时间

    # LPP
    ndatas = get_LPP_datas(ndatas, n_lsamples, n_runs)

    n_nfeat = ndatas.size(2)

    # MAP
    lam = 10
    model = GaussianModel(n_ways, lam)
    model.initFromLabelledDatas()

    alpha = 0.2
    # alpha = 0.3
    optim = MAP(alpha)

    optim.verbose = True
    optim.progressBar = True

    acc_test = optim.loop(model, n_epochs=20)

    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算经过的时间
    print(f"函数执行时间为: {elapsed_time} 秒")
    print("final accuracy found {:0.2f} +- {:0.2f}".format(*(100 * x for x in acc_test)))



