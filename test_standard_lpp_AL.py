import time
import torch
import math
import numpy as np
import scipy.sparse as sparse

from scipy.linalg import eigh
from scipy.io import savemat
torch.set_default_dtype(torch.float64)

use_gpu = torch.cuda.is_available()


# ========================================
#      loading datas

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
    P = np.zeros((n_runs, ndatas.shape[2], options['d']))
    ndatas_2 = np.zeros((n_runs, ndatas.shape[1], options['d']))

    for i in range(n_runs):
        P[i] = My_LPP(ndatas[i], W[i], options)
        ndatas_proj = np.dot(ndatas[i], P[i])  # formula 8
        proj_mean = np.mean(ndatas_proj, axis=0)
        ndatas_proj = ndatas_proj - np.tile(proj_mean, (ndatas_proj.shape[0], 1))
        ndatas_proj = My_L2Norm(ndatas_proj)
        ndatas_2[i] = ndatas_proj

    ndatas = np.array(ndatas_2)
    ndatas = torch.from_numpy(ndatas)
    return ndatas, P


def get_LPP_datas(ndatas):
    options = {'WDim': ndatas.shape[1], 'NeighborMode': 'KNN', 'WeightMode': 'HeatKernel', 'k': 7, 't': 1, 'd': 35,
               'alpha': 10}
    n_runs = ndatas.shape[0]
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
    def __init__(self, n_ways, lam, ndatas, labels, active_epoch=False):
        super(GaussianModel, self).__init__(n_ways)
        self.mus = None  # shape [n_runs][n_ways][n_nfeat]
        self.lam = lam
        self.ndatas = ndatas
        self.labels = labels
        self.active_epoch = active_epoch

    def initFromLabelledDatas(self, mus=None):
        if mus == None:
            self.mus = self.ndatas.reshape(n_runs, n_shot + n_queries, n_ways, n_nfeat)[:, :n_shot, ].mean(1)
        else:
            self.mus = mus

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
        dist = (self.ndatas.unsqueeze(2) - self.mus.unsqueeze(1)).norm(dim=3).pow(2)
        r = torch.ones(n_runs, n_usamples)
        c = torch.ones(n_runs, n_ways) * n_queries
        p_xj = torch.zeros_like(dist)
        p_xj_test, _ = self.compute_optimal_transport(dist[:, n_lsamples:], r, c, epsilon=1e-6)
        p_xj[:, n_lsamples:] = p_xj_test
        if self.active_epoch == False:
            p_xj[:, :n_lsamples].fill_(0)
            p_xj[:, :n_lsamples].scatter_(2, self.labels[:, :n_lsamples].unsqueeze(2), 1)
        return p_xj

    def estimateFromMask(self, mask):
        emus = mask.permute(0, 2, 1).matmul(self.ndatas).div(mask.sum(dim=1).unsqueeze(2))
        return emus


# =========================================
#    MAP
# =========================================

class MAP:
    def __init__(self, labels, alpha=None, active=False):
        self.verbose = False
        self.progressBar = False
        self.alpha = alpha
        self.labels = labels
        self.active = active

    def getAccuracy(self, probas):
        olabels = probas.argmax(dim=2)
        matches = self.labels.eq(olabels).float()
        acc_test = matches[:, n_lsamples:].mean(1)
        m = acc_test.mean().item()
        pm = acc_test.std().item() * 1.96 / math.sqrt(n_runs)
        return m, pm

    def performEpoch(self, model):
        p_xj = model.getProbas()
        self.probas = p_xj
        # if self.verbose:
        #     print("accuracy from filtered probas", self.getAccuracy(self.probas))
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
        for epoch in range(1, n_epochs + 1):
            if self.verbose:
                print(f"----- epoch[{epoch:3d}]  lr_p: {self.alpha:.3f}")
            self.performEpoch(model)
            # if (self.progressBar): pb.update()
        # get final accuracy and return it
        if self.active == False:
            op_xj = model.getProbas()
            acc = self.getAccuracy(op_xj)
            return acc
        else:
            op_xj = model.getProbas()
            acc = self.getAccuracy(op_xj)
            olabels = op_xj.argmax(dim=2)
            return acc, olabels, op_xj


def LPP_from_P(query_data, P):
    options = {'WDim': query_data.shape[1], 'NeighborMode': 'KNN', 'WeightMode': 'HeatKernel', 'k': 7, 't': 1,
               'd': 35, 'alpha': 10}
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


def PT(ndatas, beta):
    nve_idx = np.where(ndatas.cpu().detach().numpy() < 0)
    ndatas[nve_idx] *= -1
    ndatas[:, ] = torch.pow(ndatas[:, ] + 1e-6, beta)
    ndatas[nve_idx] *= -1  # return the sign
    return ndatas

def data_preprocessing(ndatas, labels):
    beta = 0.5
    ndatas = PT(ndatas, beta)
    ndatas = QRreduction(ndatas)
    ndatas = scaleEachUnitaryDatas(ndatas)
    ndatas = centerDatas(ndatas)  # trans-mean-sub

    # switch to cuda
    ndatas = ndatas.cuda()  # fsl ndatas
    labels = labels.cuda()

    # LPP
    ndatas, _ = get_LPP_datas(ndatas)
    return ndatas, labels


def Gasussianloop(n_shot, n_queries, n_ways, ndatas, labels, active_epoch=False, active=False, mus=None):

    lam = 10
    alpha = 0.3 if n_shot == 1 else 0.2

    model = GaussianModel(n_ways, lam, ndatas, labels, active_epoch)
    model.initFromLabelledDatas(mus)

    optim = MAP(labels, alpha, active)
    optim.verbose = False
    optim.progressBar = True
    if active:
        acc_test, prelabels, prob_active = optim.loop(model, n_epochs=20)
        return acc_test, prelabels, prob_active
    acc_test = optim.loop(model, n_epochs=20)
    return acc_test

def get_entropyies(prob_active):
    prob_active = prob_active / prob_active.sum(dim=2, keepdim=True)  # 确保每个样本的概率和为1
    all_entropies = []  # 初始化一个用于存储所有任务熵值的列表
    for i in range(n_runs):  # 遍历每个任务
        current_task_probs = prob_active[i]  # 获取当前任务的概率分布
        dist = torch.distributions.Categorical(probs=current_task_probs)  # 使用概率创建分类分布
        entropies = dist.entropy()  # 计算并存储当前任务的所有样本的熵
        all_entropies.append(entropies)
    all_entropies_tensor = torch.stack(all_entropies)   # 将列表转换为torch张量
    return all_entropies_tensor


def reorder(task_indices, num=4):
    new_order_indices = []  # 初始化新的索引列表
    num_categories = 5  # 有5个类别
    elements_per_category = num  # 每个类别选择num个元素
    for i in range(elements_per_category):  # 按照新的顺序收集索引
        for j in range(num_categories):
            index = j * elements_per_category + i  # 计算当前元素在原始列表中的位置
            new_order_indices.append(task_indices[index])  # 添加到新列表中
    return new_order_indices


def check_plabels(top_labels, num=4):
    compare_labels = [0, 1, 2, 3, 4] * num
    compare_top_labels = list(top_labels)
    if compare_top_labels != compare_labels:
        print("False label is exist!")


def selectdata(entropies, ndatas, prelabels, nlabels, n_ways, num, verbose=False, select=1):
    # 初始化存储所选数据和标签的容器
    selected_data = []
    selected_labels = []
    n_runs = ndatas.size(0)
    # 遍历每个任务
    for task_index in range(n_runs):
        # 获取当前任务的熵、数据、伪标签和真实标签
        task_entropies = entropies[task_index]
        task_data = ndatas[task_index]
        task_prelabels = prelabels[task_index]
        task_labels = nlabels[task_index]
        # 初始化存储当前任务所选索引的列表
        task_indices = []
        # 遍历每个类别
        for label in range(n_ways):
            # 找到当前类别的所有样本的索引
            label_indices = (task_prelabels == label).nonzero(as_tuple=True)[0]
            if len(label_indices) < 4:  # 如果当前类别没有样本，输出，继续下一个类别
                print("less than 4！")
                continue
            label_entropies = task_entropies[label_indices]  # 获取当前类别样本的熵

            # idea1 select by entropy level
            if select == 1:
                threshold = torch.quantile(label_entropies, 0.5)  # 计算20%的熵阈值, 这里用量化值来表示20%的阈值
                qualified_indices = torch.where(label_entropies >= threshold)[0] # 获取不小于20%熵阈值的样本
                # 如果满足条件的样本不足num个，可以选择处理的逻辑，比如继续、警告或者使用所有满足条件的样本
                if len(qualified_indices) < num:
                    print(
                        f"Warning: Not enough samples above the threshold for label {label}. Needed {num}, got {len(qualified_indices)}.")
                    continue
                # 对合格样本的熵进行排序并获取熵最小的num个样本的索引
                _, top_label_indices = torch.topk(label_entropies[qualified_indices], num, largest=False)
                selected_indices = qualified_indices[top_label_indices]  # 从合格的索引中选择top标签索引
                task_indices.extend(label_indices[selected_indices].tolist())  # 将这些索引添加到当前任务的索引列表中
            else:
                # 计算熵的中位数（50%分位数）
                median_entropy = torch.median(label_entropies)
                # 计算每个样本熵与中位数的差的绝对值
                diff_from_median = torch.abs(label_entropies - median_entropy)
                # 对差值进行排序，但同时获取原始熵值的索引（为了在相同差值情况下选择熵值较低的样本）
                sorted_diffs, sorted_indices = torch.sort(diff_from_median)
                # 选择差异最小的num个样本，但如果差异相同，则选择熵值较低的样本
                selected_for_label = []
                for i in range(num):
                    if i >= len(sorted_indices):  # 防止索引越界
                        break
                    # 选择当前最接近中位数的样本
                    current_closest_idx = sorted_indices[i]
                    selected_for_label.append(label_indices[current_closest_idx].item())
                task_indices.extend(selected_for_label)

        task_indices = reorder(task_indices, num=num)  # reorder

        # _, task_indices = torch.topk(entropies, 20, largest=True)  # largest=False最小
        # task_indices = [0, 31, 22, 23, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

        top_data = task_data[task_indices]
        top_labels = task_labels[task_indices]
        if verbose == True:
            check_plabels(top_labels, num=num)
        selected_data.append(top_data)
        selected_labels.append(top_labels)
    selected_data = torch.stack(selected_data).cuda()
    selected_labels = torch.stack(selected_labels).cuda()
    return selected_data, selected_labels


def get_mus(active_ndatas, active_nlabels, n_lsamples):
    support_datas = active_ndatas[:, :n_lsamples, :]
    support_labels = active_nlabels[:, :n_lsamples]
    mus = torch.zeros((n_runs, n_ways, n_nfeat)).cuda()  # 初始化一个张量来存储计算出的均值
    for task_index in range(n_runs):  # 遍历每个任务
        task_data = support_datas[task_index]  # 当前任务的数据
        task_labels = support_labels[task_index]  # 当前任务的标签
        for class_index in range(n_ways):  # 遍历每个类别
            class_indices = (task_labels == class_index).nonzero(as_tuple=True)[0]  # 找到当前类别的所有样本
            class_data = task_data[class_indices]  # 选择对应的数据
            mus[task_index, class_index] = class_data.mean(0)  # 计算当前类别的均值，并将其存储在mus张量中
    return mus

if __name__ == '__main__':
    # ---- data loading
    num = 4  # active learning选择的样本个数
    n_shot = 5
    n_ways = 5
    n_queries = 15
    n_unlabelled = 100
    n_lsamples = n_ways * n_shot  # 25个已经标记的支持集(全部用于fsl，第一个用于afsl)
    n_usamples = n_ways * n_queries  # 75个查询集
    active_samples = n_ways * n_unlabelled  # 150个未标记的支持集
    n_samples = n_lsamples + n_usamples + active_samples  # 250个样本
    fsl_train_samples = n_lsamples + n_usamples  # 小样本训练25+75个

    import FSLTask
    cfg = {'shot': n_shot, 'ways': n_ways, 'queries': n_queries + n_unlabelled}  # 5-shot 5-way 45查询集
    dataset = r"cifar"
    FSLTask.loadDataSet(dataset)
    FSLTask.setRandomStates(cfg)
    n_runs = FSLTask._maxRuns
    all_ndatas = FSLTask.GenerateRunSet(cfg=cfg)
    all_ndatas = all_ndatas.permute(0, 2, 1, 3).reshape(n_runs, n_samples, -1)
    labels = torch.arange(n_ways).view(1, 1, n_ways).expand(n_runs, n_shot + n_queries + n_unlabelled, 5).clone().view(n_runs,
                                                                                                        n_samples)

    # partition for afsl learning
    ndatas, active_data = all_ndatas[:, :fsl_train_samples, :], all_ndatas[:, fsl_train_samples:, :]  # fsl训练的数据和主动学习额外数据
    labels, active_label = labels[:, :fsl_train_samples], labels[:, fsl_train_samples:]  # 训练的标签和主动学习标签
    query_datas = ndatas[:, n_lsamples:, :].cuda()
    query_labels = labels[:, n_lsamples:].cuda()
    active_ndatas = torch.cat((ndatas[:, :((n_shot - num) * n_ways), :], active_data), dim=1)  # afsl的初始数据
    active_nlabels = torch.cat((labels[:, :((n_shot - num) * n_ways)], active_label), dim=1)  # afsl的初始标签
    print(ndatas.shape, active_ndatas.shape, active_data.shape, labels.shape, active_nlabels.shape, active_label.shape)

    # step1: fsl
    # Power transform
    ndatas, labels = data_preprocessing(ndatas, labels)
    n_nfeat = ndatas.size(2)

    # MAP
    acc_test = Gasussianloop(n_shot, n_queries, n_ways, ndatas, labels)
    print("fsl final accuracy with 15 queries: {:0.2f}±{:0.2f}".format(*(100 * x for x in acc_test)))

    # step2: afsl
    active_ndatas_fsl, active_nlabels = data_preprocessing(active_ndatas, active_nlabels)
    active_ndatas = active_ndatas.cuda()
    # step2: afsl  for every task, choose 20 samples, and return active datas and labels
    # step2.1 use fsl 5-way-1-shot experiment to get active prob
    n_shot = 5 - num
    temp = n_queries
    n_queries = n_unlabelled
    n_lsamples = n_shot * n_ways
    n_usamples = n_queries * n_ways

    acc_test, prelabels, prob_active = Gasussianloop(n_shot, n_queries, n_ways, active_ndatas_fsl, active_nlabels, active_epoch=True, active=True)
    print("afsl accuracy1 with 45 queries: {:0.2f}±{:0.2f}".format(*(100 * x for x in acc_test)))

    prob_active = prob_active[:, n_lsamples:, :]
    prelabels = prelabels[:, n_lsamples:]
    print(prob_active.shape, prelabels.shape)

    # step2.2 for every task, use entrophy to get 20 samples
    # prob_active = torch.rand(n_runs, active_samples, n_ways)
    entropies = get_entropyies(prob_active)
    selected_data, selected_labels = selectdata(entropies, active_data, prelabels, active_label, n_ways, num, verbose=True, select=2)

    # step2.3 update afsl data
    d_split1, d_split2 = torch.split(active_ndatas, [n_lsamples, active_samples], dim=1)
    l_split1, l_split2 = torch.split(active_nlabels, [n_lsamples, active_samples], dim=1)
    support_datas = torch.cat([d_split1, selected_data], dim=1)
    support_labels = torch.cat([l_split1, selected_labels], dim=1)

    active_ndatas = torch.cat([support_datas, query_datas], dim=1).cpu()
    active_nlabels = torch.cat([support_labels, query_labels], dim=1).cpu()

    # step3 afsl
    n_shot = 5
    n_queries = temp
    n_lsamples = n_ways * n_shot
    n_usamples = n_queries * n_ways

    active_ndatas, active_nlabels = data_preprocessing(active_ndatas, active_nlabels)  # 数据预处理
    start_time = time.time()  # 记录开始时间
    mus = get_mus(active_ndatas, active_nlabels, n_lsamples)  # 获取支持集均值

    # 使用1-shot
    # active_ndatas = torch.cat([mus, active_ndatas[:, n_lsamples:, :]], dim=1)
    # active_nlabels = labels[:, 20:]
    # n_shot = 1
    # n_lsamples = n_ways * n_shot
    acc_test = Gasussianloop(n_shot, n_queries, n_ways, active_ndatas, active_nlabels, mus=mus)
    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算经过的时间
    print(f"函数执行时间为: {elapsed_time} 秒")
    print("afsl accuracy2: {:0.2f}±{:0.2f}".format(*(100 * x for x in acc_test)))
