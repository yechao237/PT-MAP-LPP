import numpy as np
import torch
import time
import scipy.sparse as sparse
import torch.optim as optim
import torch.nn as nn

from scipy.linalg import eigh
import scipy as sp
from scipy.stats import t
use_gpu = torch.cuda.is_available()

# ========================================
#      loading datas


# 标准化
def My_L2Norm(x):
    y = x / np.tile(np.sqrt(np.sum(x ** 2, axis=1, keepdims=True)).reshape(-1, 1), (1, x.shape[1]))
    return y


# LPP
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


# 欧氏距离
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


# 构建W矩阵
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
            row, col = np.unravel_index(index, (options['WDim'], options['WDim']))  # 将索引i转换为对应的行、列下标，注意在Python中，行、列下标从0开始
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


# 中心化
def centerDatas(datas):
    ''''''
    '''
    对输入的数据进行中心化和归一化处理。
    首先，它从每个样本中减去其均值（中心化）。然后，它通过除以其L2范数（也称为欧几里得长度或二范数）来归一化样本，使得每个样本的长度（或范数）为1。
    这个处理在机器学习中经常使用，可以帮助减小原始特征的尺度对模型的影响，使得模型更容易训练。
    '''
    datas[:, :n_lsamples] = datas[:, :n_lsamples, :] - datas[:, :n_lsamples].mean(1, keepdim=True)
    datas[:, :n_lsamples] = datas[:, :n_lsamples, :] / torch.norm(datas[:, :n_lsamples, :], 2, 2)[:, :, None]
    datas[:, n_lsamples:] = datas[:, n_lsamples:, :] - datas[:, n_lsamples:].mean(1, keepdim=True)
    datas[:, n_lsamples:] = datas[:, n_lsamples:, :] / torch.norm(datas[:, n_lsamples:, :], 2, 2)[:, :, None]
    return datas


# 标准化、单位化
def scaleEachUnitaryDatas(datas):
    norms = datas.norm(dim=2, keepdim=True)
    return datas / norms


# QR分解
def QRreduction(datas):
    ndatas = torch.linalg.qr(datas.permute(0, 2, 1)).R
    ndatas = ndatas.permute(0, 2, 1)
    return ndatas


# 初始化高斯模型
def initGaussianModel(n_ways, lam, n_runs, n_shot, n_queries, n_nfeat, ndatas, list_indices):
    # mus = ndatas.reshape(n_runs, n_shot + n_queries, n_ways, n_nfeat)[:, :n_shot, ].mean(1)
    # 将list_indices中的每个值加上25
    list_indices = [[idx + 25 for idx in sublist] for sublist in list_indices]
    mus = []
    for i in range(n_ways):
        start_idx = i * (n_shot + n_queries)
        end_idx = start_idx + n_shot
        # 选择start_idx:end_idx范围内的样本
        selected_data = ndatas[start_idx:end_idx]
        # 从list_indices中选择对应的索引
        additional_data = ndatas[list_indices[i]]
        # 将这两部分的数据合并
        combined_data = torch.cat([selected_data, additional_data], dim=0)
        # 计算合并后的数据的均值
        mus.append(combined_data.mean(0))

    mus = torch.stack(mus).cuda()
    # mus = ndatas.reshape(n_shot + n_queries, n_ways, n_nfeat)[:n_shot, ].mean(0).cuda()
    model = {"n_ways": n_ways, "mus": mus, "lam": lam}
    return model


# 更新高斯模型
def updateGaussianModel(model, estimate, alpha):
    Dmus = estimate - model["mus"]
    model["mus"] = model["mus"] + alpha * (Dmus)
    return model


# sinkhorn
def compute_optimal_transport(M, r, c, lam, epsilon=1e-6):
    r = r.cuda()
    c = c.cuda()
    n, m = M.shape
    P = torch.exp(- lam * M)
    P /= P.view((-1)).sum().unsqueeze(0).unsqueeze(0)
    u = torch.zeros(n).cuda()
    maxiters = 1000
    iters = 1
    # normalize this matrix
    while torch.max(torch.abs(u - P.sum(1))) > epsilon:
        u = P.sum(1)
        P *= (r / u).view((-1, 1))
        P *= (c / P.sum(0)).view((1, -1))
        if iters == maxiters:
            break
        iters = iters + 1
    return P, torch.sum(P * M)


# 执行LPP，包括数据类型转换等
def LPP(ndatas, n_lsamples, options, W):
    ndatas = ndatas.cpu().numpy()
    supportX = ndatas[:n_lsamples, :].squeeze()
    queryX = ndatas[n_lsamples:, :].squeeze()
    # print(supportX.shape, queryX.shape)
    # print(np.concatenate((supportX, queryX)).shape, W.shape)
    P = My_LPP(np.concatenate((supportX, queryX)), W, options)
    domainS_proj = np.dot(supportX, P)
    domainT_proj = np.dot(queryX, P)
    proj_mean = np.mean(np.concatenate((domainS_proj, domainT_proj)), axis=0)
    domainS_proj = domainS_proj - np.tile(proj_mean, (domainS_proj.shape[0], 1))
    domainT_proj = domainT_proj - np.tile(proj_mean, (domainT_proj.shape[0], 1))
    domainS_proj = My_L2Norm(domainS_proj)
    domainT_proj = My_L2Norm(domainT_proj)
    supportX_2 = domainS_proj
    queryX_2 = domainT_proj
    ndatas = np.concatenate((supportX_2, queryX_2), axis=0)  # 在第一维上进行拼接
    ndatas = torch.from_numpy(ndatas)
    return ndatas

# 获取概率矩阵
def getProbasGaussianModel(model, ndatas, labels, n_lsamples, n_queries, W):
    n_samples, n_nfeat = ndatas.size()
    # compute squared dist to centroids [n_samples][n_ways]
    ndatas = ndatas.cuda()
    model["mus"]= model["mus"].cuda()
    dist = (ndatas.unsqueeze(1)-model["mus"].unsqueeze(0)).norm(dim=2).pow(2)
    p_xj = torch.zeros_like(dist)
    r = torch.ones(n_samples - n_lsamples)
    c = torch.ones(model["n_ways"]) * n_queries
    p_xj_test, sum_weight = compute_optimal_transport(dist[n_lsamples:], r, c, model["lam"], epsilon=1e-6)

    # loss_list 使用sinkhorn计算出的概率矩阵和距离相乘
    # weight_dist = p_xj_test * dist[n_lsamples:]
    # loss_list = [torch.sum(weight_dist[i]).cpu().numpy() / sum_weight.cpu().numpy() for i in range(n_samples - n_lsamples)]

    p_xj[n_lsamples:] = p_xj_test
    p_xj[:n_lsamples].fill_(0)
    p_xj[:n_lsamples].scatter_(1, labels[:n_lsamples].unsqueeze(1), 1)
    return p_xj, W

# 返回全连接层
def weight_imprinting(X, Y, model):
    no_classes = Y.max() + 1
    imprinted = torch.zeros(no_classes, X.shape[1])
    for i in range(no_classes):
        idx = np.where(Y == i)
        tmp = torch.mean(X[idx], dim=0)
        tmp = tmp / tmp.norm(p=2)
        imprinted[i, :] = tmp
    model.weight.data = imprinted
    return model

# 标签去噪，获取损失值
def label_denoising(opt, support, support_ys, query, query_ys_pred):
    all_embeddings = np.concatenate((support, query), axis=0)
    input_size = all_embeddings.shape[1]
    X = torch.tensor(all_embeddings, dtype=torch.float32, requires_grad=True)
    all_ys = np.concatenate((support_ys, query_ys_pred), axis=0)
    Y = torch.tensor(all_ys, dtype=torch.long)
    output_size = support_ys.max() + 1
    start_lr = 0.1
    end_lr = 0.1
    cycle = 50
    step_size_lr = (start_lr - end_lr) / cycle
    # print(input_size, output_size.item())
    lambda1 = lambda x: start_lr - (x % cycle) * step_size_lr
    o2u = nn.Linear(input_size, output_size.item())
    o2u = weight_imprinting(torch.Tensor(all_embeddings[:support_ys.shape[0]]), support_ys, o2u)

    optimizer = optim.SGD(o2u.parameters(), 1, momentum=0.9, weight_decay=5e-4)
    scheduler_lr = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    criterion = nn.CrossEntropyLoss(reduction='none')
    loss_statistics = torch.zeros(all_ys.shape, requires_grad=True)
    lr_progression = []
    for epoch in range(1000):
        output = o2u(X)
        optimizer.zero_grad()
        loss_each = criterion(output, Y)
        loss_each = loss_each  # * weights
        loss_all = torch.mean(loss_each)
        loss_all.backward()
        loss_statistics = loss_statistics + loss_each / (opt['denoising_iterations'])
        optimizer.step()
        scheduler_lr.step()
        lr_progression.append(optimizer.param_groups[0]['lr'])
    return loss_statistics, lr_progression


# 均值向量
def estimateFromMaskGaussianModel(mask, ndatas, select):
    ''''''
    '''
    估计均值向量
    通过样本的归属信息 (mask) 计算每个类别的中心坐标。mask 中的每个元素表示相应的样本是否属于相应的类别。
    这个函数的结果可以用于 updateGaussianModel 函数更新类别的中心坐标。
    '''
    # 使用select选择mask和ndatas的行
    mask = mask[select]
    ndatas = ndatas[select]
    mask = mask.double().cuda()  # 将 mask 转换为 Double 类型
    ndatas = ndatas.double().cuda()  # 将 ndatas 转换为 Double 类型
    # emus = mask.permute(0,2,1).matmul(ndatas).div(mask.sum(dim=1).unsqueeze(2))
    emus = mask.permute(1, 0).matmul(ndatas).div(mask.sum(dim=0).unsqueeze(1))
    return emus

# 获取一个task的准确率
def getAccuracyGaussianModel_labelacc(olabels, labels, acc, iter, verbose2=True):
    matches = labels.eq(olabels).float()  # 与真实标签比较，得到匹配的标签
    acc.append(matches.mean().item())
    if verbose2:
        print("accuracy from filtered probas task ", iter, " ", matches.mean().item())
    return acc

# 打印选择伪标签迭代中选择的伪标签的准确率
def getAccuracyGaussianModel_iteracc(olabels, labels, iter):
    matches = labels.eq(olabels).float()  # 与真实标签比较，得到匹配的标签
    print("accuracy from filtered probas iter ", iter, " ", matches.mean().item())


# 执行一轮高斯模型迭代
def performEpochGaussianModel(model, ndatas_lpp, labels, n_lsamples, n_queries, alpha, W, select):
    p_xj, W = getProbasGaussianModel(model, ndatas_lpp, labels, n_lsamples, n_queries, W)  # 使用原本的查询集数据获取概率矩阵
    emus = estimateFromMaskGaussianModel(p_xj, ndatas_lpp, select)  # 使用更新后的支持集去更新均值
    model = updateGaussianModel(model, emus, alpha)  # 使用均值去更新高斯模型
    return model, p_xj

# 求均值和标准差
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), sp.stats.sem(a)
    h = se * t._ppf((1+confidence)/2., n-1)
    return m, h

# 执行高斯模型迭代和选择伪标签迭代
def loopGaussianModel(ndatas, labels, n_shot, n_ways, n_queries, alpha, n_epochs=20, verbose1=False, verbose2=True):
    num_task, n_sum = ndatas.shape[0], ndatas.shape[1]
    n_lsamples = n_ways * n_shot
    n_usamples = n_ways * n_queries

    # MAP
    lam = 10
    acc = []
    for i in range(num_task):
        idatas = ndatas[i]
        ilabels = labels[i]

        # LPP
        options = {'WDim': n_sum, 'NeighborMode': 'KNN', 'WeightMode': 'HeatKernel', 'k': 7, 't': 1,
                   'ReducedDim': 35,
                   'alpha': 10}

        # iterative label selection
        params = {'best_samples': 3, 'no_samples': np.array(np.repeat(float(n_usamples / n_ways), n_ways)), 'denoising_iterations': 1000}

        # 增加一个迭代标签清理/选择伪标签的功能，更新model["mus"]和n_shot, n_queries, n_lsamples, n_usamples
        iterations = n_usamples
        support_features = idatas[:n_lsamples].cpu().numpy()
        query_features = idatas[n_lsamples:].cpu().numpy()
        support_ys = ilabels[:n_lsamples].cpu().numpy()
        query_ys = ilabels[n_lsamples:].cpu().numpy()
        query_features_real = query_features
        # LPP
        W = My_constructW(idatas, options)
        idatas_lpp = LPP(idatas, n_lsamples, options, W)
        # iterative label selection
        select = np.ones(100, dtype=bool)
        select[25:] = False
        list_indices = [[],[],[],[],[]]
        for j in range(iterations):
            # Gaussian Model 每次选择之后是否需要重新初始化
            n_nfeat = idatas_lpp.size(1)
            model = initGaussianModel(n_ways, lam, n_runs, n_shot, n_queries, n_nfeat, idatas_lpp, list_indices)

            # print(model["mus"].shape, idatas_lpp.shape)
            for epoch in range(1, n_epochs + 1):
                if verbose1:
                    print(f"----- task[{i + 1:3d}], epoch[{epoch:3d}]  lr_p: {alpha:.3f}")
                model, p_xj = performEpochGaussianModel(model, idatas_lpp, ilabels, n_lsamples, n_queries, alpha, W, select)
            query_ys_pred = p_xj[n_lsamples:].argmax(dim=1).cpu().numpy()  # 每次预测全部查询集标签
            # iterative label selection
            # 1 label_denoising
            loss_statistics, _ = label_denoising(params, support_features, support_ys, query_features_real,
                                                 query_ys_pred)
            un_loss_statistics = loss_statistics[support_ys.shape[0]:].detach().numpy()  # np.amax(P, 1)

            # 2 sinkhorn 权重
            # un_loss_statistics = loss_list

            rank = sp.stats.rankdata(un_loss_statistics, method='ordinal')
            indices, list_indices, ys = rank_per_class(support_ys.max() + 1, rank, query_ys_pred, params)
            print(len(indices))
            # 找到使用伪标签的样本
            pseudo_mask = np.in1d(np.arange(query_features_real.shape[0]), indices)
            if len(indices) > 60:
                getAccuracyGaussianModel_iteracc(torch.from_numpy(query_ys_pred[pseudo_mask]).cuda(),
                                                 torch.from_numpy(query_ys[pseudo_mask]).cuda(), j)
                break
            select[-75:] = pseudo_mask
            if verbose2:
                getAccuracyGaussianModel_iteracc(torch.from_numpy(query_ys_pred[pseudo_mask]).cuda(),torch.from_numpy(query_ys[pseudo_mask]).cuda(), j)
        acc = getAccuracyGaussianModel_labelacc(torch.from_numpy(query_ys_pred).cuda(), torch.from_numpy(query_ys).cuda(), acc, i, verbose2=True)
    return mean_confidence_interval(acc)

# 对损失值排序，输出伪标签中与5个class中值最小的3个,共15个
def rank_per_class(no_cls, rank, ys_pred, params):
    list_indices = []
    list_ys = []
    for i in range(no_cls):
        cur_idx = np.where(ys_pred == i)
        y = np.ones((no_cls,)) * i
        class_rank = rank[cur_idx]
        class_rank_sorted = sp.stats.rankdata(class_rank, method='ordinal')
        class_rank_sorted[class_rank_sorted > params['best_samples']] = 0
        indices = np.nonzero(class_rank_sorted)
        list_indices.append(cur_idx[0][indices[0]])
        list_ys.append(y)
    idxs = np.concatenate(list_indices, axis=0)
    ys = np.concatenate(list_ys, axis=0)
    params['best_samples'] += 3
    return idxs, list_indices, ys

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
    FSLTask.loadDataSet("miniimagenet_both")
    FSLTask.setRandomStates(cfg)
    n_runs = FSLTask._maxRuns
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
    # ndatas = ndatas.cuda()
    labels = labels.cuda()

    alpha = 0.2
    # alpha = 0.3

    start_time = time.time()  # 记录开始时间
    acc_mine, acc_std = loopGaussianModel(ndatas, labels, n_shot, n_ways, n_queries, alpha,
                                 n_epochs=20, verbose1=False)
    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算经过的时间
    print(f"函数执行时间为: {elapsed_time} 秒")

    print('final accuracy: {:0.2f} +- {:0.2f}, shots: {}, queries: {}'.format(acc_mine * 100,acc_std * 100, n_shot,n_queries))
