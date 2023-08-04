import time
import torch
import math
import scipy as sp
import numpy as np
import scipy.sparse as sparse

from scipy.linalg import eigh
from scipy.stats import t
from tqdm.notebook import tqdm

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


# 欧氏距离
def My_CosDist(fea_a, fea_b):
    fea_a = np.array(fea_a)
    fea_b = np.array(fea_b)
    dot_product = np.dot(fea_a, fea_b.T)
    norm_a = np.linalg.norm(fea_a, axis=1)
    norm_b = np.linalg.norm(fea_b, axis=1)
    return 1 - dot_product / (norm_a * norm_b)


# 构建W矩阵
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


# 执行LPP，包括数据类型转换等
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


# 构建W矩阵
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
    ''''''
    '''
    对全部100维数据进行归一化处理。具体的，它计算了数据的二范数，并将数据的每个元素除以其对应的范数值。
    '''
    norms = datas.norm(dim=2, keepdim=True)
    return datas / norms


# QR分解
def QRreduction(datas):
    ''''''
    '''
    对输入的数据执行QR分解，并返回QR分解中的上三角矩阵R。
    QR分解是一种矩阵分解的方法，它将一个矩阵分解为一个正交矩阵（其列向量是正交的）和一个上三角矩阵。
    在此函数中，我们只关心上三角矩阵R，因为它可以为我们提供输入数据的一些重要信息。
    torch.linalg.qr对最后两个维度进行 QR 分解，第一个为想要分解的维度，如(1280,100)，就是对1280维数据的信息进行QR分解
    '''
    ndatas = torch.linalg.qr(datas.permute(0, 2, 1)).R
    ndatas = ndatas.permute(0, 2, 1)
    return ndatas


class Model:  # 定义一个名为"Model"的类
    ''''''
    '''
    定义了一个名为 Model 的类，其主要作用是为所有的模型定义一个通用的基础结构。在这个例子中，它仅有一个属性：n_ways，用来表示分类的类别数量。
    这个类的主要功能是提供一个可以被其他模型类（如GaussianModel）继承的基类。在这个基类中，通过__init__方法定义了模型所需要的一些基本属性和参数。
    每当创建一个新的Model类的实例时，都会自动调用__init__方法，将指定的n_ways值赋值给实例的属性self.n_ways。
    这样，每个实例都会有它自己的n_ways属性，这是区分不同实例的一个重要方式。(继承自这个基类Model的不同类，使用的n_ways均为自己定义的不同的值)
    '''

    def __init__(self, n_ways):  # 类的初始化函数，输入参数是类别的数量
        self.n_ways = n_ways  # 将类别的数量存储在类的实例属性self.n_ways中


# ---------  GaussianModel
class GaussianModel(Model):
    ''''''
    '''
    这个类实现了高斯模型的相关计算，它的目的是实现一个高斯模型，用于计算每个样本属于每个类别的概率，然后通过最大后验估计 (MAP) 更新模型的参数。
    高斯模型的特点：1.2.假设数据服从多元高斯分布，然后通过最大化似然函数来估计参数（这里的参数就是均值向量 mus），
    3.最后根据这些参数来计算每个样本属于每个类别的概率。
    '''

    def __init__(self, n_ways, lam):
        ''''''
        '''
        初始化
        类的初始化函数，这个函数的主要目的是初始化 GaussianModel 类的对象。
        首先通过 super 调用了父类的 __init__ 方法，设置了类别的数量 n_ways。
        接着，初始化了类别的中心坐标 self.mus 为 None，这个变量在后续的训练过程中会被更新。最后，设置了用于计算最优传输矩阵的拉格朗日乘子 lam。
        '''
        super(GaussianModel, self).__init__(n_ways)  # 调用父类 Model 的初始化函数，设置类别的数量
        self.mus = None  # 初始化类别中心坐标为空，它的形状应为 [n_runs][n_ways][n_nfeat]
        self.lam = lam  # 设置用于计算最优传输矩阵的拉格朗日乘子

    def initFromLabelledDatas(self):
        ''''''
        '''
        数据初始化
        用于从已标记的数据集中初始化类别的中心坐标 (self.mus)。
        这个方法的目的是利用每个类别的支持集样本计算出该类别的均值向量，作为类别的中心。
        1.根据部分标记的数据初始化模型的均值向量 mus，这相当于在高斯模型中通过最大化似然函数来估计模型的参数。(替换为高斯混合模型的参数)
        '''
        # 首先，ndatas被reshape成一个4D张量，其形状为[n_runs, n_shot + n_queries, n_ways, n_nfeat]
        # 其中n_runs是运行次数，n_shot + n_queries是每个task中一个class的样本总数，n_ways是类别数，n_nfeat是特征数量
        # 然后，对于每个任务的每个类别，计算支持集样本的均值，得到每个类别的中心坐标，存储在self.mus中
        self.mus = ndatas.reshape(n_runs, n_shot + n_queries, n_ways, n_nfeat)[:, :n_shot, ].mean(1)

    def updateFromEstimate(self, estimate, alpha):
        ''''''
        '''
        更新参数
        根据新计算出的类别中心坐标来更新模型当前的类别中心坐标 self.mus。
        estimate 是新计算出的类别中心坐标，而 alpha 是更新的步长，表示新计算出的类别中心坐标对当前类别中心坐标更新的影响力度。
        在机器学习中，步长常常用于控制每次参数更新的幅度，也就是学习率。这里的 alpha 就类似于学习率，决定了模型参数每次更新的幅度。
        2.更新模型的均值向量 mus，这也是在不断地优化模型的参数。(替换为高斯混合模型的参数更新)
        '''
        Dmus = estimate - self.mus  # 计算新的类别中心坐标与当前类别中心坐标的差值
        self.mus = self.mus + alpha * (Dmus)  # 根据差值和步长，更新当前类别中心坐标

    def compute_optimal_transport(self, M, r, c, epsilon=1e-6):
        ''''''
        '''
        计算最优传输
        使用Sinkhorn迭代方法计算最优传输矩阵。
        最优传输问题是在给定的成本矩阵M，以及源和目标的分布r和c的条件下，寻找一种最优的传输方案，使得总成本最小。
        M为(1000,75,5)的距离矩阵，r为(1000, 75)的全1矩阵，c为(1000, 5)的全15矩阵
        Sinkhorn迭代是一种解决离散最优传输问题的有效方法。在每一次迭代中，都通过对P的更新使得其行和列的和分别接近源分布r和目标分布c，直到满足一定的精度为止。
        '''
        r = r.cuda()  # 将源分布的向量移动到GPU上 (1000, 75)
        c = c.cuda()  # 将目标分布的向量移动到GPU上 (1000, 5)
        n_runs, n, m = M.shape  # 获取成本矩阵的形状 (1000, 75, 5)
        P = torch.exp(- self.lam * M)  # 计算初始的传输矩阵 (1000, 75, 5)
        # P.view((n_runs, -1)) (1000, 375); P.view((n_runs, -1)).sum(1) (1000,); P.view((n_runs, -1)).sum(1).unsqueeze(1) (1000, 1)
        P /= P.view((n_runs, -1)).sum(1).unsqueeze(1).unsqueeze(1)  # 对传输矩阵进行归一化处理，使得每个task的75*5个概率，和为1 (1000, 75, 5)
        u = torch.zeros(n_runs, n).cuda()  # 初始化一个在GPU上的零向量u，用于存储计算过程中的中间结果 (1000, 75)
        maxiters = 1000  # 设置最大迭代次数
        iters = 1  # 初始化迭代次数
        while torch.max(torch.abs(u - P.sum(2))) > epsilon:  # 如果u和P沿第三维度求和的结果的最大差值大于epsilon，则继续迭代
            u = P.sum(2)  # 计算P沿第三维度的和，更新u (1000, 75)
            P *= (r / u).view((n_runs, -1, 1))  # 对P进行更新，P的第三维度乘一个数，使得其沿第三维度的和等于源分布r (1000, 75, 5) 第三维度的和为1
            P *= (c / P.sum(1)).view((n_runs, 1, -1))  # 对P进行更新，P的第二维度乘一个数，使得其沿第二维度的和等于目标分布c (1000, 75, 5) 第二维度的和为15
            if iters == maxiters:  # 如果达到最大迭代次数，则跳出循环
                break
            iters = iters + 1  # 更新迭代次数
        return P, torch.sum(P * M)  # 返回最优传输矩阵P和对应的最小总成本

    def getProbas(self):
        ''''''
        '''
        获取概率
        用于计算每个样本属于每个类别的概率。
        首先，该函数计算每个样本到每个类别中心的距离，然后使用 compute_optimal_transport 函数来计算概率分布。
        之后，它使用支持集样本的真实标签来修正概率分布。最后，函数返回每个样本属于每个类别的概率。
        3.根据当前的模型参数（即均值向量 mus）来计算每个样本对应每个类别的概率。(替换为高斯混合模型计算概率)
        这就像在高斯模型中，一旦得到了模型的参数，就可以计算出任何一个样本属于每个类别的概率。
        '''
        # compute squared dist to centroids [n_runs][n_samples][n_ways]
        # 计算样本到每个类别中心的距离，结果的形状是 [n_runs][n_samples][n_ways] (1000, 80, 1, 80) - (1000, 1, 5, 80)
        dist = (ndatas.unsqueeze(2) - self.mus.unsqueeze(1)).norm(dim=3).pow(2)  # 每个样本减去每个中心，再对数据标准化，然后平方
        # 初始化概率矩阵，形状和 dist 相同
        p_xj = torch.zeros_like(dist)
        # 初始化 r，形状是 [n_runs, n_usamples]，每个元素都是 1 (1000, 75)
        r = torch.ones(n_runs, n_usamples)
        # 初始化 c，形状是 [n_runs, n_ways]，每个元素都是 n_queries (1000, 5)
        c = torch.ones(n_runs, n_ways) * n_queries
        # 使用 compute_optimal_transport 函数计算查询集样本的概率分布
        p_xj_test, _ = self.compute_optimal_transport(dist[:, n_lsamples:], r, c, epsilon=1e-6)
        # 将计算出的概率分布赋值给 p_xj 的查询集对应位置
        p_xj[:, n_lsamples:] = p_xj_test
        # 将 p_xj 的支持集设为 0 (1000, 80, 5)
        p_xj[:, :n_lsamples].fill_(0)
        # 根据支持集样本的真实标签修正 p_xj 的前 n_lsamples 列
        # scatter_ 是 PyTorch 中的一个函数，用于根据索引向张量（tensor）的特定位置填充值。
        # 第一个参数是维度，在这里是 2，表示沿着第三个维度（在 Python 中，索引从 0 开始）进行操作。
        # 第二个参数是一个索引张量，它指定了在第一个参数所指定的维度上，哪些位置的值需要被改变。
        # 在这里，索引张量是 labels[:, :n_lsamples].unsqueeze(2)。值为0,1,2,3,4
        # 最后一个参数是要填充的值，在这里是 1。
        p_xj[:, :n_lsamples].scatter_(2, labels[:, :n_lsamples].unsqueeze(2), 1)
        # 返回每个样本属于每个类别的概率
        return p_xj

    def estimateFromMask(self, mask):
        ''''''
        '''
        估计均值向量
        通过样本的归属信息 (mask) 计算每个类别的中心坐标。mask 中的每个元素表示相应的样本是否属于相应的类别。
        这个函数的结果可以用于 updateFromEstimate 函数更新类别的中心坐标。
        '''
        # 首先，对 mask (1000, 80, 5)进行转置，使其形状与 ndatas（全局变量，包含所有样本的数据） (1000, 100, 100)相匹配
        # 根据 PyTorch 中的 matmul 函数的定义，如果两个张量都是 3-D，则它们的矩阵乘法会被计算为批处理矩阵乘法。
        # 最后两个维度被看作是矩阵的维度，第一个维度是批处理的维度。
        # 然后使用 matmul（矩阵乘法）函数，计算 mask 和 ndatas 的乘积 (1000, 5, 100)
        # 这一步相当于计算每个类别的所有样本的数据的总和
        emus = mask.permute(0, 2, 1).matmul(ndatas)

        # 接下来，使用 div 函数将上一步计算得到的总和除以每个类别的样本数 (1000, 5, 1)
        # 这一步相当于计算每个类别的所有样本的数据的平均值，即类别的中心坐标
        # mask.sum(dim=1) 计算的是每个类别的样本数，unsqueeze(2) 是为了保证形状匹配
        emus = emus.div(mask.sum(dim=1).unsqueeze(2))

        # 返回计算得到的类别中心坐标 (1000, 5, 80)
        return emus


# =========================================
#    MAP
# =========================================

#    MAP
class MAP:
    ''''''
    '''
    Maximum a Posteriori，也就是最大后验估计。这个类负责在给定模型和数据的情况下进行模型的训练和评估。
    '''

    def __init__(self, alpha=None):
        ''''''
        '''
        初始化
        类的构造函数或初始化方法。当创建这个类的新实例时，这个方法会被自动调用。
        在这个 MAP 类的 __init__ 方法中，它接受一个可选的参数 alpha，并设置了类的三个属性：verbose，progressBar 和 alpha。
        '''
        self.verbose = False  # 设置 verbose 属性为 False，表示默认不输出训练过程中每一轮的准确率
        self.progressBar = False  # 设置 progressBar 属性为 False，表示默认不显示训练进度条
        self.alpha = alpha  # 设置 alpha 属性，如果没有给定 alpha 参数，alpha 属性默认为 None

    def getAccuracy(self, probas):
        ''''''
        '''
        计算精度
        计算模型的精度。probas (1000, 80, 5)
        首先根据模型的预测概率来计算预测的标签，然后将预测标签和真实标签进行比对以判断预测的正确性，最后计算出模型的平均精度以及95%置信区间的精度。
        '''
        olabels = probas.argmax(dim=2)  # 对预测概率进行argmax操作，通过取5个概率值中最大概率的位置，获取预测的标签 (1000, 80)
        matches = labels.eq(olabels).float()  # 比较预测标签和真实标签是否匹配，并将结果转换为浮点数 (1000, 80)
        acc_test = matches[:, n_lsamples:].mean(1)  # 计算每个运行（run）的测试精度，只考虑查询集样本 (1000)
        m = acc_test.mean().item()  # 计算所有运行的平均精度
        pm = acc_test.std().item() * 1.96 / math.sqrt(n_runs)  # 计算所有运行精度的95%置信区间
        return m, pm  # 返回平均精度和置信区间

    def getAccuracy_label(self, pslabel):
        matches = labels.eq(pslabel).float()  # 与真实标签比较，得到匹配的标签
        return matches[n_lsamples:].mean().item()

    def performEpoch(self, model, epochInfo=None):
        ''''''
        '''
        执行训练周期
        执行一次训练周期（一个epoch）。
        在一个训练周期中，首先获取模型的预测概率，然后基于这个预测概率计算模型的新参数估计值，并使用这个新的估计值更新模型的参数。
        这个函数也会根据 verbose 参数来决定是否打印出每个训练周期结束后的模型精度。
        '''
        p_xj = model.getProbas()  # 从模型获取预测的概率
        self.probas = p_xj  # 将获取的概率赋值给类属性probas
        if self.verbose:  # 如果设置了详细模式
            print("accuracy from filtered probas", self.getAccuracy(self.probas))  # 计算并打印当前预测概率对应的模型精度
        m_estimates = model.estimateFromMask(self.probas)  # 根据预测概率计算模型的新参数估计值
        # 更新模型的参数
        model.updateFromEstimate(m_estimates, self.alpha)  # 使用新的估计值更新模型的参数
        if self.verbose:  # 如果设置了详细模式
            op_xj = model.getProbas()  # 再次获取模型的预测概率
            acc = self.getAccuracy(op_xj)  # 计算更新参数后的模型精度
            print("output model accuracy", acc)  # 打印更新参数后的模型精度

    def loop(self, model, n_epochs=20):
        ''''''
        '''
        循环训练
        进行多个训练周期（epochs）的循环训练。
        在每个训练周期中，它调用 performEpoch 函数来更新模型的参数，并根据 progressBar 参数来决定是否显示训练进度。
        最后，这个函数会返回训练结束后的模型精度。
        '''
        self.probas = model.getProbas()  # 获取模型预测的概率
        if self.verbose:  # 如果设置了 verbose
            print("initialisation model accuracy", self.getAccuracy(self.probas))  # 打印初始模型的准确率
        if self.progressBar:  # 如果设置了进度条
            if type(self.progressBar) == bool:  # 如果进度条的类型是 bool
                pb = tqdm(total=n_epochs)  # 则创建一个新的进度条
            else:
                pb = self.progressBar  # 否则使用已经设置的进度条
        for epoch in range(1, n_epochs + 1):  # 对于每一个训练周期
            if self.verbose:  # 如果设置了 verbose
                print(f"----- epoch[{epoch:3d}]  lr_p: {alpha:.3f}")  # 打印当前的训练周期和学习率
            self.performEpoch(model, epochInfo=(epoch, n_epochs))  # 执行一次训练周期
            if (self.progressBar): pb.update()  # 如果设置了进度条，则更新进度条
        # get final accuracy and return it
        op_xj = model.getProbas()  # 获取最后的模型预测概率
        acc = self.getAccuracy(op_xj)  # 计算最后的模型准确率
        return acc  # 返回最后的模型准确率


# 返回准确率的平均值和误差
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), sp.stats.sem(a)
    h = se * t._ppf((1 + confidence) / 2., n - 1)
    return m, h


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
    # dataset = r"cifarsf"
    # savemat(f'{dataset}_wrn_{n_shot}shots.mat', mdict={'features': ndatas.cpu().detach().numpy(), 'labels':labels.cpu().detach().numpy()})
    # print(111)
    start_time = time.time()  # 记录开始时间

    # LPP
    ndatas = get_LPP_datas(ndatas, n_lsamples, n_runs)

    n_nfeat = ndatas.size(2)

    # Gaussian Mixup
    import os

    os.environ['OMP_NUM_THREADS'] = '1'

    from sklearn.mixture import GaussianMixture

    # 转换为numpy数组
    ndatas_np = ndatas.cpu().numpy()
    labels_np = labels.cpu().numpy()

    from sklearn.linear_model import LogisticRegression

    # 用于保存伪标签的列表
    pseudo_labels = []
    acc = []
    a = MAP(0.2)
    # 迭代每个任务
    for task in range(ndatas_np.shape[0]):
        data = ndatas_np[task]
        label = labels_np[task]

        # 将前25个样本与后75个样本分开
        train_data = data[:25]
        test_data = data[25:]

        # 使用5个高斯分量对所有样本进行聚类
        gmm = GaussianMixture(n_components=5, random_state=42)
        gmm.fit(data)

        # 对前25个样本的聚类标签进行分类，将其映射到实际标签
        gmm_train_labels = gmm.predict(train_data)
        clf = LogisticRegression(random_state=42).fit(gmm_train_labels.reshape(-1, 1), label[:25])

        # 对后75个样本的聚类标签进行预测
        gmm_test_labels = gmm.predict(test_data)
        # 使用分类器将聚类标签映射到实际标签
        test_labels = clf.predict(gmm_test_labels.reshape(-1, 1))
        # 合并实际标签与伪标签
        task_labels = np.concatenate((label[:25], test_labels))

        acc.append(a.getAccuracy_label(torch.from_numpy(task_labels).cuda()))
        print("task", task, a.getAccuracy_label(torch.from_numpy(task_labels).cuda()))
    print(mean_confidence_interval(acc))

    # MAP
    lam = 10
    model = GaussianModel(n_ways, lam)
    model.initFromLabelledDatas()

    alpha = 0.2
    # alpha = 0.3
    optim = MAP(alpha)

    optim.verbose = True
    optim.progressBar = True

    # acc_test = optim.loop(model, n_epochs=20)
    # print("final accuracy found {:0.2f} +- {:0.2f}".format(*(100*x for x in acc_test)))
