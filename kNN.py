import numpy as np
import operator
def createDataSet():
    """
    生成样本数据集
    """
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    # 样本的属性值, 这个数据集每个样本有两个属性
    labels = ['A', 'A', 'B', 'B']
    # 每个样本对应的一个标签, 也就是一个结果
    return group, labels

def classify0(inX, dataSet, labels, k):
    """
    inX未知标签的数据集, dataSet样本数据集, labels样本数据集对应的标签, k前个点中出现频率最多的, 
    返回标记好的标签, 其中k最好是个偶数, 虽然不是也可以, 个人建议.
    """
    dataSetSize = dataSet.shape[0] 
    # 得到数据集样本的个数
    # dataSetSize can be changed by size of labels.
    diffMat = np.tile(inX, (dataSetSize, 1))-dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    # print(type(distances))
    sortedDistIndices = distances.argsort()
    # print(sortedDistIndices)
    classCount ={}
    # 这里用的是字典
    for i in range(k):
        # print(i)
        # print(sortedDistIndices[i])
        voteIlabel = labels[sortedDistIndices[i]]
        # print(voteIlabel)
        classCount[voteIlabel] = classCount.get(voteIlabel, 0)+1
        #dict.get(a, b) if dict hasnt a, return b.
    print(classCount)
    sortedClassCount = sorted(classCount.items(),
      key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0] 

def file2matrix(filename):
    fr = open(filename)
    # 打开文件
    arrayOLines = fr.readlines()
    # 将文件一行一行的读入
    numoberOfLines = len(arrayOLines)
    # 得到行数, 实际上也是样本总数
    returnMat = np.zeros((numoberOfLines,3))
    # 初始化一个3列的列表准备保存特征向量, 使用np的原因是更好操作
    classLabelVector = []
    #初始化存放结果的列表, 因为只有一类, 所以就用了python自带的列表
    index = 0
    for line in arrayOLines:
        line = line.strip()
        # 把每行的头尾的回车和空格删去
        listFromLine = line.split('\t')
        # 以Tab为单位切分txt文件
        returnMat[index, :] = listFromLine[0:3]
        # 将切割好的结果存在特征向量中, 之所以这里不用改变格式, 
        # 在于之前生成是, 已经是zeros了
        # if(index<10):
            # print(listFromLine[-1])
        classLabelVector.append(int(listFromLine[-1]))
        # 将结果保存, 注意格式, 很奇怪上面没有这一步
        index += 1
    return returnMat, classLabelVector

def autoNorm(dataSet):
    """
    正则化样本属性, 输出正则化后的样本属性, 属性中最大值和最小值的差别
    和属性中的最小值
    """
    minVals = dataSet.min(0)
    # 求出每一列, 也就是每个属性的最小值
    maxVals = dataSet.max(0)
    # 求出每一列, 也就是每个属性的最大值
    ranges = maxVals - minVals
    # 求出每一列中最大值和最小值之间的距离
    normDataSet = np.zeros(np.shape(dataSet))
    # 初始化样本属性空间
    m = dataSet.shape[0]
    # 求出样本数据总数, 实际上就是行数
    normDataSet = dataSet-np.tile(minVals, (m,1))
    normDataSet = normDataSet/np.tile(ranges, (m,1))
    # 标准化公式: (x-min)/range
    return normDataSet, ranges, minVals

def datingClassTest():
    hoRatio = 0.1
    # 选择的测试样本的比例
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    # 样本的总数
    numTestVecs = int(m*hoRatio)
    # 选取的测试样本数, 一个int*float, 结果是float, 可能有分数, 所以要转换一下格式
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResults = classify0(normMat[i, :], normMat[numTestVecs:m, :],
        datingLabels[numTestVecs:m], 3)
        # 选定从0到numTestVecs的数据为测试数据, 其他数据均为输入数据
        # 这个有点问题, 每次都要重新训练一遍, 很浪
        print("the classifier came back with: %d, the real answer is : %d"
        % (classifierResults, datingLabels[i]))
        if (classifierResults!=datingLabels[i]):
            errorCount += 1
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))
    return (errorCount/float(numTestVecs))
# group, labels = createDataSet()
