import numpy as np
import operator
def creatDataSet():
    group = np.array([[1.0, 1.0], [1.0, 1.1], [0.0, 0.0], [0.1, 0.0]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def kNN(inX, dataSet, Labels, k):
    n_samples = dataSet.shape[0]
    diff = np.tile(inX, (n_samples, 1))-dataSet
    diff = diff**2
    diff = diff.sum(axis=1)
    diff = diff**0.5
    sortedIndex = diff.argsort()
    results = {}
    for i in range(k):
        m_labels = labels[sortedIndex[i]]
        results[m_labels] = results.get(m_labels, 0)+1
    sortedResult = sorted(results.items(), key=operator.itemgetter(1), reverse=True)
    return sortedResult[0][0]

group, labels = creatDataSet()
n = kNN([0,0], group, labels, 3)
print(n)
