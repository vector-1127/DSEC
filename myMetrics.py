from sklearn import metrics
import numpy as np

def count_occurrence(list):
    d = {}
    for i in list:
        if i in d:
            d[i] += 1
        else:
            d[i] = 1
    return d

def NMI(y_true,y_pred):
    return metrics.normalized_mutual_info_score(y_true, y_pred)

def ARI(y_true,y_pred):
    return metrics.adjusted_rand_score(y_true, y_pred)


def PURITY(y_true, y_pred):
    purity = 0
    ids = sorted(set(y_pred))
    matching = 0
    for id in ids:
        indices = [i for i, j in enumerate(y_pred) if j == id]
        cluster = [y_true[i] for i in indices]
        occ = count_occurrence(cluster)
        matching += max(occ.values())
    purity =  matching / float(len(y_true))
    return purity 

def ACC(y_true,y_pred):
    Y_pred = y_pred
    Y = y_true
    from sklearn.utils.linear_assignment_ import linear_assignment
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
        ind = linear_assignment(w.max() - w)
    return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size,ind

if __name__=='__main__':
    print('main')


