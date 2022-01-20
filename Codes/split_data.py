
import numpy as np

def split_by_group(X_all, y_all, group, cap, k):

    All = []
    for i in range(len(X_all)):
        line = []
        line.append(X_all[i])
        line.append(y_all[i])
        All.append(line)
    # =============================================================================

    split_all = []
    for i in range(group):
        line = []
        for j in range(cap):
            line.append(All[i*cap+j])
        split_all.append(line)

    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for i in range(group):
        X_traint, y_traint, X_testt, y_testt = kfold_split(k, split_all[i])
        if i==0:
            for w in range(k):
                X_train.append(list(X_traint[w]))
                y_train.append(list(y_traint[w]))
                X_test.append(list(X_testt[w]))
                y_test.append(list(y_testt[w]))
        else:
            for w in range(k):
                for j in range(len(X_traint[w])):
                    X_train[w].append(X_traint[w][j])
                for j in range(len(y_traint[w])):
                    y_train[w].append(y_traint[w][j])
                for j in range(len(X_testt[w])):
                    X_test[w].append(X_testt[w][j])
                for j in range(len(y_testt[w])):
                    y_test[w].append(y_testt[w][j])

    for w in range(k):
        X_train[w] = np.array(X_train[w])
        y_train[w] = np.array(y_train[w])
        X_test[w] = np.array(X_test[w])
        y_test[w] = np.array(y_test[w])
    
    
    return X_train, y_train, X_test, y_test


def kfold_split(k, All):

    ##########################################################################
    import numpy as np
    from sklearn.model_selection import RepeatedKFold
    random_state = 12883823
    rkf = RepeatedKFold(n_splits=k, n_repeats=1, random_state=random_state)
    ##########################################################################

    ##########################################################################
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    kfold_indeX = rkf.split(All)

    for train_indeX, test_indeX in kfold_indeX:
            Xk = []
            yk = []
            for j in range(len(train_indeX)):
                Xk.append(All[train_indeX[j]][0])
                yk.append(All[train_indeX[j]][1])
            X_train.append(np.array(Xk))
            y_train.append(np.array(yk))

            Xk2 = []
            yk2 = []
            for j in range(len(test_indeX)):
                Xk2.append(All[test_indeX[j]][0])
                yk2.append(All[test_indeX[j]][1])
            X_test.append(np.array(Xk2))
            y_test.append(np.array(yk2))

    ##########################################################################
    
    return X_train, y_train, X_test, y_test


def train_test_split(X_all, y_all):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    print('set_train:', len(X_train), 'set_test:', len(X_test))
    print('ok')
    
    return X_train, y_train, X_test, y_test


def kfold_split_group(X_all, y_all, group, cap, k=5):
    # =============================================================================
    All = []
    for i in range(len(X_all)):
        line = []
        line.append(X_all[i])
        line.append(y_all[i])
        All.append(line)
    # =============================================================================

    # =============================================================================
#     group = 20
#     cap = 1000
    split_all = []
    for i in range(group):
        line = []
        for j in range(cap):
            line.append(All[i*cap+j])
        split_all.append(line)

#     k = 5
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for i in range(group):
        X_traint, y_traint, X_testt, y_testt = kfold_split(k, split_all[i])
        if i==0:
            for w in range(k):
                X_train.append(list(X_traint[w]))
                y_train.append(list(y_traint[w]))
                X_test.append(list(X_testt[w]))
                y_test.append(list(y_testt[w]))
        else:
            for w in range(k):
                for j in range(len(X_traint[w])):
                    X_train[w].append(X_traint[w][j])
                for j in range(len(y_traint[w])):
                    y_train[w].append(y_traint[w][j])
                for j in range(len(X_testt[w])):
                    X_test[w].append(X_testt[w][j])
                for j in range(len(y_testt[w])):
                    y_test[w].append(y_testt[w][j])

    for w in range(k):
        X_train[w] = np.array(X_train[w])
        y_train[w] = np.array(y_train[w])
        X_test[w] = np.array(X_test[w])
        y_test[w] = np.array(y_test[w])

    print('data ready')
    return X_train, y_train, X_test, y_test
    # =============================================================================
