

def count_scores(model, X):

    k = len(model)

    # =============================================================================
    scores = []
    for i in range(k):
        if i==0:
            scores = [x[1] for x in model[i].predict_proba(X)]
        elif i!=4:
            score = [x[1] for x in model[i].predict_proba(X)]
            for j in range(len(scores)):
                scores[j] = scores[j] + score[j]
        else:
            score = [x[1] for x in model[i].predict_proba(X)]
            for j in range(len(scores)):
                scores[j] = (scores[j] + score[j])/k
                
    return scores
    # =============================================================================
    

def pre_results(model, X):

    scores = count_scores(model, X)
    
    results = scores
    
    for i in range(len(results)):
        if results[i]>0.5:
            results[i]=1
        else:
            results[i]=0
    
    return results
    
    
def eva_acc(model, X, y):

    # =============================================================================
    # compute on remaining test data
    
    results = pre_results(model, X)
#     pipe_pred_test = model[0].predict_classes(X)
    
    n = 0
    for i in range(len(y)):
        if results[i]==y[i]:
            n+=1
        
    acc = n/len(y)
    return acc
    # =============================================================================

    
def eva_cross_acc(model,X,y):
    k = len(model)
    
    for i in range(k):
        y_pre  = model[i].predict_classes(X)
        acc = com_accuracy(y_pre, y)
        print("The acc of model ",i,":",acc)
        
def eva_cross_acc_all(model,X_train,X_test,y_train,y_test):
    k = len(model)
    
    for i in range(k):
        y_train_pre  = model[i].predict_classes(X_train[i])
        acc_train = com_accuracy_cross(y_train_pre, y_train[i])
        y_test_pre  = model[i].predict_classes(X_test[i])
        acc_test = com_accuracy_cross(y_test_pre, y_test[i])
        
        print("The acc of model ",i,": train: ",acc_train," ,test: ",acc_test)

def com_accuracy_cross(y1,y2):
    n = 0
    length_y = len(y1)
    
    for i in range(length_y):
        if y1[i]==y2[i][1]:
            n += 1
            
    accuracy = n/length_y
    return accuracy

def eva_table(model, X, y):

    # =============================================================================
    # compute on remaining test data
    
    results = pre_results(model, X)
#     pipe_pred_test = model[0].predict_classes(X)
    from sklearn.metrics import classification_report
    print(classification_report(y_true=y, y_pred = results))
    # =============================================================================


def eva_roc(model, X, y):

    # =============================================================================
    from sklearn.metrics import roc_curve, auc
    from sklearn import metrics

    scores = count_scores(model, X)

    fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=1)

    auc = metrics.auc(fpr, tpr)

    import matplotlib.pyplot as plt
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    # =============================================================================


def eva_pr(model, X, y):

    # =============================================================================
    from sklearn.metrics import precision_recall_curve
    import matplotlib.pyplot as plt

    scores = count_scores(model, X)
    
    precision, recall, _ =precision_recall_curve(y, scores)
    plt.plot(recall, precision,color='navy')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.plot(recall, precision)
    plt.title("Precision-Recall")

    plt.show()


    # =============================================================================
    
    
def com_accuracy(y1, y2):

    n = 0
    length_y = len(y1)
    
    for i in range(length_y):
        if y1[i]==y2[i]:
            n += 1
            
    accuracy = n/length_y
    return accuracy