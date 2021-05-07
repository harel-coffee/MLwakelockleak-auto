from sklearn.metrics import accuracy_score
from yellowbrick.classifier import ClassificationReport


def apply_linear_regression(X_train, Y_train, X_test, Y_test):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(penalty='l2', C=0.1)
    print("[*] Training...")
    print(clf, type(X_train), X_train.shape, Y_train.shape)
    result = clf.fit(X_train, Y_train)
    predicted = clf.predict(X_test)
    accuracy = accuracy_score(Y_test, predicted, normalize=True)
    print("LogisticRegression accuracy: ",accuracy)  ##///////Accuracy=0.44444444//////
    from sklearn.metrics import classification_report
    report = classification_report(Y_test,predicted)
    print("LogisticRegression Report: ", report)
    show_confusion_matrix(Y_test, result.predict(X_test))
    label_show_confusion_matrix(Y_test, result.predict(X_test))
    plot_roc_curve(clf, X_test, Y_test)
    plot_precision_recall_curve(clf, X_test, Y_test)
    return accuracy

##/////////////////Not able to install xgboost
def apply_xgboost(X_train, Y_train, X_test, Y_test):
    from xgboost import XGBClassifier
    clf = XGBClassifier(max_depth=2, gamma=2, eta=0.8, reg_alpha=0.5, reg_lembda=0.5)
    print("[*] Training...")
    print(clf, type(X_train), X_train.shape, Y_train.shape)
    result = clf.fit(X_train, Y_train)
    print("XGBoost accuracy: ",accuracy_score(Y_test, result.predict(X_test), normalize=True))  ##///////Accuracy=0.44444444//////
def apply_smote(X_train, Y_train, X_test, Y_test):
    from imblearn.over_sampling import SMOTE ##instruction not working
    smote = SMOTE(ratio='minority', n_jobs=-1)
    import pandas as pd
    df = pd.DataFrame()
    df['target'] = Y_train
    df['target'].value_counts().plot(kind='bar', title='Count(target)')
    X_sm, Y_sm = smote.fit_resample(X_train, Y_train)
    df = pd.DataFrame(X_sm, columns=['Leak', 'Clean'])
    df['target'] = Y_sm
    df['target'].value_counts().plot(kind='bar', title='Count(target)')

def show_confusion_matrix(y_test, y_pred):
    from sklearn.metrics import confusion_matrix, classification_report
    from matplotlib import pyplot as plt
    conf_mat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    labels = ['Clean', 'Leak']
    print("confusion matrix:\n", conf_mat)
    # print("Report:", classification_report(y_test, y_pred))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def label_show_confusion_matrix(y_test, y_pred):
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
    from matplotlib import pyplot as plt

    labels = ['Clean', 'Leak']

    # print("Report:", classification_report(y_test, y_pred))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ConfusionMatrixDisplay(confusion_matrix(y_pred, y_test, labels=[0,1]), display_labels=labels).plot(values_format=".0f", ax=ax,cmap=plt.cm.Blues)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def plot_roc_curve(clf, X_test, y_test):
    from sklearn import metrics
    import matplotlib.pyplot as plt
    metrics.plot_roc_curve(clf, X_test, y_test)
    plt.show()

def plot_precision_recall_curve(clf, X_test, y_test):
    from sklearn import metrics
    import matplotlib.pyplot as plt
    metrics.plot_precision_recall_curve(clf, X_test, y_test)
    plt.show()

def apply_naive_bayes(X_train, Y_train, X_test, Y_test):
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    print("[*] Training...")
    print(clf, type(X_train), X_train.shape, Y_train.shape)
    result = clf.fit(X_train, Y_train)
    predicted = clf.predict(X_test)
    accuracy = accuracy_score(Y_test, predicted, normalize=True)
    print("Naive-Bayes accuracy: ", accuracy) ##///////Accuracy=0.44444444//////
    from sklearn.metrics import classification_report
    report = classification_report(Y_test, predicted)
    print("Naive-Bayes Report: ", report)
    show_confusion_matrix(Y_test, result.predict(X_test))
    label_show_confusion_matrix(Y_test, result.predict(X_test))
    plot_roc_curve(clf, X_test, Y_test)
    plot_precision_recall_curve(clf, X_test, Y_test)
    # show_confusion_matrix(Y_test, result.predict(X_test))
    # plot_roc_curve(clf, X_test, Y_test)
    # plot_precision_recall_curve(clf, X_test, Y_test)
    return accuracy


def apply_linear_SVC(X_train, Y_train, X_test, Y_test):
    from sklearn.svm import LinearSVC
    clf = LinearSVC(random_state=0)
    print("[*] Training...")
    print(clf, type(X_train), X_train.shape, Y_train.shape)
    result = clf.fit(X_train, Y_train)
    predicted = clf.predict(X_test)
    accuracy = accuracy_score(Y_test, predicted, normalize=True)
    print("LinearSVC accuracy: ", accuracy) ##///////////Accuracy=0.55555555//////
    from sklearn.metrics import classification_report
    report = classification_report(Y_test, predicted)
    print("LinearSVC Report: ", report)
    show_confusion_matrix(Y_test, result.predict(X_test))
    label_show_confusion_matrix(Y_test, result.predict(X_test))
    plot_roc_curve(clf, X_test, Y_test)
    plot_precision_recall_curve(clf, X_test, Y_test)
    # show_confusion_matrix(Y_test, result.predict(X_test))
    # plot_roc_curve(clf, X_test, Y_test)
    # plot_precision_recall_curve(clf, X_test, Y_test)
    return accuracy


def apply_K_Neighbors_Classifier(X_train, Y_train, X_test, Y_test):
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=3)
    print("[*] Training...")
    print(clf, type(X_train), X_train.shape, Y_train.shape)
    result = clf.fit(X_train, Y_train)
    predicted = clf.predict(X_test)
    accuracy = accuracy_score(Y_test, predicted, normalize=True)
    print("KNeighbors accuracy: ", accuracy)  ##///////////Accuracy=0.77777777//////
    from sklearn.metrics import classification_report
    report = classification_report(Y_test, predicted)
    print("KNeighbors Report: ", report)
    show_confusion_matrix(Y_test, result.predict(X_test))
    label_show_confusion_matrix(Y_test, result.predict(X_test))
    plot_roc_curve(clf, X_test, Y_test)
    plot_precision_recall_curve(clf, X_test, Y_test)
    # show_confusion_matrix(Y_test, result.predict(X_test))
    # plot_roc_curve(clf, X_test, Y_test)
    # plot_precision_recall_curve(clf, X_test, Y_test)
    return accuracy


def apply_Ridge_Classifier(X_train, Y_train, X_test, Y_test):
    from sklearn.linear_model import RidgeClassifier
    clf = RidgeClassifier(alpha=0.7)
    print("[*] Training...")
    print(clf, type(X_train), X_train.shape, Y_train.shape)
    result = clf.fit(X_train, Y_train)
    predicted = clf.predict(X_test)
    accuracy = accuracy_score(Y_test, predicted, normalize=True)
    print("Ridge Classifer accuracy: ", accuracy)  ##///////////Accuracy=0.960000//////
    from sklearn.metrics import classification_report
    report = classification_report(Y_test, predicted)
    print("Ridge Classifier Report: ", report)
    show_confusion_matrix(Y_test, result.predict(X_test))
    label_show_confusion_matrix(Y_test, result.predict(X_test))
    plot_roc_curve(clf, X_test, Y_test)
    plot_precision_recall_curve(clf, X_test, Y_test)
    # show_confusion_matrix(Y_test, result.predict(X_test))
    # plot_roc_curve(clf, X_test, Y_test)
    # plot_precision_recall_curve(clf, X_test, Y_test)
    return accuracy


def apply_Bagged_Decision_Tree(X_train, Y_train, X_test, Y_test):
    from sklearn.ensemble import BaggingClassifier
    clf = BaggingClassifier(n_estimators=100)
    print("[*] Training...")
    print(clf, type(X_train), X_train.shape, Y_train.shape)
    result = clf.fit(X_train, Y_train)
    predicted = clf.predict(X_test)
    accuracy = accuracy_score(Y_test, predicted, normalize=True)
    print("Bagged Decision Tree accuracy: ", accuracy)  ##///////////Accuracy=0.981667//////
    from sklearn.metrics import classification_report
    report = classification_report(Y_test, predicted)
    print("BDT Report: ", report)
    show_confusion_matrix(Y_test, result.predict(X_test))
    label_show_confusion_matrix(Y_test, result.predict(X_test))
    plot_roc_curve(clf, X_test, Y_test)
    plot_precision_recall_curve(clf, X_test, Y_test)
    # show_confusion_matrix(Y_test, result.predict(X_test))
    # plot_roc_curve(clf, X_test, Y_test)
    # plot_precision_recall_curve(clf, X_test, Y_test)
    return accuracy


def apply_Random_Forest(X_train, Y_train, X_test, Y_test):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, max_features='sqrt')
    print("[*] Training...")
    print(clf, type(X_train), X_train.shape, Y_train.shape)
    result = clf.fit(X_train, Y_train)
    predicted = clf.predict(X_test)
    accuracy = accuracy_score(Y_test, predicted, normalize=True)
    print("Random Forest accuracy: ", accuracy)  ##///////////Accuracy=0.980000//////
    from sklearn.metrics import classification_report
    report = classification_report(Y_test, predicted)
    print("Random Forest Report: ", report)
    show_confusion_matrix(Y_test, result.predict(X_test))
    label_show_confusion_matrix(Y_test, result.predict(X_test))
    plot_roc_curve(clf, X_test, Y_test)
    plot_precision_recall_curve(clf, X_test, Y_test)
    # show_confusion_matrix(Y_test, result.predict(X_test))
    # plot_roc_curve(clf, X_test, Y_test)
    # plot_precision_recall_curve(clf, X_test, Y_test)
    return accuracy


def apply_Gradient_Boosting(X_train, Y_train, X_test, Y_test):
    from sklearn.ensemble import GradientBoostingClassifier
    clf = GradientBoostingClassifier(learning_rate=0.1, max_depth=9, n_estimators=1000, subsample=1.0)
    print("[*] Training...")
    print(clf, type(X_train), X_train.shape, Y_train.shape)
    result = clf.fit(X_train, Y_train)
    predicted = clf.predict(X_test)
    accuracy = accuracy_score(Y_test, predicted, normalize=True)
    print("Gradient Boosting accuracy: ", accuracy)  ##///////////Accuracy=0.982500//////
    from sklearn.metrics import classification_report
    report = classification_report(Y_test, predicted)
    print("Gradient Boosting Report: ", report)
    show_confusion_matrix(Y_test, result.predict(X_test))
    label_show_confusion_matrix(Y_test, result.predict(X_test))
    plot_roc_curve(clf, X_test, Y_test)
    plot_precision_recall_curve(clf, X_test, Y_test)
    # show_confusion_matrix(Y_test, result.predict(X_test))
    # plot_roc_curve(clf, X_test, Y_test)
    # plot_precision_recall_curve(clf, X_test, Y_test)
    return accuracy

def apply_XGBoost(X_train, Y_train, X_test, Y_test):
    # First XGBoost model for Pima Indians dataset
    # from numpy import loadtxt
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    # load data
    # dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")
    # split data into X and y
    # X = dataset[:, 0:8]
    # Y = dataset[:, 8]
    # split data into train and test sets
    # seed = 7
    # test_size = 0.33
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    # fit model no training data
    model = XGBClassifier()
    model.fit(X_train, Y_train)
    # make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(Y_test, predictions)
    print("XGBoost Accuracy: %.2f%%" % (accuracy * 100.0))


def visualize_data(clf, X_train, Y_train, X_test, Y_test):
    visualizer = ClassificationReport(clf, classes=['Leak', 'Clean'])
    visualizer.fit(X_train, Y_train)
    visualizer.score(X_test, Y_test)
    g = visualizer.show()


def show_classification(X_train, Y_train, X_test, Y_test):
    from sklearn.datasets import make_classification
    X, y = make_classification(n_classes=2, class_sep=1.5, weights=[0.9, 0.1], n_informative=3, n_redundant=1, flip_y=0, n_features=20, n_clusters_per_class=1, n_samples=100, random_state=10)
    import pandas as pd
    df = pd.DataFrame(X)
    df['target'] = y
    df.target.value_counts().plot(kind='bar', title='Count(target)')


def plot_2d_space(X, y, label='Classes'):
    import numpy as np
    from matplotlib import pyplot as plt
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(X[y==l, 0], X[y==l, 1], c=c, label=l, marker=m)
    plt.title(label)
    plt.legend(loc='upper right')
    plt.show()


def show_PCA(X, y):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X = pca.fit_transform(X)
    plot_2d_space(X, y, 'Imbalanced data (2 PCA components')


def random_under_sampler(X, y):
    from imblearn.under_sampling import RandomUnderSampler
    rus = RandomUnderSampler(return_indices=True) ##//////Not working
    X_rus, y_rus, id_rus = rus.fit_sample(X, y)
    print('Removed indexes:', id_rus)
    plot_2d_space(X_rus, y_rus, 'Random under-sampling')


def tomeklinks_under_sampler(X, y):
    from imblearn.under_sampling import TomekLinks
    tl = TomekLinks(return_indices=True, ratio='majority') ##/////Not working no return_indices
    X_tl, y_tl, id_tl = tl.fit_sample(X, y)
    print('Removed indexes:', id_tl)
    plot_2d_space(X_tl, y_tl, 'Tomek links under-sampling')


def cluster_centroids_under_sampler(X, y):
    from imblearn.under_sampling import ClusterCentroids

    cc = ClusterCentroids(sampling_strategy='auto') ##///////////Not working, no ratio///////
    X_cc, y_cc = cc.fit_sample(X, y)

    # plot_2d_space(X_cc, y_cc, 'Cluster Centroids under-sampling')
    return X_cc, y_cc


def random_over_sampling(X, y):
    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler()
    X_ros, y_ros = ros.fit_sample(X, y) ##////////Not Working

    print(X_ros.shape[0] - X.shape[0], 'new random picked points')
    # plot_2d_space(X_ros, y_ros, 'Random over-sampling')
    return X_ros, y_ros

def smote_over_sampling(X, y):
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(sampling_strategy='minority')#(ratio='minority') ##/////////Not working at ratio
    X_sm, y_sm = smote.fit_sample(X, y)

    # plot_2d_space(X_sm, y_sm, 'SMOTE over-sampling')
    return X_sm, y_sm

def smote_over_sampling_under_sampling(X, y):
    from imblearn.combine import SMOTETomek

    smt = SMOTETomek(sampling_strategy='minority')#(ratio='auto') ##//////////Not working at ratio
    X_smt, y_smt = smt.fit_sample(X, y)

    # plot_2d_space(X_smt, y_smt, 'SMOTE + Tomek links')
    return X_smt, y_smt

def apply_simple_smote(X, y):
    from imblearn.over_sampling import SMOTE
    from collections import Counter
    simple_smote = SMOTE()
    print(Counter(y))
    X_smt, y_smt = simple_smote.fit_sample(X, y)
    print(Counter(y_smt))
    return X_smt, y_smt

def apply_simple_adasyn(X, y):
    from imblearn.over_sampling import ADASYN
    from collections import Counter
    simple_adasyn = ADASYN(sampling_strategy='minority')
    print(Counter(y))
    X_smt, y_smt = simple_adasyn.fit_sample(X, y)
    print(Counter(y_smt))
    return X_smt, y_smt


    # plot_2d_space(X_smt, y_smt, 'SMOTE + Tomek links')

def apply_over_random_under_sample_smote(X, y):
    # Oversample with SMOTE and random undersample for imbalanced dataset
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline
    from matplotlib import pyplot
    from numpy import where
    # counter = Counter(y)
    # print(counter)
    # define pipeline
    over = SMOTE(sampling_strategy=0.5)
    under = RandomUnderSampler(sampling_strategy=0.5)
    # over = SMOTE(ratio=0.1)
    # under = RandomUnderSampler(ratio=0.5)
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)
    # transform the dataset
    X_smt, y_smt = pipeline.fit_resample(X, y)
    return X_smt, y_smt
    # summarize the new class distribution
    # counter = Counter(y)
    # print(counter)
    # scatter plot of examples by class label
    # for label, _ in counter.items():
    #     row_ix = where(y == label)[0]
    #     pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
    # pyplot.legend()
    # pyplot.show()


def apply_different_smote(X, y):
    from collections import Counter
    print("No sampling=", Counter(y))
    X_smt, y_smt = smote_over_sampling(X, y) ##////////////Simple Smote
    # X_smt, y_smt = smote_over_sampling(X, y) ##///////////Same as SMOTE
    # X_smt, y_smt = smote_over_sampling_under_sampling(X, y) ##/////////////Smote over under sampling Tomek///
    # X_smt, y_smt = apply_over_random_under_sample_smote(X, y) ##/////////////Smote over random under sampling///
    # X_smt, y_smt = random_over_sampling(X, y) ##/////////////Random over sampling///
    print("Over Sampling=",Counter(y_smt))
    X_train, X_test, y_train, y_test = randomize_dataset(X_smt, y_smt)
    return apply_different_model(X_train, y_train, X_test, y_test)


    ##/////////////////////K-Fold and running 16 times to plot boxplot graph/////////
    # apply_kfold_LR(X_smt, y_smt)
    # apply_kfold_RC(X_smt, y_smt)
    # apply_kfold_KNN(X_smt, y_smt)
    # apply_kfold_SVC(X_smt, y_smt)
    # apply_kfold_BDT(X_smt, y_smt)
    # apply_kfold_RF(X_smt, y_smt)
    ##///////////////////////////////////////////////////////////////////////////////
    ##/////////////Grid Search to find hyper parameters//////////////////////////////
    # apply_grid_search_LR(X_smt, y_smt)
    # apply_grid_search_RidgeClassifier(X_smt, y_smt)
    # apply_grid_search_KNN(X_smt, y_smt)
    # apply_grid_search_SVC(X_smt, y_smt)
    # apply_grid_search_BaggingClassifier(X_smt, y_smt)
    # apply_grid_search_RandomForest(X_smt, y_smt)
    # apply_grid_search_GradientBoosting(X_smt, y_smt) ## Takes lot of time
    ##//////////////////////////////////////////////////////////////////////////////////////
    ##////////////Over_Random_Under_Simple Sampling SMOTE///////////////////////////////////
    # X_smt, y_smt = apply_over_random_under_sample_smote(X, y)
    # X_train, X_test, y_train, y_test = randomize_dataset(X_smt, y_smt)
    # nb, sv, knr, lr = apply_different_model(X_train, y_train, X_test, y_test)
    ##//////////////////////////////////////////////////////////////////////////////////////
    ##////////////Over_Sampling_Under Sampling SMOTE////////////////////////////////////////
    # X_smt, y_smt = smote_over_sampling_under_sampling(X, y)
    # X_train, X_test, y_train, y_test = randomize_dataset(X_smt, y_smt)
    # nb, sv, knr, lr = apply_different_model(X_train, y_train, X_test, y_test)
    ##//////////////////////////////////////////////////////////////////////////////////////
    ##///////////////////////Over_Sampling_SMOTE////////////////////////////////////////////
    # X_smt, y_smt = smote_over_sampling(X, y)
    # X_train, X_test, y_train, y_test = randomize_dataset(X_smt, y_smt)
    # nb, sv, knr, lr = apply_different_model(X_train, y_train, X_test, y_test)
    ##//////////////////////////////////////////////////////////////////////////////////////
    ##////////////////////Random_Over_Sampling_SMOTE////////////////////////////////////////
    # X_smt, y_smt = random_over_sampling(X, y)
    # X_train, X_test, y_train, y_test = randomize_dataset(X_smt, y_smt)
    # nb, sv, knr, lr = apply_different_model(X_train, y_train, X_test, y_test)

    # return nb, sv, knr, lr

def apply_grid_search_LR(X, y):
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    solvers = ['newton-cg', 'lbfgs', 'liblinear']
    penalty = ['l2']
    c_values = [100, 10, 1.0, 0.1, 0.01]
    ##//////////////////////Added////////////////////////////////////////////////
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y)
    ##////////////////////////////////////////////////////////////////////////////
    # define grid search
    grid = dict(solver=solvers, penalty=penalty, C=c_values)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, cv=cv, scoring='accuracy', error_score=0)
    # grid_result = grid_search.fit(X, y)
    grid_result = grid_search.fit(X_train, y_train)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    ##////////////////////Added/////////////////////////////////////////////////
    print("test-set score: {:.3f}".format(grid_result.score(X_test, y_test)))
    ##/////////////////////////////////////////////////////////////////////////
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


def apply_grid_search_KNN(X, y):
    # example of grid searching key hyperparametres for KNeighborsClassifier
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.model_selection import GridSearchCV
    from sklearn.neighbors import KNeighborsClassifier
    # define models and parameters
    model = KNeighborsClassifier()
    n_neighbors = range(1, 21, 2)
    weights = ['uniform', 'distance']
    metric = ['euclidean', 'manhattan', 'minkowski']
    ##//////////////////////Added////////////////////////////////////////////////
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.33, stratify=y)
    ##////////////////////////////////////////////////////////////////////////////
    # define grid search
    grid = dict(n_neighbors=n_neighbors, weights=weights, metric=metric)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, cv=cv, scoring='accuracy', error_score=0)
    # grid_result = grid_search.fit(X, y)
    grid_result = grid_search.fit(X_train, y_train)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    ##////////////////////Added/////////////////////////////////////////////////
    print("test-set score: {:.3f}".format(grid_result.score(X_test, y_test)))
    ##/////////////////////////////////////////////////////////////////////////
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

def apply_grid_search_SVC(X, y):
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    # define model and parameters
    model = SVC()
    kernel = ['poly', 'rbf', 'sigmoid']
    C = [50, 10, 1.0, 0.1, 0.01]
    gamma = ['scale']
    ##//////////////////////Added////////////////////////////////////////////////
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.33, stratify=y)
    ##////////////////////////////////////////////////////////////////////////////
    # define grid search
    grid = dict(kernel=kernel, C=C, gamma=gamma)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, cv=cv, scoring='accuracy', error_score=0)
    # grid_result = grid_search.fit(X, y)
    grid_result = grid_search.fit(X_train, y_train)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    ##////////////////////Added/////////////////////////////////////////////////
    print("test-set score: {:.3f}".format(grid_result.score(X_test, y_test)))
    ##/////////////////////////////////////////////////////////////////////////
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

def apply_validation_curve_grid_search_SVC(X, y):
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import validation_curve
    from sklearn.svm import SVC
    import numpy as np
    # define model and parameters
    model = SVC()
    kernel = ['poly', 'rbf', 'sigmoid']
    C = [50, 10, 1.0, 0.1, 0.01]
    gamma = ['scale']
    ##//////////////////////Added////////////////////////////////////////////////
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.33, stratify=y)
    ##////////////////////////////////////////////////////////////////////////////
    # define grid search
    grid = dict(kernel=kernel, C=C, gamma=gamma)
    from sklearn.model_selection import ShuffleSplit
    train_scores, test_scores = validation_curve(model, X, y, param_name='gamma', param_range=np.logspace(-6,-1,5), scoring='accuracy')
    # grid_result = grid_search.fit(X, y)
    print("Training Score= ", train_scores)
    print("Testing Score= ", test_scores)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    import matplotlib.pyplot as plt
    # plt.title("Validation Curve with SVM")
    plt.xlabel(r"$\gamma$")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.semilogx(np.logspace(-6,-1,5), train_scores_mean, label="Training score",
                 color="green", lw=lw, marker='v')
    plt.fill_between(np.logspace(-6,-1,5), train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="green", lw=lw)
    plt.semilogx(np.logspace(-6,-1,5), test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw, marker='o')
    plt.fill_between(np.logspace(-6,-1,5), test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.99, top=0.99)
    plt.grid(linestyle='--')
    plt.savefig("Validation.png", dpi=500)
    plt.show()


def apply_grid_search_BaggingClassifier(X, y):
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import BaggingClassifier
    # define models and parameters
    model = BaggingClassifier()
    n_estimators = [10, 100, 1000]
    ##//////////////////////Added////////////////////////////////////////////////
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.33, stratify=y)
    ##////////////////////////////////////////////////////////////////////////////
    # define grid search
    grid = dict(n_estimators=n_estimators)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, cv=cv, scoring='accuracy',error_score=0)
    # grid_result = grid_search.fit(X, y)
    grid_result = grid_search.fit(X_train, y_train)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    ##////////////////////Added/////////////////////////////////////////////////
    print("test-set score: {:.3f}".format(grid_result.score(X_test, y_test)))
    ##/////////////////////////////////////////////////////////////////////////
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

def apply_grid_search_GradientBoosting(X, y):
# def apply_grid_search_GradientBoosting(X_train, X_test, y_train, y_test):
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import GradientBoostingClassifier
    # define models and parameters
    model = GradientBoostingClassifier()
    n_estimators = [10, 100, 1000]
    learning_rate = [0.001, 0.01, 0.1]
    subsample = [0.5, 0.7, 1.0]
    max_depth = [3, 7, 9]
    ##//////////////////////Added////////////////////////////////////////////////
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
    ##////////////////////////////////////////////////////////////////////////////
    # define grid search
    grid = dict(learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, max_depth=max_depth)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, cv=cv, scoring='accuracy', error_score=0)
    # grid_result = grid_search.fit(X, y)
    grid_result = grid_search.fit(X_train, y_train) ##///////////Added
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    ##////////////////////Added/////////////////////////////////////////////////
    print("test-set score: {:.3f}".format(grid_result.score(X_test, y_test)))
    ##/////////////////////////////////////////////////////////////////////////
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

# def apply_grid_search_RandomForest(X_train, X_test, y_train, y_test):
def apply_grid_search_RandomForest(X, y):
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    # define models and parameters
    model = RandomForestClassifier()
    n_estimators = [10, 100, 1000]
    max_features = ['sqrt', 'log2']
    # define grid search
    ##//////////////////////Added////////////////////////////////////////////////
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.33, stratify=y)
    ##////////////////////////////////////////////////////////////////////////////
    grid = dict(n_estimators=n_estimators, max_features=max_features)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, cv=cv, scoring='accuracy', error_score=0)
    grid_result = grid_search.fit(X_train, y_train) ##//////////Changed
    # grid_result = grid_search.fit(X, y)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    ##////////////////////Added/////////////////////////////////////////////////
    print("test-set score: {:.3f}".format(grid_result.score(X_test, y_test)))
    ##/////////////////////////////////////////////////////////////////////////
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

def apply_grid_search_RidgeClassifier(X, y):
    # example of grid searching key hyperparametres for ridge classifier
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import RidgeClassifier
    # define models and parameters
    model = RidgeClassifier()
    alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # define grid search
    grid = dict(alpha=alpha)
    ##//////////////////////Added////////////////////////////////////////////////
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y)
    ##////////////////////////////////////////////////////////////////////////////
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, cv=cv, scoring='accuracy', error_score=0)
    # grid_result = grid_search.fit(X, y)
    grid_result = grid_search.fit(X_train, y_train)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    ##////////////////////Added/////////////////////////////////////////////////
    print("test-set score: {:.3f}".format(grid_result.score(X_test, y_test)))
    ##/////////////////////////////////////////////////////////////////////////

    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

def apply_kfold_LR(X, y):
    from sklearn.model_selection import RepeatedKFold
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression
    from numpy import mean
    from scipy.stats import sem
    repeats = range(1, 16)
    results = list()
    from matplotlib import pyplot
    for r in repeats:
        # evaluate using a given number of repeats
        # scores = evaluate_model(X, y, r)
        cv = RepeatedKFold(n_splits=10, n_repeats=r, random_state=1)
        # model = LogisticRegression()
        model = LogisticRegression(C=1.0, penalty='l2', solver='liblinear')

        scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv)
        # summarize
        print('>%d mean=%.4f se=%.3f' % (r, mean(scores), sem(scores)))
        # store
        results.append(scores)
    pyplot.boxplot(results, labels=[str(r) for r in repeats], showmeans=True)
    pyplot.show()

def apply_kfold_RC(X, y):
    from sklearn.model_selection import RepeatedKFold
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import RidgeClassifier
    from numpy import mean
    from scipy.stats import sem
    repeats = range(1, 16)
    results = list()
    from matplotlib import pyplot
    for r in repeats:
        # evaluate using a given number of repeats
        # scores = evaluate_model(X, y, r)
        cv = RepeatedKFold(n_splits=10, n_repeats=r, random_state=1)
        # model = LogisticRegression()
        model = RidgeClassifier(alpha=0.7)

        scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv)
        # summarize
        print('>%d mean=%.4f se=%.3f' % (r, mean(scores), sem(scores)))
        # store
        results.append(scores)
    pyplot.boxplot(results, labels=[str(r) for r in repeats], showmeans=True)
    pyplot.show()


def apply_kfold_KNN(X, y):
    from sklearn.model_selection import RepeatedKFold
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    from numpy import mean
    from scipy.stats import sem
    repeats = range(1, 16)
    results = list()
    from matplotlib import pyplot
    for r in repeats:
        # evaluate using a given number of repeats
        # scores = evaluate_model(X, y, r)
        cv = RepeatedKFold(n_splits=10, n_repeats=r, random_state=1)
        # model = LogisticRegression()
        model = KNeighborsClassifier(metric='euclidean', n_neighbors=3, weights='distance')

        scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv)
        # summarize
        print('>%d mean=%.4f se=%.3f' % (r, mean(scores), sem(scores)))
        # store
        results.append(scores)
    pyplot.boxplot(results, labels=[str(r) for r in repeats], showmeans=True)
    pyplot.show()

def apply_kfold_SVC(X, y):
    from sklearn.model_selection import RepeatedKFold
    from sklearn.model_selection import cross_val_score
    from sklearn.svm import SVC
    from numpy import mean
    from scipy.stats import sem
    repeats = range(1, 16)
    results = list()
    from matplotlib import pyplot
    for r in repeats:
        # evaluate using a given number of repeats
        # scores = evaluate_model(X, y, r)
        cv = RepeatedKFold(n_splits=10, n_repeats=r, random_state=1)
        # model = LogisticRegression()
        model = SVC(C=10, gamma='scale', kernel='rbf')

        scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv)
        # summarize
        print('>%d mean=%.4f se=%.3f' % (r, mean(scores), sem(scores)))
        # store
        results.append(scores)
    pyplot.boxplot(results, labels=[str(r) for r in repeats], showmeans=True)
    pyplot.show()
    # fig = pyplot.figure()
    # fig.savefig('SVC.png')

def apply_kfold_BDT(X, y):
    from sklearn.model_selection import RepeatedKFold
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import BaggingClassifier
    from numpy import mean
    from scipy.stats import sem
    repeats = range(1, 16)
    results = list()
    from matplotlib import pyplot
    for r in repeats:
        # evaluate using a given number of repeats
        # scores = evaluate_model(X, y, r)
        cv = RepeatedKFold(n_splits=10, n_repeats=r, random_state=1)
        # model = LogisticRegression()
        model = BaggingClassifier(n_estimators=100)

        scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv)
        # summarize
        print('>%d mean=%.4f se=%.3f' % (r, mean(scores), sem(scores)))
        # store
        results.append(scores)
    pyplot.boxplot(results, labels=[str(r) for r in repeats], showmeans=True)
    pyplot.show()

def apply_kfold_RF(X, y):
    from sklearn.model_selection import RepeatedKFold
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    from numpy import mean
    from scipy.stats import sem
    repeats = range(1, 16)
    results = list()
    from matplotlib import pyplot
    for r in repeats:
        # evaluate using a given number of repeats
        # scores = evaluate_model(X, y, r)
        cv = RepeatedKFold(n_splits=10, n_repeats=r, random_state=1)
        # model = LogisticRegression()
        model = RandomForestClassifier(max_features='sqrt', n_estimators=100)

        scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv)
        # summarize
        print('>%d mean=%.4f se=%.3f' % (r, mean(scores), sem(scores)))
        # store
        results.append(scores)
    pyplot.boxplot(results, labels=[str(r) for r in repeats], showmeans=True)
    pyplot.show()

    # from sklearn import model_selection
    # from sklearn.model_selection import KFold
    #
    # kfold = model_selection.KFold(n_splits=10, random_state=100)
    # from sklearn.linear_model import LogisticRegression
    #
    # model_kfold = LogisticRegression()
    # result_kfold = model_selection.cross_val_score(model_kfold, fdf_normalized, y, cv=kfold)
    # print("Accuracy: %.2f%%" % (result_kfold.mean()*100))


def apply_different_model(X_train, y_train, X_test, y_test):
    nb = apply_naive_bayes(X_train, y_train, X_test, y_test)
    sv = apply_linear_SVC(X_train, y_train, X_test, y_test)
    knr = apply_K_Neighbors_Classifier(X_train, y_train, X_test, y_test)
    lr = apply_linear_regression(X_train, y_train, X_test, y_test)
    rc = apply_Ridge_Classifier(X_train, y_train, X_test, y_test)
    bdt = apply_Bagged_Decision_Tree(X_train, y_train, X_test, y_test)
    rf = apply_Random_Forest(X_train, y_train, X_test, y_test)
    sgb = apply_Gradient_Boosting(X_train, y_train, X_test, y_test)
    return nb, sv, knr, lr, rc, bdt, rf, sgb

def randomize_dataset(X, Y):
        """ Randomly split the dataset in training and testing sets
        """
        n = len(Y)
        train_size = int(0.8 * n)
        # index = range(n) ##/////////////////Orignal//////////
        index = list(range(n))  ##/////////////////Changed//////////
        print(n, train_size, index)
        from random import shuffle
        shuffle(index)
        train_index = sorted(index[:train_size])
        test_index = sorted(index[train_size:])

        X_train = X[train_index, :]
        X_test = X[test_index, :]
        Y_train = Y[train_index]
        Y_test = Y[test_index]
        print("train=", len(Y_train))
        print("test=", len(Y_test))
        return X_train, X_test, Y_train, Y_test

def check_smote(X, y):
    nb_list = []
    sv_list = []
    knr_list = []
    lr_list = []
    rc_list = []
    bdt_list = []
    rf_list = []
    sgb_list = []
    for i in range(10):
        nb, sv, knr, lr, rc, bdt, rf, sgb = apply_different_smote(X_smt, y_smt)
        nb_list.append(nb)
        sv_list.append(sv)
        knr_list.append(knr)
        lr_list.append(lr)
        rc_list.append(rc)
        bdt_list.append(bdt)
        rf_list.append(rf)
        sgb_list.append(sgb)
    import matplotlib.pyplot as plt
    print('NB=',  nb_list)
    print('SV=',  sv_list)
    print('KNR=',  knr_list)
    print('LR=',  lr_list)
    print('RC=',  rc_list)
    print('BDT=',  bdt_list)
    print('RF=',  rf_list)
    print('SGB=',  sgb_list)
    plt.plot(nb_list)
    plt.plot(sv_list)
    plt.plot(knr_list)
    plt.plot(lr_list)
    plt.plot(rc_list)
    plt.plot(bdt_list)
    plt.plot(rf_list)
    plt.plot(sgb_list)
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Iteration')
    plt.legend(['Naive-Bay', 'Linear SVC', 'K-Neighbor', 'Logistic Regression', 'Ridge Classifier', 'Bagged Decision Tree', 'Random Forest', 'Stochastic Gradient Boosting'])
    plt.show()

from numpy import load
##//////////////////No SMOTE applied X and y////////////////////////////////
# dict_data = load('X_test.npz')
# X_smt = dict_data['arr_0']
# dict_lbl = load('Y_test.npz')
# y_smt = dict_lbl['arr_0']
# print(len(y_smt))
# check_smote(X_smt, y_smt)

##///////////Testing////////////////////////////////////////
# X, y = apply_simple_smote(X_smt, y_smt)
# print(len(y))
# X_train, X_test, Y_train, Y_test = randomize_dataset(X,y)
# apply_linear_regression(X_train, Y_train, X_test, Y_test)
##//////////////////////////////////////////////////////////

# apply_grid_search_RandomForest(X, y)
# apply_grid_search_RidgeClassifier(X_train, y_train)
# apply_grid_search_GradientBoosting(X_train, y_train) ##////Taking long time so i stopped
# X_train, X_test, y_train, y_test = randomize_dataset(X, y)
# apply_grid_search_RandomForest(X_train, X_test, y_train, y_test)
# apply_grid_search_RandomForest(X, y)
# apply_grid_search_SVC(X, y)
# apply_grid_search_LR(X, y)
# apply_grid_search_RidgeClassifier(X, y)
# apply_grid_search_KNN(X, y)
# apply_grid_search_BaggingClassifier(X, y)
# apply_grid_search_GradientBoosting(X, y) #need to check with split, take long time
# X_train, X_test, y_train, y_test = randomize_dataset(X, y)
# apply_XGBoost(X_train, y_train, X_test, y_test) ## Same
# apply_xgboost(X_train, y_train, X_test, y_test) ## Same



##//////////////////SMOTE applied data and label////////////////////////////////
dict_data = load('data.npz')
X_smt = dict_data['arr_0']
dict_lbl = load('label.npz')
y_smt = dict_lbl['arr_0']
print(len(y_smt))
apply_validation_curve_grid_search_SVC(X_smt, y_smt)
##//////////////////////////////////////////////////////////////////////////////////
##///////////get accuracy of 0.9600 and loss of 0.1527//////////////////////////////
##//////////////////////////////////////////////////////////////////////////////////
##//////////X_val and Y_val are for validation, copied from the orignal data////////
##//////X_test and Y_test are for test, without removing validation from this data//
# dict_data = load('X_test.npz')
# X = dict_data['arr_0']
# dict_lbl = load('Y_test.npz')
# y = dict_lbl['arr_0']
# dict_data_val = load('X_val.npz')
# X_val = dict_data_val['arr_0']
# dict_lbl_val = load('Y_val.npz')
# y_val = dict_lbl_val['arr_0']
# from myMLAlgo import apply_simple_smote
# X_smt, y_smt = apply_simple_smote(X, y)
# # apply_different_NN(X_smt, y_smt)
# apply_basic_NN_validate(X_smt, y_smt, X_val, y_val)
##////////////////////////////////////////////////////////////////////////////////////////
##///////////get accuracy of 1.000 and loss of 0.00548////////////////////////////////////
##//////////////////////////////////////////////////////////////////////////////////
##//////////X_val1 and Y_val1 are for validation, removed from the orignal data/////
##//////////X_test1 and Y_test1 are for test, removed validation from this data/////
# dict_data = load('X_test1.npz')
# X = dict_data['arr_0']
# dict_lbl = load('Y_test1.npz')
# y = dict_lbl['arr_0']
# dict_data_val = load('X_val1.npz')
# X_val = dict_data_val['arr_0']
# dict_lbl_val = load('Y_val1.npz')
# y_val = dict_lbl_val['arr_0']
# from myMLAlgo import apply_simple_smote
# X_smt, y_smt = apply_simple_smote(X, y)
# # apply_different_NN(X_smt, y_smt)
# apply_basic_NN_validate(X_smt, y_smt, X_val, y_val)
# # print(len(y_smt))
##////////////////////////////////////////////////////////////////////////////////////////
##///////////get accuracy of 0.800 and loss of 1.3098/////////////////////////////////////
##////////////////////////////////////////////////////////////////////////////////////////