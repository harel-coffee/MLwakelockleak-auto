from sklearn.metrics import accuracy_score

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
    label_show_confusion_matrix(Y_test, result.predict(X_test))

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
    label_show_confusion_matrix(Y_test, result.predict(X_test))
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
    label_show_confusion_matrix(Y_test, result.predict(X_test))
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
    label_show_confusion_matrix(Y_test, result.predict(X_test))
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
    label_show_confusion_matrix(Y_test, result.predict(X_test))
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
    label_show_confusion_matrix(Y_test, result.predict(X_test))
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
    label_show_confusion_matrix(Y_test, result.predict(X_test))
    return accuracy


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
    X_sm, y_sm = smote.fit_resample(X, y)

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
    over = SMOTE(sampling_strategy=0.5)
    under = RandomUnderSampler(sampling_strategy=0.5)
    # over = SMOTE(ratio=0.1)
    # under = RandomUnderSampler(ratio=0.5)
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)
    # transform the dataset
    X_smt, y_smt = pipeline.fit_resample(X, y)
    return X_smt, y_smt


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

def check_smote(X_smt, y_smt):
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
    print('KNN=',  knr_list)
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
    plt.legend(['Naive-Bay', 'Linear SVC', 'K-NearNeighbor', 'Logistic Regression', 'Ridge Classifier', 'Bagged Decision Tree', 'Random Forest', 'Stochastic Gradient Boosting'])
    plt.show()
from numpy import load
def load_data_plot(path, x_data='X_test.npz', y_data='Y_test.npz'):
    ##//////////////////No SMOTE applied X and y////////////////////////////////
    import os
    x_data = os.path.join(os.getcwd()+path, x_data)
    y_data = os.path.join(os.getcwd()+path, y_data)
    dict_data = load(x_data)
    X_smt = dict_data['arr_0']
    dict_lbl = load(y_data)
    y_smt = dict_lbl['arr_0']
    print(len(y_smt))
    check_smote(X_smt, y_smt)

