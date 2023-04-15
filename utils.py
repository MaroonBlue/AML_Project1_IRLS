import numpy as np
np.random.seed(123)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from LogisticRegression_with_IRLS_acc import LogisticRegression_with_IRLS
import matplotlib.pyplot as plt
from scipy.stats import bernoulli

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

import seaborn as sns
sns.set(rc={'figure.figsize':(11,9)})

# Part 2 of the project

def sigmoid(z):
    """Sigmoid function for logistic regression."""

    return 1.0 / (1.0 + np.exp(-z))

def generate_dataset1(size, num_features=2):
    """Generates a dataset with a given size and number of features.
    
    Args:
        size (int): Number of observations.
        num_features (int): Number of features. """
    
    np.random.seed(123)
    first_dataset_X = np.random.normal(size=(size, num_features))
    first_dataset_y = np.random.binomial(1, sigmoid(np.dot(np.column_stack((np.ones(size), first_dataset_X)), np.array([i for i in range (num_features+1)]))))
    return first_dataset_X, first_dataset_y


def generate_dataset2(size):
    """Generates a dataset with a given size and the number of features.

    Args:
        size (int): Number of observations."""

    b = np.array([2.3, 4.5, 0.345])
    second_dataset_X = np.zeros((size,3))
    for i in range(size):
        second_dataset_X[i][0]=np.random.normal(0,1,1)
        second_dataset_X[i][1]=np.random.normal(3,2,1)
        second_dataset_X[i][2]=np.random.normal(-5,3,1)
    p=(1/(1+np.exp(-(np.dot(second_dataset_X, b.T)))))
    second_dataset_y= bernoulli.rvs(p, size=size)
    return second_dataset_X, second_dataset_y


def generate_dataset3(m):
    """Generates a dataset with a given size and the number of features.
    
    Args:
        m (int): Mean of the second class."""

    dataset=np.zeros((1000, 3))
    for i in range(500):
        dataset[i,0]=np.random.normal(loc=0.0, scale=1.0)
        dataset[i,1]=np.random.normal(loc=0.0, scale=1.0)
        dataset[i,2]=0
        dataset[i+500,0]=np.random.normal(loc=m, scale=0.5)
        dataset[i+500,1]=np.random.normal(loc=m, scale=0.7)
        dataset[i+500,2]=1
    return dataset

def generate_dataset4(variance):
    """Generates a dataset with a given size and the number of features.
    
    Args:
        variance (float): Variance of the noise."""

    dataset=np.zeros((1000, 3))
    for i in range(500):
        x_1=np.random.uniform(low=-1, high=1)
        x_2=np.sqrt(1-x_1**2)
        dataset[i,0]=x_1
        if i%2==0:
            dataset[i,1]=x_2*(-1)
        else:
            dataset[i,1]=x_2
        dataset[i,2]=0
        x_1=np.random.uniform(low=-2, high=2)
        x_2=np.sqrt(4-x_1**2)
        dataset[i+500,0]=x_1
        if i%2==0:
            dataset[i+500,1]=x_2*(-1)
        else:
            dataset[i+500,1]=x_2
        dataset[i+500,2]=1
    dataset[:, 0:2]=dataset[:, 0:2]+np.random.normal(loc=0.0, scale=variance, size=(1000, 2))
    return dataset

def generate_xor(size):
    """Generates a XOR dataset with a given size.

    Args:
        size (int): Number of observations."""

    xor_X = np.random.uniform(-1, 1, (size, 2))
    xor_y = np.logical_xor(xor_X[:, 0] > 0, xor_X[:, 1] > 0).astype(int)
    return xor_X, xor_y


def run_experiments(X, y, iterations=10):
    """Runs the experiments for the given dataset.

    Args:
        X (numpy.ndarray): The dataset.
        y (numpy.ndarray): The labels.
        iterations (int): Number of iterations."""

    acc_all=np.zeros((iterations, 2))
    prec_all=np.zeros((iterations, 2))
    rec_all=np.zeros((iterations, 2))
    f1_all=np.zeros((iterations, 2))
    for i in range(iterations):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
        clf_without_interactions = LogisticRegression_with_IRLS()
        clf_without_interactions.fit(X_train, y_train, max_iter=500)
        y_pred_without_interactions = clf_without_interactions.predict(X_test)
        acc_all[i][0]=accuracy_score(y_test, y_pred_without_interactions)
        prec_all[i][0]=precision_score(y_test, y_pred_without_interactions)
        rec_all[i][0]=recall_score(y_test, y_pred_without_interactions)
        f1_all[i][0]=f1_score(y_test, y_pred_without_interactions)
        clf_with_interactions = LogisticRegression_with_IRLS()
        clf_with_interactions.fit(X_train, y_train, interaction_ids = [[0,1],[0,2],[1,2]], max_iter=500)
        y_pred_with_interactions = clf_with_interactions.predict(X_test)
        acc_all[i][1]=accuracy_score(y_test, y_pred_with_interactions)
        prec_all[i][1]=precision_score(y_test, y_pred_with_interactions)
        rec_all[i][1]=recall_score(y_test, y_pred_with_interactions)
        f1_all[i][1]=f1_score(y_test, y_pred_with_interactions)
    return acc_all, prec_all, rec_all, f1_all


def make_boxplot(acc, prec, rec, f1, name, dataset_name):
    """Makes a boxplot for the given metrics.

    Args:
        acc (numpy.ndarray): Accuracy.
        prec (numpy.ndarray): Precision.
        rec (numpy.ndarray): Recall.
        f1 (numpy.ndarray): F1 score.
        name (str): Name of the metric.
        dataset_name (str): Name of the dataset."""
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{name} for models for {dataset_name}', fontsize=20)
    axs[0, 0].boxplot(acc)
    axs[0, 0].set_title('Accuracy', fontsize=15)
    axs[0, 0].set_xticklabels(['Without interactions', 'With interactions'], fontsize=12)
    axs[0, 1].boxplot(prec)
    axs[0, 1].set_title('Precision', fontsize=15)
    axs[0, 1].set_xticklabels(['Without interactions', 'With interactions'], fontsize=12)
    axs[1, 0].boxplot(rec)
    axs[1, 0].set_title('Recall', fontsize=15)
    axs[1, 0].set_xticklabels(['Without interactions', 'With interactions'], fontsize=12)
    axs[1, 1].boxplot(f1)
    axs[1, 1].set_title('F1-score', fontsize=15)
    axs[1, 1].set_xticklabels(['Without interactions', 'With interactions'], fontsize=12)


def check_performance(X, y, dataset_name, if_interactions=False, interaction_ids=None):
    """Checks the performance of the model.

    Args:
        X (numpy.ndarray): The dataset.
        y (numpy.ndarray): The labels.
        dataset_name (str): Name of the dataset.
        if_interactions (bool): Whether to use interactions or not.
        interaction_ids (list): List of ids of the features to interact."""

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = LogisticRegression_with_IRLS()
    if if_interactions:
        clf.fit(X_train, y_train, max_iter=500, interaction_ids=interaction_ids)
    else:
        clf.fit(X_train, y_train, max_iter=500)
    y_pred = clf.predict(X_test)
    acc= accuracy_score(y_test, y_pred)
    print(f'Accuracy: {acc}')

    #draw results

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    ax[0].scatter(X_test[:,0], X_test[:,1], c=y_test, cmap='winter')
    ax[0].set_title('Original dataset', fontsize=20)
    ax[1].scatter(X_test[:,0], X_test[:,1], c=y_pred, cmap='winter')
    ax[1].set_title('Predicted classes', fontsize=20)
    fig.suptitle(dataset_name, fontsize=20)
    plt.show()


### nwm czy używamy to co niżej


def draw_boundary(X, y, coef, iter_num):
    """Draws the boundary of the model.

    Args:
        X (numpy.ndarray): The dataset.
        y (numpy.ndarray): The labels.
        coef (numpy.ndarray): Coefficients of the model.
        iter_num (int): Number of iterations."""

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='winter')
    x1 = np.linspace(-2, 2, 100)
    m=-(coef[0])/coef[1]
    c= - (coef[2])/coef[1]
    y = m*x1 + c
    ax.plot(x1, y)
    ax.set_title(f'Boundary after {iter_num} iterations', fontsize=20)
    plt.show()


# Part 3 of the project

def metrics_lr(X_train, X_test, y_train, y_test):
    """Calculates the metrics for the logistic regression model.
    
    Args:
        X_train: The training dataset.
        X_test: The testing dataset.
        y_train: The training labels.
        y_test: The testing labels."""

    lr=LogisticRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    acc=accuracy_score(y_test, y_pred)
    prec=precision_score(y_test, y_pred)
    rec=recall_score(y_test, y_pred)
    f1=f1_score(y_test, y_pred)
    return acc, prec, rec, f1

def metrics_LDA(X_train, X_test, y_train, y_test):
    """Calculates the metrics for the linear discriminant analysis model.
    
    Args:
        X_train: The training dataset.
        X_test: The testing dataset.
        y_train: The training labels.
        y_test: The testing labels."""

    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    y_pred = lda.predict(X_test)
    acc=accuracy_score(y_test, y_pred)
    prec=precision_score(y_test, y_pred)
    rec=recall_score(y_test, y_pred)
    f1=f1_score(y_test, y_pred)
    return acc, prec, rec, f1

def metrics_QDA(X_train, X_test, y_train, y_test):
    """Calculates the metrics for the quadratic discriminant analysis model.

    Args:
        X_train: The training dataset.
        X_test: The testing dataset.
        y_train: The training labels.
        y_test: The testing labels."""

    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_train, y_train)
    y_pred = qda.predict(X_test)
    acc=accuracy_score(y_test, y_pred)
    prec=precision_score(y_test, y_pred)
    rec=recall_score(y_test, y_pred)
    f1=f1_score(y_test, y_pred)
    return acc, prec, rec, f1

def metrics_KNN(X_train, X_test, y_train, y_test):
    """Calculates the metrics for the K-nearest neighbors model.

    Args:
        X_train: The training dataset.
        X_test: The testing dataset.
        y_train: The training labels.
        y_test: The testing labels."""

    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc=accuracy_score(y_test, y_pred)
    prec=precision_score(y_test, y_pred)
    rec=recall_score(y_test, y_pred)
    f1=f1_score(y_test, y_pred)
    return acc, prec, rec, f1

def metrics_IRLS(X_train, X_test, y_train, y_test):
    """Calculates the metrics for the IRLS model.

    Args:
        X_train: The training dataset.
        X_test: The testing dataset.
        y_train: The training labels.
        y_test: The testing labels."""

    clf = LogisticRegression_with_IRLS()
    X_train_np=np.array(X_train)
    X_test_np=np.array(X_test)
    y_train_np=np.array(y_train)
    y_test_np=np.array(y_test)
    clf.fit(X_train_np,y_train_np, max_iter=1000)
    y_pred = clf.predict(X_test_np)
    acc=accuracy_score(y_test_np, y_pred)
    prec=precision_score(y_test_np, y_pred)
    rec=recall_score(y_test_np, y_pred)
    f1=f1_score(y_test_np, y_pred)
    return acc, prec, rec, f1

def metrics_IRLS_interactions(X_train, X_test, y_train, y_test):
    """Calculates the metrics for the IRLS model with interactions.

    Args:
        X_train: The training dataset.
        X_test: The testing dataset.
        y_train: The training labels.
        y_test: The testing labels."""

    clf = LogisticRegression_with_IRLS()
    X_train_np=np.array(X_train)
    X_test_np=np.array(X_test)
    y_train_np=np.array(y_train)
    y_test_np=np.array(y_test)
    clf.fit(X_train_np,y_train_np, [[0,1], [1,2]], max_iter=1000)
    y_pred = clf.predict(X_test_np)
    acc=accuracy_score(y_test_np, y_pred)
    prec=precision_score(y_test_np, y_pred)
    rec=recall_score(y_test_np, y_pred)
    f1=f1_score(y_test_np, y_pred)
    return acc, prec, rec, f1

def run_experiments_part3(X, y, iterations=10):
    """Runs the experiments for part 3.

    Args:
        X: The dataset.
        y: The labels.
        iterations: The number of iterations to run the experiments."""

    acc_all=np.zeros((iterations, 6))
    prec_all=np.zeros((iterations, 6))
    rec_all=np.zeros((iterations, 6))
    f1_all=np.zeros((iterations, 6))
    for i in range(iterations):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
        acc_all[i, 0], prec_all[i, 0], rec_all[i, 0], f1_all[i, 0] = metrics_LDA(X_train, X_test, y_train, y_test)
        acc_all[i, 1], prec_all[i, 1], rec_all[i, 1], f1_all[i, 1] = metrics_QDA(X_train, X_test, y_train, y_test)
        acc_all[i, 2], prec_all[i, 2], rec_all[i, 2], f1_all[i, 2] = metrics_KNN(X_train, X_test, y_train, y_test)
        acc_all[i, 3], prec_all[i, 3], rec_all[i, 3], f1_all[i, 3] = metrics_IRLS(X_train, X_test, y_train, y_test)
        acc_all[i, 4], prec_all[i, 4], rec_all[i, 4], f1_all[i, 4] = metrics_IRLS_interactions(X_train, X_test, y_train, y_test)
        acc_all[i, 5], prec_all[i, 5], rec_all[i, 5], f1_all[i, 5] = metrics_lr(X_train, X_test, y_train, y_test)
    return acc_all, prec_all, rec_all, f1_all

def make_boxplot_part3(data, name, dataset_name):
    """Makes a boxplot for the metrics.

    Args:
        data: The data to plot.
        name: The name of the metric.
        dataset_name: The name of the dataset."""

    fig_auc=sns.boxplot(data=data, palette='Blues_d')
    fig_auc.set_title(f'{name} for different models for {dataset_name}', fontsize=20)
    fig_auc.set_ylabel(f'{name}', fontsize=15)
    fig_auc.set_xlabel('Model', fontsize=15)
    fig_auc.set_xticklabels(['LDA', 'QDA', 'KNN', 'IRLS', 'IRLS_interactions', 'Linear Regression'], fontsize=12)