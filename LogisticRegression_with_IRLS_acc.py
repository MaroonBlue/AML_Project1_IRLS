import numpy as np
from numpy.linalg import inv
from sklearn.metrics import accuracy_score

class LogisticRegression_with_IRLS:
    """Linear Regression with Iterative Reweighted Least Squares optimization."""

    def __init__(self, X_test=None, y_test=None, test_flag=False):
        """Initializes the model

        Args:
            X_test (numpy.ndarray): The test dataset, needed if test_flag is True.
            y_test (numpy.ndarray): The test labels, needed if test_flag is True.
            test_flag (bool): If True, the model will be tested on the test dataset."""
                
        # nr of instances
        self.n = None
        
        # nr of variables
        self.p = None
        
        # vector of betas
        self.beta = None

        # if test
        self.test_flag = test_flag
        
        # X test
        self.X_test = X_test

        # y test
        self.y_test = y_test 
        
    def fit(self, X, y, interaction_ids = [], max_iter=100):
        """ Fits the model using training data.

        Args:
            X (numpy.ndarray): The training dataset.
            y (numpy.ndarray): The training labels.
            interaction_ids (list): A list of tuples, each tuple containing the ids of the variables to be multiplied.
            max_iter (int): The maximum number of iterations of the IRLS algorithm."""
        
        # test if X matrix and y vector of the same length
        np.testing.assert_equal(np.shape(X)[0], len(y))
        
        # test if y has only classes 0 and 1
        np.testing.assert_equal(np.unique(y),[0,1])
        
        # n - nr of instances
        self.n = np.shape(X)[0]
                
        
        # p - nr of variables
        self.p = np.shape(X)[1]
        
        self.interaction_ids = interaction_ids

        # for testing more parameters
        if self.test_flag:
            self.beta, self.X, self.acc_all, self.tol_all = self.IRLS_with_interactions(X, y, self.interaction_ids, max_iter=max_iter)
        else:
            self.beta, self.X = self.IRLS_with_interactions(X, y, self.interaction_ids, max_iter=max_iter)

    def add_interactions(self, X, interaction_ids):
        """ Adds interactions to the dataset.
        
        Args:
            X (numpy.ndarray): The dataset.
            interaction_ids (list): A list of tuples, each tuple containing the ids of the variables to be multiplied."""

        for interaction in interaction_ids:
            new_col = X[:,interaction[0]]*X[:,interaction[1]]
            X = np.insert(X, X.shape[1], new_col, axis=1)
        return X

    def IRLS_with_interactions(self, X, y, interaction_ids, max_iter=100, w_init = 0.5, d = 0.0001, tolerance = 0.001):
        """ Fits the model using training data.
        
        Args:
            X (numpy.ndarray): The training dataset.
            y (numpy.ndarray): The training labels.
            interaction_ids (list): A list of tuples, each tuple containing the ids of the variables to be multiplied.
            max_iter (int): The maximum number of iterations of the IRLS algorithm.
            w_init (float): The initial value of the weights.
            d (float): The value of the d parameter.
            tolerance (float): Tolerance threshold for the algorithm to stop."""

        if self.test_flag:
            acc_all = []
            tol_all = []

        X = self.add_interactions(X, interaction_ids)
        n,p = X.shape
        delta = np.array(np.repeat(d, n)).reshape(1,n)
        w = np.repeat(w_init, n)
        W = np.diag( w )
        B_new = np.dot(inv( X.T.dot(W).dot(X) ), 
                ( X.T.dot(W).dot(y)))
        for i in range(max_iter):
            B_old = B_new
            _w = abs(y - X.dot(B_old)).T
            w = float(1)/np.maximum( delta, _w )
            W = np.diag( w[0] )
            B_new = np.dot(inv( X.T.dot(W).dot(X)), 
                    ( X.T.dot(W).dot(y) ) )
            tol = sum( abs( B_new - B_old ) ) 
            if self.test_flag:
                acc_all.append(accuracy_score(self.y_test, self.predict_test(self.X_test, B_new)))
                tol_all.append(tol)
            
            if tol < tolerance:
                if self.test_flag:
                    return B_new, X, acc_all, tol_all
                return B_new, X
            
        if self.test_flag:
            return B_new, X, acc_all, tol_all
        return B_new, X
    
    def predict_proba(self, Xtest):
        """ Predicts posterior probabilities for class 1 for observations whose featurevalues are in rows of the matrix Xtest.
        
        Args:
            Xtest (numpy.ndarray): The test dataset."""

        Xtest = self.add_interactions(Xtest, self.interaction_ids)
        pi = (np.exp(np.dot(Xtest, self.beta.T)))/(1+np.exp(np.dot(Xtest, self.beta.T)))
        
        return pi
    
    def predict(self, Xtest):
        """ Assigns the predicted class (0 or 1) for observations whose feature values are in rows of the matrix Xtest.
        
        Args:
            Xtest (numpy.ndarray): The test dataset."""
        
        pi = self.predict_proba(Xtest)
        y_pred = np.multiply([pi>0.5], 1)[0];
        
        return y_pred
    
    def get_params(self):
        """Returns a list containing the estimated parameters beta."""
    
        return self.beta
    
    def predict_proba_test(self, Xtest, beta):
        """ Predicts posterior probabilities for class 1 for observations whose feature values are in rows of the matrix Xtest.
        
        Args:
            Xtest (numpy.ndarray): The test dataset.
            beta (numpy.ndarray): The estimated beta."""
        
        Xtest = self.add_interactions(Xtest, self.interaction_ids)
        pi = (np.exp(np.dot(Xtest, beta.T)))/(1+np.exp(np.dot(Xtest, beta.T)))
        
        return pi
    
    def predict_test(self, Xtest, beta):
        """ Assigns the predicted class (0 or 1) for observations whose feature values are in rows of the matrix Xtest.
        
        Args:
            Xtest (numpy.ndarray): The test dataset.
            beta (numpy.ndarray): The estimated beta."""

        pi = self.predict_proba_test(Xtest, beta)
        y_pred = np.multiply([pi>0.5], 1)[0];
        
        return y_pred