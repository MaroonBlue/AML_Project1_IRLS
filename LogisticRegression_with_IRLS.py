import numpy as np
from numpy.linalg import inv

class LogisticRegression_with_IRLS:
    """Linear Regression with Iterative Reweighted Least Squares optimization"""
    def __init__(self):
                
        # nr of instances
        self.n = None
        
        # nr of variables
        self.p = None
        
        # vector of betas
        self.beta = None
        
        
    def fit(self, X, y, interaction_ids = [], max_iter=100):
        """ Fits the model using training data"""
        
        # test if X matrix and y vector of the same length
        np.testing.assert_equal(np.shape(X)[0], len(y))
        
        # test if y has only classes 0 and 1
        np.testing.assert_equal(np.unique(y),[0,1])
        
        
        # n - nr of instances
        self.n = np.shape(X)[0]
                
        
        # p - nr of variables
        self.p = np.shape(X)[1]
        
        self.interaction_ids = interaction_ids
        self.beta, self.X = self.IRLS_with_interactions(X, y, self.interaction_ids, max_iter=max_iter)

    def add_interactions(self, X, interaction_ids):
        for interaction in interaction_ids:
            new_col = X[:,interaction[0]]*X[:,interaction[1]]
            X = np.insert(X, X.shape[1], new_col, axis=1)
        return X

    def IRLS_with_interactions(self, X, y, interaction_ids, max_iter=100, w_init = 1, d = 0.0001, tolerance = 0.001):
        
        X = self.add_interactions(X, interaction_ids)
        n,p = X.shape
        delta = np.array(np.repeat(d, n)).reshape(1,n)
        w = np.repeat(w_init, n)
        W = np.diag( w )
        B_new = np.dot(inv( X.T.dot(W).dot(X) ), 
                ( X.T.dot(W).dot(y)))
        for _ in range(max_iter):
            B_old = B_new
            _w = abs(y - X.dot(B_old)).T
            w = float(1)/np.maximum( delta, _w )
            W = np.diag( w[0] )
            B_new = np.dot(inv( X.T.dot(W).dot(X)), 
                    ( X.T.dot(W).dot(y) ) )
            tol = sum( abs( B_new - B_old ) ) 
            
            if tol < tolerance:
                return B_new, X
        return B_new, X
    
    def predict_proba(self, Xtest):
        """ Predicts posterior probabilities for class 1 for observations whose featurevalues are in rows of the matrix Xtest"""
        Xtest = self.add_interactions(Xtest, self.interaction_ids)
        pi = (np.exp(np.dot(Xtest, self.beta.T)))/(1+np.exp(np.dot(Xtest, self.beta.T)))
        
        return pi
    
    def predict(self, Xtest):
        """ Assigns the predicted class (0 or 1) for observations whose feature values are in rows of the matrix Xtest"""
        pi = self.predict_proba(Xtest)
        y_pred = np.multiply([pi>0.5], 1)[0];
        
        return y_pred
    
    def get_params(self):
        """returns a list containing the estimated parameters beta """
    
        return self.beta