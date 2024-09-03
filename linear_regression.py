import numpy as np

class LinearRegression():
    
    def __init__(self, learning_rate=0.00001, epochs=40):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights, self.bias = None, None
        self.losses, self.train_accuracies = [], []
        
    def gradient_descent(self, X, Y, Y_pred):
        """
        Args:
        X: Array with n samples with m features. Observed values.
        Y: Array with target variable corresponding to X
        Y_pred: The prediction of the target variable
        """
        n = X.shape[0]
        grad_w = -2*np.dot(X.T, Y-Y_pred)/n
        grad_b = -2*(Y-Y_pred)/n
        return grad_w, grad_b
    
    def MSE(self, Y, Y_pred):
        n = np.shape(Y)
        return (Y-Y_pred)/n
    

    def fit(self, X, Y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<n,m>): a matrix of floats with
                n rows (#samples) and m columns (#features)
            Y (array<n>): a vector of floats. The observed target variables
        """
        n = X.shape[0]
        self.weights = np.zeros([n])
        self.bias = 0
        for i in range(self.epochs):
            Y_pred = np.dot(X, self.weights) + self.bias
            grad_w, grad_b = self.gradient_descent(X, Y, Y_pred)  
            self.weights = self.weights - self.learning_rate*grad_w
            self.bias = self.bias - self.learning_rate*grad_b
            current_loss = self.MSE(Y, Y_pred)  
            print(f'Epoch: {i}. Current loss: {current_loss}')
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats
        """
        # TODO: Implement
        raise NotImplementedError("The predict method is not implemented yet.")

import pandas as pd
data = pd.read_csv('mission1.csv')
lr = LinearRegression()
lr.fit(data['Net_Activity'], data['Energy'])

#lr.predict(data['Net_Activity'])