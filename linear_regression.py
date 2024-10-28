import numpy as np

class LinearRegression():
    
    def __init__(self, learning_rate=0.001, epochs=40):
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
        grad_w = -(2*(X.T).dot(Y-Y_pred))/n
        grad_b = -2*np.sum(Y-Y_pred)/n
        return grad_w, grad_b
    

    def MSE(self, Y, Y_pred):
        return (np.square(Y - Y_pred)).mean()
        

    def fit(self, X, Y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<n,m>): a matrix of floats with
                n rows (#samples) and m columns (#features)
            Y (array<n>): a vector of floats. The observed target variables
        """
        self.weights = np.zeros(1) 
        self.bias = 0
        for i in range(self.epochs):
            Y_pred = self.predict( X ) 
            grad_w, grad_b = self.gradient_descent(X, Y, Y_pred)  
            self.weights = self.weights - self.learning_rate*grad_w
            self.bias = self.bias - self.learning_rate*grad_b
            current_loss = self.MSE(Y, Y_pred)  
            self.losses.append(current_loss)
            # print(f'Epoch: {i}. Current loss: {current_loss}')


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
        return X*self.weights+ self.bias
        

