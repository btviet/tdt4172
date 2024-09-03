import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.1, epochs=40):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights, self.bias = None, None
        self.losses, self.train_accuracies = [], []
    
    def sigmoid_function(self, x):
        return 1/(1+np.e**(-x))
    
    def _compute_loss(self, Y, Y_pred):
        print("wo")

    def compute_gradients(self, X, Y, Y_pred):
        print("wo")

    def fit(self, X, Y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<n,m>): a matrix of floats with
                n rows (#samples) and m columns (#features)
            Y (array<n>): a vector of floats. The observed target variables
        """
        n = X.shape[0] # Number of data points
        
        self.weights = np.zeros(1) # np.zeros(n)

        self.bias = 0
        for i in range(self.epochs):
            Y_pred = self.predict( X ) 
            grad_w, grad_b = self.gradient_descent(X, Y, Y_pred)  
            self.weights = self.weights - self.learning_rate*grad_w
            self.bias = self.bias - self.learning_rate*grad_b
            current_loss = self.MSE(Y, Y_pred)  
            self.losses.append(current_loss)
            # print(f'Epoch: {i}. Current loss: {current_loss}')

import pandas as pd
data = pd.read_csv('mission2.csv')
train = data[data['split'] == 'train']
test = data[data['split'] == 'test']


lr = LogisticRegression()
lr.fit(train[['x0', 'x1']], train['y'])

