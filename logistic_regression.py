import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=40):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights, self.bias = None, None
        self.losses, self.train_accuracies = [], []
    
    def _sigmoid_function(self, x):
        return 1/(1+np.e**(-x))
    
    def _sigmoid(self, X):
        return np.array([self._sigmoid_function(x) for x in X])

    def _compute_loss(self, Y, Y_pred):
        #print(Y, Y.shape)
        #print(Y.T)
        #print(np.log(self._sigmoid(Y_pred)))
        ledd0 = -Y.T.dot(np.log(self._sigmoid(Y_pred)))
        ledd1 = (1-Y).T.dot(np.log(1-self._sigmoid(Y_pred)))
        return np.mean(ledd0+ledd1)

    def compute_gradients(self, X, Y, Y_pred):
        n = X.shape[0] # Number of samples
        grad_w = np.dot(X.T, (Y_pred-Y))/n
        grad_b = np.sum(Y_pred-Y)/n
        return grad_w, grad_b
    
    def accuracy(self, Y, Y_pred):
        #print("Y:\n", Y)
        #print("Y_pred. \n", Y_pred)
        #print(Y == Y_pred)
        #print("accuracy: ", np.mean(Y == Y_pred))
        return np.mean(Y == Y_pred)

    def fit(self, X, Y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<n,m>): a matrix of floats with
                n rows (#samples) and m columns (#features)
            Y (array<n>): a vector of floats. The observed target variables
        """
        n = X.shape[0] # Number of data points
        m = X.shape[1] # Number of features
        self.weights = np.zeros(m) 

        self.bias = 0
        for i in range(self.epochs):
            # lin_model = np.dot(X, self.weights) + self.bias
            z = np.dot(X, self.weights) + self.bias # Output from linear model
            Y_pred = self._sigmoid(z)
            grad_w, grad_b = self.compute_gradients(X, Y, Y_pred)  
            self.weights = self.weights - self.learning_rate*grad_w
            self.bias = self.bias - self.learning_rate*grad_b
            current_loss = self._compute_loss(Y, Y_pred)  
            self.losses.append(current_loss)
            Y_binary = [1 if _y > 0.5 else 0 for _y in Y_pred]
            current_accuracy = self.accuracy(Y, Y_binary)
            #print("current_accuracy: ", current_accuracy)
            self.train_accuracies.append(current_accuracy)
            #print(self.train_accuracies)
            #print(f'Epoch: {i}. Current loss: {current_loss}')
            #print("Weight: ", self.weights, ". Bias: ", self.bias)

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
            z = np.dot(X, self.weights) + self.bias #X*self.weights+ self.bias
            y_sigmoid = self._sigmoid(z)
            #print("X:", X)
            #print("weights:", self.weights)
            #print("bias: ", self.bias)
            #print("z:",  z)
            return [1 if _y > 0.5 else 0 for _y in y_sigmoid]
            #return np.dot(X, self.weights) + self.bias
            




if __name__ == "__main__" :      
    import pandas as pd
    data = pd.read_csv('mission2.csv')
    train = data[data['split'] == 'train']
    test = data[data['split'] == 'test']


    lr = LogisticRegression()
    lr.fit(train[['x0', 'x1']], train['y'])

    predictions = lr.predict(test[['x0', 'x1']])
    print("Accuracy: ", lr.accuracy(test['y'], predictions))

