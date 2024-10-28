import numpy as np 
  
import pandas as pd 
  
  
import matplotlib.pyplot as plt 
  
# Linear Regression 
  
class LinearRegression() : 
      
    def __init__( self, learning_rate, iterations ) : 
          
        self.learning_rate = learning_rate 
          
        self.iterations = iterations 
          
    # Function for model training 
              
    def fit( self, X, Y ) : 
          
        # no_of_training_examples, no_of_features 

        self.m = np.shape(X)[0]
        self.n = 1
        print(self.m, self.n)
        # weight initialization 
          
        self.W = np.zeros( self.n ) 
          
        self.b = 1
          
        self.X = X 
          
        self.Y = Y 
          
          
        # gradient descent learning 
                  
        for i in range( self.iterations ) : 
            
            self.update_weights() 
            print(i, self.W, self.b)  
        return self
      
    # Helper function to update weights in gradient descent 
      
    def update_weights( self ) : 
             
        Y_pred = self.predict( self.X ) 
        current_loss = np.mean(np.square(Y_pred - self.Y))
        print("loss: ", current_loss)
        # calculate gradients   
      
        dW = - ( 2 * ( self.X.T ).dot( self.Y - Y_pred )  ) / self.m 
       
        db = - 2 * np.sum( self.Y - Y_pred ) / self.m  
          
        # update weights 
      
        self.W = self.W - self.learning_rate * dW 
      
        self.b = self.b - self.learning_rate * db 
          
        return self

    # Hypothetical function  h( x )  
      
    def predict( self, X ) : 
      
        return X*self.W + self.b 
     
  
# driver code 
  
def main() : 
      
    # Importing dataset 
 
    data = pd.read_csv('mission1.csv')
    lr = LinearRegression(learning_rate=0.0001, iterations=40)
    lr.fit(data['Net_Activity'], data['Energy'])

      
      

     
if __name__ == "__main__" :  
      
    main()