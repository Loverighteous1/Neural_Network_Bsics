import numpy as np
import matplotlib.pyplot as plt
# generate data
var = 0.2
n = 800
class_0_a = var * np.random.randn(n//4,2)
class_0_b =var * np.random.randn(n//4,2) + (2,2)

class_1_a = var* np.random.randn(n//4,2) + (0,2)
class_1_b = var * np.random.randn(n//4,2) +  (2,0)

X = np.concatenate([class_0_a, class_0_b,class_1_a,class_1_b], axis =0)
Y = np.concatenate([np.zeros((n//2,1)), np.ones((n//2,1))])
X.shape, Y.shape
# shuffle the data
rand_perm = np.random.permutation(n)

X = X[rand_perm, :]
Y = Y[rand_perm, :]

X = X.T
Y = Y.T
X.shape, Y.shape

# train test split
ratio = 0.8
X_train = X [:, :int (n*ratio)]
Y_train = Y [:, :int (n*ratio)]

X_test = X [:, int (n*ratio):]
Y_test = Y [:, int (n*ratio):]

plt.scatter(X_train[0,:], X_train[1,:], c=Y_train[0,:])
plt.show()

class Clas_1hidlayer:

    def sigmoid(self,z):
       sigmoid_func = 1/(1+ np.exp(-z))
       return sigmoid_func

    def d_sigmoid(self,z):
       d_sig_z = self.sigmoid(z) * (1 - self.sigmoid(z))
       return d_sig_z

    def loss(self,y_pred, Y):
        loss_func = np.divide(- np.sum(Y * np.log(y_pred) + (1 - Y) * np.log(1 - y_pred)),Y.shape[1])
        return  loss_func
    #Iniatialisation of parameters
    def init_params(self):
        h0, h1, h2 = 2, 10, 1
       # Initialization with Xavier initialization technique 
        W1 = np.random.randn(h1, h0)
        W2 = np.random.randn(h2, h1)
        b1 = np.random.randn(h1, h2)
        b2 = np.random.randn(h2,h2)
        return W1, W2, b1, b2
    #Forward
    def forward_pass(self,X, W1,W2, b1, b2):
        Z1 = W1.dot(X) + b1
        A1 = self.sigmoid(Z1)
        Z2 = W2.dot(A1) + b2
        A2 = self.sigmoid(Z2)
        return A2, Z2, A1, Z1
    #Backward
    def backward_pass(self,X,Y, A2, Z2, A1, Z1, W1, W2, b1, b2):
        dl_A2 = -(Y - A2)/(A2 *(1 - A2))
        dA2_Z2 = self.d_sigmoid(Z2)
        dZ2_A1 = W2
        dA1_Z1 = self.d_sigmoid(Z1)
        dZ1_W1 = X.T
        dZ2_W2 = A1.T

        dW1 = (dZ2_A1.T * dl_A2 * dA2_Z2 * dA1_Z1) @ X.T
        dW2 = (dl_A2 * dA2_Z2 ) @ dZ2_W2
        db1 = ((dl_A2*dA2_Z2) @ (dZ2_A1.T * dA1_Z1).T).T
        db2 = np.sum(A2-Y,axis=1,keepdims=True)#dl_A2 @ dA2_Z2.T
        
        return dW1, dW2, db1, db2
    #Accuracy
    def accuracy(self,y_pred, y):
        y = y.reshape((-1,1))
        y_pred = y_pred.reshape((-1,1))
        return np.mean((y == y_pred))
    #Prediction
    def predict(self,X,W1,W2, b1, b2):
        A2, _, _, _ = self.forward_pass(X, W1,W2, b1, b2)
        A2 = (A2 >= 0.5).astype(int)
        return A2
    #Update Parameters
    def update(self,W1, W2, b1, b2,dW1, dW2, db1, db2, alpha ):
        W1 = W1 - alpha * dW1
        W2 = W2 - alpha * dW2
        b1 = b1 - alpha * db1
        b1 = b2 - alpha * db2

        return W1, W2, b1, b2
    #Plot
    def plot_decision_boundary(self,W1, W2, b1, b2):
        x = np.linspace(-0.5, 2.5,100 )
        y = np.linspace(-0.5, 2.5,100 )
        xv , yv = np.meshgrid(x,y)
        xv.shape , yv.shape
        X_ = np.stack([xv,yv],axis = 0)
        X_ = X_.reshape(2,-1)
        A2, Z2, A1, Z1 = self.forward_pass(X_, W1, W2, b1, b2)
        plt.figure()
        plt.scatter(X_[0,:], X_[1,:], c= A2)
        plt.show()
    #Training
    def training_loop(self,alpha,n_epochs):
        #alpha = 0.001
        W1, W2, b1, b2 = self.init_params()
        #n_epochs = 10000
        train_loss = []
        test_loss = []
        for i in range(n_epochs):
            ## forward pass
            A2, Z2, A1, Z1 = self.forward_pass(X_train, W1,W2, b1, b2)
            ## backward pass
            dW1, dW2, db1, db2 = self.backward_pass(X_train,Y_train, A2, Z2, A1, Z1, W1, W2, b1, b2)
            ## update parameters
            W1, W2, b1, b2 = self.update(W1, W2, b1, b2,dW1, dW2, db1, db2, alpha )

            ## save the train loss
            train_loss.append(self.loss(A2, Y_train))
            ## compute test loss
            A2, Z2, A1, Z1 = self.forward_pass(X_test, W1, W2, b1, b2)
            test_loss.append(self.loss(A2, Y_test))

            ## plot boundary
            if i %1000 == 0:
                self.plot_decision_boundary(W1, W2, b1, b2)

            ## plot train et test losses
        plt.plot(train_loss)
        plt.plot(test_loss)
        plt.show()

        y_pred = self.predict(X_train, W1, W2, b1, b2)
        train_accuracy = self.accuracy(y_pred, Y_train)
        print ("train accuracy :", train_accuracy)
        y_pred = self.predict(X_test, W1, W2, b1, b2)
        test_accuracy = self.accuracy(y_pred, Y_test)
        print ("test accuracy :", test_accuracy)
nn=Clas_1hidlayer()
nn.training_loop(0.001,10000)   