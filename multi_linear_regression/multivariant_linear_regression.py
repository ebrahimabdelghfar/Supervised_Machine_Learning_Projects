from ctypes.wintypes import HINSTANCE
import numpy as np
import matplotlib.pyplot as plt 

x_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

class multi_variant_linear_regression:
    def __init__(self) -> None:
        pass
    def compute_cost(self,x_vec,y_vec,w_vec,b):
        number_of_sample = np.shape(y_vec)[0]
        predicted = (np.dot(w_vec,np.transpose(x_vec)))+b
        sum_error_squared = np.sum((predicted-y_vec)**2)
        cost=((1/(2*number_of_sample))*sum_error_squared)
        return cost
        
    def compute_gradiant(self,x_vec,y_vec,w_vec,b):
        dj_dw=0
        dj_db=0

        number_of_sample = np.shape(y_vec)[0]
        predicted = np.dot(w_vec,np.transpose(x_vec)) + b
        err = np.reshape(predicted-y_vec,(1,np.shape(x_vec)[0]))
        dj_dw = (1/number_of_sample)*np.sum(np.dot(err,x_vec),axis=0)
        dj_db = (1/number_of_sample)*np.sum(predicted-y_vec)

        return dj_dw,dj_db

    def train_model(self,x,y,learning_rate,epoch,accurcy_of_model):
        cost_history=[]
        w=np.zeros((1,np.shape(x)[1]))
        b=0
        #back propagation value
        dj_dw = 0
        dj_db = 0
        #end
        for i in range(epoch):
            cost_history.append(self.compute_cost(x,y,w,b))
            dj_dw,dj_db=self.compute_gradiant(x,y,w,b)
            counter_of_repeatation=0
            # Update Parameters using w, b, alpha and gradient
            w= w - (learning_rate*dj_dw)
            b= b - (learning_rate*dj_db)
            if(i>1):    
                if(cost_history[i] == cost_history[i-1]):
                    counter_of_repeatation+=1
                if((abs(cost_history[i] - cost_history[i-1])<accurcy_of_model)):
                    print("converged")
                    break
            #end

        return w, b , cost_history 

multi_regression = multi_variant_linear_regression()

w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
b=785.1811367994083
w,b,history =multi_regression.train_model(x_train,y_train,learning_rate=5.0e-7,epoch=100000,accurcy_of_model=1e-5)
#print(compute_cost(x_train,y_train,w_init,b))
plt.plot(history)
plt.show()