import numpy as np
import matplotlib.pyplot as plt 

x_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

def compute_cost(x_vec,y_vec,A,B,C,D):
    number_of_sample = np.shape(y_vec)[0]
    predicted = (np.dot(A,np.transpose(x_vec**3)))+(np.dot(B,np.transpose(x_vec**2)))+(np.dot(C,np.transpose(x_vec)))+D
    sum_error_squared = np.sum((predicted-y_vec)**2)
    cost=((1/(2*number_of_sample))*sum_error_squared)
    return cost

def compute_gradiant(x_vec,y_vec,A,B,C,D):
    dj_dw=0
    dj_db=0

    number_of_sample = np.shape(y_vec)[0]
    predicted = (np.dot(A,np.transpose(x_vec**3)))+(np.dot(B,np.transpose(x_vec**2)))+(np.dot(C,np.transpose(x_vec)))+D
    err = np.reshape(predicted-y_vec,(1,np.shape(x_vec)[0]))
    dj_dA = (1/number_of_sample)*np.sum(np.dot(err,x_vec**3),axis=0)
    dj_dB = (1/number_of_sample)*np.sum(np.dot(err,x_vec**2),axis=0)
    dj_dC = (1/number_of_sample)*np.sum(np.dot(err,x_vec),axis=0)
    dj_dD = (1/number_of_sample)*np.sum(err)

    return dj_dA,dj_dB,dj_dC,dj_dD

def train_model(x,y,learning_rate,epoch):
    cost_history=[]
    A=np.zeros((1,np.shape(x)[1]))
    B=np.zeros((1,np.shape(x)[1]))
    C=np.zeros((1,np.shape(x)[1]))
    D=0
    #back propagation value
    dj_dA=0
    dj_dB=0
    dj_dC=0
    dj_dD=0
    #end
    for i in range(epoch):
        cost_history.append(compute_cost(x,y,A,B,C,D))
        dj_dA,dj_dB,dj_dC,dj_dD=compute_gradiant(x,y,A,B,C,D)
        counter_of_repeatation=0
        # Update Parameters using w, b, alpha and gradient
        A = A - (learning_rate*dj_dA)
        B = B - (learning_rate*dj_dB)
        C = C - (learning_rate*dj_dC)
        D = D - (learning_rate*dj_dD)
    return A, B , C , D , cost_history 

A, B , C , D , cost_history =train_model(x_train,y_train,5e-15,150000)
fig,axs=plt.subplots(2)
axs[0].plot(cost_history)
print(cost_history[-1])
plt.show()



