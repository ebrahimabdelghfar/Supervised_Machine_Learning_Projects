import numpy as np
import matplotlib.pyplot as plt 

x_train=np.array( [6.1101, 5.5277, 8.5186, 7.0032, 5.8598, 8.3829, 7.4764, 8.5781, 6.4862, 5.0546, 5.7107, 14.164, 5.734, 8.4084, 5.6407, 5.3794, 6.3654, 5.1301, 6.4296, 7.0708, 6.1891, 20.27, 5.4901, 6.3261, 5.5649, 18.945, 12.828, 10.957, 13.176, 22.203, 5.2524, 6.5894, 9.2482, 5.8918, 8.2111, 7.9334, 8.0959, 5.6063, 12.836, 6.3534, 5.4069, 6.8825, 11.708, 5.7737, 7.8247, 7.0931, 5.0702, 5.8014, 11.7, 5.5416, 7.5402, 5.3077, 7.4239, 7.6031, 6.3328, 6.3589, 6.2742, 5.6397, 9.3102, 9.4536, 8.8254, 5.1793, 21.279, 14.908, 18.959, 7.2182, 8.2951, 10.236, 5.4994, 20.341, 10.136, 7.3345, 6.0062, 7.2259, 5.0269, 6.5479, 7.5386, 5.0365, 10.274, 5.1077, 5.7292, 5.1884, 6.3557, 9.7687, 6.5159, 8.5172, 9.1802, 6.002, 5.5204, 5.0594, 5.7077, 7.6366, 5.8707, 5.3054, 8.2934, 13.394, 5.4369])
y_train=np.array([17.592, 9.1302, 13.662, 11.854, 6.8233, 11.886, 4.3483, 12.0, 6.5987, 3.8166, 3.2522, 15.505, 3.1551, 7.2258, 0.71618, 3.5129, 5.3048, 0.56077, 3.6518, 5.3893, 3.1386, 21.767, 4.263, 5.1875, 3.0825, 22.638, 13.501, 7.0467, 14.692, 24.147, -1.22, 5.9966, 12.134, 1.8495, 6.5426, 4.5623, 4.1164, 3.3928, 10.117, 5.4974, 0.55657, 3.9115, 5.3854, 2.4406, 6.7318, 1.0463, 5.1337, 1.844, 8.0043, 1.0179, 6.7504, 1.8396, 4.2885, 4.9981, 1.4233, -1.4211, 2.4756, 4.6042, 3.9624, 5.4141, 5.1694, -0.74279, 17.929, 12.054, 17.054, 4.8852, 5.7442, 7.7754, 1.0173, 20.992, 6.6799, 4.0259, 1.2784, 3.3411, -2.6807, 0.29678, 3.8845, 5.7014, 6.7526, 2.0576, 0.47953, 0.20421, 0.67861, 7.5435, 5.3436, 4.2415, 6.7981, 0.92695, 0.152, 2.8214, 1.8451, 4.2959, 7.2029, 1.9869, 0.14454, 9.0551, 0.61705])

class one_variant_linear_regression:
    def __init__(self):
        self
        pass
    def compute_cost(self,input_set,target_set,W,B):
        #intilization of variable
        train_example=(np.shape(y_train)[0])
        total_cost = 0
        error_squared = 0
        #end

        #calculate variable
        predicted = (W*input_set)+B
        error_squared = (predicted - target_set)**2
        #end

        #computend total cost
        total_cost = (1/(2*train_example))*(np.sum(error_squared))
        #end

        return total_cost

    def compute_gradiant_decent(self,input_set,target_set,W,B):
        #intilization of variable
        train_example=(np.shape(y_train)[0])
        dj_dw = 0
        dj_db = 0
        #end

        #compute the variable
        predicted= ((W*input_set)+B)
        dj_dw = (1/train_example)*np.sum((predicted-target_set)*input_set)
        dj_db = (1/train_example)*np.sum(predicted-target_set)
        #end

        return dj_dw ,dj_db    

    def train_the_model(self,input_set , target_set , learning_rate , num_iters,accurcy_of_model):
            
        # An array to store cost J and w's at each iteration — primarily for graphing later
        J_history = []
        w_history = []
        w_in=0
        b_in=0
        counter_of_repeatation=0
        #end
        # Calculate the gradient and update the parameters
        for i in range(num_iters):
            dj_dw, dj_db = self.compute_gradiant_decent(input_set, target_set, w_in, b_in)  
            # Update Parameters using w, b, alpha and gradient
            w_in = w_in - learning_rate * dj_dw               
            b_in = b_in - learning_rate * dj_db   
            if i<num_iters:      # prevent resource exhaustion 
                cost = self.compute_cost(input_set, target_set, w_in, b_in)
                J_history.append(cost)
                w_history.append(w_in)
            #check for convergence 
            if(i>1):    
                if(w_history[i] == w_history[i-1]):
                    counter_of_repeatation+=1
                if((abs(J_history[i] - J_history[i-1])<accurcy_of_model)):
                    print("converged")
                    break
            #end

        return w_in , b_in, J_history, w_history #return w and J,w history for graphing 


# some gradient descent settings
iterations = 1500
alpha = 0.01

one_variant_regression = one_variant_linear_regression()


w,b,cost_history,w_history = one_variant_regression.train_the_model(x_train , y_train,alpha, iterations,accurcy_of_model=1e-5)

predict1 = 3.5 * w + b
print('For population = 35,000, we predict a profit of $%.2f' % (predict1*10000))

predict2 = 7.0 * w + b
print('For population = 70,000, we predict a profit of $%.2f' % (predict2*10000))

plt.plot(cost_history)
plt.show()

    