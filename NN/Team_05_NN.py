"""
Project - Part C: Neural Networks
@authors: Tal Eylon, Avihoo Menahem, Amihai Kalev, Ziv Levit
"""

############################################
##### Initial Setup
############################################

import numpy as np
import matplotlib.pyplot as plt
import math

RANDOM_SEED = 120
np.random.seed(RANDOM_SEED) # for results consistency

############################################
##### Network Architecture
############################################

n_input = 2 # number of input neurons
n_hidden = 5 # numbers of neurons in the hidden layer
n_output = 2 # number of output neurons

n_train = 800 # number of points for training the network
n_test = 100 # number of points for testing the network
n_valid = 100 # number of points for validating the network

bias =-1 # The bias we add for each layer except the output layer
learning_rate = 0.01

# bounds for the random point creation
lower_bound = 0
upper_bound = 1

# epochs
epochs = 100

############################################
##### Initialization
############################################

# A random input->hidden (n_input + bias)X(n_hidden) matrix with random numbers between -0.1 to 0.1
w_i_h = np.random.uniform(-0.1,0.1,(n_input+1,n_hidden)) # matrix dimensions: (n_input + bias)X(n_hidden)

# A random hidden->output (n_hidden + bias)X(n_output) matrix with random numbers between -0.1 to 0.1
w_h_o = np.random.uniform(-0.1,0.1,(n_hidden+1,n_output)) # matrix dimensions: (n_hidden + bias)X(n_output)

############################################
##### Functions Setup
############################################

## function g(x) - the main function we are trying to find
## x = (x1,x2)
def g(x): # x is a vector of 2X1
    a = np.log(x[0]/math.pow(12000,2))
    b = np.exp(x[1]/(12000*x[0]))
    c = math.pow(1.2,10)
    result = a + b + c
    return result

## function restrictions(x) - the restrictions function. Returns 1 if the point doesn't break any constraint, -1 otherwise.
def restrictions(x): # x is a vector of 2X1
    if (x[0] - 192000 <= 0) and (x[1] - 108330 <= 0) and (x[0] - 1.5*x[1] <= 0):
        return 1
    else:
        return -1

## Activation functions:
###### Activation function for the function we are trying to find
def f1(x,alpha=14): # ELU activation function. alpha is set to 14 as default
    if x>0:
        return x
    else:
        return alpha * (np.exp(x)-1)

def df1(x,alpha=14): # ELU derivative activation function. alpha is set to 14 as default
    if x>0:
        return 1
    else:
        return alpha * np.exp(x)

###### Activation function for the hidden layer and the restrictions function
def f2(x): # tanh activation function, x is a number!
    value = np.tanh(x)
    return value

def df2(x,value): # Derivative of tanh. value is the same value from the non-derivative function.
    return 1 - value**2


############################################
# Points creation for training, test, valid
############################################

def point_and_value(lower_bound,upper_bound):
    """
    This function creates a random point and calculates the relevant values
    :return:
    """
    point = [np.random.uniform(lower_bound,upper_bound) for i in range(n_input)]
    value = [g(point),restrictions(point)]
    return np.array([point,value])


train_set = [point_and_value(lower_bound,upper_bound) for i in range(n_train)]
test_set = [point_and_value(lower_bound,upper_bound) for i in range(n_test)]
validation_set = [point_and_value(lower_bound,upper_bound) for i in range(n_valid)]

############################################
# Training, Validation, Error
############################################
## training
def training(pair,w_i_h,w_h_o,option):
    """
    :param option: 1 - forward and backward pass, 2 - only forward pass
    :return:
    """
    # pair[0] - the point, pair[1] - the function g(x) value and the restrictions function value.

    ############### feed forward ###############

    point_with_bias = np.append(pair[0],bias) # the point is now 1X3 after adding the bias

    # matrices multiplication element-wise and function activation (f2): f = (point * w_i_h)
    # (1X3)*(3X5) -> (1X5)
    hidden_layer = f2(np.matmul(point_with_bias,w_i_h))

    # add the bias,
    # so now the hidden layer's vector will be 1X6
    hidden_layer = np.append(hidden_layer,[bias]) # bias

    # matrices multiplication element-wise and ReLU function (f1) activation, output's dimensions: 1X2
    before_output = np.matmul(hidden_layer,w_h_o)
    # run activation function for the function we are trying to find and the restrictions function:
    output = [f1(before_output[0]),f2(before_output[1])]

    err_function = output[0] - pair[1][0] # error output for the function we are trying to find
    err_restrictions = output[1] - pair[1][1] # error output for the restrictions function
    err = [err_function,err_restrictions]

    if option == 1: # for training the network with back propagation
        ############### backpropagation ###############
        output_derivatives = [df1(output[0]),df2(before_output[1],output[1])]
        dw_h_o = np.multiply(err,output_derivatives) #element-wise multiplication: dw_h_o = err * df(output)
        err_h = np.matmul(dw_h_o, w_h_o.T) # err_h = dw_h_o * w_h_o

        # dw_i_h = err_h * df2(hidden_layer) and cutting the bias
        h_after_df = df2(point_with_bias,hidden_layer[:-1])
        dw_i_h = np.multiply(err_h[:-1],h_after_df)

        ### update matrices
        w_h_o -= np.multiply(learning_rate, (dw_h_o.reshape(2,1) * hidden_layer.reshape(1,6)).reshape(6,2))

        w_i_h -= np.multiply(learning_rate,dw_i_h * np.transpose([point_with_bias]))

    return err

def validating(pair,w_i_h,w_h_o): # only feed forward for error calculation
    point_with_bias = np.append(pair[0],bias) # the point is now 1X3

    # matrices multiplication element-wise and function activation (f2): f = (point * w_i_h)
    # (1X3)*(3X5) -> (1X5)
    hidden_layer = f2(np.matmul(point_with_bias,w_i_h))

    # add the bias,
    # so now the hidden layer's vector will be 1X6
    hidden_layer = np.append(hidden_layer,[bias]) # bias

    # matrices multiplication element-wise and ReLU function (f1) activation, output's dimensions: 1X2
    before_output = np.matmul(hidden_layer,w_h_o)
    # run activation function for the function we are trying to find and the restrictions function:
    output = [f1(before_output[0]),f2(before_output[1])]

    err_function = output[0] - pair[1][0] # error output for the function we are trying to find
    err_restrictions = output[1] - pair[1][1] # error output for the restrictions function
    return [err_function,err_restrictions]

def testing(pair,w_i_h,w_h_o): # only feed forward for error calculation
    point_with_bias = np.append(pair[0],bias) # the point is now 1X3

    # matrices multiplication element-wise and function activation (f2): f = (point * w_i_h)
    # (1X3)*(3X5) -> (1X5)
    hidden_layer = f2(np.matmul(point_with_bias,w_i_h))

    # add the bias,
    # so now the hidden layer's vector will be 1X6
    hidden_layer = np.append(hidden_layer,[bias]) # bias

    # matrices multiplication element-wise and ReLU function (f1) activation, output's dimensions: 1X2
    before_output = np.matmul(hidden_layer,w_h_o)
    # run activation function for the function we are trying to find and the restrictions function:
    output = [f1(before_output[0]),f2(before_output[1])]

    err_function = (output[0] - pair[1][0])**2 # error output for the function we are trying to find
    err_restrictions = (output[1] - pair[1][1])**2 # error output for the restrictions function
    return [err_function,err_restrictions]

def forEachEpoch(w_i_h,w_h_o,pairs_lst,option):
    """
    This function works each epoch
    :param w_i_h: the w_i_h matrix
    :param w_h_o: the w_h_o matrix
    :param pairs_lst: the relevant points: can be either train_set or valid_set
    :param option: 1 - feed forward and back propagation only, 2 - training with error
    calculation, 3 - only validation with error calculation
    :return: MSE for training and MSE for validation
    """
    # initialize for each function a list
    lst_of_errors_function = []
    lst_of_errors_restrictions = []

    if option == 1: #training: forward and backward pass
        for i in range(len(pairs_lst)):
            training(pairs_lst[i],w_i_h,w_h_o,1) # forward and backward pass

    elif option == 2: #training forward only - no back propagation
        for i in range(len(pairs_lst)): # forward pass
            err_func,err_rest = training(pairs_lst[i], w_i_h, w_h_o,2)
            lst_of_errors_function.append(err_func**2)
            lst_of_errors_restrictions.append(err_rest**2)
        return [(sum(lst_of_errors_function) / len(lst_of_errors_function)),(sum(lst_of_errors_restrictions) / len(lst_of_errors_restrictions))]

    elif option == 3: #validation - feed forward only
        for i in range(len(pairs_lst)): # validation
            err_func, err_rest = validating(pairs_lst[i], w_i_h, w_h_o)
            lst_of_errors_function.append(err_func ** 2)
            lst_of_errors_restrictions.append(err_rest ** 2)
        return [(sum(lst_of_errors_function) / len(lst_of_errors_function)),(sum(lst_of_errors_restrictions) / len(lst_of_errors_restrictions))]

############################################
# Main program
############################################

# initialize a list of 3 lists for each function
err_history_function = [[],[],[]]
err_history_restrictions = [[],[],[]]

for epoch in range(epochs): # for each epoch
    # step 1: feed forward and back propagation
    forEachEpoch(w_i_h,w_h_o,train_set,1) # only train forwards and backwards

    # step 2: training with error calculation
    err_func,err_rest = forEachEpoch(w_i_h,w_h_o,train_set,2) # training
    err_history_function[0].append(err_func)
    err_history_restrictions[0].append(err_rest)

    # step 3: validation with error calculation
    err_func,err_rest = forEachEpoch(w_i_h,w_h_o,validation_set,3) # validation
    err_history_function[1].append(err_func)
    err_history_restrictions[1].append(err_rest)
    print("finished %s training & validating epoch\n" % (epoch))

# after the training & validation have finished:
# final step: calculating MSE of the test set
for epoch in range(epochs):
    err_func, err_rest = testing(test_set[epoch],w_i_h,w_h_o)
    err_history_function[2].append(err_func)
    err_history_restrictions[2].append(err_rest)
print("MSE over all test set: \nMain function: %s\nRestrictions function: %s" % ((sum(err_history_function[2]) / len(err_history_function[2])),(sum(err_history_restrictions[2]) / len(err_history_restrictions[2]))))

############################################
# Graphs
############################################

# first graph: for MSE over epochs for the main function we are trying to find
plt.plot([i for i in range(len(err_history_function[0]))], err_history_function[0], label="Training")
plt.plot([i for i in range(len(err_history_function[1]))], err_history_function[1], label="Validation")
plt.legend(ncol=2, borderaxespad=0.)
plt.title("MSE vs. Epoch - Main Function")
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.show()

# second graph: for MSE over epochs for the restrictions function
plt.plot([i for i in range(len(err_history_restrictions[0]))], err_history_restrictions[0], label="Training")
plt.plot([i for i in range(len(err_history_restrictions[1]))], err_history_restrictions[1], label="Validation")
plt.legend(ncol=2, borderaxespad=0.)
plt.title("MSE vs. Epoch - Restrictions Function")
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.show()


        
    



        

