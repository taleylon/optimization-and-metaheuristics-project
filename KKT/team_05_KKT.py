import math
import numpy as np

########################################################
#################### TEAM 05 ###########################
########################################################

#############################################
#### Functions and restriction functions
#############################################

def f(x): # The function
    '''
    The function you are minimizing / maximizing
    Example:
    x1 = x[0] # math.sqrt(math.sqrt(x[0]))/2
    x2 = x[1]
    x3 = x[2]
    return x[0]*x[1] - math.pow(x[2])
    '''
    a = np.log(x[0]/math.pow(12000,2))
    b = np.exp(x[1]/(12000*x[0]))
    c = math.pow(1.2,10)
    result = a + b + c
    return result


def df(x): # the gradient
    '''
    :param x:
    :return: a vector, the size of the x
    '''
    a = [0 for i in range(len(x))]
    a[0] = (1/x[0]) + np.exp(x[1]/(12000*x[0]))*(-(x[1]/(12000*math.pow(x[0],2))))
    a[1] = np.exp(x[1]/(12000*x[0]))*(1/(12000*x[0]))

    return a

# Your restrictions here
def g1(x):
    # Example:
    # return x[0] + x[1] + x[2] - 30
    return x[0] - 192000

def g2(x):
    return x[1] - 108330

def g3(x):
    return x[0] - 1.5*x[1]

def dg1(x): # the restrictions gradients
    '''
    :param x:
    :return: a vector, the size of the x
    '''
    a = [0 for i in range(len(x))]
    a[0] = 1
    a[1] = 0
    return a

def dg2(x):
    a = [0 for i in range(len(x))]
    a[0] = 0
    a[1] = 1

    return a

def dg3(x):
    a = [0 for i in range(len(x))]
    a[0] = 1
    a[1] = -1.5

    return a

### CREATE LIST OF FUNCTIONS IN ORDER TO HAVE A GENERIC PROCESS ###

g = [g1,g2,g3]
dg = [dg1,dg2,dg3]



#############################################
#### Implementation of Penalty method
#############################################

# You can write both, but *only one* will be checked.

# Penalty function
def alpha(x, mu): # penalty function
    # Build the vector a to include the relevant value from each restriction function
    a = [0 for i in range(len(g))]
    for i in range(len(g)):
        if g[i](x) > 0:
            a[i] = math.pow(g[i](x),2)
        else:
            a[i] = 0

    # sum the vector a to calculate the value of alpha
    return sum(a)

def dalpha(x, mu):
    # gradient of alpha equals to the sum of the gradient of each restriction function
    vector = [0 for i in range(len(x))]

    for i in range(len(dg)):
        current_dg = dg[i](x)
        for j in range(len(current_dg)):
            vector[j] += current_dg[j]

    return vector


#---
def theta(x, mu): # The main function to minimize
    return f(x) + mu*alpha(x, mu)

def dtheta(x, mu): # the gradient of theta
    # equals to the vector: df(x) + mu*dalpha(x)
    current_df = df(x)
    current_dalpha = dalpha(x,mu)

    vector = [0 for j in range(len(x))]
    for i in range(len(vector)):
      vector[i] = current_df[i] + (mu*current_dalpha[i])
    return vector



def h(y, x, mu, d): # the single-dimenstion funciton to minimize
    '''
    :param y: distance along theta's gradient
    :param x: starting point
    :param mu:
    :return: the value of theta in the appropriate point
    '''
    vector = [0 for i in range(len(x))]
    for i in range(len(vector)):
        vector[i] = x[i] + y*d[i] # the y is the distance in the direction d[i]

    current_value = theta(vector, mu) # calculate the theta value

    return current_value

# Single dimension optimization: for example, Golden Section
# will require a loop over a single dimension to find the minimium
def singleDimensionOptimization(x, mu, d):
    '''
    :param x: starting point
    :param mu:
    :return: the value of x that gives the minimum of h
    '''
    ###### Golden Section Implementation ######
    interv = [-0.1,100] # the interval for y

    ## first only calculations
    a = interv[0] + 0.382*(interv[1]-interv[0])
    b = interv[0] + 0.618*(interv[1]-interv[0])

    ## retrieve the single-dimension function values for a,b in the given direction d
    fa = h(a, x, mu, d)
    fb = h(b, x, mu, d)

    N=1000  # for 1000 iterations
    for i in range(N):
        if fa<fb: #
            interv[1] = b
            b = a
            a = interv[0] + 0.382*(interv[1]-interv[0])
        else:
            interv[0] = a
            a = b
            b = interv[0] + 0.618*(interv[1]-interv[0])
        fa = h(a, x, mu, d)
        fb = h(b, x, mu, d)

    # return the mean of the interval
    # as the value of y, the distance in which the theta is getting minimum

    return sum(interv)/len(interv)


# THE FUNCTION
# Progressing until convergence and returning the answer
# - Go over the appropriate mu
# - for each mu, find the point that minimizes theta
# - check if solves the Mathematical Problem
# *** print all values along the loop: x, mu, f, alpha/B, theta

def norma(x1,x2): # stopping criterion
    norma_vec = [0 for i in range(len(x1))]
    for i in range(len(x1)):
        norma_vec[i] = x1[i] - x2[i]

    a = [0 for i in range(len(x1))]
    for i in range(len(a)):
        a[i] += norma_vec[i]**2

    # calculate the norma, given two points.
    return sum(a)**(1/2)


#############################################
#### The main optimize function
#############################################
def Optimize():
    global x
    epsilon = 0.01
    k=0
    xk = x # initial first point
    new_x = [0 for i in range(len(xk))] # the x that gets updated in the iterations, in the same length as xk
    print("[k , mu , [x1 , x2] , f(x) , alpha(x) , theta(x) , mu*alpha(x)]")
    for mu in [0, 0.1, 1, 10, 100, 1000, 10000]:
        # calculate current values for theta, alpha, mu*alpha, f(x) for the current iteration
        current_theta = theta(xk, mu)
        current_alpha = alpha(xk, mu)
        current_mualpha = mu*alpha(xk,mu)
        current_f = f(xk)

        # counter for iterations inside the cyclic coordination loop
        stop = 0

        # Begin with the cyclic coordination method:
        while True:
            for i in range(len(xk)): # for each direction, calculate the relevant value in the new_x
                current_d = [0 for i in range(len(xk))] # the direction vector
                current_d[i] = 1 # the current direction, according to i
                current_lambda = singleDimensionOptimization(xk, mu, current_d) # get the distance in which theta is
                # getting minimum, using Golden Section result
                new_x[i] = xk[i] + current_lambda*current_d[i] # update the new_x in the vector according
                # to the current direction

            current_norma = norma(xk, new_x) # stopping criterion
            xk = new_x # now x_k = x_k+1
            new_x = [0 for i in range(len(xk))] # new_x is again an empty vector with 0
            stop+=1 # increase the current iterations

            ## stop the loop if: 1. stopping criterion is small enough 2. we have reached 10 iterations
            if current_norma < epsilon or stop==10: # is it good enough?
                break

        result = [k, mu, xk, current_f, current_alpha, current_theta, current_mualpha] # current results
        print(result)
        print("Number of iterations for cyclic coordination method: ",stop)
        k+=1 # increase the penalty method iterations by 1

x = [10,5]
Optimize()