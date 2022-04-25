import numpy as np

def act(x):
    return 0 if x < 0.5 else 1

def go(house, rock, attr):
    x = np.array([house, rock, attr])
    w11 = [0.3, 0.3, 0]
    w12 = [0.4, -0.5, 1]
    weight1 = np.array([w11, w12]) # matrix 2*3
    weight2 = np.array([-1, 1]) # vector 1*3

    sum_hidden = np.dot(weight1, x)    # calculate the sum at the inputs of neurons in the hidden layer
    print("The values of the sums on the neurons of the hidden layer: "+str(sum_hidden))

    out_hidden = np.array([act(x) for x in sum_hidden])
    print("Values at the outputs of neurons in the hidden layer: "+str(out_hidden))

    sum_end = np.dot(weight2, out_hidden)
    y = act(sum_end)

    return y

house = 1
rock = 0
attr = 1

res = go(house, rock, attr)
if res == 1:
    print("I like you")
else:
    print("Let's call")