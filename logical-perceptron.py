from cgi import test
import numpy as np

test_inputs = [[0,0], [0,1], [1,0], [1,1]]

WEIGHT_1 = 1
WEIGHT_2 = 1
BIAS = -2

def perceptron(weights, bias, values):
    inputs = np.array(values)
    w = np.array(weights)
    return np.sum(inputs * w) + bias


for t in test_inputs:
    r = perceptron(
        [WEIGHT_1, WEIGHT_2],
        BIAS,
        t
    )
    print(str(t) + ' -> ' + str(r))