import numpy as np

LEARNING_RATE = 0.1

test_inputs = [[1.0,1.0]]
desired_outputs= [1.0]

weight_1 = 3
weight_2 = 4
bias = -10

def perceptron(weights, bias, input):
    return np.sum(input * weights) + bias

def calibrate_perceptron(weights, bias, inputs, desired_outputs):
    
    w = weights
    b = bias
    is_input_calibrated = [False for _ in inputs]

    print('WEIGHTS: ' + str(w))
    print('BIAS: ' + str(b))

    step = 0

    while(not np.all(is_input_calibrated)):
        step += 1

        print()
        print('STEP: ' + str(step))

        for i, input in enumerate(inputs):
            print('input: ' + str(i) + ' -> ' + str(input))
            current_result = perceptron(w, b, input)
            print('result: ' + str(current_result))
            print('desired: ' + str(desired_outputs[i]))
            if (desired_outputs[i] == current_result):
                print('CALIBRATED!')
                is_input_calibrated[i] = True

            if (current_result < desired_outputs[i]):
                print('less than desired, adjusting up')
                w = w + (input * LEARNING_RATE)
                b = b + LEARNING_RATE
                print('new weights: ' + str(w))
                print('new bias: ' + str(b))
                is_input_calibrated[i] = False

            if (current_result > desired_outputs[i]):
                print('less than desired, adjusting up')
                w = w - (input * LEARNING_RATE)
                b = b - LEARNING_RATE
                print('new weights: ' + str(w))
                print('new bias: ' + str(b))
                is_input_calibrated[i] = False

    return (w, b)

for t in test_inputs:
    r = perceptron(
        np.array([weight_1, weight_2]),
        bias,
        np.array(t)
    )
    print(str(t) + ' -> ' + str(r))

print('calibrating perceptron')
new_weights, new_bias = calibrate_perceptron(
    np.array([weight_1, weight_2], dtype=np.float64), 
    bias, 
    np.array(test_inputs), 
    desired_outputs
)
print()
print('--------------------------------------------------------------')
print('new weights: ' + str(new_weights))
print('new bias: ' + str(new_bias))

for t in test_inputs:
    r = perceptron(
        np.array([weight_1, weight_2], dtype=np.float64),
        bias,
        np.array(t)
    )
    print(str(t) + ' -> ' + str(r))
