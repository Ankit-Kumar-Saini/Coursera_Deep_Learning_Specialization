import numpy as np
from test_utils import single_test, multiple_test

def forward_propagation_test(target):
    x, theta = 2, 4
    expected_output = 8
    test_cases = [
        {
            "name": "equation_output_check",
            "input": [x, theta],
            "expected": expected_output,
            "error": "Wrong output"
        } 
    ]
    
    single_test(test_cases, target) 
        
def backward_propagation_test(target):
    x, theta = 2, 4
    expected_output = 2
    test_cases = [
        {
            "name": "equation_output_check",
            "input": [x, theta],
            "expected": expected_output,
            "error": "Wrong output"
        } 
    ]
    
    single_test(test_cases, target)

def gradient_check_test(target):
    x, theta = 2, 4
    expected_output = 2.919335883291695e-10
    test_cases = [
        {
            "name": "equation_output_check",
            "input": [x, theta],
            "expected": expected_output,
            "error": "Wrong output"
        } 
    ]
    
    single_test(test_cases, target)
    
def gradient_check_n_test(target, parameters, gradients, X, Y):
    expected_output_1 = 0.2850931567761623
    expected_output_2 = 1.1890913024229996e-07
    test_cases = [
        {
            "name": "multiple_equation_output_check",
            "input": [parameters, gradients, X, Y],
            "expected": [expected_output_1, expected_output_2],
            "error": "Wrong output"
        } 
    ]
    
    single_test(test_cases, target)
    
def predict_test(target):
    np.random.seed(1)
    X = np.random.randn(2, 3)
    parameters = {'W1': np.array([[-0.00615039,  0.0169021 ],
        [-0.02311792,  0.03137121],
        [-0.0169217 , -0.01752545],
        [ 0.00935436, -0.05018221]]),
     'W2': np.array([[-0.0104319 , -0.04019007,  0.01607211,  0.04440255]]),
     'b1': np.array([[ -8.97523455e-07],
        [  8.15562092e-06],
        [  6.04810633e-07],
        [ -2.54560700e-06]]),
     'b2': np.array([[  9.14954378e-05]])}
    expected_output = np.array([[True, False, True]])

    test_cases = [
        {
            "name":"datatype_check",
            "input": [parameters, X],
            "expected": expected_output,
            "error":"Data type mismatch"
        },
        {
            "name": "shape_check",
            "input": [parameters, X],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [parameters, X],
            "expected": expected_output,
            "error": "Wrong output"
        } 
    ]
    
    single_test(test_cases, target)

    
