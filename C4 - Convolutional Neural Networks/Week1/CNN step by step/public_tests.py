import numpy as np
from test_utils import single_test, multiple_test
from outputs import *


def zero_pad_test(target):
    np.random.seed(1)
    x = np.random.randn(4, 3, 3, 2)
    pad = 2
    expected_output = expected_output = np.array(zero_pad_output0)

    test_cases = [
        {
            "name": "datatype_check",
            "input": [x, pad],
            "expected": expected_output,
            "error":"Datatype mismatch."
        },
        {
            "name": "equation_output_check",
            "input": [x, pad],
            "expected": expected_output,
            "error": "Wrong output"
        }
    ]

    single_test(test_cases, target)


def conv_single_step_test(target):

    np.random.seed(1)
    a_slice_prev = np.random.randn(4, 4, 3)
    W = np.random.randn(4, 4, 3)
    b = np.random.randn(1, 1, 1)
    expected_output = np.float64(-6.999089450680221)
    test_cases = [
        {
            "name": "datatype_check",
            "input": [a_slice_prev, W, b],
            "expected": expected_output,
            "error":"Datatype mismatch"
        },
        {
            "name": "shape_check",
            "input": [a_slice_prev, W, b],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [a_slice_prev, W, b],
            "expected": expected_output,
            "error": "Wrong output"
        }
    ]

    multiple_test(test_cases, target)


def conv_forward_test(target):
    # Test 1
    A_prev = np.random.randn(2, 5, 7, 4)
    W = np.random.randn(3, 3, 4, 8)
    b = np.random.randn(1, 1, 1, 8)
    
    Z, cache_conv = target(A_prev, W, b, {"pad" : 3, "stride": 1})
    Z_shape = Z.shape
    assert Z_shape[0] == A_prev.shape[0], f"m is wrong. Current: {Z_shape[0]}.  Expected: {A_prev.shape[0]}"
    assert Z_shape[1] == 9, f"n_H is wrong. Current: {Z_shape[1]}.  Expected: 9"
    assert Z_shape[2] == 11, f"n_W is wrong. Current: {Z_shape[2]}.  Expected: 11"
    assert Z_shape[3] == W.shape[3], f"n_C is wrong. Current: {Z_shape[3]}.  Expected: {W.shape[3]}"

    # Test 2 
    Z, cache_conv = target(A_prev, W, b, {"pad" : 0, "stride": 2})
    assert(Z.shape == (2, 2, 3, 8)), "Wrong shape. Don't hard code the pad and stride values in the function"
    
    # Test 3
    W = np.random.randn(5, 5, 4, 8)
    b = np.random.randn(1, 1, 1, 8)
    Z, cache_conv = target(A_prev, W, b, {"pad" : 6, "stride": 1})
    Z_shape = Z.shape
    print(Z_shape)
    assert Z_shape[0] == A_prev.shape[0], f"m is wrong. Current: {Z_shape[0]}.  Expected: {A_prev.shape[0]}"
    assert Z_shape[1] == 13, f"n_H is wrong. Current: {Z_shape[1]}.  Expected: 9"
    assert Z_shape[2] == 15, f"n_W is wrong. Current: {Z_shape[2]}.  Expected: 11"
    assert Z_shape[3] == W.shape[3], f"n_C is wrong. Current: {Z_shape[3]}.  Expected: {W.shape[3]}"


    np.random.seed(1)
    A_prev = np.random.randn(2, 5, 7, 4)
    W = np.random.randn(3, 3, 4, 8)
    b = np.random.randn(1, 1, 1, 8)
    hparameters = {"pad": 1,
                   "stride": 2}
    expected_Z = np.array(conv_forward_output0)
    expected_cache = (A_prev, W, b, hparameters)
    expected_output = (expected_Z, expected_cache)
    test_cases = [
        {
            "name": "datatype_check",
            "input": [A_prev, W, b, hparameters],
            "expected": expected_output,
            "error":"Datatype mismatch"
        },
        {
            "name": "shape_check",
            "input": [A_prev, W, b, hparameters],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [A_prev, W, b, hparameters],
            "expected": expected_output,
            "error": "Wrong output"
        }
    ]

    multiple_test(test_cases, target)


def pool_forward_test(target):
    
    # Test 1
    A_prev = np.random.randn(2, 5, 7, 3)
    A, cache = target(A_prev, {"stride" : 2, "f": 2}, mode = "average")
    A_shape = A.shape
    assert A_shape[0] == A_prev.shape[0], f"Test 1 - m is wrong. Current: {A_shape[0]}.  Expected: {A_prev.shape[0]}"
    assert A_shape[1] == 2, f"Test 1 - n_H is wrong. Current: {A_shape[1]}.  Expected: 2"
    assert A_shape[2] == 3, f"Test 1 - n_W is wrong. Current: {A_shape[2]}.  Expected: 3"
    assert A_shape[3] == A_prev.shape[3], f"Test 1 - n_C is wrong. Current: {A_shape[3]}.  Expected: {A_prev.shape[3]}"
    
    # Test 2
    A_prev = np.random.randn(4, 5, 7, 4)
    A, cache = target(A_prev, {"stride" : 1, "f": 5}, mode = "max")
    A_shape = A.shape
    assert A_shape[0] == A_prev.shape[0], f"Test 2 - m is wrong. Current: {A_shape[0]}.  Expected: {A_prev.shape[0]}"
    assert A_shape[1] == 1, f"Test 2 - n_H is wrong. Current: {A_shape[1]}.  Expected: 1"
    assert A_shape[2] == 3, f"Test 2 - n_W is wrong. Current: {A_shape[2]}.  Expected: 3"
    assert A_shape[3] == A_prev.shape[3], f"Test 2 - n_C is wrong. Current: {A_shape[3]}.  Expected: {A_prev.shape[3]}"
    
    # Test 3
    np.random.seed(1)
    A_prev = np.random.randn(2, 5, 5, 3)
    hparameters = {"stride": 1, "f": 3}
    expected_cache = (A_prev, hparameters)

    expected_A_max = np.array(pool_forward_output0)

    expected_output_max = (expected_A_max, expected_cache)

    expected_A_average = np.array(pool_forward_output1)
    expected_output_average = (expected_A_average, expected_cache)
    test_cases = [
        {
            "name": "datatype_check",
            "input": [A_prev, hparameters, "max"],
            "expected": expected_output_max,
            "error":"Datatype mismatch in MAX-Pool"
        },
        {
            "name": "shape_check",
            "input": [A_prev, hparameters, "max"],
            "expected": expected_output_max,
            "error": "Wrong shape in MAX-Pool"
        },
        {
            "name": "equation_output_check",
            "input": [A_prev, hparameters, "max"],
            "expected": expected_output_max,
            "error": "Wrong output in MAX-Pool"
        },
        {
            "name": "datatype_check",
            "input": [A_prev, hparameters, "average"],
            "expected": expected_output_average,
            "error":"Datatype mismatch in AVG-Pool"
        },
        {
            "name": "shape_check",
            "input": [A_prev, hparameters, "average"],
            "expected": expected_output_average,
            "error": "Wrong shape in AVG-Pool"
        },
        {
            "name": "equation_output_check",
            "input": [A_prev, hparameters, "average"],
            "expected": expected_output_average,
            "error": "Wrong output in AVG-Pool"
        }
    ]

    multiple_test(test_cases, target)

######################################
############## UNGRADED ##############
######################################


def conv_backward_test(target):

    test_cases = [
        {
            "name": "datatype_check",
            "input": [parameters, cache, X, Y],
            "expected": expected_output,
            "error":"The function should return a numpy array."
        },
        {
            "name": "shape_check",
            "input": [parameters, cache, X, Y],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [parameters, cache, X, Y],
            "expected": expected_output,
            "error": "Wrong output"
        }
    ]

    multiple_test(test_cases, target)


def create_mask_from_window_test(target):

    test_cases = [
        {
            "name": "datatype_check",
            "input": [parameters, grads],
            "expected": expected_output,
            "error":"Data type mismatch"
        },
        {
            "name": "shape_check",
            "input": [parameters, grads],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [parameters, grads],
            "expected": expected_output,
            "error": "Wrong output"
        }
    ]

    multiple_test(test_cases, target)


def distribute_value_test(target):
    test_cases = [
        {
            "name": "datatype_check",
            "input": [X, Y, n_h],
            "expected": expected_output,
            "error":"Data type mismatch"
        },
        {
            "name": "shape_check",
            "input": [X, Y, n_h],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [X, Y, n_h],
            "expected": expected_output,
            "error": "Wrong output"
        }
    ]

    multiple_test(test_cases, target)


def pool_backward_test(target):

    test_cases = [
        {
            "name": "datatype_check",
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
