#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_neural
----------------------------------

Tests for `neural` module.
"""

import unittest

from neural import neural
import numpy as np
from scipy.io import loadmat


class TestNeural(unittest.TestCase):

    def setUp(self):
        pass

    def test_xor(self):
        """Basic test to train network on XOR function."""
        nn = neural.NeuralNetwork()
        xor_data = [[[1,0],1],[[0,1],1],[[0,0],0],[[1,1],0]]
        nn.train(xor_data)
        for item in xor_data:
            assert nn.predict(item[0]) == item[1]

    def test_xor_alternate_input(self):
        """Tests giving X and y matrices as inputs rather than list of cases."""
        nn = neural.NeuralNetwork()
        X = np.matrix("[1 0; 0 1; 0 0; 1 1]")
        y = np.matrix("[1; 1; 0; 0]")
        xor_data_alternate = [X, y]
        nn.train(xor_data_alternate)
        xor_data = [[[1,0],1],[[0,1],1],[[0,0],0],[[1,1],0]]
        for item in xor_data:
            assert nn.predict(item[0]) == item[1]

    def test_multilabel(self):
        """Divides complex number plane into 3 regions based on angle."""
        nn = neural.NeuralNetwork(num_labels=3)
        test_data = []
        for x in range(50):
            i = np.random.rand()*2-1
            j = np.random.rand()*2-1
            angle = np.angle(i+j*1j, deg=True)
            result = int(np.floor(angle % 360 / 120))
            test_data.append([[i, j], result])
        nn.train(test_data)
        predict_data = [[[0.5,0.5],0],[[-0.5,0],1],[[0.5,-0.5],2]]
        for item in predict_data:
            assert nn.predict(item[0]) == item[1]

    def test_xor_parallel(self):
        """Tests returning a matrix of predictions given multiple cases to test after training."""
        nn = neural.NeuralNetwork()
        xor_data = [[[1,0],1],[[0,1],1],[[0,0],0],[[1,1],0]]
        nn.train(xor_data)
        a = nn.predict([item[0] for item in xor_data])
        b = [item[1] for item in xor_data]
        for i in range(len(a)):
            assert a[i] == b[i]

    def test_xor_multiple_hidden(self):
        """Tests using multiple hidden layers."""
        nn = neural.NeuralNetwork(hidden_layers=(25,25))
        xor_data = [[[1,0],1],[[0,1],1],[[0,0],0],[[1,1],0]]
        nn.train(xor_data)
        for item in xor_data:
            assert nn.predict(item[0]) == item[1]

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()