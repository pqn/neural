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
        nn = neural.NeuralNetwork()
        xor_data = [[[1,0],1],[[0,1],1],[[0,0],0],[[1,1],0]]
        nn.train(xor_data)
        for item in xor_data:
            assert nn.predict(item[0]) == item[1]

    def test_multilabel(self):
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
        nn = neural.NeuralNetwork()
        xor_data = [[[1,0],1],[[0,1],1],[[0,0],0],[[1,1],0]]
        nn.train(xor_data)
        a = nn.predict([item[0] for item in xor_data])
        b = [item[1] for item in xor_data]
        for i in range(len(a)):
            assert a[i] == b[i]

    def test_xor_multiple_hidden(self):
        nn = neural.NeuralNetwork(hidden_layers=(25,25))
        xor_data = [[[1,0],1],[[0,1],1],[[0,0],0],[[1,1],0]]
        nn.train(xor_data)
        for item in xor_data:
            assert nn.predict(item[0]) == item[1]

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()