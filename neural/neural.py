#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import fmin_ncg

class NeuralNetwork:
    """A simple neural network."""

    def __init__(self, hidden_layers=(25,), reg_lambda=0, num_labels=2):
        """Instantiates the class."""
        self.__hidden_layers = tuple(hidden_layers)
        self.__lambda = reg_lambda
        if num_labels > 2:
            self.__num_labels = num_labels
        else:
            self.__num_labels = 1

    def train(self, training_set, iterations=500):
        """Trains itself using the sequence data."""
        self.__X = np.matrix([example[0] for example in training_set])
        if self.__num_labels == 1:
            self.__y = np.matrix([example[1] for example in training_set]).reshape((-1, 1))
        else:
            eye = np.eye(self.__num_labels)
            self.__y = np.matrix([eye[example[1]] for example in training_set])
        self.__m = self.__X[0].shape[0]
        self.__input_layer_size = self.__X[0].shape[1]
        self.__sizes = [self.__input_layer_size]
        self.__sizes.extend(self.__hidden_layers)
        self.__sizes.append(self.__num_labels)
        param_size = 0
        for count in range(len(self.__sizes) - 1):
            param_size += (self.__sizes[count] + 1) * self.__sizes[count+1]
        print(param_size)
        # self.__thetas = self.__roll(fmin_ncg(self.__cost, np.zeros(param_size), self.__costGrad))

    def __cost(self, params):
        """Computes cost function."""

    def __costGrad(self, params):
        """Computes cost function derivative."""

    def __roll(self, unrolled):
        """Converts parameter array back into matrices."""
        rolled = []
        index = 0
        for count in range(len(self.__sizes) - 1):
            in_size = self.__sizes[count]
            out_size = self.__sizes[count+1]
            theta_unrolled = np.matrix(unrolled[index:index+(in_size+1)*out_size])
            theta_rolled = theta_unrolled.reshape((out_size, in_size+1))
            rolled.append(theta_rolled)
            index += (in_size + 1) * out_size
        return rolled


    def __unroll(self, rolled):
        """Converts parameter matrices into an array."""
        return np.array(np.concatenate([matrix.reshape(-1) for matrix in rolled], axis=1)).reshape(-1)
