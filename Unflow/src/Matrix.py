import os
import numpy as np

class Matrix():
    def __init__(self, dir):
        self.dir = dir

    def input_matrix(self):
        self._odometry_txt = open(self.dir).read()

    def return_matrix(self, num):
        self.input_matrix()
        return np.matrix(self._odometry_txt.split('\n')[num]).reshape(3,4)