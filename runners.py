# -*- coding: utf-8 -*-

"""
    File name    :    runners
    Date         :    10/07/2019
    Description  :    {TODO}
    Author       :    VickeeX
"""
import numpy as np
from multiprocessing import Queue
from multiprocessing.sharedctypes import RawArray
from ctypes import c_uint, c_float, c_double


class Runners(object):
    NUMPY_TO_C_DTYPE = {np.float32: c_float, np.float64: c_double, np.uint8: c_uint}

    def __init__(self, EmulatorRunner, emulators, workers, variables, queue):
        self.variables = [self._get_shared(var) for var in variables]
        self.workers = workers
        self.queue = queue
        self.putSignal = [Queue() for _ in range(workers)]
        self.getSignal = Queue()

        self.runners = [EmulatorRunner(i, emulators, vars, self.putSignal[i], self.getSignal) for i, (emulators, vars)
                        in enumerate(zip(emulators, zip(*[var for var in self.variables])))]

    def _get_shared(self, array):
        """
        Returns a RawArray backed numpy array that can be shared between processes.
        :param array: the array to be shared
        :return: the RawArray backed numpy array
        """

        dtype = self.NUMPY_TO_C_DTYPE[array.dtype.type]

        shape = array.shape
        shared = RawArray(dtype, array.reshape(-1))
        return np.frombuffer(shared, dtype).reshape(shape)

    def start(self):
        for r in self.runners:
            r.start()

        while True:
            self.update_environments()
            self.wait_updated()
            self.queue.put(self.variables)

    def stop(self):
        for q in self.putSignal:
            q.put(None)

    #
    # def get_shared_variables(self):
    #     return self.variables

    def update_environments(self):
        for q in self.putSignal:
            q.put(True)

    def wait_updated(self):
        for _ in range(self.workers):
            self.getSignal.get()
