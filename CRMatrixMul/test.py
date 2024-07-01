import ctypes
import numpy as np
import os
import io
import sys
import unittest
from contextlib import redirect_stdout

def parse_test_file(filename):
    with open(filename, 'r') as file:
        content = file.read().strip()

    matrices_str = content.split('\n\n')
    matrices = [np.array([list(map(float, row.split())) for row in mat.split('\n')], dtype=np.float32) for mat in matrices_str]
    
    return matrices

def matrix_to_string(matrix):
    rows, cols = matrix.shape
    result = "Matrix:\n"
    for i in range(rows):
        for j in range(cols):
            result += f"{matrix[i, j]:.5f} "
        result += "\n"
    return result

class TestMatrixMultiplication(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        os.add_dll_directory(os.getcwd())
        os.add_dll_directory("C:\\Windows\\System32")
        os.add_dll_directory("E:\\Nvidia\\bin")

        cls.matrixLib = ctypes.CDLL('.\\x64\\Release\\CRMatrixMul.dll')

        cls.matrixLib.matrixMultiplyCPU.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
            ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]
        cls.matrixLib.matrixMultiplyCPU.restype = None

        cls.matrixLib.matrixMultiplyGPU.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
            ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]
        cls.matrixLib.matrixMultiplyGPU.restype = None

        cls.matrixLib.matrixToString.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
            ctypes.c_int, ctypes.c_int
        ]
        cls.matrixLib.matrixToString.restype = ctypes.c_char_p

    def run_test_case(self, test_file):
        matrices = parse_test_file(test_file)
        A, B, expected_C = matrices

        m, k1 = A.shape
        k2, n = B.shape
        self.assertEqual(k1, k2, "Incompatible dimensions for matrix multiplication")
        k = k1

        A = A.ravel(order='F')
        B = B.ravel(order='F')
        expected_C = expected_C.ravel(order='F')
        C_CPU = np.zeros((m * n), dtype=np.float32, order='F')
        C_GPU = np.zeros((m * n), dtype=np.float32, order='F')

        self.matrixLib.matrixMultiplyCPU(A, B, C_CPU, m, n, k)
        self.matrixLib.matrixMultiplyGPU(A, B, C_GPU, m, n, k)

        np.testing.assert_allclose(C_CPU, expected_C, rtol=1e-5, atol=1e-8, err_msg=f"Test failed for {test_file}")
        np.testing.assert_allclose(C_GPU, expected_C, rtol=1e-5, atol=1e-8, err_msg=f"Test failed for {test_file}")
        
        output = self.matrixLib.matrixToString(C_CPU, m, n)
        output = output.decode('utf-8')
        self.assertEqual(matrix_to_string(matrices[2]), output)

def add_test_cases():
    test_dir = r'.\test cases'
    for filename in os.listdir(test_dir):
        if filename.endswith('.txt'):
            test_file = os.path.join(test_dir, filename)
            test_name = f'test_{os.path.splitext(filename)[0]}'
            test_func = lambda self, tf=test_file: self.run_test_case(tf)
            setattr(TestMatrixMultiplication, test_name, test_func)

add_test_cases()


if __name__ == '__main__':
    unittest.main()

	