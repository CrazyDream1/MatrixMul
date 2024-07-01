import ctypes
import numpy as np
import os

def parse_test_file(filename):
    with open(filename, 'r') as file:
        content = file.read().strip()
    
    # Split the content by double newlines to separate matrices
    matrices_str = content.split('\n\n')
    
    # Convert the string matrices to numpy arrays
    matrices = [np.array([list(map(float, row.split())) for row in mat.split('\n')], dtype=np.float32) for mat in matrices_str]
    
    return matrices

def unit_test(lib, test_file):
    matrices = parse_test_file(test_file)
    
    A = matrices[0]
    B = matrices[1]
    expected_C = matrices[2].ravel(order='F')
    
    m, k1 = A.shape
    k2, n = B.shape
    
    if k1 != k2:
        raise ValueError("Incompatible dimensions for matrix multiplication")
    k = k1

    A = A.ravel(order='F')
    B = B.ravel(order='F')
    C = np.zeros((m * n), dtype=np.float32, order='F')

    lib.matrixMultiplyCPU(A, B, C, m, n, k)

    if np.allclose(C, expected_C):
        print(f"Test {test_file} passed!")
    else:
        print(f"Test {test_file} failed!")
        print("Expected:")
        print(expected_C)
        print("Got:")
        print(C)

os.add_dll_directory(os.getcwd())
os.add_dll_directory("C:\\Windows\\System32")
os.add_dll_directory("E:\\Nvidia\\bin")

matrixLib = ctypes.CDLL('.\\x64\\Release\\CRMatrixMul.dll')

matrixLib.matrixMultiplyCPU.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
    ctypes.c_int, ctypes.c_int, ctypes.c_int
]
matrixLib.matrixMultiplyCPU.restype = None

matrixLib.matrixMultiplyGPU.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
    ctypes.c_int, ctypes.c_int, ctypes.c_int
]
matrixLib.matrixMultiplyGPU.restype = None

matrixLib.printMatrix.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
    ctypes.c_int, ctypes.c_int
]
matrixLib.printMatrix.restype = None

test_dir = r'.\test cases'

for filename in os.listdir(test_dir):
    if filename.endswith('.txt'):
        test_file = os.path.join(test_dir, filename)
        unit_test(matrixLib, test_file)