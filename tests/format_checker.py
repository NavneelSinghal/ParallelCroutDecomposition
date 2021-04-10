#!/usr/bin/env python
# coding: utf-8
import sys
import numpy as np

with open(sys.argv[1], "r") as f:
    A = f.read()

with open(sys.argv[2], "r") as f:
    L = f.read()

with open(sys.argv[3], "r") as f:
    U = f.read()

def convert2numpy(mat):
    mat = mat.split("\n")
    mat = [i.strip().split() for i in mat]
    return np.array(mat).astype(float)

A_dash = np.matmul(convert2numpy(L), convert2numpy(U))


if np.allclose(convert2numpy(A), A_dash) and abs(np.linalg.det(convert2numpy(U)) - 1) < 1e-3:
    print("Valid Crout Decomposition")
else:
    print("Invalid Crout Decomposition")

