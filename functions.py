import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import cv2

def QR(A):
    n = A.shape[1]
    Q = np.zeros_like(A, dtype=float)
    R = np.zeros((n, n), dtype=float)

    for i in range(n):
        v = A[:, i]
        for j in range(i):
            R[j, i] = np.dot(Q[:, j], A[:, i])
            v = v - R[j, i] * Q[:, j]
        R[i, i] = np.linalg.norm(v)
        Q[:, i] = v / R[i, i]
    return Q, R
    

def calculate_eigval_eigvec(A, max_iter=1000, tol=1e-10):

    n = A.shape[0]
    A_k = A.copy()
    Q_total = np.eye(n)

    for _ in range(max_iter):
        Q, R = QR(A_k)
        A_k = R @ Q
        Q_total = Q_total @ Q

        # stop condition check
        off_diagonal = A_k - np.diag(np.diag(A_k))
        if np.all(np.abs(off_diagonal) < tol):
            break

    eigvals = np.diag(A_k)
    eigvecs = Q_total
    return eigvals, eigvecs


def complete_orthonormal_columns(U):
    m, n = U.shape
    rank = np.linalg.matrix_rank(U)
    Q = []

    # normalize columns
    for i in range(n):
        col = U[:, i]
        norm = np.linalg.norm(col)
        if norm > 1e-10:
            Q.append(col / norm)

    #add new orthonormal vectors
    while len(Q) < m:
        new_vec = np.random.rand(m)
        for q in Q:
            new_vec -= np.dot(new_vec, q) * q
        norm = np.linalg.norm(new_vec)
        if norm > 1e-10:
            Q.append(new_vec / norm)

    return np.stack(Q, axis=1)


def SVD(A):

    #calculate eigenvalues and eigenvectors of A.T @ A
    A = np.array(A)
    ATA = A.T @ A
    eigval , eigvec = calculate_eigval_eigvec(ATA)

    #sorting eigenvalues and eigenvectors
    idx = np.argsort(eigval)[::-1]
    eigval = eigval[idx]
    eigvec = eigvec[:, idx]

    V = eigvec

    #singular values
    singular_values = np.sqrt(np.maximum(eigval, 0))

    m, n = A.shape
    Sigma = np.zeros((m, n))
    for i in range(min(m, n)):
        Sigma[i, i] = singular_values[i]

    #calculating U matrix
    U = np.zeros((m, m))
    for i in range(len(singular_values)):
        if singular_values[i] > 1e-10:  # sıfıra bölünmeyi önlemek için
            u_i = (1 / singular_values[i]) * A @ V[:, i]
            U[:, i] = u_i

        U = complete_orthonormal_columns(U)

    return U, Sigma, V.T   


def umeyama_alignment(src_points, dst_points):
    """
    src_points: NxD numpy array (örneğin mat2'den alınan eşleşen kaynak noktalar)
    dst_points: NxD numpy array (örneğin mat1'den alınan hedef noktalar)
    with_scaling: ölçek faktörü hesaplansın mı?
    """
    assert src_points.shape == dst_points.shape
    n, dim = src_points.shape

    # Ortalama merkezlerini al
    mu_src = np.mean(src_points, axis=0)
    mu_dst = np.mean(dst_points, axis=0)

    # Merkezlenmiş noktalar
    src_centered = src_points - mu_src
    dst_centered = dst_points - mu_dst

    # Kovaryans matrisi
    cov_matrix = dst_centered.T @ src_centered / n

    # SVD
    U, D, Vt = SVD(cov_matrix)
    
    R = U @ Vt
    if np.linalg.det(R) < 0:
        print("mirror detected")
        Vt[-1, : ] *= -1
        R = U @ Vt

    # Çeviri vektörü
    t = mu_dst - R @ mu_src

    # Sonuç: R, t, ölçek
    return R, t




