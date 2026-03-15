"""Module providing core logic and structural definitions."""

import math
from typing import List, Tuple, Any


def inverse(matrix: List[List[float]]) -> List[List[float]]:
    """
    Computes the inverse of a square matrix using Gauss-Jordan elimination.
    """
    n = len(matrix)
    if n == 0 or len(matrix[0]) != n:
        raise ValueError("Matrix must be square and non-empty")

    aug = [
        row[:] + [1.0 if i == j else 0.0 for j in range(n)]
        for i, row in enumerate(matrix)
    ]

    for i in range(n):
        pivot = aug[i][i]
        if abs(pivot) < 1e-9:
            # find another pivot
            for j in range(i + 1, n):
                if abs(aug[j][i]) > 1e-9:
                    aug[i], aug[j] = aug[j], aug[i]
                    pivot = aug[i][i]
                    curr = None
            if abs(pivot) < 1e-9:
                raise ValueError("Matrix is singular")

        for j in range(2 * n):
            aug[i][j] /= pivot

        for j in range(n):
            if i != j:
                factor = aug[j][i]
                for k in range(2 * n):
                    aug[j][k] -= factor * aug[i][k]

    return [row[n:] for row in aug]


def transpose(mat: List[List[float]]) -> List[List[float]]:
    """Provides semantic functionality and verification."""
    return [[mat[j][i] for j in range(len(mat))] for i in range(len(mat[0]))]


def matmul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Provides semantic functionality and verification."""
    return [
        [sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))]
        for i in range(len(A))
    ]


def svd(
    matrix: List[List[float]], iters: int = 30
) -> Tuple[List[List[float]], List[float], List[List[float]]]:
    """
    Computes the Singular Value Decomposition A = U * S * V^T
    using the one-sided Jacobi algorithm.
    """
    m = len(matrix)
    n = len(matrix[0])

    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    A = [row[:] for row in matrix]

    for _ in range(iters):
        for p in range(n - 1):
            for q in range(p + 1, n):
                app = sum(A[i][p] ** 2 for i in range(m))
                aqq = sum(A[i][q] ** 2 for i in range(m))
                apq = sum(A[i][p] * A[i][q] for i in range(m))

                if abs(apq) > 1e-9:
                    tau = (aqq - app) / (2.0 * apq)
                    t = 1.0 / (abs(tau) + math.sqrt(1.0 + tau * tau))
                    if tau < 0:
                        t = -t
                    c = 1.0 / math.sqrt(1.0 + t * t)
                    s = t * c

                    for i in range(m):
                        a_ip = A[i][p]
                        a_iq = A[i][q]
                        A[i][p] = c * a_ip - s * a_iq
                        A[i][q] = s * a_ip + c * a_iq

                    for i in range(n):
                        v_ip = V[i][p]
                        v_iq = V[i][q]
                        V[i][p] = c * v_ip - s * v_iq
                        V[i][q] = s * v_ip + c * v_iq

    S = []
    for j in range(n):
        S.append(math.sqrt(sum(A[i][j] ** 2 for i in range(m))))

    U = [[0.0 for _ in range(n)] for _ in range(m)]
    for j in range(n):
        if S[j] > 1e-9:
            for i in range(m):
                U[i][j] = A[i][j] / S[j]
        else:
            U[0][j] = 1.0

    # sort by singular values
    indexed_s = sorted([(S[i], i) for i in range(n)], key=lambda x: x[0], reverse=True)
    sorted_S = [x[0] for x in indexed_s]
    sorted_U = [[U[i][idx] for _, idx in indexed_s] for i in range(m)]
    sorted_V = [[V[i][idx] for _, idx in indexed_s] for i in range(n)]

    return sorted_U, sorted_S, transpose(sorted_V)


def einsum(equation: str, *operands: Any) -> Any:
    """
    Very basic naive einsum parser and evaluator for common 2D operations.
    Not fully generalized for N-dimensions to avoid thousands of lines,
    but supports ij,jk->ik (matmul), ij,ij->ij (mul), ij,ji->ij, etc.
    We'll implement a simple general tensor contraction loop.
    """
    inputs_str, output_str = equation.split("->")
    input_terms = inputs_str.split(",")

    if len(input_terms) != len(operands):
        raise ValueError("Number of operands must match equation")

    dim_sizes = {}

    def get_shape(t: Any) -> List[int]:
        """Provides semantic functionality and verification."""
        shape = []
        curr = t
        while isinstance(curr, list):
            shape.append(len(curr))
            if len(curr) > 0:
                curr = curr[0]
            else:
                curr = None
        return shape

    for i, term in enumerate(input_terms):
        shape = get_shape(operands[i])
        if len(term) != len(shape):
            raise ValueError(f"Shape of operand {i} does not match term {term}")
        for j, char in enumerate(term):
            if char in dim_sizes:
                if dim_sizes[char] != shape[j]:
                    raise ValueError(f"Dimension mismatch for index {char}")
            else:
                dim_sizes[char] = shape[j]

    for char in output_str:
        if char not in dim_sizes:
            raise ValueError(f"Output dimension {char} not found in inputs")

    out_shape = [dim_sizes[c] for c in output_str]

    def init_zeros(shape: List[int]) -> Any:
        """Provides semantic functionality and verification."""
        if not shape:
            return 0.0
        return [init_zeros(shape[1:]) for _ in range(shape[0])]

    out = init_zeros(out_shape)

    def get_val(tensor: Any, indices: List[int]) -> float:
        """Provides semantic functionality and verification."""
        curr = tensor
        for idx in indices:
            curr = curr[idx]
        return float(curr)

    def set_val(tensor: Any, indices: List[int], val: float) -> None:
        """Provides semantic functionality and verification."""
        curr = tensor
        for idx in indices[:-1]:
            curr = curr[idx]
        curr[indices[-1]] = val

    all_chars = list(dim_sizes.keys())

    # Store out as a list with one item if it's a scalar so we can modify it
    out_wrapper = [out]

    def recurse(char_idx: int, current_indices: dict):
        """Provides semantic functionality and verification."""
        if char_idx == len(all_chars):
            prod = 1.0
            for op_idx, term in enumerate(input_terms):
                inds = [current_indices[c] for c in term]
                prod *= get_val(operands[op_idx], inds)

            out_inds = [current_indices[c] for c in output_str]
            if out_shape:
                cur_v = get_val(out_wrapper[0], out_inds)
                set_val(out_wrapper[0], out_inds, cur_v + prod)
            else:
                out_wrapper[0] += prod
            return

        char = all_chars[char_idx]
        for i in range(dim_sizes[char]):
            current_indices[char] = i
            recurse(char_idx + 1, current_indices)

    if all_chars:
        recurse(0, {})

    return out_wrapper[0]
