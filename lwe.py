import numpy as np
import numpy.typing as npt
import cmath

# def lwe(pk:npt.NDArray[np.int256], sk:npt.NDArray[np.int256]) -> npt.NDArray[np.int256]:
#    pass


class CKKSEncoder:

    def __init__(self, scale: np.float128, tol: np.float128 = 1E-10):
        self.scale = scale
        self.tol = tol

    @staticmethod
    def get_cyclonomical_polynom_roots(n: int):
        return [np.exp((2 * np.pi * k + np.pi)*1j/n) for k in range(n)]

    @staticmethod
    def get_matrix_at_cyclonomical_poly_roots(n: int):
        roots = CKKSEncoder.get_cyclonomical_polynom_roots(n)
        return np.array([r**i for r in roots for i in range(n)]).reshape(n, n)

    @staticmethod
    def sigma_inverse(x: npt.NDArray[np.complex128], complex_round_tol: np.float128) -> np.polynomial.Polynomial:
        """
        The inverse embedding transaformation
        sigma^{-1}: C^N -> C[X]/(X^N + 1)

        To encode a vector into a Polynomial in R[X]/(X^N+1) one needs to find
        the polynomial such that evaluating this polynomial on the roots of the 
        cyclonomic polynomial is equal to the given vector
        """
        n = len(x)
        A = CKKSEncoder.get_matrix_at_cyclonomical_poly_roots(n)
        return np.polynomial.Polynomial(np.real_if_close(np.linalg.solve(A, x), complex_round_tol))

    @staticmethod
    def sigma(mu: np.polynomial.Polynomial) -> npt.NDArray[np.complex128]:
        """
        The embedding transaformation
        sigma: C[X]/(X^N + 1) -> C^N

        To decode a polynomial into a vector in C^n one needs to evaluate the
        polynomial on the roots of the cyclonomic polynomial X^N+1
        """
        n = mu.degree() + 1
        roots = CKKSEncoder.get_cyclonomical_polynom_roots(n)
        return np.array([mu(x) for x in roots])

    @staticmethod
    def pi(x: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
        """
        Projector operator
        H -> C^{N/2}
        """
        n = len(x)
        if (n % 2 != 0):
            raise Exception("Only works with arrays with even length.")
        return x[:n//2]

    @staticmethod
    def pi_inverse(x: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
        """
        Projector operator
        C^{N/2} -> H
        """
        # the invese order in the conjugate is to keep the counter-clockwise ordering
        return np.concatenate([x, np.conj(x[::-1])])

    @staticmethod
    def inner_prod(x: npt.NDArray[np.complex128], y: npt.NDArray[np.complex128], complex_round_tol: np.float128) -> npt.NDArray[np.complex128]:
        return np.real_if_close(x.dot(y.conj()), tol=complex_round_tol)

    @staticmethod
    def round(x: npt.NDArray[np.float128]) -> npt.NDArray[np.int64]:
        """
        Elementwise random rounding procedure 
        """
        x_floor = x - np.floor(x)
        x_round_error = np.vectorize(
            lambda c: np.random.choice([c, c-1], 1, p=[1-c, c]))(x_floor)
        return np.vectorize(int)(x - x_round_error)

    def encode(self, x: npt.NDArray[np.complex128]) -> np.polynomial.Polynomial:
        # expand from C^N/2 to C^N to get all roots of the cyclonomic polynomial
        x_in_H = self.pi_inverse(x)
        # multiply by delta to reduce the effect of the discretization
        scaled_x_in_H = self.scale * x_in_H
        A = CKKSEncoder.get_matrix_at_cyclonomical_poly_roots(
            len(scaled_x_in_H))
        a_norm2 = np.diag(CKKSEncoder.inner_prod(
            A, A.T, complex_round_tol=self.tol))
        x_in_z = CKKSEncoder.inner_prod(
            scaled_x_in_H, A, complex_round_tol=self.tol)/a_norm2
        rounded_x_in_z = self.round(x_in_z)
        x_in_sigma_z = A.dot(rounded_x_in_z)
        return self.sigma_inverse(x_in_sigma_z, self.tol)

    def decode(self, x: np.polynomial.Polynomial) -> npt.NDArray[np.complex128]:
        return(self.pi(self.sigma(x/self.scale)))
