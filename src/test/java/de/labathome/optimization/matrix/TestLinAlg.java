package de.labathome.optimization.matrix;

import static de.labathome.optimization.RelAbsAssertions.assertArrayRelAbsEquals;
import static de.labathome.optimization.RelAbsAssertions.assertRelAbsEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;
import org.scipy.optimize.minimize.matrix.CholResult;
import org.scipy.optimize.minimize.matrix.DenseMatrix;
import org.scipy.optimize.minimize.matrix.LinAlg;
import org.scipy.optimize.minimize.matrix.QRResult;
import org.scipy.optimize.minimize.matrix.SVDResult;

class TestLinAlg {

	// ---------- LU solve ----------

	@Test
	void solveSquareSystem() {
		// A = [[2, 1], [1, 3]], b = [3, 4]; expected x = [1, 1].
		DenseMatrix A = DenseMatrix.fromRows(new double[][] {{2, 1}, {1, 3}});
		DenseMatrix b = DenseMatrix.column(3.0, 4.0);
		DenseMatrix x = LinAlg.solve(A, b);
		assertArrayRelAbsEquals(new double[] {1.0, 1.0}, x.toColumnArray(), 1e-12);
	}

	@Test
	void solveDoesNotMutateInputs() {
		DenseMatrix A = DenseMatrix.fromRows(new double[][] {{2, 1}, {1, 3}});
		DenseMatrix b = DenseMatrix.column(3.0, 4.0);
		double[] aBefore = A.data().clone();
		double[] bBefore = b.data().clone();
		LinAlg.solve(A, b);
		assertArrayRelAbsEquals(aBefore, A.data(), 0.0);
		assertArrayRelAbsEquals(bBefore, b.data(), 0.0);
	}

	@Test
	void solveMultipleRightHandSides() {
		// Identity solve: A = I, RHS arbitrary, x == RHS.
		DenseMatrix I = DenseMatrix.eye(3);
		DenseMatrix B = DenseMatrix.fromRows(new double[][] {
				{1, 2}, {3, 4}, {5, 6}});
		DenseMatrix X = LinAlg.solve(I, B);
		assertArrayRelAbsEquals(B.toDoubleArray(), X.toDoubleArray(), 1e-12);
	}

	@Test
	void solveRejectsNonSquare() {
		assertThrows(IllegalArgumentException.class,
				() -> LinAlg.solve(DenseMatrix.zeros(2, 3), DenseMatrix.column(0, 0)));
	}

	@Test
	void solveOnSingularMatrixThrows() {
		DenseMatrix A = DenseMatrix.fromRows(new double[][] {{1, 2}, {2, 4}});
		DenseMatrix b = DenseMatrix.column(1.0, 2.0);
		assertThrows(ArithmeticException.class, () -> LinAlg.solve(A, b));
	}

	// ---------- QR ----------

	@Test
	void qrFactorsSquareMatrix() {
		DenseMatrix A = DenseMatrix.fromRows(new double[][] {
				{1, 1, 0},
				{1, 0, 1},
				{0, 1, 1},
		});
		QRResult qr = LinAlg.qr(A);
		// Q has orthonormal columns
		DenseMatrix QtQ = (DenseMatrix) qr.Q().transpose().mtimes(qr.Q());
		assertArrayRelAbsEquals(DenseMatrix.eye(3).toDoubleArray(),
				QtQ.toDoubleArray(), 1e-12);
		// R upper triangular
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < i; ++j) {
				assertRelAbsEquals(0.0, qr.R().get(i, j), 1e-15);
			}
		}
		// Q * R reconstructs A
		DenseMatrix QR = (DenseMatrix) qr.Q().mtimes(qr.R());
		assertArrayRelAbsEquals(A.toDoubleArray(), QR.toDoubleArray(), 1e-12);
	}

	@Test
	void qrFactorsTallMatrix() {
		DenseMatrix A = DenseMatrix.fromRows(new double[][] {
				{1, 2},
				{3, 4},
				{5, 6},
				{7, 8},
		});
		QRResult qr = LinAlg.qr(A);
		assertEquals(4, qr.Q().rows());
		assertEquals(2, qr.Q().cols());
		assertEquals(2, qr.R().rows());
		assertEquals(2, qr.R().cols());
		// Q^T Q = I (orthonormal columns)
		DenseMatrix QtQ = (DenseMatrix) qr.Q().transpose().mtimes(qr.Q());
		assertArrayRelAbsEquals(DenseMatrix.eye(2).toDoubleArray(),
				QtQ.toDoubleArray(), 1e-12);
		// Q * R = A
		DenseMatrix QR = (DenseMatrix) qr.Q().mtimes(qr.R());
		assertArrayRelAbsEquals(A.toDoubleArray(), QR.toDoubleArray(), 1e-12);
	}

	@Test
	void qrSolveSquareIsLeastSquares() {
		DenseMatrix A = DenseMatrix.fromRows(new double[][] {{2, 1}, {1, 3}});
		DenseMatrix b = DenseMatrix.column(3.0, 4.0);
		DenseMatrix x = LinAlg.qr(A).solve(b);
		assertArrayRelAbsEquals(new double[] {1.0, 1.0}, x.toColumnArray(), 1e-10);
	}

	@Test
	void qrRejectsShortFatMatrix() {
		assertThrows(IllegalArgumentException.class,
				() -> LinAlg.qr(DenseMatrix.zeros(2, 3)));
	}

	// ---------- SVD ----------

	@Test
	void svdReconstructsSquareMatrix() {
		DenseMatrix A = DenseMatrix.fromRows(new double[][] {
				{4, 0},
				{3, -5},
		});
		SVDResult svd = LinAlg.svd(A);
		// Reconstruct U * diag(s) * Vt
		DenseMatrix Sigma = DenseMatrix.zeros(2, 2);
		for (int i = 0; i < svd.s().length; ++i) Sigma.set(i, i, svd.s()[i]);
		DenseMatrix recon = (DenseMatrix) svd.U().mtimes(Sigma).mtimes(svd.Vt());
		assertArrayRelAbsEquals(A.toDoubleArray(), recon.toDoubleArray(), 1e-12);
		// U^T U = I, V V^T = I
		assertArrayRelAbsEquals(DenseMatrix.eye(2).toDoubleArray(),
				((DenseMatrix) svd.U().transpose().mtimes(svd.U())).toDoubleArray(), 1e-12);
		assertArrayRelAbsEquals(DenseMatrix.eye(2).toDoubleArray(),
				((DenseMatrix) svd.Vt().mtimes(svd.Vt().transpose())).toDoubleArray(), 1e-12);
	}

	@Test
	void svdSingularValuesAreNonIncreasing() {
		DenseMatrix A = DenseMatrix.fromRows(new double[][] {
				{1, 0, 2},
				{0, 3, 4},
		});
		SVDResult svd = LinAlg.svd(A);
		double[] s = svd.s();
		for (int i = 1; i < s.length; ++i) {
			assertTrue(s[i - 1] >= s[i] - 1e-15,
					"singular values must be non-increasing: " + s[i - 1] + " < " + s[i]);
			assertTrue(s[i] >= 0.0, "singular values must be non-negative");
		}
	}

	@Test
	void svdRankRecognisesRankDeficient() {
		// Rank 1 matrix: one row is a scalar multiple of another.
		DenseMatrix A = DenseMatrix.fromRows(new double[][] {
				{1, 2, 3},
				{2, 4, 6},
				{3, 6, 9},
		});
		SVDResult svd = LinAlg.svd(A);
		double sigmaMax = svd.s()[0];
		double tol = Math.max(3, 3) * Math.ulp(sigmaMax);
		assertEquals(1, svd.rank(tol));
	}

	@Test
	void svdReciprocalSingularValuesZeroesOutBelowTol() {
		// Rank 1 matrix again
		DenseMatrix A = DenseMatrix.fromRows(new double[][] {
				{1, 2}, {2, 4},
		});
		SVDResult svd = LinAlg.svd(A);
		double[] r = svd.reciprocalSingularValues();
		assertEquals(2, r.length);
		assertTrue(r[0] > 0.0);
		assertEquals(0.0, r[1], "near-zero singular value must map to 0");
	}

	// ---------- Cholesky ----------

	@Test
	void choleskyReconstructsSPD() {
		// SPD: A = L L^T with L = [[2, 0], [1, 3]] => A = [[4, 2], [2, 10]]
		DenseMatrix A = DenseMatrix.fromRows(new double[][] {{4, 2}, {2, 10}});
		CholResult chol = LinAlg.cholesky(A);
		// The "U" upper triangle of factor holds R such that A = R^T R.
		// R should be [[2, 1], [0, 3]] modulo signs.
		double[] f = chol.factor();
		// Column-major: f[0]=R[0,0], f[1]=R[1,0]=junk-or-zero, f[2]=R[0,1], f[3]=R[1,1]
		assertRelAbsEquals(2.0, f[0], 1e-12);
		assertRelAbsEquals(1.0, f[2], 1e-12);
		assertRelAbsEquals(3.0, f[3], 1e-12);
	}

	@Test
	void choleskySolveCorrect() {
		DenseMatrix A = DenseMatrix.fromRows(new double[][] {{4, 2}, {2, 10}});
		// Solve A x = b for b = [10, 22], expected x = [2, 1.8] (verify: 4*2+2*1.8 = 11.6, hmm)
		// Recompute: A*x = [4*2+2*1.8, 2*2+10*1.8] = [11.6, 22], so b should be [11.6, 22].
		// Easier: pick x = [1, 1], then b = [6, 12].
		DenseMatrix b = DenseMatrix.column(6.0, 12.0);
		CholResult chol = LinAlg.cholesky(A);
		DenseMatrix x = chol.solve(b);
		assertArrayRelAbsEquals(new double[] {1.0, 1.0}, x.toColumnArray(), 1e-12);
		// Solve again -- captured factor must still work.
		DenseMatrix b2 = DenseMatrix.column(2.0, 4.0);
		// A*x = [2, 4]: 4x1 + 2x2 = 2, 2x1 + 10x2 = 4. Solve: x2 = (4 - 2x1)/10; sub: 4x1 + 2(4-2x1)/10 = 2 -> 40x1 + 8 - 4x1 = 20 -> 36x1 = 12 -> x1 = 1/3, x2 = (4-2/3)/10 = 10/3/10 = 1/3.
		DenseMatrix x2 = chol.solve(b2);
		assertArrayRelAbsEquals(new double[] {1.0 / 3.0, 1.0 / 3.0}, x2.toColumnArray(), 1e-12);
	}

	@Test
	void choleskyRejectsNonSPD() {
		// Indefinite: A = [[1, 2], [2, 1]] has det=-3
		DenseMatrix A = DenseMatrix.fromRows(new double[][] {{1, 2}, {2, 1}});
		assertThrows(ArithmeticException.class, () -> LinAlg.cholesky(A));
	}

	// ---------- BLAS rank updates ----------

	@Test
	void syrAgreesWithManualRankOneUpdate() {
		// A starts as 4x4 zeros. After A := A + alpha x x^T with alpha = 2, x = [1,2,3,4],
		// A should equal 2 * outer(x, x).
		DenseMatrix A = DenseMatrix.zeros(4, 4);
		double[] x = {1, 2, 3, 4};
		LinAlg.syr(A, 2.0, x);
		double[][] expected = new double[4][4];
		for (int i = 0; i < 4; ++i) {
			for (int j = 0; j < 4; ++j) {
				expected[i][j] = 2.0 * x[i] * x[j];
			}
		}
		assertArrayRelAbsEquals(expected, A.toDoubleArray(), 1e-15);
	}

	@Test
	void syrPreservesSymmetry() {
		// Add two rank-1 updates to a symmetric base; result must stay symmetric.
		DenseMatrix A = DenseMatrix.fromRows(new double[][] {
				{1, 2, 0},
				{2, 1, 0},
				{0, 0, 1},
		});
		LinAlg.syr(A, 1.5, new double[] {0.5, 0.5, 0.5});
		for (int i = 0; i < 3; ++i) {
			for (int j = i + 1; j < 3; ++j) {
				assertRelAbsEquals(A.get(i, j), A.get(j, i), 1e-15);
			}
		}
	}

	@Test
	void syr2AgreesWithManualRankTwoUpdate() {
		// A starts as 4x4 with some content; add alpha (x y^T + y x^T).
		DenseMatrix A = DenseMatrix.eye(4);
		double[] x = {1, 0, 1, 0};
		double[] y = {0, 1, 0, 1};
		double alpha = 3.0;
		double[][] before = A.toDoubleArray();
		LinAlg.syr2(A, alpha, x, y);
		double[][] expected = new double[4][4];
		for (int i = 0; i < 4; ++i) {
			for (int j = 0; j < 4; ++j) {
				expected[i][j] = before[i][j] + alpha * (x[i] * y[j] + y[i] * x[j]);
			}
		}
		assertArrayRelAbsEquals(expected, A.toDoubleArray(), 1e-15);
	}

	@Test
	void syrRejectsLengthMismatch() {
		DenseMatrix A = DenseMatrix.eye(3);
		assertThrows(IllegalArgumentException.class,
				() -> LinAlg.syr(A, 1.0, new double[] {1, 2}));
	}

	@Test
	void syr2RejectsLengthMismatch() {
		DenseMatrix A = DenseMatrix.eye(3);
		assertThrows(IllegalArgumentException.class,
				() -> LinAlg.syr2(A, 1.0, new double[] {1, 2, 3}, new double[] {1, 2}));
	}
}
