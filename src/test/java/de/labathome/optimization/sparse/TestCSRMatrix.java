package de.labathome.optimization.sparse;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.scipy.optimize.minimize.sparse.CSCMatrix;
import org.scipy.optimize.minimize.sparse.CSRMatrix;

import de.labathome.optimization.RelAbsAssertions;

/**
 * Unit tests for {@link CSRMatrix}. Reference values are computed from
 * scipy.sparse on the same inputs and baked in below; the regenerator script
 * lives in {@code src/test/python/regenerate_references.py}.
 */
class TestCSRMatrix {

	private static final double TOL = 1.0e-14;

	@Test
	void testFromDenseAndBack() {
		double[][] dense = {
				{ 1.0, 0.0, 2.0 },
				{ 0.0, 0.0, 3.0 },
				{ 4.0, 5.0, 0.0 },
		};
		CSRMatrix a = CSRMatrix.fromDense(dense);
		Assertions.assertEquals(3, a.rows());
		Assertions.assertEquals(3, a.cols());
		Assertions.assertEquals(5, a.nnz());
		Assertions.assertArrayEquals(new int[] {0, 2, 3, 5}, a.indptr());
		Assertions.assertArrayEquals(new int[] {0, 2, 2, 0, 1}, a.indices());
		RelAbsAssertions.assertArrayRelAbsEquals(new double[] {1.0, 2.0, 3.0, 4.0, 5.0}, a.data(), TOL);
		RelAbsAssertions.assertArrayRelAbsEquals(dense, a.toDense(), TOL);
	}

	@Test
	void testEye() {
		CSRMatrix i3 = CSRMatrix.eye(3);
		Assertions.assertEquals(3, i3.rows());
		Assertions.assertEquals(3, i3.cols());
		Assertions.assertEquals(3, i3.nnz());
		double[][] expected = { {1, 0, 0}, {0, 1, 0}, {0, 0, 1} };
		RelAbsAssertions.assertArrayRelAbsEquals(expected, i3.toDense(), TOL);
	}

	@Test
	void testEyeZero() {
		CSRMatrix i0 = CSRMatrix.eye(0);
		Assertions.assertEquals(0, i0.rows());
		Assertions.assertEquals(0, i0.nnz());
	}

	@Test
	void testMatvec() {
		// A = [[1, 0, 2], [0, 0, 3], [4, 5, 0]]
		double[][] dense = {
				{ 1.0, 0.0, 2.0 },
				{ 0.0, 0.0, 3.0 },
				{ 4.0, 5.0, 0.0 },
		};
		CSRMatrix a = CSRMatrix.fromDense(dense);
		double[] x = {2.0, 3.0, 5.0};
		// A x = [1*2 + 2*5, 3*5, 4*2 + 5*3] = [12, 15, 23]
		RelAbsAssertions.assertArrayRelAbsEquals(new double[] {12.0, 15.0, 23.0}, a.matvec(x), TOL);
	}

	@Test
	void testRmatvec() {
		double[][] dense = {
				{ 1.0, 0.0, 2.0 },
				{ 0.0, 0.0, 3.0 },
				{ 4.0, 5.0, 0.0 },
		};
		CSRMatrix a = CSRMatrix.fromDense(dense);
		double[] x = {1.0, 2.0, 3.0};
		// A^T x = [1*1 + 0*2 + 4*3, 0*1 + 0*2 + 5*3, 2*1 + 3*2 + 0*3] = [13, 15, 8]
		RelAbsAssertions.assertArrayRelAbsEquals(new double[] {13.0, 15.0, 8.0}, a.rmatvec(x), TOL);
	}

	@Test
	void testMultiplyScalar() {
		CSRMatrix a = CSRMatrix.fromDense(new double[][] {{1, 2}, {3, 4}});
		double[][] expected = { {2.5, 5.0}, {7.5, 10.0} };
		RelAbsAssertions.assertArrayRelAbsEquals(expected, a.multiply(2.5).toDense(), TOL);
	}

	@Test
	void testTransposeIsCSCWithSwappedShape() {
		double[][] dense = {
				{ 1.0, 0.0, 2.0 },
				{ 0.0, 0.0, 3.0 },
				{ 4.0, 5.0, 0.0 },
		};
		CSRMatrix a = CSRMatrix.fromDense(dense);
		CSCMatrix at = a.transpose();
		Assertions.assertEquals(3, at.rows());
		Assertions.assertEquals(3, at.cols());
		// transpose of A
		double[][] expected = {
				{ 1.0, 0.0, 4.0 },
				{ 0.0, 0.0, 5.0 },
				{ 2.0, 3.0, 0.0 },
		};
		RelAbsAssertions.assertArrayRelAbsEquals(expected, at.toDense(), TOL);
	}

	@Test
	void testCSRtoCSCRoundTrip() {
		double[][] dense = {
				{ 1.0, 0.0, 2.0 },
				{ 0.0, 0.0, 3.0 },
				{ 4.0, 5.0, 0.0 },
				{ 0.0, 6.0, 0.0 },
		};
		CSRMatrix a = CSRMatrix.fromDense(dense);
		CSCMatrix asCsc = a.toCSC();
		Assertions.assertEquals(a.rows(), asCsc.rows());
		Assertions.assertEquals(a.cols(), asCsc.cols());
		Assertions.assertEquals(a.nnz(), asCsc.nnz());
		RelAbsAssertions.assertArrayRelAbsEquals(dense, asCsc.toDense(), TOL);
		// Round trip
		CSRMatrix back = asCsc.toCSR();
		RelAbsAssertions.assertArrayRelAbsEquals(dense, back.toDense(), TOL);
	}

	@Test
	void testConstructorRejectsBadInputs() {
		Assertions.assertThrows(IllegalArgumentException.class,
				() -> new CSRMatrix(2, 2, new int[] {0, 1}, new int[] {0}, new double[] {1.0}));
		Assertions.assertThrows(IllegalArgumentException.class,
				() -> new CSRMatrix(2, 2, new int[] {0, 1, 2}, new int[] {0, 1}, new double[] {1.0}));
		Assertions.assertThrows(IllegalArgumentException.class,
				() -> new CSRMatrix(1, 2, new int[] {0, 1}, new int[] {5}, new double[] {1.0}));
		// Non-monotone column indices within a row (1 then 0)
		Assertions.assertThrows(IllegalArgumentException.class,
				() -> new CSRMatrix(1, 3, new int[] {0, 2}, new int[] {1, 0}, new double[] {1.0, 2.0}));
	}
}
