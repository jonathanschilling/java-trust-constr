package de.labathome.optimization.sparse;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import de.labathome.trustconstr.sparse.CSCMatrix;
import de.labathome.trustconstr.sparse.CSRMatrix;

import de.labathome.optimization.RelAbsAssertions;

class TestCSCMatrix {

	private static final double TOL = 1.0e-14;

	@Test
	void testFromDenseCSCLayout() {
		// A = [[1, 0, 2], [0, 0, 3], [4, 5, 0]]
		// CSC stores by column:
		//   col 0: rows {0, 2} values {1, 4}
		//   col 1: rows {2}    values {5}
		//   col 2: rows {0, 1} values {2, 3}
		// indptr = [0, 2, 3, 5], indices = [0, 2, 2, 0, 1], data = [1, 4, 5, 2, 3]
		double[][] dense = {
				{ 1.0, 0.0, 2.0 },
				{ 0.0, 0.0, 3.0 },
				{ 4.0, 5.0, 0.0 },
		};
		CSCMatrix a = CSCMatrix.fromDense(dense);
		Assertions.assertArrayEquals(new int[] {0, 2, 3, 5}, a.indptr());
		Assertions.assertArrayEquals(new int[] {0, 2, 2, 0, 1}, a.indices());
		RelAbsAssertions.assertArrayRelAbsEquals(new double[] {1.0, 4.0, 5.0, 2.0, 3.0}, a.data(), TOL);
		RelAbsAssertions.assertArrayRelAbsEquals(dense, a.toDense(), TOL);
	}

	@Test
	void testEye() {
		double[][] expected = { {1, 0, 0}, {0, 1, 0}, {0, 0, 1} };
		RelAbsAssertions.assertArrayRelAbsEquals(expected, CSCMatrix.eye(3).toDense(), TOL);
	}

	@Test
	void testMatvec() {
		double[][] dense = {
				{ 1.0, 0.0, 2.0 },
				{ 0.0, 0.0, 3.0 },
				{ 4.0, 5.0, 0.0 },
		};
		CSCMatrix a = CSCMatrix.fromDense(dense);
		double[] x = {2.0, 3.0, 5.0};
		RelAbsAssertions.assertArrayRelAbsEquals(new double[] {12.0, 15.0, 23.0}, a.matvec(x), TOL);
	}

	@Test
	void testRmatvec() {
		double[][] dense = {
				{ 1.0, 0.0, 2.0 },
				{ 0.0, 0.0, 3.0 },
				{ 4.0, 5.0, 0.0 },
		};
		CSCMatrix a = CSCMatrix.fromDense(dense);
		double[] x = {1.0, 2.0, 3.0};
		RelAbsAssertions.assertArrayRelAbsEquals(new double[] {13.0, 15.0, 8.0}, a.rmatvec(x), TOL);
	}

	@Test
	void testTransposeAndCSCtoCSR() {
		double[][] dense = {
				{ 1.0, 0.0, 2.0, 7.0 },
				{ 0.0, 0.0, 3.0, 0.0 },
				{ 4.0, 5.0, 0.0, 8.0 },
		};
		CSCMatrix a = CSCMatrix.fromDense(dense);
		CSRMatrix at = a.transpose();
		Assertions.assertEquals(4, at.rows());
		Assertions.assertEquals(3, at.cols());
		double[][] expected = {
				{ 1.0, 0.0, 4.0 },
				{ 0.0, 0.0, 5.0 },
				{ 2.0, 3.0, 0.0 },
				{ 7.0, 0.0, 8.0 },
		};
		RelAbsAssertions.assertArrayRelAbsEquals(expected, at.toDense(), TOL);

		CSRMatrix asCsr = a.toCSR();
		RelAbsAssertions.assertArrayRelAbsEquals(dense, asCsr.toDense(), TOL);
	}

	@Test
	void testConstructorRejectsBadInputs() {
		Assertions.assertThrows(IllegalArgumentException.class,
				() -> new CSCMatrix(2, 2, new int[] {0, 1}, new int[] {0}, new double[] {1.0}));
		Assertions.assertThrows(IllegalArgumentException.class,
				() -> new CSCMatrix(2, 1, new int[] {0, 1}, new int[] {3}, new double[] {1.0}));
	}
}
