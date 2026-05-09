package de.labathome.optimization.matrix;

import static de.labathome.optimization.RelAbsAssertions.assertArrayRelAbsEquals;
import static de.labathome.optimization.RelAbsAssertions.assertRelAbsEquals;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;
import org.scipy.optimize.minimize.matrix.DenseMatrix;
import org.scipy.optimize.minimize.matrix.Matrix;
import org.scipy.optimize.minimize.matrix.SparseMatrix;

class TestDenseMatrix {

	@Test
	void zerosFactoryProducesEmptyBuffer() {
		DenseMatrix m = DenseMatrix.zeros(3, 4);
		assertEquals(3, m.rows());
		assertEquals(4, m.cols());
		assertEquals(12, m.data().length);
		for (double v : m.data()) assertEquals(0.0, v);
	}

	@Test
	void eyeFactoryProducesIdentity() {
		DenseMatrix I = DenseMatrix.eye(3);
		assertEquals(3, I.rows());
		assertEquals(3, I.cols());
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				assertEquals(i == j ? 1.0 : 0.0, I.get(i, j));
			}
		}
	}

	@Test
	void columnFactoryStoresAsColumnVector() {
		DenseMatrix v = DenseMatrix.column(1.0, 2.0, 3.0);
		assertEquals(3, v.rows());
		assertEquals(1, v.cols());
		for (int i = 0; i < 3; ++i) {
			assertEquals((double) (i + 1), v.get(i, 0));
		}
	}

	@Test
	void rowFactoryStoresAsRowVector() {
		DenseMatrix v = DenseMatrix.row(1.0, 2.0, 3.0);
		assertEquals(1, v.rows());
		assertEquals(3, v.cols());
		for (int j = 0; j < 3; ++j) {
			assertEquals((double) (j + 1), v.get(0, j));
		}
	}

	@Test
	void fromRowsRepacksAsColumnMajor() {
		double[][] src = {
				{1.0, 2.0, 3.0},
				{4.0, 5.0, 6.0},
		};
		DenseMatrix m = DenseMatrix.fromRows(src);
		assertEquals(2, m.rows());
		assertEquals(3, m.cols());
		// Column-major layout: data[0]=A[0,0], data[1]=A[1,0], data[2]=A[0,1], ...
		assertArrayEquals(new double[] {1.0, 4.0, 2.0, 5.0, 3.0, 6.0}, m.data());
	}

	@Test
	void fromColumnMajorWrapsBufferWithoutCopying() {
		double[] buf = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
		DenseMatrix m = DenseMatrix.fromColumnMajor(2, 3, buf);
		assertSame(buf, m.data(), "fromColumnMajor should not copy the buffer");
		// Mutating the matrix shows up in the original array.
		m.set(0, 0, 99.0);
		assertEquals(99.0, buf[0]);
	}

	@Test
	void fromColumnMajorRejectsWrongLength() {
		assertThrows(IllegalArgumentException.class,
				() -> DenseMatrix.fromColumnMajor(2, 3, new double[5]));
	}

	@Test
	void getAndSetMatchColumnMajorIndexing() {
		DenseMatrix m = DenseMatrix.zeros(3, 4);
		m.set(2, 1, 7.5);
		assertEquals(7.5, m.get(2, 1));
		// Column-major: data[2 + 1*3] = data[5] should hold the value.
		assertEquals(7.5, m.data()[5]);
	}

	@Test
	void getAsDoubleAndSetAsDoubleAgreeWithIntApi() {
		DenseMatrix m = DenseMatrix.zeros(3, 3);
		m.setAsDouble(4.0, 1L, 2L);
		assertEquals(4.0, m.get(1, 2));
		assertEquals(4.0, m.getAsDouble(1L, 2L));
	}

	@Test
	void copyIsADeepClone() {
		DenseMatrix m = DenseMatrix.column(1.0, 2.0, 3.0);
		DenseMatrix c = m.copy();
		assertNotSame(m.data(), c.data());
		c.set(0, 0, 99.0);
		assertEquals(1.0, m.get(0, 0));
	}

	@Test
	void mtimesDenseDenseAgreesWithReference() {
		// 3x3 * 3x2 = 3x2
		DenseMatrix A = DenseMatrix.fromRows(new double[][] {
				{1.0, 2.0, 3.0},
				{4.0, 5.0, 6.0},
				{7.0, 8.0, 9.0},
		});
		DenseMatrix B = DenseMatrix.fromRows(new double[][] {
				{1.0, 0.0},
				{0.0, 1.0},
				{1.0, 1.0},
		});
		// A*B by hand:
		// row 0: [1+0+3, 0+2+3] = [4, 5]
		// row 1: [4+0+6, 0+5+6] = [10, 11]
		// row 2: [7+0+9, 0+8+9] = [16, 17]
		double[][] expected = {{4, 5}, {10, 11}, {16, 17}};
		Matrix C = A.mtimes(B);
		assertEquals(3, C.getRowCount());
		assertEquals(2, C.getColumnCount());
		assertArrayRelAbsEquals(expected, ((DenseMatrix) C).toDoubleArray(), 1e-15);
	}

	@Test
	void mtimesRectangularAgreesWithReference() {
		// 2x4 * 4x3 = 2x3
		DenseMatrix A = DenseMatrix.fromRows(new double[][] {
				{1, 2, 3, 4},
				{5, 6, 7, 8},
		});
		DenseMatrix B = DenseMatrix.fromRows(new double[][] {
				{1, 0, 1},
				{0, 1, 1},
				{1, 1, 0},
				{1, 0, 0},
		});
		double[][] expected = {
				{1 + 0 + 3 + 4, 0 + 2 + 3 + 0, 1 + 2 + 0 + 0},  // {8, 5, 3}
				{5 + 0 + 7 + 8, 0 + 6 + 7 + 0, 5 + 6 + 0 + 0},  // {20, 13, 11}
		};
		assertArrayRelAbsEquals(expected, ((DenseMatrix) A.mtimes(B)).toDoubleArray(), 1e-15);
	}

	@Test
	void mtimesAgainstNonDenseMatrixUsesFallback() {
		// Build a sparse matrix with the same content as B above and check mtimes still works.
		DenseMatrix A = DenseMatrix.fromRows(new double[][] {
				{1, 2, 3},
				{4, 5, 6},
		});
		SparseMatrix S = (SparseMatrix) SparseMatrix.Factory.zeros(3, 2);
		S.setAsDouble(1.0, 0L, 0L);
		S.setAsDouble(1.0, 1L, 1L);
		S.setAsDouble(1.0, 2L, 0L);
		S.setAsDouble(1.0, 2L, 1L);
		// A*S = [[1+0+3, 0+2+3], [4+0+6, 0+5+6]] = [[4,5], [10,11]]
		Matrix C = A.mtimes(S);
		assertArrayRelAbsEquals(new double[][] {{4, 5}, {10, 11}},
				((DenseMatrix) C).toDoubleArray(), 1e-15);
	}

	@Test
	void plusMinusAgreeOnDenseDense() {
		DenseMatrix A = DenseMatrix.fromRows(new double[][] {{1, 2}, {3, 4}});
		DenseMatrix B = DenseMatrix.fromRows(new double[][] {{10, 20}, {30, 40}});
		assertArrayRelAbsEquals(new double[][] {{11, 22}, {33, 44}},
				((DenseMatrix) A.plus(B)).toDoubleArray(), 1e-15);
		assertArrayRelAbsEquals(new double[][] {{-9, -18}, {-27, -36}},
				((DenseMatrix) A.minus(B)).toDoubleArray(), 1e-15);
	}

	@Test
	void timesAndDivideScalar() {
		DenseMatrix A = DenseMatrix.column(1.0, 2.0, 3.0);
		assertArrayRelAbsEquals(new double[] {2.0, 4.0, 6.0},
				((DenseMatrix) A.times(2.0)).toColumnArray(), 1e-15);
		assertArrayRelAbsEquals(new double[] {0.5, 1.0, 1.5},
				((DenseMatrix) A.divide(2.0)).toColumnArray(), 1e-15);
	}

	@Test
	void timesElementwiseHadamard() {
		DenseMatrix A = DenseMatrix.fromRows(new double[][] {{1, 2}, {3, 4}});
		DenseMatrix B = DenseMatrix.fromRows(new double[][] {{2, 0}, {0, 2}});
		assertArrayRelAbsEquals(new double[][] {{2, 0}, {0, 8}},
				((DenseMatrix) A.times((Matrix) B)).toDoubleArray(), 1e-15);
	}

	@Test
	void transposeRoundTrip() {
		DenseMatrix A = DenseMatrix.fromRows(new double[][] {
				{1, 2, 3},
				{4, 5, 6},
		});
		DenseMatrix At = (DenseMatrix) A.transpose();
		assertEquals(3, At.rows());
		assertEquals(2, At.cols());
		// First transpose entries
		for (int i = 0; i < A.rows(); ++i) {
			for (int j = 0; j < A.cols(); ++j) {
				assertEquals(A.get(i, j), At.get(j, i));
			}
		}
		// Round trip
		DenseMatrix Att = (DenseMatrix) At.transpose();
		assertArrayRelAbsEquals(A.toDoubleArray(), Att.toDoubleArray(), 1e-15);
	}

	@Test
	void norm2FrobeniusAndNormInf() {
		DenseMatrix A = DenseMatrix.fromRows(new double[][] {{3, 0}, {0, 4}});
		// Frobenius = sqrt(9 + 16) = 5
		assertRelAbsEquals(5.0, A.norm2(), 1e-15);
		assertRelAbsEquals(4.0, A.normInf(), 1e-15);
		DenseMatrix B = DenseMatrix.column(-7.0, 1.0, 2.0);
		assertRelAbsEquals(7.0, B.normInf(), 1e-15);
	}

	@Test
	void absInPlaceMutates() {
		DenseMatrix A = DenseMatrix.fromRows(new double[][] {{-1, 2}, {3, -4}});
		DenseMatrix r = A.absInPlace();
		assertSame(A, r);
		assertArrayRelAbsEquals(new double[][] {{1, 2}, {3, 4}}, A.toDoubleArray(), 1e-15);
	}

	@Test
	void scaleInPlaceMutates() {
		DenseMatrix A = DenseMatrix.column(1.0, 2.0, 3.0);
		A.scaleInPlace(-2.0);
		assertArrayRelAbsEquals(new double[] {-2.0, -4.0, -6.0}, A.toColumnArray(), 1e-15);
	}

	@Test
	void subMatrixSlicesInclusive() {
		DenseMatrix A = DenseMatrix.fromRows(new double[][] {
				{1, 2, 3, 4},
				{5, 6, 7, 8},
				{9, 10, 11, 12},
		});
		DenseMatrix S = A.subMatrix(1, 2, 1, 2);
		assertEquals(2, S.rows());
		assertEquals(2, S.cols());
		assertArrayRelAbsEquals(new double[][] {{6, 7}, {10, 11}}, S.toDoubleArray(), 1e-15);
		// Slice independence: mutate S, A unchanged.
		S.set(0, 0, 99.0);
		assertEquals(6.0, A.get(1, 1));
	}

	@Test
	void selectColumnsBuildsFreshMatrix() {
		DenseMatrix A = DenseMatrix.fromRows(new double[][] {
				{1, 2, 3, 4},
				{5, 6, 7, 8},
		});
		DenseMatrix S = A.selectColumns(3, 0, 2);
		assertArrayRelAbsEquals(new double[][] {{4, 1, 3}, {8, 5, 7}}, S.toDoubleArray(), 1e-15);
	}

	@Test
	void selectRowsBuildsFreshMatrix() {
		DenseMatrix A = DenseMatrix.fromRows(new double[][] {
				{1, 2},
				{3, 4},
				{5, 6},
				{7, 8},
		});
		DenseMatrix S = A.selectRows(2, 0);
		assertArrayRelAbsEquals(new double[][] {{5, 6}, {1, 2}}, S.toDoubleArray(), 1e-15);
	}

	@Test
	void copyFromMatrixDensifiesSparseInput() {
		SparseMatrix S = (SparseMatrix) SparseMatrix.Factory.zeros(3, 3);
		S.setAsDouble(1.0, 0L, 0L);
		S.setAsDouble(2.0, 1L, 1L);
		S.setAsDouble(3.0, 2L, 2L);
		DenseMatrix D = DenseMatrix.copyFromMatrix(S);
		assertArrayRelAbsEquals(new double[][] {{1, 0, 0}, {0, 2, 0}, {0, 0, 3}},
				D.toDoubleArray(), 1e-15);
	}

	@Test
	void copyFromMatrixOnDenseInputProducesIndependentCopy() {
		DenseMatrix A = DenseMatrix.column(1.0, 2.0);
		DenseMatrix B = DenseMatrix.copyFromMatrix(A);
		assertNotSame(A.data(), B.data());
		B.set(0, 0, 99.0);
		assertEquals(1.0, A.get(0, 0));
	}

	@Test
	void mtimesShapeMismatchThrows() {
		DenseMatrix A = DenseMatrix.zeros(2, 3);
		DenseMatrix B = DenseMatrix.zeros(2, 2);
		assertThrows(IllegalArgumentException.class, () -> A.mtimes(B));
	}

	@Test
	void shapeMismatchOnPlusThrows() {
		DenseMatrix A = DenseMatrix.zeros(2, 3);
		DenseMatrix B = DenseMatrix.zeros(3, 2);
		assertThrows(IllegalArgumentException.class, () -> A.plus(B));
	}

	@Test
	void toRowMajorRoundTrip() {
		DenseMatrix A = DenseMatrix.fromRows(new double[][] {{1, 2, 3}, {4, 5, 6}});
		double[] rowMajor = A.toRowMajor();
		assertArrayEquals(new double[] {1, 2, 3, 4, 5, 6}, rowMajor);
	}
}
