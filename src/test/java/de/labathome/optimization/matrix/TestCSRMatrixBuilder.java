package de.labathome.optimization.matrix;

import static de.labathome.optimization.RelAbsAssertions.assertArrayRelAbsEquals;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;
import de.labathome.trustconstr.sparse.CSRMatrix;

class TestCSRMatrixBuilder {

	@Test
	void emptyBuilderProducesEmptyCSR() {
		CSRMatrix m = CSRMatrix.builder(3, 4).build();
		assertEquals(3, m.rows());
		assertEquals(4, m.cols());
		assertEquals(0, m.nnz());
		double[][] dense = m.toDense();
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 4; ++j) {
				assertEquals(0.0, dense[i][j]);
			}
		}
	}

	@Test
	void singleEntryRoundTripsThroughBuild() {
		CSRMatrix m = CSRMatrix.builder(2, 2)
				.set(1, 0, 7.0)
				.build();
		assertEquals(1, m.nnz());
		assertArrayRelAbsEquals(new double[][] {{0, 0}, {7, 0}}, m.toDense(), 1e-15);
	}

	@Test
	void duplicateSetOverwrites() {
		CSRMatrix m = CSRMatrix.builder(2, 2)
				.set(0, 0, 1.0)
				.set(0, 0, 2.0)
				.set(0, 0, 3.0)
				.build();
		assertEquals(1, m.nnz());
		assertEquals(3.0, m.toDense()[0][0]);
	}

	@Test
	void settingZeroRemovesPreviousEntry() {
		CSRMatrix.Builder b = CSRMatrix.builder(2, 2)
				.set(0, 0, 5.0)
				.set(1, 1, 8.0);
		assertEquals(2, b.nnz());
		b.set(0, 0, 0.0);
		assertEquals(1, b.nnz());
		CSRMatrix m = b.build();
		assertEquals(1, m.nnz());
		assertArrayRelAbsEquals(new double[][] {{0, 0}, {0, 8}}, m.toDense(), 1e-15);
	}

	@Test
	void getReturnsZeroForUnsetCells() {
		CSRMatrix.Builder b = CSRMatrix.builder(3, 3).set(1, 2, 4.5);
		assertEquals(4.5, b.get(1, 2));
		assertEquals(0.0, b.get(0, 0));
	}

	@Test
	void columnIndicesAreSortedWithinEachRow() {
		CSRMatrix m = CSRMatrix.builder(2, 4)
				// Insert in non-canonical order:
				.set(0, 3, 30.0)
				.set(0, 1, 10.0)
				.set(0, 2, 20.0)
				.set(1, 0, 100.0)
				.set(1, 3, 300.0)
				.build();
		int[] indptr = m.indptr();
		int[] indices = m.indices();
		double[] data = m.data();
		assertArrayEquals(new int[] {0, 3, 5}, indptr);
		// Row 0: columns sorted 1, 2, 3 with their values
		assertArrayEquals(new int[] {1, 2, 3, 0, 3}, indices);
		assertArrayEquals(new double[] {10.0, 20.0, 30.0, 100.0, 300.0}, data);
	}

	@Test
	void buildMatchesFromDense() {
		double[][] dense = {
				{0, 1, 0, 2},
				{3, 0, 0, 0},
				{0, 0, 4, 5},
		};
		CSRMatrix viaBuilder = CSRMatrix.builder(3, 4)
				.set(0, 1, 1.0)
				.set(0, 3, 2.0)
				.set(1, 0, 3.0)
				.set(2, 2, 4.0)
				.set(2, 3, 5.0)
				.build();
		CSRMatrix viaDense = CSRMatrix.fromDense(dense);
		assertEquals(viaDense.nnz(), viaBuilder.nnz());
		assertArrayEquals(viaDense.indptr(), viaBuilder.indptr());
		assertArrayEquals(viaDense.indices(), viaBuilder.indices());
		assertArrayRelAbsEquals(viaDense.data(), viaBuilder.data(), 1e-15);
	}

	@Test
	void builderRejectsOutOfRangeIndices() {
		CSRMatrix.Builder b = CSRMatrix.builder(2, 2);
		assertThrows(IndexOutOfBoundsException.class, () -> b.set(2, 0, 1.0));
		assertThrows(IndexOutOfBoundsException.class, () -> b.set(0, -1, 1.0));
	}

	@Test
	void buildIsIdempotentForDistinctCalls() {
		// Calling build() twice on the same builder produces equivalent CSR matrices.
		// Note: it's not specified that they share storage; just that they compare equal.
		CSRMatrix.Builder b = CSRMatrix.builder(3, 3)
				.set(0, 0, 1.0)
				.set(1, 1, 2.0)
				.set(2, 2, 3.0);
		CSRMatrix m1 = b.build();
		CSRMatrix m2 = b.build();
		assertNotSame(m1, m2);
		assertArrayRelAbsEquals(m1.toDense(), m2.toDense(), 1e-15);
	}
}
