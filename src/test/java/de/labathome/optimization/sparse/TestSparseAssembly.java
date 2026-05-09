package de.labathome.optimization.sparse;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.scipy.optimize.minimize.sparse.CSRMatrix;
import org.scipy.optimize.minimize.sparse.SparseAssembly;

import de.labathome.optimization.RelAbsAssertions;

class TestSparseAssembly {

	private static final double TOL = 1.0e-14;

	@Test
	void testVstack() {
		CSRMatrix a = CSRMatrix.fromDense(new double[][] {
				{ 1.0, 2.0 },
				{ 0.0, 3.0 },
		});
		CSRMatrix b = CSRMatrix.fromDense(new double[][] {
				{ 4.0, 0.0 },
				{ 5.0, 6.0 },
				{ 0.0, 7.0 },
		});
		CSRMatrix v = SparseAssembly.vstack(a, b);
		double[][] expected = {
				{ 1.0, 2.0 },
				{ 0.0, 3.0 },
				{ 4.0, 0.0 },
				{ 5.0, 6.0 },
				{ 0.0, 7.0 },
		};
		RelAbsAssertions.assertArrayRelAbsEquals(expected, v.toDense(), TOL);
	}

	@Test
	void testVstackRejectsColumnMismatch() {
		CSRMatrix a = CSRMatrix.fromDense(new double[][] { {1.0, 2.0} });
		CSRMatrix b = CSRMatrix.fromDense(new double[][] { {1.0, 2.0, 3.0} });
		Assertions.assertThrows(IllegalArgumentException.class, () -> SparseAssembly.vstack(a, b));
	}

	@Test
	void testHstack() {
		CSRMatrix a = CSRMatrix.fromDense(new double[][] {
				{ 1.0, 2.0 },
				{ 0.0, 3.0 },
		});
		CSRMatrix b = CSRMatrix.fromDense(new double[][] {
				{ 4.0, 0.0, 5.0 },
				{ 6.0, 7.0, 0.0 },
		});
		CSRMatrix h = SparseAssembly.hstack(a, b);
		double[][] expected = {
				{ 1.0, 2.0, 4.0, 0.0, 5.0 },
				{ 0.0, 3.0, 6.0, 7.0, 0.0 },
		};
		RelAbsAssertions.assertArrayRelAbsEquals(expected, h.toDense(), TOL);
	}

	@Test
	void testHstackRejectsRowMismatch() {
		CSRMatrix a = CSRMatrix.fromDense(new double[][] { {1.0}, {2.0} });
		CSRMatrix b = CSRMatrix.fromDense(new double[][] { {1.0} });
		Assertions.assertThrows(IllegalArgumentException.class, () -> SparseAssembly.hstack(a, b));
	}

	@Test
	void testBlockArrayKKTPattern() {
		// [[I_3, A^T], [A, 0]] -- the projection KKT shape from projections.py:99
		CSRMatrix i3 = CSRMatrix.eye(3);
		double[][] aDense = {
				{ 1.0, 2.0, 3.0 },
				{ 0.0, 1.0, 0.0 },
		};
		CSRMatrix a = CSRMatrix.fromDense(aDense);
		CSRMatrix at = a.transpose().toCSR();
		CSRMatrix kkt = SparseAssembly.blockArray(new CSRMatrix[][] {
				{ i3, at },
				{ a,  null },
		});
		double[][] expected = {
				{ 1.0, 0.0, 0.0, 1.0, 0.0 },
				{ 0.0, 1.0, 0.0, 2.0, 1.0 },
				{ 0.0, 0.0, 1.0, 3.0, 0.0 },
				{ 1.0, 2.0, 3.0, 0.0, 0.0 },
				{ 0.0, 1.0, 0.0, 0.0, 0.0 },
		};
		RelAbsAssertions.assertArrayRelAbsEquals(expected, kkt.toDense(), TOL);
	}

	@Test
	void testBlockArrayRejectsNullBand() {
		CSRMatrix a = CSRMatrix.eye(2);
		Assertions.assertThrows(IllegalArgumentException.class,
				() -> SparseAssembly.blockArray(new CSRMatrix[][] { { a, null }, { null, null } }));
	}

	@Test
	void testAssembleJacobianWithSlacks() {
		// J_eq is 1x3, J_ineq is 2x3, s has length 2.
		// Result shape: 3 rows, 5 cols -- [[J_eq, 0_{1x2}], [J_ineq, diag(s)]]
		double[][] jEqDense = {
				{ 1.0, 2.0, 0.0 },
		};
		double[][] jIneqDense = {
				{ 0.0, 3.0, 4.0 },
				{ 5.0, 0.0, 6.0 },
		};
		double[] s = {7.0, 8.0};
		CSRMatrix jEq = CSRMatrix.fromDense(jEqDense);
		CSRMatrix jIneq = CSRMatrix.fromDense(jIneqDense);
		CSRMatrix j = SparseAssembly.assembleJacobianWithSlacks(jEq, jIneq, s);
		Assertions.assertEquals(3, j.rows());
		Assertions.assertEquals(5, j.cols());
		double[][] expected = {
				{ 1.0, 2.0, 0.0, 0.0, 0.0 },
				{ 0.0, 3.0, 4.0, 7.0, 0.0 },
				{ 5.0, 0.0, 6.0, 0.0, 8.0 },
		};
		RelAbsAssertions.assertArrayRelAbsEquals(expected, j.toDense(), TOL);
	}

	@Test
	void testAssembleJacobianWithSlacksMatchesGenericBlockArray() {
		// Cross-check the optimised path against the equivalent block_array call.
		double[][] jEqDense = {
				{ 1.0, 0.0, 0.0, 2.0 },
				{ 0.0, 3.0, 0.0, 0.0 },
		};
		double[][] jIneqDense = {
				{ 0.0, 0.0, 4.0, 5.0 },
				{ 6.0, 0.0, 0.0, 0.0 },
				{ 0.0, 7.0, 8.0, 0.0 },
		};
		double[] s = {9.0, 10.0, 11.0};
		CSRMatrix jEq = CSRMatrix.fromDense(jEqDense);
		CSRMatrix jIneq = CSRMatrix.fromDense(jIneqDense);

		CSRMatrix optimised = SparseAssembly.assembleJacobianWithSlacks(jEq, jIneq, s);
		// Build the same thing via the generic blockArray API.
		CSRMatrix diagS = CSRMatrix.fromDense(new double[][] {
				{ s[0], 0.0,  0.0  },
				{ 0.0,  s[1], 0.0  },
				{ 0.0,  0.0,  s[2] },
		});
		CSRMatrix generic = SparseAssembly.blockArray(new CSRMatrix[][] {
				{ jEq,   null  },
				{ jIneq, diagS },
		});
		RelAbsAssertions.assertArrayRelAbsEquals(generic.toDense(), optimised.toDense(), TOL);
	}
}
