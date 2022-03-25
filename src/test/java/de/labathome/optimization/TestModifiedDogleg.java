package de.labathome.optimization;

import org.junit.jupiter.api.Test;

class TestModifiedDogleg {

	@Test
	void testCauchyPointEqualToNewtonPoint() {
		final double tolerance = 1.0e-15;

		final double[][] A = { { 1.0, 8.0 } };
		final double[] b = { -16.0 };

		// TODO: need Projections now

	}

}
