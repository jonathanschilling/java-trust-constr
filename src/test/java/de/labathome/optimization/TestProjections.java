package de.labathome.optimization;

import org.junit.jupiter.api.Test;

public class TestProjections {

	@Test
	void testNullspaceAndLeastSquaresSparse() {

		double[][] A = {
				{1, 2, 3, 4, 0, 5, 0, 7},
				{0, 8, 7, 0, 1, 5, 9, 0},
				{1, 0, 0, 0, 0, 1, 2, 3}
		};

		double[][] testPoints = {
				{1, 2, 3, 4, 5, 6, 7, 8},
                {1, 10, 3, 0, 1, 6, 7, 8},
                {1.12, 10, 0, 0, 100000, 6, 0.7, 8}
		};

	}

}
