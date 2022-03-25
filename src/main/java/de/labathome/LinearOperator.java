package de.labathome;

public interface LinearOperator {

	/** number of cols == input dimensionality */
	public int n();

	/** number of rows == output dimensionality */
	public int m();

	/**
	 * Apply the linear operator to a given vector.
	 *
	 * @param x [n] vector
	 * @return [m] result
	 */
	public double[] apply(double[] x);

	public default double[][] mat() {

		// TODO: fix this...

		int n = this.n();
		int m = this.m();
		double[][] A = new double[m][n];
		for (int i=0; i<n; ++i) {
			double[] e_i = new double[m];
			e_i[i] = 1.0;
			double[] e_i_result = this.apply(e_i);
			for (int j=0; j<m; ++j) {
				A[j][i] = e_i_result[j];
			}
		}

		return A;
	}

}
