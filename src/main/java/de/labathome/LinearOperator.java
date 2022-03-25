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

}
