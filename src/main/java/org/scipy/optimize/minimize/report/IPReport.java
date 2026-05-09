package org.scipy.optimize.minimize.report;

/**
 * Trust-region interior-point progress report (10 columns).
 * Counterpart to scipy's {@code _trustregion_constr/report.py:IPReport}.
 *
 * <p>Columns: {@code niter | f evals | CG iter | obj func | tr radius |
 * opt | c viol | penalty | barrier param | CG stop}.
 */
public class IPReport extends ReportBase {

	private static final String[] COLUMN_NAMES = {
			"niter", "f evals", "CG iter", "obj func", "tr radius",
			"opt", "c viol", "penalty", "barrier param", "CG stop"};
	private static final int[] COLUMN_WIDTHS = {7, 7, 7, 13, 10, 10, 10, 10, 13, 7};
	private static final String[] ITERATION_FORMATS = {
			"7d", "7d", "7d", "+13.4e", "10.2e", "10.2e", "10.2e",
			"10.2e", "13.2e", "7d"};

	@Override
	protected String[] columnNames() {
		return COLUMN_NAMES;
	}

	@Override
	protected int[] columnWidths() {
		return COLUMN_WIDTHS;
	}

	@Override
	protected String[] iterationFormats() {
		return ITERATION_FORMATS;
	}
}
