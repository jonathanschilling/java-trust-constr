package de.labathome.trustconstr.report;

/**
 * Basic 7-column progress report used for bound-constrained problems.
 * Counterpart to scipy's {@code _trustregion_constr/report.py:BasicReport}.
 *
 * <p>Columns: {@code niter | f evals | CG iter | obj func | tr radius |
 * opt | c viol}.
 */
public class BasicReport extends ReportBase {

	private static final String[] COLUMN_NAMES = {
			"niter", "f evals", "CG iter", "obj func",
			"tr radius", "opt", "c viol"};
	private static final int[] COLUMN_WIDTHS = {7, 7, 7, 13, 10, 10, 10};
	private static final String[] ITERATION_FORMATS = {
			"7d", "7d", "7d", "+13.4e", "10.2e", "10.2e", "10.2e"};

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
