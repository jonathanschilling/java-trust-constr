package org.scipy.optimize.minimize.report;

import java.io.PrintStream;
import java.util.Locale;

/**
 * Abstract base for trust-constr per-iteration progress reports.
 * Counterpart to scipy's {@code _trustregion_constr/report.py:ReportBase}.
 *
 * <p>Concrete subclasses ({@link BasicReport}, {@link SQPReport},
 * {@link IPReport}) define column names, widths, and per-iteration
 * format strings; this base handles the actual rendering of the header,
 * data rows, and footer.
 */
public abstract class ReportBase {

	/** Per-column header labels. */
	protected abstract String[] columnNames();

	/** Per-column display widths (number of characters per cell). */
	protected abstract int[] columnWidths();

	/**
	 * Per-column {@link java.util.Formatter} format specs (e.g. {@code "^7"}
	 * for a centered 7-char field, {@code "^+13.4e"} for centered scientific
	 * notation). The leading {@code %} and trailing {@code s}/{@code f}/etc.
	 * are added by {@link #printIteration}.
	 */
	protected abstract String[] iterationFormats();

	/**
	 * Print the column header line and a separator. Output goes to
	 * {@link System#out} (writeable target tests can override
	 * {@link System#setOut(PrintStream)}).
	 */
	public void printHeader() {
		printHeader(System.out);
	}

	/**
	 * Print the column header to a specified target (used by tests).
	 *
	 * @param out target stream
	 */
	public void printHeader(PrintStream out) {
		String[] names = columnNames();
		int[] widths = columnWidths();
		StringBuilder fmt = new StringBuilder("|");
		for (int w : widths) {
			fmt.append("%-").append(w).append("s|");
		}
		fmt.append('\n');
		// Center each name in its width: pad with spaces on each side.
		String[] centered = new String[names.length];
		for (int i = 0; i < names.length; ++i) {
			centered[i] = center(names[i], widths[i]);
		}
		out.printf(Locale.ENGLISH, fmt.toString(), (Object[]) centered);
		StringBuilder sep = new StringBuilder("|");
		for (int w : widths) {
			for (int k = 0; k < w; ++k) sep.append('-');
			sep.append('|');
		}
		out.println(sep);
	}

	/**
	 * Print a single iteration row to {@link System#out}.
	 *
	 * @param values per-column values, in the order matching
	 *               {@link #columnNames()} and {@link #iterationFormats()}
	 */
	public void printIteration(Object... values) {
		printIteration(System.out, values);
	}

	/**
	 * Print a single iteration row to a specified target (used by tests).
	 *
	 * @param out    target stream
	 * @param values per-column values, in column order
	 */
	public void printIteration(PrintStream out, Object... values) {
		String[] formats = iterationFormats();
		if (values.length != formats.length) {
			throw new IllegalArgumentException("expected " + formats.length
					+ " values, got " + values.length);
		}
		StringBuilder fmt = new StringBuilder("|");
		for (String f : formats) {
			fmt.append('%').append(f).append('|');
		}
		fmt.append('\n');
		out.printf(Locale.ENGLISH, fmt.toString(), values);
	}

	/** Print a blank line as the report footer. */
	public void printFooter() {
		printFooter(System.out);
	}

	/**
	 * Print the footer to a specified target (used by tests).
	 *
	 * @param out target stream
	 */
	public void printFooter(PrintStream out) {
		out.println();
	}

	private static String center(String s, int width) {
		if (s.length() >= width) return s;
		int total = width - s.length();
		int left = total / 2;
		int right = total - left;
		StringBuilder sb = new StringBuilder(width);
		for (int i = 0; i < left; ++i) sb.append(' ');
		sb.append(s);
		for (int i = 0; i < right; ++i) sb.append(' ');
		return sb.toString();
	}
}
