/*
 * Copyright 2026 Jonathan Schilling
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package de.labathome.trustconstr.report;

/**
 * Equality-constrained SQP progress report (9 columns).
 * Counterpart to scipy's {@code _trustregion_constr/report.py:SQPReport}.
 *
 * <p>Columns: {@code niter | f evals | CG iter | obj func | tr radius |
 * opt | c viol | penalty | CG stop}.
 */
public class SQPReport extends ReportBase {

	private static final String[] COLUMN_NAMES = {
			"niter", "f evals", "CG iter", "obj func", "tr radius",
			"opt", "c viol", "penalty", "CG stop"};
	private static final int[] COLUMN_WIDTHS = {7, 7, 7, 13, 10, 10, 10, 10, 7};
	private static final String[] ITERATION_FORMATS = {
			"7d", "7d", "7d", "+13.4e", "10.2e", "10.2e", "10.2e", "10.2e", "7d"};

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
