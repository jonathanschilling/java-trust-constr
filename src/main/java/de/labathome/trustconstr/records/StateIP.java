package de.labathome.trustconstr.records;

/**
 * IP-path extension of {@link State}: adds the log-barrier coefficient and
 * the barrier-subproblem inner-loop tolerance carried alongside the standard
 * outer-iteration fields.
 */
public class StateIP extends State {

	/** Current log-barrier coefficient. */
	public double barrierParameter;

	/** Current barrier-subproblem tolerance. */
	public double barrierTolerance;
}
