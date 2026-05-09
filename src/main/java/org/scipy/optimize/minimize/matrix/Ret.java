package org.scipy.optimize.minimize.matrix;

/**
 * Mirrors the UJMP {@code Ret} enum so {@code abs(Ret.LINK)},
 * {@code lt(Ret.LINK, b)} etc. continue to compile after the import swap.
 *
 * <p>UJMP semantics:
 * <ul>
 *   <li>{@code LINK} — return a lazy view (no copy).</li>
 *   <li>{@code NEW} — return a fresh copy of the result.</li>
 *   <li>{@code ORIG} — mutate the receiver in place.</li>
 * </ul>
 *
 * <p>This in-tree port always returns a fresh result regardless of the
 * {@code Ret} value (the value is accepted only for API compatibility).
 * The handful of {@code Ret.ORIG} call sites in the trust-constr port were
 * rewritten to assign the returned matrix back to the variable.
 */
public enum Ret {
	LINK,
	NEW,
	ORIG
}
