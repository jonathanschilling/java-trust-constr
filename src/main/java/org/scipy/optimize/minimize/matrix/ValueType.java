package org.scipy.optimize.minimize.matrix;

/**
 * Stand-in for UJMP's {@code org.scipy.optimize.minimize.matrix.ValueType}.
 *
 * <p>The trust-constr port only ever uses {@link #DOUBLE}, so this enum
 * exists purely to keep call sites that branch on {@code getValueType()}
 * compiling without rewriting them.
 */
public enum ValueType {
	DOUBLE,
	FLOAT,
	INT,
	LONG,
	BOOLEAN,
	STRING,
	OBJECT
}
