package org.scipy.optimize.minimize.records;

import org.ujmp.core.Matrix;

public final class Sparsity {

	private Matrix A;
	private int[] sparsityGroups;

	public Sparsity(Matrix A, int[] sparsityGroups) {
		this.A = A;
		this.sparsityGroups = sparsityGroups;
	}

	public Matrix A() {
		return A;
	}

	public int[] sparsityGroups() {
		return sparsityGroups;
	}
}
