package org.scipy.optimize.minimize.records;

import org.scipy.optimize.minimize.NumDiff;
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

	public static Sparsity of(Matrix A) {
		int[] sparsityGroups = NumDiff.groupColumns(A);
		return new Sparsity(A, sparsityGroups);
	}

	public static Sparsity of(Matrix A, int[] order) {
		int[] sparsityGroups = NumDiff.groupColumns(A, order);
		return new Sparsity(A, sparsityGroups);
	}
}
