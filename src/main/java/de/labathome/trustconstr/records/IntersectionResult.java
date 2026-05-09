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
package de.labathome.trustconstr.records;

/**
 * Result of intersecting a parametric segment {@code x(t) = z + t*d} with a
 * trust-region or box constraint. The segment lies inside the constraint set
 * for {@code tA &le; t &le; tB} when {@link #intersect()} is {@code true}.
 */
public final class IntersectionResult {

	private double tA;
	private double tB;
	private boolean intersect;

	/**
	 * @param tA        lower end of the {@code t}-interval
	 * @param tB        upper end of the {@code t}-interval
	 * @param intersect {@code true} iff the segment enters the feasible region
	 */
	public IntersectionResult(double tA, double tB, boolean intersect) {
		this.tA = tA;
		this.tB = tB;
		this.intersect = intersect;
	}

	/** @return lower end of the {@code t}-interval */
	public double tA() {
		return tA;
	}

	/** @return upper end of the {@code t}-interval */
	public double tB() {
		return tB;
	}

	/** @return whether the segment enters the feasible region at all */
	public boolean intersect() {
		return intersect;
	}
}
