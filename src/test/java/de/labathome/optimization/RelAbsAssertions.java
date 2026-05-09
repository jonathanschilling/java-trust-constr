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
package de.labathome.optimization;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.fail;

import java.util.Locale;

/**
 * JUnit-style assertions that compare floating-point values with the
 * "relative-or-absolute" error metric {@code |actual - expected| / (1 + |expected|)}
 * (Gill, Murray &amp; Wright, "Practical Optimization", 1984). For values much
 * smaller than 1 this behaves like an absolute-difference comparison; for
 * values much larger than 1 it behaves like a relative-difference comparison.
 *
 * <p>Lives in test sources because no production code depends on it.
 */
public final class RelAbsAssertions {

	private RelAbsAssertions() {}

	public static double relAbsError(double expected, double actual) {
		return Math.abs(actual - expected) / (1.0 + Math.abs(expected));
	}

	public static boolean assertRelAbsEquals(double expected, double actual, double tolerance) {
		return assertRelAbsEquals(expected, actual, tolerance, "");
	}

	public static boolean assertRelAbsEquals(double expected, double actual, double tolerance, String locString) {
		if ((!Double.isFinite(expected) && Double.isFinite(actual))
				|| (Double.isFinite(expected) && !Double.isFinite(actual))) {
			fail(String.format(Locale.ENGLISH,
					locString + " expected %g, actual %g --> one of them is not finite!",
					expected, actual));
			return false;
		} else if (!Double.isFinite(expected) && !Double.isFinite(actual)) {
			fail(String.format(Locale.ENGLISH,
					locString + " expected %g, actual %g --> both are not finite!",
					expected, actual));
			return false;
		}
		double err = relAbsError(expected, actual);
		if (err > tolerance) {
			fail(String.format(Locale.ENGLISH,
					locString + " expected %g, actual %g (rel/abs error %g, tolerance %g)",
					expected, actual, err, tolerance));
			return false;
		}
		return true;
	}

	public static boolean assertArrayRelAbsEquals(double[] expected, double[] actual, double tolerance) {
		return assertArrayRelAbsEquals(expected, actual, tolerance, "");
	}

	public static boolean assertArrayRelAbsEquals(double[] expected, double[] actual, double tolerance, String locString) {
		if (expected == null) {
			assertNull(actual);
			return true;
		}
		assertNotNull(actual);
		assertEquals(expected.length, actual.length);
		boolean allGood = true;
		for (int i = 0; i < expected.length; ++i) {
			allGood &= assertRelAbsEquals(expected[i], actual[i], tolerance,
					locString + "[" + i + "]");
		}
		return allGood;
	}

	public static boolean assertArrayRelAbsEquals(double[][] expected, double[][] actual, double tolerance) {
		return assertArrayRelAbsEquals(expected, actual, tolerance, "");
	}

	public static boolean assertArrayRelAbsEquals(double[][] expected, double[][] actual, double tolerance, String locString) {
		if (expected == null) {
			assertNull(actual);
			return true;
		}
		assertNotNull(actual);
		assertEquals(expected.length, actual.length);
		boolean allGood = true;
		for (int i = 0; i < expected.length; ++i) {
			allGood &= assertArrayRelAbsEquals(expected[i], actual[i], tolerance,
					locString + "[" + i + "]");
		}
		return allGood;
	}
}
