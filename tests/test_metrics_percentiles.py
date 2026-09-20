"""
Tests for MetricsCollector Percentile Calculation.
Verifies boundary conditions (empty list, single item) and linear interpolation on distributions.
"""
import unittest
from app.orchestrator.metrics_collector import MetricsCollector

class TestMetricsPercentiles(unittest.TestCase):

    def test_percentile_empty_list(self):
        """Verifies empty list returns 0.0 for any percentile."""
        self.assertEqual(MetricsCollector._percentile([], 0.5), 0.0)
        self.assertEqual(MetricsCollector._percentile([], 0.99), 0.0)

    def test_percentile_single_value(self):
        """Verifies single item list returns that exact value for all percentiles."""
        val = 145.50
        self.assertEqual(MetricsCollector._percentile([val], 0.0), val)
        self.assertEqual(MetricsCollector._percentile([val], 0.5), val)
        self.assertEqual(MetricsCollector._percentile([val], 0.90), val)
        self.assertEqual(MetricsCollector._percentile([val], 0.99), val)
        self.assertEqual(MetricsCollector._percentile([val], 1.0), val)

    def test_percentile_linear_interpolation(self):
        """Verifies linear interpolation accurately computes standard percentiles."""
        # 10 values from 10 to 100
        values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        # p50: index = (10-1)*0.5 = 4.5 -> (50 + 60)/2 = 55.0
        self.assertEqual(MetricsCollector._percentile(values, 0.50), 55.0)
        # p90: index = 9 * 0.9 = 8.1 -> 90 + 0.1*(100-90) = 91.0
        self.assertEqual(MetricsCollector._percentile(values, 0.90), 91.0)
        # p95: index = 9 * 0.95 = 8.55 -> 90 + 0.55*10 = 95.5
        self.assertEqual(MetricsCollector._percentile(values, 0.95), 95.5)
        # p99: index = 9 * 0.99 = 8.91 -> 90 + 0.91*10 = 99.1
        self.assertEqual(MetricsCollector._percentile(values, 0.99), 99.1)

if __name__ == "__main__":
    unittest.main()
