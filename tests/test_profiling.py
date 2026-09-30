import unittest

from crowd_tracker.profiling import StageProfiler


class ProfilingTests(unittest.TestCase):
    def test_reports_recorded_stage(self):
        profiler = StageProfiler()
        with profiler.measure("tracking"):
            pass
        report = profiler.summary()
        self.assertEqual(report[0]["stage"], "tracking")
        self.assertEqual(report[0]["calls"], 1)
        self.assertGreaterEqual(report[0]["mean_ms"], 0)
