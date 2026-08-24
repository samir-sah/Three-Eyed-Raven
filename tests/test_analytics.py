import unittest

from crowd_tracker.analytics import ZoneAnalytics
from crowd_tracker.config import ZoneConfig
from crowd_tracker.models import TrackObservation


class AnalyticsTests(unittest.TestCase):
    def test_zone_and_alert_are_triggered_once_until_recovery(self):
        analytics = ZoneAnalytics([ZoneConfig("gate", [[0, 0], [100, 0], [100, 100], [0, 100]], 2)])
        first_frame = [
            TrackObservation("camera_1", 0, 0.0, 1, (10, 10, 30, 80), 0.9, analytics.zones_for_bbox((10, 10, 30, 80))),
            TrackObservation("camera_1", 0, 0.0, 2, (40, 10, 60, 80), 0.9, analytics.zones_for_bbox((40, 10, 60, 80))),
        ]
        self.assertEqual(analytics.alerts(first_frame)[0].zone_id, "gate")
        self.assertEqual(analytics.alerts(first_frame), [])
        analytics.alerts([])
        self.assertEqual(analytics.alerts(first_frame)[0].count, 2)


if __name__ == "__main__":
    unittest.main()
