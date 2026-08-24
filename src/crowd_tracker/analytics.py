"""Zone occupancy and alert logic built from local track observations."""

from __future__ import annotations

from collections import defaultdict

from .config import ZoneConfig
from .models import TrackObservation, ZoneAlert


class ZoneAnalytics:
    def __init__(self, zones: list[ZoneConfig]) -> None:
        self._zones = zones
        self._above_threshold: set[str] = set()

    def zones_for_bbox(self, bbox_xyxy: tuple[int, int, int, int]) -> tuple[str, ...]:
        x1, y1, x2, y2 = bbox_xyxy
        foot_point = ((x1 + x2) // 2, y2)
        matches = []
        for zone in self._zones:
            if _point_in_polygon_or_boundary(foot_point, zone.points):
                matches.append(zone.id)
        return tuple(matches)

    def alerts(self, observations: list[TrackObservation]) -> list[ZoneAlert]:
        observed_ids: dict[str, set[int]] = defaultdict(set)
        for observation in observations:
            for zone_id in observation.zones:
                observed_ids[zone_id].add(observation.track_id)

        alerts: list[ZoneAlert] = []
        for zone in self._zones:
            count = len(observed_ids[zone.id])
            if count >= zone.occupancy_alert_threshold:
                if zone.id not in self._above_threshold:
                    first = observations[0] if observations else None
                    if first is not None:
                        alerts.append(ZoneAlert(
                            camera_id=first.camera_id,
                            frame_index=first.frame_index,
                            timestamp_seconds=first.timestamp_seconds,
                            zone_id=zone.id,
                            count=count,
                            threshold=zone.occupancy_alert_threshold,
                        ))
                    self._above_threshold.add(zone.id)
            else:
                self._above_threshold.discard(zone.id)
        return alerts

    def draw(self, frame, counts: dict[str, int]) -> None:
        import cv2
        import numpy as np

        for zone in self._zones:
            polygon = np.asarray(zone.points, dtype=np.int32)
            count = counts.get(zone.id, 0)
            active = count >= zone.occupancy_alert_threshold
            color = (0, 0, 255) if active else (255, 180, 0)
            cv2.polylines(frame, [polygon], True, color, 2)
            x, y = zone.points[0]
            cv2.putText(
                frame,
                f"{zone.id}: {count}/{zone.occupancy_alert_threshold}",
                (x, max(24, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )


def _point_in_polygon_or_boundary(point: tuple[int, int], polygon: list[list[int]]) -> bool:
    """Ray-casting membership test that treats polygon edges as inside."""
    px, py = point
    inside = False
    previous_x, previous_y = polygon[-1]
    for current_x, current_y in polygon:
        if _is_on_segment(px, py, previous_x, previous_y, current_x, current_y):
            return True
        intersects = (current_y > py) != (previous_y > py)
        if intersects:
            crossing_x = (previous_x - current_x) * (py - current_y) / (previous_y - current_y) + current_x
            if px < crossing_x:
                inside = not inside
        previous_x, previous_y = current_x, current_y
    return inside


def _is_on_segment(px: int, py: int, x1: int, y1: int, x2: int, y2: int) -> bool:
    cross = (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)
    if cross != 0:
        return False
    return min(x1, x2) <= px <= max(x1, x2) and min(y1, y2) <= py <= max(y1, y2)
