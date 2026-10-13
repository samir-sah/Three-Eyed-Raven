// Counts come from every processed image, including frames with no detections.
export function countSummary(rows) {
  const timeline = rows.filter((row) => Number.isInteger(row.count) && row.count >= 0 && Number.isFinite(row.timestamp_seconds))
    .map((row) => ({ frame: row.frame_index, seconds: row.timestamp_seconds, count: row.count }));
  const busiest = timeline.reduce((best, row) => row.count > best.count ? row : best, { frame: 0, seconds: 0, count: 0 });
  return { timeline, peakCount: busiest.count, peakFrame: busiest.frame, peakSeconds: busiest.seconds,
    meanCount: timeline.length ? timeline.reduce((sum, row) => sum + row.count, 0) / timeline.length : 0 };
}

export function sampleCountTimeline(rows, buckets = 12) {
  if (rows.length <= buckets * 2) return rows;
  const selected = new Set([0, rows.length - 1]);
  const width = Math.ceil(rows.length / buckets);
  for (let start = 0; start < rows.length; start += width) {
    let low = start, high = start;
    for (let index = start; index < Math.min(start + width, rows.length); index++) {
      if (rows[index].count < rows[low].count) low = index;
      if (rows[index].count > rows[high].count) high = index;
    }
    selected.add(low); selected.add(high);
  }
  return [...selected].sort((a, b) => a - b).map((index) => rows[index]);
}

export function countLogComplete(rows, expectedFrames) {
  return Number.isInteger(expectedFrames) && expectedFrames > 0 && rows.length === expectedFrames && rows.every((row, index) =>
    row.frame_index === index && Number.isFinite(row.timestamp_seconds) && row.timestamp_seconds >= 0 &&
    (index === 0 || row.timestamp_seconds >= rows[index - 1].timestamp_seconds) && Number.isInteger(row.count) && row.count >= 0 &&
    Array.isArray(row.boxes_scores) && row.boxes_scores.length === row.count);
}
