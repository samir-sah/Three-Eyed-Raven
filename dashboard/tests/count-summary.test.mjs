import test from "node:test";
import assert from "node:assert/strict";
import { countSummary, sampleCountTimeline, countLogComplete } from "../lib/count-summary.mjs";
test("empty frames and camera-relative peak timestamps retained", () => {
  const summary = countSummary([{ frame_index: 0, timestamp_seconds: 0, count: 0 }, { frame_index: 1, timestamp_seconds: .1, count: 4 }, { frame_index: 2, timestamp_seconds: .2, count: 0 }]);
  assert.equal(summary.peakCount, 4);
  assert.equal(summary.peakSeconds, .1);
  assert.equal(summary.timeline.length, 3);
  assert.equal(summary.meanCount, 4 / 3);
  assert.equal(countSummary([]).meanCount, 0);
});
test("missing, duplicated or inconsistent count logs are not verified zero counts", () => {
  const rows = [{ frame_index: 0, timestamp_seconds: 0, count: 0, boxes_scores: [] }, { frame_index: 1, timestamp_seconds: .1, count: 1, boxes_scores: [[0, 0, 1, 1, .8]] }];
  assert.equal(countLogComplete(rows, 2), true);
  assert.equal(countLogComplete([], 2), false);
  assert.equal(countLogComplete([rows[0], rows[0]], 2), false);
  assert.equal(countLogComplete([{ ...rows[0], count: 5 }, rows[1]], 2), false);
});
test("downsampled chart retains brief peaks, troughs and endpoints", () => {
  const rows = Array.from({ length: 100 }, (_, frame) => ({ frame, seconds: frame / 10, count: frame === 17 ? 29 : frame === 21 ? 0 : 8 }));
  const sampled = sampleCountTimeline(rows);
  assert.equal(sampled[0].frame, 0);
  assert.equal(sampled.at(-1).frame, 99);
  assert.ok(sampled.some((row) => row.frame === 17));
  assert.ok(sampled.some((row) => row.frame === 21));
});
