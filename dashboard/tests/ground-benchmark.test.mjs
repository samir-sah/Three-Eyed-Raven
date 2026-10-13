import test from "node:test";
import assert from "node:assert/strict";
import { normalizeGroundComparison } from "../lib/ground-benchmark.js";

test("frozen score-target report uses report rows, never calibration or the training adapter", () => {
  const fixed = { threshold: .35, recall: .66, count_mae: 2.83 };
  const result = normalizeGroundComparison({
    target_reached: true, limitation: "development only",
    reports: [{ name: "winner", profile: { checkpoint: "yolo11x.pt", imgsz: 1536, augment: true },
      detection: { map50: .7096 }, fixed, calibrated: { threshold: .35, fixed },
      raw_counts: { mae: 2.9 }, speed: { images_per_second: 3.47 } }],
    evaluations: [{ name: "do_not_use_adapter" }],
    calibrations: [{ name: "do_not_use_calibration" }],
  }, { protocol: { neutralization: "ignored regions" } }, "winner");
  assert.equal(result.evaluations.length, 1);
  assert.equal(result.evaluations[0].detection.map50, .7096);
  assert.equal(result.evaluations[0].isActive, true);
  assert.equal(result.evaluations[0].default, fixed);
  assert.equal(result.evaluations[0].speed.decode_predict_fps, 3.47);
  assert.equal(result.evaluations[0].resolution, 1536);
  assert.equal(result.protocol.limitations, "development only");
  assert.equal(result.evaluations[0].rawCounts.mae, 2.9);
});

test("historical ground report schema is still readable without falsely marking a new candidate active", () => {
  const fixed = { threshold: .35, recall: .53 };
  const tuned = { threshold: .2, recall: .6 };
  const result = normalizeGroundComparison({ protocol: { limitations: "historical" }, evaluations: [
    { name: "old", checkpoint: "yolo11s.pt", resolution: 960, report_detection: { map50: .645 },
      report_default: fixed, report_tuned: tuned, selected_threshold: .2, speed: { decode_predict_fps: 35 } },
  ] }, {}, "different");
  assert.equal(result.evaluations[0].isActive, false);
  assert.equal(result.evaluations[0].default, fixed);
  assert.equal(result.evaluations[0].tuned, tuned);
  assert.equal(result.protocol.limitations, "historical");
});
