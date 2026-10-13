import test from "node:test";
import assert from "node:assert/strict";
import { chooseSelection, recordingKey, recordingsFromRuns, selectedRun } from "../lib/recordings.mjs";

const clip = (id, source, date = "2026-10-10T00:00:00Z") => ({ id, hasVideo: true, updatedAt: date, metrics: { source } });

test("different model runs on the same Windows source are one recording", () => {
  const runs = [clip("model_a", "S:\\video\\crowd.mp4"), clip("model_b", "s:/video/crowd.mp4")];
  assert.equal(recordingsFromRuns(runs).length, 1);
});

test("a newer unrelated video or model output does not replace the pinned run", () => {
  const original = clip("old", "S:/crowd.mp4");
  const saved = { runId: original.id, recordingKey: recordingKey(original) };
  const runs = [clip("upload_new", "S:/new.mp4", "2026-10-13T00:00:00Z"), clip("new_model", "S:/crowd.mp4", "2026-10-13T00:00:00Z"), original];
  assert.deepEqual(chooseSelection(runs, saved, null), saved);
  assert.equal(selectedRun(runs, saved).id, "old");
});

test("first use prefers uploaded footage over a newer sample", () => {
  assert.equal(chooseSelection([clip("sample", "S:/sample.mp4", "2026-10-13T00:00:00Z"), clip("upload_1", "S:/mine.mp4")], null, null).runId, "upload_1");
});

test("missing saved footage never silently falls back to another recording", () => {
  const saved = { runId: "removed", recordingKey: "missing" };
  const runs = [clip("demo", "S:/demo.mp4")];
  assert.deepEqual(chooseSelection(runs, null, saved), saved);
  assert.equal(selectedRun(runs, saved), null);
});

test("unplayable benchmarks are excluded and both camera sources define a pair", () => {
  const pair = { ...clip("pair", "unused"), isMulti: true, streams: [{ source: "S:/a.mp4" }, { source: "S:/b.mp4" }] };
  const other = { ...pair, id: "other", streams: [{ source: "S:/a.mp4" }, { source: "S:/c.mp4" }] };
  assert.equal(recordingsFromRuns([pair, other, { ...clip("metrics", "S:/x"), hasVideo: false }]).length, 2);
});
