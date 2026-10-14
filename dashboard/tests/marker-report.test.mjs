import test from "node:test";
import assert from "node:assert/strict";
import { mkdir, mkdtemp, rm, symlink, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { isMarkerCode, isMarkerUploadId, markerRunDirectory, markerSummaryPath, markerEvidencePath, markerTagPath, validateMarkerDemo, validateMarkerReport, readMarkerReport } from "../lib/marker-report.mjs";

const RUN = "upload_20261013180000_123abc";
function summary() {
  return { analysis_type: "anonymous_detection", config: { marker_demo: { enabled: true, confirmed_volunteer_demo: true, allowed_codes: ["V01", "V02", "V03", "V04"], min_reads: 3 } },
    cameras: ["aerial", "ground"].map((camera_id) => ({ camera_id, source_fps: 25, frames_processed: 100 })) };
}
function observation(code, first_frame = 2, last_frame = 10, read_frames = 3) {
  return { first_frame, last_frame, first_seconds: first_frame / 25, last_seconds: last_frame / 25, read_frames, evidence_file: `${code}.jpg` };
}
function camera(observations = {}, duplicate_codes = []) {
  return { frames_processed: 100, frames_scanned: 100, read_failures: 0, observations, duplicate_codes };
}
function report() {
  return { method: "explicit_qr_codes", scope: "controlled_marker_demo", min_reads: 3, shared_count: 999, shared_codes: ["V99"],
    cameras: { aerial: camera({ V01: observation("V01"), V02: observation("V02"), V03: observation("V03", 5, 5, 1) }),
      ground: camera({ V01: observation("V01", 3, 12, 4), V02: observation("V02"), V03: observation("V03"), V04: observation("V04") }, ["V02"]) } };
}
async function workspace(t) {
  const directory = await mkdtemp(path.join(os.tmpdir(), "raven-marker-report-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  return directory;
}

test("default, off and legacy runs have no QR report or zero-match conclusion", () => {
  for (const value of [{}, { analysis_type: "legacy_tracking", config: {} }, { config: { marker_demo: { enabled: false } } },
    { config: { marker_demo: { enabled: 1 } } }]) {
    assert.equal(readMarkerReport("nonexistent-run", value), null);
    assert.equal(validateMarkerReport(report(), value, RUN), null);
  }
});

test("minimum reads defaults to three only when omitted, with exact controlled opt-in fields", () => {
  const config = summary().config.marker_demo;
  delete config.min_reads;
  assert.deepEqual(validateMarkerDemo(config), { codes: ["V01", "V02", "V03", "V04"], minReads: 3 });
  for (const min_reads of [null, "3", 0, -1, 2.5, 20001]) assert.throws(() => validateMarkerDemo({ ...config, min_reads }));
  for (const allowed_codes of [[], ["V00"], ["V100"], ["V01", "V01"], ["Samir"], ["v01"], null]) assert.throws(() => validateMarkerDemo({ ...config, allowed_codes }));
  assert.throws(() => validateMarkerDemo({ ...config, confirmed_volunteer_demo: false }), /explicitly confirmed/);
  assert.throws(() => validateMarkerDemo({ ...config, reid_checkpoint: "people.pt" }), /unsupported fields/);
});

test("code and upload identifiers reject newlines, path characters and non-string coercion", () => {
  assert.equal(isMarkerCode("V01"), true); assert.equal(isMarkerCode("V99"), true);
  assert.equal(isMarkerUploadId(RUN), true);
  for (const value of ["V01\n", "V01\r", "V01\r\n", "V01 ", "../V01", "V00", "V100", null, { toString: () => "V01" }]) {
    assert.equal(isMarkerCode(value), false);
  }
  for (const value of [`${RUN}\n`, `${RUN}\r`, `../${RUN}`, RUN.toUpperCase(), null, { toString: () => RUN }]) {
    assert.equal(isMarkerUploadId(value), false);
  }
  const malformed = summary(); malformed.config.marker_demo.allowed_codes = ["V01\n"];
  assert.throws(() => validateMarkerDemo(malformed.config.marker_demo), /registered codes/);
  assert.throws(() => validateMarkerReport(report(), summary(), `${RUN}\n`), /anonymous-upload/);
});

test("shared codes are recomputed from eligible observations, not forged summary totals", () => {
  const result = validateMarkerReport(report(), summary(), RUN);
  assert.equal(result.status, "ready"); assert.equal(result.readerVerified, true);
  assert.equal(result.sharedCount, 1); assert.deepEqual(result.sharedCodes, ["V01"]);
  assert.deepEqual(result.ambiguousCodes, ["V02"]);
  assert.deepEqual(result.cameras[0].eligibleCodes, ["V01", "V02"]);
  assert.deepEqual(result.cameras[1].eligibleCodes, ["V01", "V03", "V04"]);
  assert.equal(result.cameras[0].observedCount, 3); assert.equal(result.cameras[1].observedCount, 4);
  assert.equal(result.evidence[0].views[0].evidenceUrl, `/api/marker-evidence/${RUN}/aerial/V01`);
  assert.match(result.warning, /not recognition of people or proof of identity/);
  assert.match(result.ambiguityWarning, /V02/);
});

test("duplicates in either view are globally excluded from shared evidence", () => {
  for (const id of ["aerial", "ground"]) {
    const value = report(); value.cameras.aerial.duplicate_codes = []; value.cameras.ground.duplicate_codes = [];
    value.cameras[id].duplicate_codes = ["V01", "V02"];
    assert.deepEqual(validateMarkerReport(value, summary(), RUN).sharedCodes, []);
  }
});

test("duplicate-only codes with null tag crops do not invalidate other shared codes", () => {
  for (const id of ["aerial", "ground"]) {
    const value = report();
    value.cameras[id].observations.V02.evidence_file = null;
    value.cameras[id].duplicate_codes = ["V02"];
    const result = validateMarkerReport(value, summary(), RUN);
    assert.equal(result.status, "ready"); assert.equal(result.readerVerified, true);
    assert.deepEqual(result.sharedCodes, ["V01"]); assert.equal(result.sharedCount, 1);
    assert.equal(result.cameras.find((camera) => camera.id === id).observations.find((row) => row.code === "V02").evidenceUrl, null);
    assert.equal(result.evidence.some((row) => row.code === "V02"), false);
  }
});

test("null tag crops do not bypass read reliability or duplicate checks", () => {
  const reliable = report(); reliable.cameras.aerial.observations.V01.evidence_file = null;
  const result = validateMarkerReport(reliable, summary(), RUN);
  assert.deepEqual(result.sharedCodes, ["V01"]);
  assert.equal(result.evidence[0].views[0].evidenceUrl, null);
  const weak = report(); weak.cameras.aerial.observations.V01 = { ...observation("V01", 5, 5, 1), evidence_file: null };
  assert.deepEqual(validateMarkerReport(weak, summary(), RUN).sharedCodes, []);
  const invalid = report(); invalid.cameras.aerial.observations.V01 = { ...observation("V01"), read_frames: 0, evidence_file: null };
  assert.throws(() => validateMarkerReport(invalid, summary(), RUN), /read bounds/);
});

test("a valid scan finding no codes is a measured zero, distinct from unavailable", () => {
  const value = report(); value.cameras = { aerial: camera(), ground: camera() };
  const result = validateMarkerReport(value, summary(), RUN);
  assert.equal(result.sharedCount, 0); assert.equal(result.status, "ready"); assert.deepEqual(result.evidence, []);
});

test("partial scan failures remain visible and do not imply absent tags", () => {
  const value = report(); value.cameras.aerial.read_failures = 2;
  const result = validateMarkerReport(value, summary(), RUN);
  assert.equal(result.status, "ready"); assert.equal(result.sharedCount, 1);
  assert.equal(result.cameras[0].readFailures, 2);
  assert.match(result.coverageWarning, /aerial: 2\/100/);
  assert.match(result.coverageWarning, /do not establish absence/);
});

test("wrong scope, method, threshold, unknown codes and camera roles are rejected", () => {
  for (const [key, value] of [["scope", "reidentification"], ["method", "appearance_matching"], ["min_reads", 2]]) {
    assert.throws(() => validateMarkerReport({ ...report(), [key]: value }, summary(), RUN));
  }
  const unknown = report(); unknown.cameras.aerial.observations.V99 = observation("V99");
  assert.throws(() => validateMarkerReport(unknown, summary(), RUN), /Unregistered/);
  const duplicate = report(); duplicate.cameras.aerial.duplicate_codes = ["V99"];
  assert.throws(() => validateMarkerReport(duplicate, summary(), RUN), /registered codes/);
  const extra = report(); extra.cameras.third_camera = camera();
  assert.throws(() => validateMarkerReport(extra, summary(), RUN), /unsupported fields/);
  const legacy = summary(); legacy.analysis_type = "legacy_tracking";
  assert.throws(() => validateMarkerReport(report(), legacy, RUN), /anonymous-upload/);
  assert.throws(() => validateMarkerReport(report(), summary(), "../old-run"), /anonymous-upload/);
});

test("frame ranges, read counts and matching nominal timestamps must fit camera metadata", () => {
  for (const change of [{ first_frame: -1 }, { first_frame: 11 }, { last_frame: 100 }, { read_frames: 0 }, { read_frames: 10 },
    { first_seconds: -1 }, { first_seconds: .3 }, { last_seconds: 4 }, { last_seconds: NaN }]) {
    const value = report(); Object.assign(value.cameras.aerial.observations.V01, change);
    assert.throws(() => validateMarkerReport(value, summary(), RUN), /bounds|timestamps/);
  }
  for (const change of [{ frames_processed: 99 }, { frames_scanned: 0 }, { frames_scanned: 101 }, { read_failures: -1 }, { read_failures: 100 }]) {
    const value = report(); Object.assign(value.cameras.aerial, change);
    assert.throws(() => validateMarkerReport(value, summary(), RUN), /coverage|bounds/);
  }
  const failures = report(); failures.cameras.aerial.frames_scanned = 3; failures.cameras.aerial.read_failures = 1;
  assert.throws(() => validateMarkerReport(failures, summary(), RUN), /read bounds/);
  for (const source_fps of [0, -1, NaN, "25"]) {
    const meta = summary(); meta.cameras[0].source_fps = source_fps;
    assert.throws(() => validateMarkerReport(report(), meta, RUN), /coverage|bounds/);
  }
});

test("tag-only filenames cannot traverse folders or refer to identity/face crops", () => {
  for (const evidence_file of ["../V01.jpg", "face.jpg", "V02.jpg", "V01.png", "V01.jpg\n"]) {
    const value = report(); value.cameras.aerial.observations.V01.evidence_file = evidence_file;
    assert.throws(() => validateMarkerReport(value, summary(), RUN), /tag-only evidence/);
  }
  const extra = report(); extra.cameras.aerial.observations.V01.face_crop = "face.jpg";
  assert.throws(() => validateMarkerReport(extra, summary(), RUN), /unsupported fields/);
  const value = report(); value.identity = "unexpected advisory data";
  assert.equal(Object.hasOwn(validateMarkerReport(value, summary(), RUN), "identity"), false);
});

test("opted-in missing/corrupt/inconsistent reports are unavailable, never zero", async (t) => {
  const directory = await workspace(t);
  let result = readMarkerReport(directory, summary());
  assert.equal(result.status, "unavailable"); assert.equal(result.sharedCount, null); assert.equal(result.readerVerified, false);
  await writeFile(path.join(directory, "markers.json"), "{invalid");
  assert.equal(readMarkerReport(directory, summary()).sharedCount, null);
  const value = report(); value.min_reads = 2;
  await writeFile(path.join(directory, "markers.json"), JSON.stringify(value));
  assert.equal(readMarkerReport(directory, summary()).status, "unavailable");
});

test("valid report on disk is sanitized and adds only code-tag evidence", async (t) => {
  const directory = await workspace(t);
  await writeFile(path.join(directory, "markers.json"), JSON.stringify(report()));
  const result = readMarkerReport(directory, summary());
  assert.equal(result.status, "ready"); assert.deepEqual(result.sharedCodes, ["V01"]);
  assert.match(result.evidence[0].views[0].evidenceUrl, /\/aerial\/V01$/);
  assert.equal(Object.hasOwn(result, "shared_count"), false);
});

test("regular run, summary, tag and evidence paths resolve inside their declared directories", async (t) => {
  const root = await workspace(t);
  const run = path.join(root, "artifacts", RUN);
  const evidence = path.join(run, "aerial", "marker_evidence", "V01.jpg");
  const tag = path.join(root, "dashboard", "public", "marker-tags", "V01.png");
  await mkdir(path.dirname(evidence), { recursive: true });
  await mkdir(path.dirname(tag), { recursive: true });
  await writeFile(path.join(run, "summary.json"), "{}");
  await writeFile(evidence, "generated tag-only test fixture");
  await writeFile(tag, "generated printable test fixture");
  assert.equal(markerRunDirectory(root, RUN), run);
  assert.equal(markerSummaryPath(run), path.join(run, "summary.json"));
  assert.equal(markerEvidencePath(run, "aerial", "V01"), evidence);
  assert.equal(markerTagPath(path.join(root, "dashboard"), "V01"), tag);
  assert.throws(() => markerRunDirectory(root, `../${RUN}`), /identifier/);
  assert.throws(() => markerEvidencePath(run, "../aerial", "V01"), /identifier/);
  assert.throws(() => markerTagPath(path.join(root, "dashboard"), "V01\n"), /identifier/);
});

test("run, camera and evidence-directory junctions cannot redirect tag evidence", async (t) => {
  for (const level of ["artifacts", "run", "camera", "evidence"]) {
    const root = await workspace(t);
    const outside = path.join(root, "outside");
    await mkdir(outside);
    const run = path.join(root, "artifacts", RUN);
    let redirected;
    if (level === "artifacts") redirected = path.join(root, "artifacts");
    if (level === "run") { await mkdir(path.dirname(run)); redirected = run; }
    if (level === "camera") { await mkdir(run, { recursive: true }); redirected = path.join(run, "aerial"); }
    if (level === "evidence") { await mkdir(path.join(run, "aerial"), { recursive: true }); redirected = path.join(run, "aerial", "marker_evidence"); }
    await symlink(outside, redirected, "junction");
    if (level === "artifacts" || level === "run") assert.throws(() => markerRunDirectory(root, RUN), /links|redirects/);
    else assert.throws(() => markerEvidencePath(run, "aerial", "V01"), /links|redirects/);
  }
});

test("printable-tag parent junctions are rejected, too", async (t) => {
  const root = await workspace(t);
  const outside = path.join(root, "outside"); await mkdir(outside);
  const dashboard = path.join(root, "dashboard"); await mkdir(path.join(dashboard, "public"), { recursive: true });
  await symlink(outside, path.join(dashboard, "public", "marker-tags"), "junction");
  assert.throws(() => markerTagPath(dashboard, "V01"), /links|redirects/);
});

test("redirected report ancestry is unavailable rather than read outside the run", async (t) => {
  const root = await workspace(t);
  const outside = path.join(root, "outside"); await mkdir(outside);
  await writeFile(path.join(outside, "markers.json"), JSON.stringify(report()));
  const alias = path.join(root, "alias"); await symlink(outside, alias, "junction");
  assert.equal(readMarkerReport(alias, summary()).status, "unavailable");
});
