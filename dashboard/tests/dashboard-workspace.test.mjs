import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { validateDashboardWorkspace, parseDashboardWorkspace, readDashboardWorkspace, isCurrentWorkspaceJob, isCurrentWorkspaceUpload } from "../lib/dashboard-workspace.mjs";

const ROOT = path.resolve(os.tmpdir(), "raven-upload-workspace-fixture");
const RUN = "upload_20261013170100_a1b2c3";
function marker() {
  return { schema_version: 1, scope: "fresh_anonymous_uploads", workspace_id: "3b40aa02-62e2-40f4-8c81-dc1fe0aea7ec", created_at: "2026-10-13T17:00:50.639Z" };
}
function job() {
  return { runId: RUN, workspaceId: marker().workspace_id, status: "completed", createdAt: "2026-10-13T17:01:00.000Z", sourceNames: { aerial: "my-drone.mp4", ground: "my-cctv.mp4" } };
}
function summary() {
  const cameras = ["aerial", "ground"].map((id) => ({ id, source: path.join(ROOT, "data", "uploads", RUN, `${id}.mp4`) }));
  return {
    run_type: "multi_stream", analysis_type: "anonymous_detection", completed_at: "2026-10-13T17:02:00.000Z",
    config: { dashboard_workspace_id: marker().workspace_id, upload_run_id: RUN, cameras },
    cameras: cameras.map((camera) => ({ camera_id: camera.id, source: camera.source })),
  };
}
function accepted(result = summary(), upload = job(), workspace = marker(), runId = RUN) {
  return isCurrentWorkspaceUpload(runId, result, upload, workspace, ROOT);
}

test("valid workspace is sanitized without changing the saved stable UUID", () => {
  const value = marker();
  assert.deepEqual(validateDashboardWorkspace(value), value);
  assert.deepEqual(parseDashboardWorkspace(JSON.stringify(value)), value);
  assert.notEqual(validateDashboardWorkspace(value), value);
  assert.equal(validateDashboardWorkspace({ ...value, workspace_id: value.workspace_id.toUpperCase() }).workspace_id, value.workspace_id);
});

test("invalid JSON, schemas, scope, UUID and creation times fail closed", () => {
  assert.throws(() => parseDashboardWorkspace("{broken"), /invalid JSON/);
  for (const value of [null, [], {}, "old", { ...marker(), another: true }]) assert.throws(() => validateDashboardWorkspace(value));
  for (const [key, value] of [
    ["schema_version", 2], ["schema_version", "1"], ["scope", "all_runs"],
    ["workspace_id", "legacy"], ["workspace_id", "3b40aa02-62e2-10f4-8c81-dc1fe0aea7ec"],
    ["created_at", "not a date"], ["created_at", "2026-02-31T17:00:50.639Z"], ["created_at", null],
  ]) assert.throws(() => validateDashboardWorkspace({ ...marker(), [key]: value }));
});

test("filesystem marker remains stable; missing or corrupt file never discovers old data", async (t) => {
  const root = await mkdtemp(path.join(os.tmpdir(), "raven-workspace-test-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  assert.throws(() => readDashboardWorkspace(root), /unavailable/);
  await writeFile(path.join(root, "dashboard_workspace.json"), JSON.stringify(marker()));
  assert.deepEqual(readDashboardWorkspace(root), marker());
  assert.deepEqual(readDashboardWorkspace(root), marker());
  await writeFile(path.join(root, "dashboard_workspace.json"), "{broken");
  assert.throws(() => readDashboardWorkspace(root), /invalid JSON/);
});

test("only marked fresh jobs are available to polling, including processing and failures", () => {
  for (const status of ["queued", "processing", "completed", "failed"]) assert.equal(isCurrentWorkspaceJob({ ...job(), status }, marker()), true);
  for (const invalid of [null, {}, { ...job(), workspaceId: null }, { ...job(), workspaceId: "2b40aa02-62e2-40f4-8c81-dc1fe0aea7ec" },
    { ...job(), createdAt: "2026-10-01T17:01:00.000Z" }, { ...job(), createdAt: "unknown" },
    { ...job(), runId: "two_stream_demo" }, { ...job(), runId: "../upload_20261013170100_a1b2c3" }, { ...job(), status: "unknown" }]) {
    assert.equal(isCurrentWorkspaceJob(invalid, marker()), false);
  }
});

test("a newly received completed anonymous two-view upload is accepted", () => {
  assert.equal(accepted(), true);
  const upperExtension = summary();
  upperExtension.config.cameras[0].source = path.join(ROOT, "data", "uploads", RUN, "aerial.MOV");
  upperExtension.cameras[0].source = upperExtension.config.cameras[0].source;
  assert.equal(accepted(upperExtension), true);
});

test("legacy/demo/old anonymous runs never become a fallback even when some markers match", () => {
  for (const change of [{ run_type: "single_stream" }, { analysis_type: "legacy_tracking" }, { analysis_type: undefined }]) {
    assert.equal(accepted({ ...summary(), ...change }), false);
  }
  const old = summary(); delete old.config.dashboard_workspace_id;
  assert.equal(accepted(old), false);
  const wrong = summary(); wrong.config.dashboard_workspace_id = "2b40aa02-62e2-40f4-8c81-dc1fe0aea7ec";
  assert.equal(accepted(wrong), false);
  assert.equal(accepted(summary(), job(), null), false);
  assert.equal(accepted(summary(), null), false);
  assert.equal(accepted(summary(), job(), marker(), "live_preview"), false);
});

test("unfinished, time-incoherent or differently named upload results are excluded", () => {
  for (const status of ["queued", "processing", "failed"]) assert.equal(accepted(summary(), { ...job(), status }), false);
  assert.equal(accepted({ ...summary(), completed_at: undefined }), false);
  assert.equal(accepted({ ...summary(), completed_at: "2026-10-13T17:00:00.000Z" }), false);
  const differentRun = summary(); differentRun.config.upload_run_id = "upload_20261013170100_000000";
  assert.equal(accepted(differentRun), false);
  assert.equal(accepted(summary(), { ...job(), sourceNames: { aerial: "", ground: "cctv.mp4" } }), false);
});

test("reprocessed old source paths cannot masquerade as newly received uploads", () => {
  for (const source of [path.join(ROOT, "data", "uploads", "old_run", "aerial.mp4"), path.join(ROOT, "data", "uploads", RUN, "ground.mp4"),
    path.join(ROOT, "data", "uploads", RUN, "aerial.exe"), path.join(ROOT, "data", "uploads", RUN, "nested", "aerial.mp4"), "aerial.mp4"]) {
    const result = summary(); result.config.cameras[0].source = source; result.cameras[0].source = source;
    assert.equal(accepted(result), false);
  }
  const mismatch = summary(); mismatch.cameras[0].source = path.join(ROOT, "elsewhere", "aerial.mp4");
  assert.equal(accepted(mismatch), false);
});

test("malformed and duplicated camera roles are rejected without throwing", () => {
  for (const cameras of [undefined, [], [null, null], [summary().cameras[0], summary().cameras[0]]]) {
    assert.equal(accepted({ ...summary(), cameras }), false);
  }
  const result = summary(); result.config.cameras = [result.config.cameras[0], result.config.cameras[0]];
  assert.equal(accepted(result), false);
});
