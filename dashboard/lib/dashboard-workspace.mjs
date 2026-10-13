/** Fresh uploads are isolated from archived demos without deleting any artifacts. */
import { readFileSync } from "node:fs";
import path from "node:path";

const WORKSPACE_FIELDS = ["schema_version", "scope", "workspace_id", "created_at"];
const UUID_V4 = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const UPLOAD_ID = /^upload_\d{14}_[0-9a-f]{6}$/;
const VIDEO_FILE = /^(aerial|ground)\.(mp4|avi|mov|mkv|mpg|mpeg|webm)$/i;

function object(value) { return value !== null && typeof value === "object" && !Array.isArray(value); }
function timestamp(value) { return typeof value === "string" ? Date.parse(value) : NaN; }
function normalizedPath(value) { return path.resolve(value).toLowerCase(); }

/** Reject missing, malformed or incompatible markers; never infer an old workspace. */
export function validateDashboardWorkspace(value) {
  if (!object(value) || Object.keys(value).length !== WORKSPACE_FIELDS.length ||
      WORKSPACE_FIELDS.some((key) => !Object.hasOwn(value, key))) {
    throw new Error("Dashboard workspace must contain the required schema, scope, UUID and creation time.");
  }
  if (value.schema_version !== 1 || value.scope !== "fresh_anonymous_uploads") {
    throw new Error("Dashboard workspace must use schema version 1 and fresh_anonymous_uploads scope.");
  }
  if (typeof value.workspace_id !== "string" || !UUID_V4.test(value.workspace_id)) {
    throw new Error("Dashboard workspace requires a valid UUID v4 marker.");
  }
  const created = timestamp(value.created_at);
  if (!Number.isFinite(created) || new Date(created).toISOString() !== value.created_at) {
    throw new Error("Dashboard workspace creation time must be a valid canonical UTC timestamp.");
  }
  return { ...value, workspace_id: value.workspace_id.toLowerCase() };
}

export function parseDashboardWorkspace(source) {
  let value;
  try { value = JSON.parse(source); }
  catch (error) { throw new Error("The dashboard workspace marker contains invalid JSON.", { cause: error }); }
  return validateDashboardWorkspace(value);
}

export function readDashboardWorkspace(projectRoot) {
  let source;
  try { source = readFileSync(path.join(projectRoot, "dashboard_workspace.json"), "utf8"); }
  catch (error) {
    throw new Error("The dashboard workspace marker is unavailable. Restore dashboard_workspace.json before uploading videos.", { cause: error });
  }
  return parseDashboardWorkspace(source);
}

/** Jobs in an old browser session must not bring an archived run back into view. */
export function isCurrentWorkspaceJob(job, workspace) {
  try {
    const marker = validateDashboardWorkspace(workspace);
    return object(job) && UPLOAD_ID.test(job.runId) && job.workspaceId === marker.workspace_id &&
      ["queued", "processing", "completed", "failed"].includes(job.status) &&
      Number.isFinite(timestamp(job.createdAt)) && timestamp(job.createdAt) >= timestamp(marker.created_at);
  } catch { return false; }
}

/** Only completed two-view analysis of files received in this workspace is visible. */
export function isCurrentWorkspaceUpload(runId, summary, job, workspace, projectRoot) {
  try {
    if (!isCurrentWorkspaceJob(job, workspace) || job.runId !== runId || job.status !== "completed") return false;
    if (!object(summary) || summary.run_type !== "multi_stream" || summary.analysis_type !== "anonymous_detection") return false;
    const config = summary.config;
    if (!object(config) || config.dashboard_workspace_id !== job.workspaceId || config.upload_run_id !== runId) return false;
    if (!Number.isFinite(timestamp(summary.completed_at)) || timestamp(summary.completed_at) < timestamp(job.createdAt)) return false;
    if (!object(job.sourceNames) || ["aerial", "ground"].some((role) => typeof job.sourceNames[role] !== "string" || !job.sourceNames[role].trim())) return false;
    if (!Array.isArray(config.cameras) || config.cameras.length !== 2 || !Array.isArray(summary.cameras) || summary.cameras.length !== 2) return false;
    const expectedDirectory = normalizedPath(path.join(projectRoot, "data", "uploads", runId));
    for (const role of ["aerial", "ground"]) {
      const cameras = config.cameras.filter((camera) => object(camera) && camera.id === role);
      const results = summary.cameras.filter((camera) => object(camera) && camera.camera_id === role);
      if (cameras.length !== 1 || results.length !== 1) return false;
      const source = cameras[0].source;
      if (typeof source !== "string" || !path.isAbsolute(source) || typeof results[0].source !== "string" || normalizedPath(source) !== normalizedPath(results[0].source)) return false;
      if (normalizedPath(path.dirname(source)) !== expectedDirectory) return false;
      const file = path.basename(source).match(VIDEO_FILE);
      if (!file || file[1].toLowerCase() !== role) return false;
    }
    return true;
  } catch { return false; }
}
