/** Evidence for explicit, consented QR tags only; never appearance or identity matching. */
import { lstatSync, readFileSync, realpathSync } from "node:fs";
import path from "node:path";

const ROLES = ["aerial", "ground"];
const CODE = /^V(?:0[1-9]|[1-9][0-9])$/;
const SAFE_RUN = /^[A-Za-z0-9_-]{1,120}$/;
const METHOD = "explicit_qr_codes";
const SCOPE = "controlled_marker_demo";
const WARNING = "These are explicit QR-tag observations in a confirmed volunteer demonstration, not recognition of people or proof of identity. A shared tag does not establish camera-clock synchronization; duplicate tags cannot be counted as shared evidence.";

export function isMarkerCode(value) { return typeof value === "string" && value.length === 3 && CODE.test(value); }
export function isMarkerUploadId(value) { return typeof value === "string" && value.length === 28 && /^upload_\d{14}_[0-9a-f]{6}$/.test(value); }

function checkedDirectory(value) {
  const expected = path.resolve(value), info = lstatSync(expected);
  if (!info.isDirectory() || info.isSymbolicLink() || path.relative(expected, realpathSync(expected)) !== "") {
    throw new Error("Marker directories may not be links or filesystem redirects.");
  }
  return expected;
}
function checkedFile(value) {
  const expected = path.resolve(value), info = lstatSync(expected);
  if (!info.isFile() || info.isSymbolicLink() || path.relative(expected, realpathSync(expected)) !== "") {
    throw new Error("Marker files must be regular local files, not links or redirects.");
  }
  return expected;
}

/** Check the full run ancestry before opening even its summary metadata. */
export function markerRunDirectory(projectRoot, runId) {
  if (!isMarkerUploadId(runId)) throw new Error("Invalid marker upload identifier.");
  const trustedRoot = realpathSync(path.resolve(projectRoot));
  const artifacts = checkedDirectory(path.join(trustedRoot, "artifacts"));
  return checkedDirectory(path.join(artifacts, runId));
}
export function markerSummaryPath(runDirectory) { return checkedFile(path.join(checkedDirectory(runDirectory), "summary.json")); }
export function markerEvidencePath(runDirectory, camera, code) {
  if (!ROLES.includes(camera) || !isMarkerCode(code)) throw new Error("Invalid marker evidence identifier.");
  const cameraDirectory = checkedDirectory(path.join(checkedDirectory(runDirectory), camera));
  const evidenceDirectory = checkedDirectory(path.join(cameraDirectory, "marker_evidence"));
  return checkedFile(path.join(evidenceDirectory, `${code}.jpg`));
}
export function markerTagPath(dashboardRoot, code) {
  if (!isMarkerCode(code)) throw new Error("Invalid printable marker identifier.");
  const trustedRoot = realpathSync(path.resolve(dashboardRoot));
  const publicDirectory = checkedDirectory(path.join(trustedRoot, "public"));
  const tagsDirectory = checkedDirectory(path.join(publicDirectory, "marker-tags"));
  return checkedFile(path.join(tagsDirectory, `${code}.png`));
}

function object(value) { return value !== null && typeof value === "object" && !Array.isArray(value); }
function integer(value, minimum = 0) { return Number.isSafeInteger(value) && value >= minimum; }
function exactFields(value, fields, label) {
  if (!object(value) || Object.keys(value).length !== fields.length || fields.some((key) => !Object.hasOwn(value, key))) {
    throw new Error(`${label} has missing or unsupported fields.`);
  }
}
function codeList(value, allowed, label) {
  if (!Array.isArray(value) || value.some((code) => !isMarkerCode(code) || (allowed && !allowed.has(code))) || new Set(value).size !== value.length) {
    throw new Error(`${label} must contain unique registered codes V01–V99.`);
  }
  return [...value].sort();
}

export function validateMarkerDemo(config) {
  if (!object(config) || config.enabled !== true) return null;
  const fields = ["enabled", "confirmed_volunteer_demo", "allowed_codes", ...(Object.hasOwn(config, "min_reads") ? ["min_reads"] : [])];
  exactFields(config, fields, "Marker demo configuration");
  if (config.confirmed_volunteer_demo !== true) throw new Error("A controlled volunteer demonstration must be explicitly confirmed.");
  const codes = codeList(config.allowed_codes, null, "Registered markers");
  if (!codes.length) throw new Error("At least one anonymous QR code must be registered.");
  const minReads = Object.hasOwn(config, "min_reads") ? config.min_reads : 3;
  if (!integer(minReads, 1) || minReads > 20000) throw new Error("Minimum marker reads must be an integer from 1 to 20,000.");
  return { codes, minReads };
}

function unavailable(reason) {
  return { status: "unavailable", readerVerified: false, method: METHOD, scope: SCOPE, sharedCount: null, sharedCodes: [], cameras: [], evidence: [], reason, warning: WARNING };
}

/** Recompute intersections from validated per-camera records; reported totals are advisory. */
export function validateMarkerReport(report, summary, runId) {
  const demo = validateMarkerDemo(summary?.config?.marker_demo);
  if (!demo) return null;
  if (summary?.analysis_type !== "anonymous_detection" || typeof runId !== "string" || !runId.length || runId.length > 120 || !SAFE_RUN.test(runId)) throw new Error("Marker reports require a valid anonymous-upload run.");
  if (!object(report) || report.method !== METHOD || report.scope !== SCOPE || report.min_reads !== demo.minReads) {
    throw new Error("Marker report method, scope or minimum-read policy differs from the confirmed configuration.");
  }
  exactFields(report.cameras, ROLES, "Marker camera report");
  if (!Array.isArray(summary.cameras) || summary.cameras.length !== 2) throw new Error("Exactly two camera summaries are required.");
  const allowed = new Set(demo.codes);
  const cameras = ROLES.map((id) => {
    const metas = summary.cameras.filter((camera) => object(camera) && camera.camera_id === id);
    if (metas.length !== 1) throw new Error("Camera summary roles must be unique aerial and ground views.");
    const meta = metas[0], camera = report.cameras[id];
    exactFields(camera, ["frames_processed", "frames_scanned", "read_failures", "observations", "duplicate_codes"], `Marker camera ${id}`);
    if (!integer(meta.frames_processed, 1) || !Number.isFinite(meta.source_fps) || meta.source_fps <= 0 ||
        camera.frames_processed !== meta.frames_processed || !integer(camera.frames_scanned, 1) || camera.frames_scanned > meta.frames_processed ||
        !integer(camera.read_failures) || camera.read_failures >= camera.frames_scanned || !object(camera.observations)) {
      throw new Error(`Marker scan coverage or camera bounds are unavailable for ${id}.`);
    }
    const duplicateCodes = codeList(camera.duplicate_codes, allowed, `Duplicate codes in ${id}`);
    const observations = Object.entries(camera.observations).map(([code, row]) => {
      if (!isMarkerCode(code) || !allowed.has(code)) throw new Error(`Unregistered marker code in ${id}.`);
      exactFields(row, ["first_frame", "last_frame", "first_seconds", "last_seconds", "read_frames", "evidence_file"], `Marker ${code}`);
      if (!integer(row.first_frame) || !integer(row.last_frame) || row.first_frame > row.last_frame || row.last_frame >= meta.frames_processed ||
          !integer(row.read_frames, 1) || row.read_frames > row.last_frame - row.first_frame + 1 || row.read_frames > camera.frames_scanned - camera.read_failures) {
        throw new Error(`Invalid marker frame/read bounds for ${code} in ${id}.`);
      }
      if (!Number.isFinite(row.first_seconds) || !Number.isFinite(row.last_seconds) || row.first_seconds < 0 || row.first_seconds > row.last_seconds ||
          row.last_seconds >= meta.frames_processed / meta.source_fps ||
          Math.abs(row.first_seconds - row.first_frame / meta.source_fps) > 1e-5 ||
          Math.abs(row.last_seconds - row.last_frame / meta.source_fps) > 1e-5) {
        throw new Error(`Invalid marker timestamps for ${code} in ${id}.`);
      }
      // Duplicate-only sightings can have valid read records but no unambiguous
      // tag crop. Their code is still excluded by the global duplicate rule.
      if (row.evidence_file !== null && row.evidence_file !== `${code}.jpg`) throw new Error(`Unsafe or mismatched tag-only evidence file for ${code}.`);
      return { code, firstFrame: row.first_frame, lastFrame: row.last_frame, firstSeconds: row.first_seconds, lastSeconds: row.last_seconds,
        readFrames: row.read_frames, evidenceUrl: row.evidence_file === null ? null : `/api/marker-evidence/${encodeURIComponent(runId)}/${id}/${code}` };
    }).sort((first, second) => first.code.localeCompare(second.code));
    const observedCodes = observations.map((row) => row.code);
    const eligibleCodes = observations.filter((row) => row.readFrames >= demo.minReads && !duplicateCodes.includes(row.code)).map((row) => row.code);
    return { id, label: meta.label || (id === "aerial" ? "Aerial / drone view" : "Ground / surveillance view"),
      framesProcessed: camera.frames_processed, framesScanned: camera.frames_scanned, readFailures: camera.read_failures,
      observedCodes, observedCount: observedCodes.length, eligibleCodes, eligibleCount: eligibleCodes.length, duplicateCodes, observations };
  });
  const ambiguousCodes = [...new Set(cameras.flatMap((camera) => camera.duplicateCodes))].sort();
  const incompleteScans = cameras.filter((camera) => camera.readFailures > 0);
  const coverageWarning = incompleteScans.length ? `Some QR scans failed (${incompleteScans.map((camera) => `${camera.id}: ${camera.readFailures}/${camera.framesScanned}`).join("; ")}). Undetected tags do not establish absence from a view.` : null;
  const sharedCodes = cameras[0].eligibleCodes.filter((code) => cameras[1].eligibleCodes.includes(code) && !ambiguousCodes.includes(code));
  const evidence = sharedCodes.map((code) => ({ code, views: cameras.map((camera) => ({ cameraId: camera.id, ...camera.observations.find((row) => row.code === code) })) }));
  return { status: "ready", readerVerified: true, method: METHOD, scope: SCOPE, minReads: demo.minReads, sharedCodes, sharedCount: sharedCodes.length,
    ambiguousCodes, ambiguityWarning: ambiguousCodes.length ? `Duplicate tags excluded from shared evidence: ${ambiguousCodes.join(", ")}.` : null,
    coverageWarning,
    cameras, evidence, warning: WARNING };
}

/** An opted-in missing/corrupt report is unavailable, not a measured zero. */
export function readMarkerReport(runDirectory, summary) {
  try {
    if (!validateMarkerDemo(summary?.config?.marker_demo)) return null;
    let report;
    try { report = JSON.parse(readFileSync(checkedFile(path.join(checkedDirectory(runDirectory), "markers.json")), "utf8")); }
    catch { return unavailable("The opted-in QR-tag report is missing or contains invalid JSON."); }
    return validateMarkerReport(report, summary, path.basename(runDirectory));
  } catch (error) { return unavailable(error.message); }
}
