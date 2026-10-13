/** Approved inference profiles for anonymous uploads only, never identity paths. */
import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { lstat, readFile } from "node:fs/promises";
import path from "node:path";

const OFFICIAL_CHECKPOINT = /^(?:yolo11|yolo26)[nsmxl]\.pt$/;
const SAFE_NAME = /^[A-Za-z0-9][A-Za-z0-9_-]{0,95}$/;
const PROFILE_FIELDS = ["name", "checkpoint", "imgsz", "tile_size", "augment", "fp16", "end2end"];

function exactFields(value, allowed, label) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} must be an object.`);
  }
  const keys = Object.keys(value);
  if (keys.some((key) => !allowed.includes(key)) || allowed.some((key) => !Object.hasOwn(value, key))) {
    throw new Error(`${label} must contain only the required anonymous-detection fields: ${allowed.join(", ")}.`);
  }
}

/** Validate policy data without filesystem access; return a sanitized profile. */
export function validateGroundPolicy(policy) {
  exactFields(policy, ["schema_version", "scope", "approved", "ground", "evidence"], "Detection policy");
  if (policy.schema_version !== 1 || policy.scope !== "anonymous_detection" || policy.approved !== true) {
    throw new Error("Detection policy must be approved schema version 1 with anonymous_detection scope.");
  }
  exactFields(policy.ground, ["profile"], "Ground policy");
  const profile = policy.ground.profile;
  exactFields(profile, PROFILE_FIELDS, "Ground detector profile");
  if (typeof profile.name !== "string" || !SAFE_NAME.test(profile.name)) {
    throw new Error("Ground detector profile needs a safe name.");
  }
  if (typeof profile.checkpoint !== "string" || !OFFICIAL_CHECKPOINT.test(profile.checkpoint)) {
    throw new Error("Ground checkpoint must be an official yolo11 or yolo26 model alias, without directories.");
  }
  if (!Number.isInteger(profile.imgsz) || profile.imgsz < 320 || profile.imgsz > 1920 || profile.imgsz % 32) {
    throw new Error("Ground imgsz must be a multiple of 32 between 320 and 1920.");
  }
  if (!Number.isInteger(profile.tile_size) || (profile.tile_size !== 0 && (
    profile.tile_size < 320 || profile.tile_size > 4096 || profile.tile_size % 32
  ))) {
    throw new Error("Ground tile_size must be zero or a multiple of 32 between 320 and 4096.");
  }
  if (["augment", "fp16", "end2end"].some((key) => typeof profile[key] !== "boolean")) {
    throw new Error("Ground augment, fp16 and end2end settings must be booleans.");
  }
  if (profile.end2end !== false) {
    throw new Error("Approved ground profiles must use end2end=false to preserve the evaluation protocol.");
  }
  exactFields(policy.evidence, ["experiment", "saved_report_map50", "checkpoint_sha256"], "Detection evidence");
  const evidence = policy.evidence;
  if (typeof evidence.experiment !== "string" || !SAFE_NAME.test(evidence.experiment)) {
    throw new Error("Detection evidence needs a safe experiment identifier.");
  }
  if (!Number.isFinite(evidence.saved_report_map50) || evidence.saved_report_map50 < 0 || evidence.saved_report_map50 > 1) {
    throw new Error("Detection evidence mAP50 must be a finite number between zero and one.");
  }
  if (typeof evidence.checkpoint_sha256 !== "string" || !/^[0-9a-f]{64}$/i.test(evidence.checkpoint_sha256)) {
    throw new Error("Detection evidence needs a 64-character hexadecimal checkpoint SHA-256.");
  }
  return Object.fromEntries(PROFILE_FIELDS.map((key) => [key, profile[key]]));
}

/** Missing policy keeps the existing detector. An invalid saved policy fails closed. */
export async function groundCameraConfig(projectRoot) {
  const root = path.resolve(projectRoot);
  const policyPath = path.join(root, "ground_detection_policy.json");
  let source;
  try {
    source = await readFile(policyPath, "utf8");
  } catch (error) {
    if (error.code === "ENOENT") return { model: path.join(root, "yolo11s.pt"), imgsz: 960 };
    throw new Error(`Cannot read the saved ground detection policy: ${error.message}`, { cause: error });
  }
  let policy;
  try {
    policy = JSON.parse(source);
  } catch (error) {
    throw new Error("The saved ground detection policy contains invalid JSON.", { cause: error });
  }
  const profile = validateGroundPolicy(policy);
  const checkpoint = path.join(root, profile.checkpoint);
  let metadata;
  try {
    metadata = await lstat(checkpoint);
  } catch (error) {
    throw new Error(`The approved ground checkpoint is unavailable: ${profile.checkpoint}.`, { cause: error });
  }
  if (!metadata.isFile()) throw new Error("The approved ground checkpoint must be a regular local file.");
  const digest = createHash("sha256");
  for await (const chunk of createReadStream(checkpoint)) digest.update(chunk);
  if (digest.digest("hex") !== policy.evidence.checkpoint_sha256.toLowerCase()) {
    throw new Error("The ground checkpoint SHA-256 differs from its approved evaluation evidence.");
  }
  return {
    model: checkpoint,
    imgsz: profile.imgsz,
    detection_profile: { ...profile, checkpoint },
  };
}
