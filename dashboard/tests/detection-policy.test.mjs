import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { groundCameraConfig, validateGroundPolicy } from "../lib/detection-policy.mjs";

const WEIGHT = Buffer.from("dummy approved detector, not executable model weights");
const SHA = createHash("sha256").update(WEIGHT).digest("hex");
function policy() {
  return {
    schema_version: 1,
    scope: "anonymous_detection",
    approved: true,
    ground: { profile: {
      name: "ground_960", checkpoint: "yolo11s.pt", imgsz: 960, tile_size: 0,
      augment: false, fp16: true, end2end: false,
    } },
    evidence: { experiment: "ground_report_20261013", saved_report_map50: .71, checkpoint_sha256: SHA },
  };
}
async function workspace(t) {
  const root = await mkdtemp(path.join(os.tmpdir(), "raven-policy-test-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  return root;
}
async function savePolicy(root, value) {
  await writeFile(path.join(root, "ground_detection_policy.json"), JSON.stringify(value));
}

test("missing policy returns the unchanged existing ground detector configuration", async (t) => {
  const root = await workspace(t);
  assert.deepEqual(await groundCameraConfig(root), { model: path.join(root, "yolo11s.pt"), imgsz: 960 });
});

test("approved policy verifies its checkpoint and returns the absolute profile", async (t) => {
  const root = await workspace(t);
  await writeFile(path.join(root, "yolo11s.pt"), WEIGHT);
  const value = policy();
  value.ground.profile.tile_size = 768;
  await savePolicy(root, value);
  const checkpoint = path.join(root, "yolo11s.pt");
  assert.deepEqual(await groundCameraConfig(root), {
    model: checkpoint, imgsz: 960,
    detection_profile: { ...value.ground.profile, checkpoint },
  });
});

test("pure validation accepts only the ten official aliases and clones the profile", () => {
  for (const family of ["yolo11", "yolo26"]) {
    for (const size of ["n", "s", "m", "x", "l"]) {
      const value = policy();
      value.ground.profile.checkpoint = `${family}${size}.pt`;
      const profile = validateGroundPolicy(value);
      assert.deepEqual(profile, value.ground.profile);
      assert.notEqual(profile, value.ground.profile);
    }
  }
});

test("unapproved policies, other scopes and unsupported schemas are rejected", () => {
  for (const [key, value] of [["schema_version", 2], ["schema_version", "1"], ["scope", "tracking"], ["approved", false], ["approved", 1]]) {
    const candidate = policy(); candidate[key] = value;
    assert.throws(() => validateGroundPolicy(candidate), /approved schema version 1/);
  }
});

test("checkpoint aliases cannot escape the project root or name arbitrary weights", () => {
  for (const checkpoint of ["../yolo11s.pt", "S:/yolo11s.pt", "folder\\yolo11s.pt", "folder/yolo11s.pt", "best.pt", "YOLO11s.pt", "yolo11s.pt\n"]) {
    const value = policy(); value.ground.profile.checkpoint = checkpoint;
    assert.throws(() => validateGroundPolicy(value), /official yolo11 or yolo26/);
  }
});

test("profile dimensions and booleans retain their exact validated types", () => {
  for (const [key, invalid] of [
    ["imgsz", 319], ["imgsz", 321], ["imgsz", 1921], ["imgsz", "960"],
    ["tile_size", -1], ["tile_size", 319], ["tile_size", 321], ["tile_size", 4097],
    ["tile_size", "0"], ["augment", 1], ["fp16", "true"], ["end2end", 0],
  ]) {
    const value = policy(); value.ground.profile[key] = invalid;
    assert.throws(() => validateGroundPolicy(value), /multiple of 32|booleans/);
  }
  for (const imgsz of [320, 1920]) {
    const value = policy(); value.ground.profile.imgsz = imgsz;
    assert.equal(validateGroundPolicy(value).imgsz, imgsz);
  }
  for (const tile of [0, 320, 4096]) {
    const value = policy(); value.ground.profile.tile_size = tile;
    assert.equal(validateGroundPolicy(value).tile_size, tile);
  }
});

test("end-to-end mode and unsafe profile names are rejected", () => {
  const value = policy(); value.ground.profile.end2end = true;
  assert.throws(() => validateGroundPolicy(value), /end2end=false/);
  for (const name of ["", "../unsafe", "with space", "x".repeat(97)]) {
    const candidate = policy(); candidate.ground.profile.name = name;
    assert.throws(() => validateGroundPolicy(candidate), /safe name/);
  }
});

test("identity, unknown and missing fields are not accepted anywhere", () => {
  for (const level of ["root", "ground", "profile", "evidence"]) {
    const value = policy();
    const target = level === "root" ? value : level === "profile" ? value.ground.profile : value[level];
    target.reid_checkpoint = "identities.pt";
    assert.throws(() => validateGroundPolicy(value), /only the required anonymous-detection fields/);
  }
  const incomplete = policy(); delete incomplete.ground.profile.fp16;
  assert.throws(() => validateGroundPolicy(incomplete), /only the required anonymous-detection fields/);
  for (const value of [null, [], {}, "anonymous_detection"]) assert.throws(() => validateGroundPolicy(value));
});

test("saved report evidence must have a real finite fraction and SHA-256", () => {
  for (const score of [NaN, Infinity, -.1, 1.1, "0.71", null]) {
    const value = policy(); value.evidence.saved_report_map50 = score;
    assert.throws(() => validateGroundPolicy(value), /finite number/);
  }
  for (const hash of ["", "a".repeat(63), "g".repeat(64), null]) {
    const value = policy(); value.evidence.checkpoint_sha256 = hash;
    assert.throws(() => validateGroundPolicy(value), /64-character hexadecimal/);
  }
  const value = policy(); value.evidence.experiment = "../another";
  assert.throws(() => validateGroundPolicy(value), /safe experiment identifier/);
});

test("invalid existing policy does not silently fall back", async (t) => {
  const root = await workspace(t);
  const value = policy(); value.approved = false;
  await savePolicy(root, value);
  await assert.rejects(groundCameraConfig(root), /approved schema version 1/);
  await writeFile(path.join(root, "ground_detection_policy.json"), "{broken");
  await assert.rejects(groundCameraConfig(root), /invalid JSON/);
});

test("missing, non-file or modified approved checkpoints fail closed", async (t) => {
  const root = await workspace(t);
  await savePolicy(root, policy());
  await assert.rejects(groundCameraConfig(root), /checkpoint is unavailable/);
  const checkpoint = path.join(root, "yolo11s.pt");
  await mkdir(checkpoint);
  await assert.rejects(groundCameraConfig(root), /regular local file/);
  await rm(checkpoint, { recursive: true });
  await writeFile(checkpoint, "modified checkpoint");
  await assert.rejects(groundCameraConfig(root), /SHA-256 differs/);
});

test("uppercase evidence hash still verifies the identical checkpoint", async (t) => {
  const root = await workspace(t);
  await writeFile(path.join(root, "yolo11s.pt"), WEIGHT);
  const value = policy(); value.evidence.checkpoint_sha256 = SHA.toUpperCase();
  await savePolicy(root, value);
  assert.equal((await groundCameraConfig(root)).model, path.join(root, "yolo11s.pt"));
});
