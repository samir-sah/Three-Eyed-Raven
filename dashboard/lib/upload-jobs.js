import { appendFile, mkdir, readFile, writeFile, open, unlink } from "node:fs/promises";
import { existsSync, createWriteStream } from "node:fs";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import path from "node:path";
import { randomUUID } from "node:crypto";
import { spawn } from "node:child_process";
import { readTrainingExperiment } from "./training-status.js";

const PROJECT_ROOT = path.resolve(process.cwd(), "..");
const UPLOAD_ROOT = path.join(PROJECT_ROOT, "data", "uploads");
const ARTIFACTS_ROOT = path.join(PROJECT_ROOT, "artifacts");
const AERIAL_CHECKPOINT = path.join(PROJECT_ROOT, "runs", "experiments", "extended_detection_20261012", "visdrone_person_960", "weights", "best.pt");
const UPLOAD_LOCK = path.join(UPLOAD_ROOT, "processing.lock");
const SAFE_RUN_ID = /^[A-Za-z0-9_-]+$/;
const SUPPORTED_EXTENSIONS = new Set([".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".webm"]);
const MAX_FILE_BYTES = 1_500_000_000;

function pythonExecutable() {
  const configured = process.env.TRACKER_PYTHON;
  if (configured) return configured;
  const localVenv = path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe");
  return existsSync(localVenv) ? localVenv : "python";
}

function jobPath(runId) { return path.join(UPLOAD_ROOT, runId, "job.json"); }

export async function readUploadJob(runId) {
  if (!SAFE_RUN_ID.test(runId)) return null;
  try {
    const value = JSON.parse(await readFile(jobPath(runId), "utf8"));
    if (value?.status === "processing") {
      try {
        const progress = JSON.parse(await readFile(path.join(ARTIFACTS_ROOT, runId, "progress.json"), "utf8"));
        if (progress.status === "processing") value.stage = `Analysing ${progress.camera} view · ${progress.frames} frames processed`;
        if (progress.status === "completed" && existsSync(path.join(ARTIFACTS_ROOT, runId, "summary.json"))) {
          value.status = "completed"; value.stage = "Run ready in dashboard";
        }
        if (progress.status === "failed") {
          value.status = "failed"; value.stage = "Processing failed"; value.error = progress.error;
        }
      } catch { /* The worker has not written its first progress update yet. */ }
    }
    return value && typeof value === "object" ? value : null;
  } catch { return null; }
}

async function writeJob(runId, update) {
  const previous = await readUploadJob(runId);
  const next = { ...previous, ...update, updatedAt: new Date().toISOString() };
  await writeFile(jobPath(runId), JSON.stringify(next, null, 2), "utf8");
  return next;
}

function validateUpload(file, role) {
  if (!file || typeof file.arrayBuffer !== "function") throw new Error(`Choose an ${role} video file.`);
  const extension = path.extname(file.name || "").toLowerCase();
  if (!SUPPORTED_EXTENSIONS.has(extension)) throw new Error(`${role} video must be MP4, AVI, MOV, MKV, MPG, MPEG, or WebM.`);
  if (!file.size) throw new Error(`${role} video is empty.`);
  if (file.size > MAX_FILE_BYTES) throw new Error(`${role} video exceeds the 1.5 GB local upload limit.`);
  return extension;
}

function runCommand(command, args, onOutput) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd: PROJECT_ROOT,
      env: { ...process.env, PYTHONPATH: path.join(PROJECT_ROOT, "src") },
      windowsHide: true,
    });
    child.stdout.on("data", (chunk) => onOutput(chunk.toString()));
    child.stderr.on("data", (chunk) => onOutput(chunk.toString()));
    child.on("error", reject);
    child.on("close", (code) => code === 0 ? resolve() : reject(new Error(`Analysis process exited with code ${code}. See the saved processing log.`)));
  });
}

async function processUploadRun(runId, configPath) {
  const logPath = path.join(UPLOAD_ROOT, runId, "processing.log");
  const log = (line) => appendFile(logPath, line, "utf8").catch(() => undefined);
  try {
    await writeJob(runId, { status: "processing", stage: "Counting people and encoding full-frame video", error: null });
    await runCommand(pythonExecutable(), ["-m", "crowd_tracker.anonymous_cli", "--config", configPath], log);
    await writeJob(runId, { status: "completed", stage: "Run ready in dashboard", artifactDir: path.join(ARTIFACTS_ROOT, runId) });
  } catch (error) {
    await writeJob(runId, { status: "failed", stage: "Processing failed", error: error instanceof Error ? error.message : "Unknown processing error" });
  } finally {
    await unlink(UPLOAD_LOCK).catch(() => undefined);
  }
}

export async function startUploadRun(formData) {
  let experiment;
  try { experiment = readTrainingExperiment(); } catch { /* No readable experiment status. */ }
  if (experiment?.pid && !["completed", "completed_with_errors", "failed"].includes(experiment.status)) {
    let alive = true;
    try { process.kill(experiment.pid, 0); } catch (error) { alive = error.code !== "ESRCH"; }
    if (alive) throw new Error("GPU training or evaluation is running. Wait for it to finish before starting another video analysis.");
  }
  const aerial = formData.get("aerial");
  const ground = formData.get("ground");
  const aerialExtension = validateUpload(aerial, "aerial");
  const groundExtension = validateUpload(ground, "ground");
  const aerialProfile = String(formData.get("aerialProfile") ?? "aerial");
  if (!["aerial", "general"].includes(aerialProfile)) throw new Error("Choose a valid aerial-video detector profile.");
  const rawMaxFrames = String(formData.get("maxFrames") ?? "600").trim();
  const maxFrames = rawMaxFrames ? Number(rawMaxFrames) : null;
  if (maxFrames !== null && (!Number.isInteger(maxFrames) || maxFrames < 1 || maxFrames > 20000)) throw new Error("Maximum frames must be a whole number between 1 and 20,000, or blank for the full videos.");
  if (aerialProfile === "aerial" && !existsSync(AERIAL_CHECKPOINT)) throw new Error("The measured aerial checkpoint is missing. Restore its saved best.pt before starting an analysis.");
  if (!existsSync(path.join(PROJECT_ROOT, "yolo11s.pt"))) throw new Error("The ground-detector checkpoint yolo11s.pt is missing.");
  await mkdir(UPLOAD_ROOT, { recursive: true });
  let lock;
  try { lock = await open(UPLOAD_LOCK, "wx"); }
  catch { throw new Error("Another upload analysis is running, or was interrupted. Wait for it to finish; if the server restarted, check the saved job before clearing processing.lock."); }

  const runId = `upload_${new Date().toISOString().replace(/[-:.TZ]/g, "").slice(0, 14)}_${randomUUID().slice(0, 6)}`;
  try {
  await lock.writeFile(JSON.stringify({ runId, pid: process.pid, startedAt: new Date().toISOString() }));
  await lock.close();
  const directory = path.join(UPLOAD_ROOT, runId);
  await mkdir(directory, { recursive: true });
  const aerialPath = path.join(directory, `aerial${aerialExtension}`);
  const groundPath = path.join(directory, `ground${groundExtension}`);
  // Do not allocate additional full-sized buffers for multi-GB video files.
  await pipeline(Readable.fromWeb(aerial.stream()), createWriteStream(aerialPath, { flags: "wx" }));
  await pipeline(Readable.fromWeb(ground.stream()), createWriteStream(groundPath, { flags: "wx" }));

  const configPath = path.join(directory, "multi_config.json");
  const config = {
    output_dir: path.join("artifacts", runId), device: "auto", confidence_threshold: 0.35,
    max_frames_per_camera: maxFrames,
    cameras: [{ id: "aerial", label: aerialProfile === "aerial" ? "Aerial / drone view" : "Camera A · ground / elevated view", source: aerialPath, model: aerialProfile === "aerial" ? AERIAL_CHECKPOINT : path.join(PROJECT_ROOT, "yolo11s.pt"), imgsz: 960 },
      { id: "ground", source: groundPath, model: path.join(PROJECT_ROOT, "yolo11s.pt"), imgsz: 960 }],
  };
  await writeFile(configPath, JSON.stringify(config, null, 2), "utf8");
  await writeFile(jobPath(runId), JSON.stringify({
    runId, status: "queued", stage: "Videos received; preparing anonymous crowd analysis", createdAt: new Date().toISOString(), updatedAt: new Date().toISOString(),
    maxFrames, sourceNames: { aerial: aerial.name, ground: ground.name },
  }, null, 2), "utf8");
  void processUploadRun(runId, configPath).catch((error) => console.error("Could not persist upload job status:", error.message));
  return { runId, status: "queued" };
  } catch (error) {
    await lock.close().catch(() => undefined);
    await unlink(UPLOAD_LOCK).catch(() => undefined);
    throw error;
  }
}
