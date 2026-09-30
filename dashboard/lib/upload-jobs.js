import { appendFile, mkdir, readFile, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";
import { randomUUID } from "node:crypto";
import { spawn } from "node:child_process";

const PROJECT_ROOT = path.resolve(process.cwd(), "..");
const UPLOAD_ROOT = path.join(PROJECT_ROOT, "data", "uploads");
const ARTIFACTS_ROOT = path.join(PROJECT_ROOT, "artifacts");
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
    child.on("close", (code) => code === 0 ? resolve() : reject(new Error(`Tracker process exited with code ${code}.`)));
  });
}

async function processUploadRun(runId, configPath) {
  const logPath = path.join(UPLOAD_ROOT, runId, "processing.log");
  const log = (line) => appendFile(logPath, line, "utf8").catch(() => undefined);
  try {
    await writeJob(runId, { status: "processing", stage: "Detecting and tracking both videos", error: null });
    await runCommand(pythonExecutable(), ["-m", "crowd_tracker.multi_cli", "--config", configPath], log);
    await writeJob(runId, { status: "processing", stage: "Preparing browser previews" });
    for (const camera of ["aerial", "ground"]) {
      await runCommand(pythonExecutable(), ["scripts/export_preview_frames.py", "--run", runId, "--camera", camera, "--stride", "5", "--width", "720"], log);
    }
    await writeJob(runId, { status: "completed", stage: "Run ready in dashboard", artifactDir: path.join(ARTIFACTS_ROOT, runId) });
  } catch (error) {
    await writeJob(runId, { status: "failed", stage: "Processing failed", error: error instanceof Error ? error.message : "Unknown processing error" });
  }
}

export async function startUploadRun(formData) {
  const aerial = formData.get("aerial");
  const ground = formData.get("ground");
  const aerialExtension = validateUpload(aerial, "aerial");
  const groundExtension = validateUpload(ground, "ground");
  const rawMaxFrames = String(formData.get("maxFrames") ?? "600").trim();
  const maxFrames = rawMaxFrames ? Number(rawMaxFrames) : null;
  if (maxFrames !== null && (!Number.isInteger(maxFrames) || maxFrames < 1 || maxFrames > 20000)) throw new Error("Maximum frames must be a whole number between 1 and 20,000, or blank for the full videos.");

  const runId = `upload_${new Date().toISOString().replace(/[-:.TZ]/g, "").slice(0, 14)}_${randomUUID().slice(0, 6)}`;
  const directory = path.join(UPLOAD_ROOT, runId);
  await mkdir(directory, { recursive: true });
  const aerialPath = path.join(directory, `aerial${aerialExtension}`);
  const groundPath = path.join(directory, `ground${groundExtension}`);
  await Promise.all([writeFile(aerialPath, Buffer.from(await aerial.arrayBuffer())), writeFile(groundPath, Buffer.from(await ground.arrayBuffer()))]);

  const configPath = path.join(directory, "multi_config.json");
  const config = {
    output_dir: path.join("artifacts", runId), model: "yolo11n.pt", device: "auto", confidence_threshold: 0.35, iou_threshold: 0.5,
    frame_stride: 1, max_frames_per_camera: maxFrames, save_video: true, tracker: "bytetrack.yaml", reid_candidate_threshold: 0.72,
    reid_top_k: 3, reid_min_box_size: 32, reconnect_attempts: 0, reconnect_delay_seconds: 0,
    cameras: [{ id: "aerial", source: aerialPath, zones: [] }, { id: "ground", source: groundPath, zones: [] }],
  };
  await writeFile(configPath, JSON.stringify(config, null, 2), "utf8");
  await writeFile(jobPath(runId), JSON.stringify({
    runId, status: "queued", stage: "Videos received; preparing tracker", createdAt: new Date().toISOString(), updatedAt: new Date().toISOString(),
    maxFrames, sourceNames: { aerial: aerial.name, ground: ground.name },
  }, null, 2), "utf8");
  void processUploadRun(runId, configPath);
  return { runId, status: "queued" };
}
