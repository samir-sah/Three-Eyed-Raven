import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import path from "node:path";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const ARTIFACTS_ROOT = path.resolve(process.cwd(), "..", "artifacts");
const TRAINING_ROOT = path.resolve(process.cwd(), "..", "runs", "detect");

function safeJson(filePath, fallback) {
  try {
    return JSON.parse(readFileSync(filePath, "utf8"));
  } catch {
    return fallback;
  }
}

function detectorLabel(modelPath) {
  if (!modelPath) return "Detector not recorded";
  if (String(modelPath).includes("visdrone_person")) return "VisDrone fine-tuned YOLO11n";
  if (path.basename(String(modelPath)) === "yolo11n.pt") return "YOLO11n (COCO)";
  return path.basename(String(modelPath));
}

function cameraLabel(cameraId) {
  if (cameraId.startsWith("aerial")) return "Aerial / drone view";
  if (cameraId.startsWith("cctv")) return "Ground / CCTV view";
  return cameraId.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function readJsonLines(filePath) {
  if (!existsSync(filePath)) return [];
  return readFileSync(filePath, "utf8")
    .split(/\r?\n/)
    .filter(Boolean)
    .flatMap((line) => {
      try {
        return [JSON.parse(line)];
      } catch {
        return [];
      }
    });
}

function sampleTimeline(frameCounts) {
  if (frameCounts.length <= 12) return frameCounts;
  const step = Math.max(1, Math.floor(frameCounts.length / 12));
  const sampled = frameCounts.filter((_, index) => index % step === 0);
  return sampled.at(-1)?.frame === frameCounts.at(-1)?.frame
    ? sampled
    : [...sampled, frameCounts.at(-1)];
}

function readPreviewFrames(directory, urlForFile) {
  if (!existsSync(directory)) return { frames: [], previewFps: 5 };
  const metadata = safeJson(path.join(directory, "metadata.json"), {});
  const frames = readdirSync(directory)
    .filter((file) => /^frame_\d+\.jpg$/.test(file))
    .sort()
    .map(urlForFile);
  const sourceFps = Number(metadata.source_fps);
  const stride = Number(metadata.preview_stride);
  return {
    frames,
    previewFps: Number.isFinite(sourceFps) && Number.isFinite(stride) && stride > 0
      ? sourceFps / stride
      : 5,
  };
}

function findFiles(directory, name, found = []) {
  if (!existsSync(directory)) return found;
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    const entryPath = path.join(directory, entry.name);
    if (entry.isDirectory()) findFiles(entryPath, name, found);
    else if (entry.isFile() && entry.name === name) found.push(entryPath);
  }
  return found;
}

function readTrainingProgress() {
  const logs = findFiles(TRAINING_ROOT, "training.log")
    .map((file) => ({ file, modifiedAt: statSync(file).mtime }))
    .sort((first, second) => second.modifiedAt - first.modifiedAt);
  const latest = logs[0];
  if (!latest) return null;

  const text = readFileSync(latest.file, "utf8");
  const epochMatches = [...text.matchAll(/\s(\d+)\/(\d+)\s+\d+(?:\.\d+)?G/g)];
  const epoch = epochMatches.at(-1);
  const currentEpoch = epoch ? Number(epoch[1]) : 0;
  const totalEpochs = epoch ? Number(epoch[2]) : 0;
  const completed = /\d+ epochs completed in/.test(text);
  const running = !completed && Date.now() - latest.modifiedAt.getTime() < 120000;
  const lastLine = text.split(/\r?\n/).filter(Boolean).at(-1) ?? "Preparing dataset and model";
  const runName = path.basename(path.dirname(latest.file));
  const resultsPath = findFiles(TRAINING_ROOT, "results.csv")
    .filter((file) => path.basename(path.dirname(file)) === runName)
    .map((file) => ({ file, modifiedAt: statSync(file).mtime }))
    .sort((first, second) => second.modifiedAt - first.modifiedAt)[0]?.file;
  const rows = resultsPath ? readFileSync(resultsPath, "utf8").trim().split(/\r?\n/) : [];
  const headers = rows.length > 1 ? rows[0].split(",").map((value) => value.trim()) : [];
  const values = rows.length > 1 ? rows.at(-1).split(",").map((value) => Number(value.trim())) : [];
  const metric = (name) => {
    const value = values[headers.indexOf(name)];
    return Number.isFinite(value) ? value : null;
  };

  return {
    name: runName.replaceAll("_", " "),
    updatedAt: latest.modifiedAt.toISOString(),
    currentEpoch: completed ? totalEpochs : currentEpoch,
    totalEpochs,
    progress: totalEpochs ? Math.round((completed ? totalEpochs : currentEpoch) / totalEpochs * 100) : 0,
    status: completed ? "completed" : running ? "running" : "paused",
    lastLine: completed
      ? `Completed ${totalEpochs}-epoch run; best checkpoint validated and ready for evaluation.`
      : lastLine.replace(/\x1B\[[0-?]*[ -\/]*[@-~]/g, "").slice(-180),
    metrics: {
      precision: metric("metrics/precision(B)"),
      recall: metric("metrics/recall(B)"),
      map50: metric("metrics/mAP50(B)"),
      map5095: metric("metrics/mAP50-95(B)"),
    },
  };
}

function readLatestMotBenchmark() {
  const summaries = findFiles(ARTIFACTS_ROOT, "benchmark_summary.json")
    .map((file) => ({ file, modifiedAt: statSync(file).mtime }))
    .sort((first, second) => second.modifiedAt - first.modifiedAt);
  const latest = summaries[0];
  if (!latest) return null;

  const summary = safeJson(latest.file, null);
  const aggregate = summary?.aggregate;
  if (!aggregate) return null;
  return {
    label: `${summary.sequences?.length ?? 0} MOT17 sequences`,
    updatedAt: latest.modifiedAt.toISOString(),
    model: summary.model ?? "Unknown detector",
    ...aggregate,
  };
}

function readRun(runId) {
  const runDirectory = path.join(ARTIFACTS_ROOT, runId);
  const summaryPath = path.join(runDirectory, "summary.json");
  if (!existsSync(summaryPath)) return null;

  const summary = safeJson(summaryPath, {});
  const observations = readJsonLines(path.join(runDirectory, "observations.jsonl"));
  const alerts = readJsonLines(path.join(runDirectory, "alerts.jsonl"));
  const evaluation = safeJson(path.join(runDirectory, "evaluation.json"), null);
  const tracksByFrame = new Map();
  const tracks = new Map();

  for (const observation of observations) {
    const frame = observation.frame_index ?? 0;
    const entry = tracksByFrame.get(frame) ?? new Set();
    entry.add(observation.track_id);
    tracksByFrame.set(frame, entry);

    const track = tracks.get(observation.track_id) ?? {
      id: observation.track_id,
      firstFrame: frame,
      lastFrame: frame,
      frames: 0,
      confidenceTotal: 0,
    };
    track.firstFrame = Math.min(track.firstFrame, frame);
    track.lastFrame = Math.max(track.lastFrame, frame);
    track.frames += 1;
    track.confidenceTotal += observation.confidence ?? 0;
    tracks.set(observation.track_id, track);
  }

  const frameCounts = [...tracksByFrame.entries()]
    .sort(([first], [second]) => first - second)
    .map(([frame, ids]) => ({ frame, count: ids.size }));
  const confidence = observations.length
    ? observations.reduce((total, item) => total + (item.confidence ?? 0), 0) / observations.length
    : 0;
  const busiest = frameCounts.reduce((best, item) => (item.count > best.count ? item : best), { frame: 0, count: 0 });
  const stableTracks = [...tracks.values()]
    .map((track) => ({ ...track, meanConfidence: track.confidenceTotal / track.frames }))
    .sort((first, second) => second.frames - first.frames)
    .slice(0, 4);
  const videoPath = path.join(runDirectory, "annotated.mp4");
  const preview = readPreviewFrames(
    path.join(runDirectory, "preview_frames"),
    (file) => `/api/frames/${encodeURIComponent(runId)}/${encodeURIComponent(file)}`,
  );

  return {
    id: runId,
    label: runId.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase()),
    updatedAt: statSync(summaryPath).mtime.toISOString(),
    hasVideo: existsSync(videoPath),
    videoUrl: `/api/artifacts/${encodeURIComponent(runId)}/annotated.mp4`,
    previewFrames: preview.frames,
    previewFps: preview.previewFps,
    detector: detectorLabel(summary.config?.model),
    metrics: {
      frames: summary.frames_processed ?? 0,
      processingFps: summary.processing_fps ?? 0,
      duration: summary.elapsed_seconds ?? 0,
      resolution: summary.source_resolution?.join(" × ") ?? "—",
      uniqueTracks: summary.unique_local_track_ids ?? tracks.size,
      averageConfidence: confidence,
      peakCount: busiest.count,
      peakFrame: busiest.frame,
      alertCount: alerts.length,
      source: summary.source ?? "—",
    },
    timeline: sampleTimeline(frameCounts),
    latencyProfile: summary.latency_profile ?? [],
    tracks: stableTracks,
    alerts: alerts.slice(-5).reverse(),
    evaluation,
  };
}

function readMultiRun(runId) {
  const runDirectory = path.join(ARTIFACTS_ROOT, runId);
  const summaryPath = path.join(runDirectory, "summary.json");
  const summary = safeJson(summaryPath, {});
  const reidReport = safeJson(path.join(runDirectory, "reid_candidates.json"), { summary: {}, candidates: [] });
  const fallbackModel = summary.config?.model;
  const streams = (summary.cameras ?? []).map((camera) => {
    const cameraDirectory = path.join(runDirectory, camera.camera_id);
    const observations = readJsonLines(path.join(cameraDirectory, "observations.jsonl"));
    const alerts = readJsonLines(path.join(cameraDirectory, "alerts.jsonl"));
    const preview = readPreviewFrames(
      path.join(cameraDirectory, "preview_frames"),
      (file) => `/api/multi-frames/${encodeURIComponent(runId)}/${encodeURIComponent(camera.camera_id)}/${encodeURIComponent(file)}`,
    );
    const frameCounts = new Map();
    for (const observation of observations) {
      const frame = observation.frame_index ?? 0;
      const ids = frameCounts.get(frame) ?? new Set();
      ids.add(observation.track_id);
      frameCounts.set(frame, ids);
    }
    const peakCount = Math.max(0, ...[...frameCounts.values()].map((ids) => ids.size));
    const peakFrame = [...frameCounts.entries()].find(([, ids]) => ids.size === peakCount)?.[0] ?? 0;
    const averageConfidence = observations.length
      ? observations.reduce((total, item) => total + (item.confidence ?? 0), 0) / observations.length
      : 0;
    return {
      id: camera.camera_id,
      label: cameraLabel(camera.camera_id),
      detector: detectorLabel(camera.model ?? fallbackModel),
      previewFrames: preview.frames,
      previewFps: preview.previewFps,
      source: camera.source,
      resolution: camera.source_resolution?.join(" × ") ?? "—",
      frames: camera.frames_processed,
      localTracks: camera.unique_local_track_ids,
      peakCount,
      peakFrame,
      averageConfidence,
      alertCount: alerts.length,
      alerts: alerts.slice(-5).reverse(),
    };
  });
  const detectorLabels = [...new Set(streams.map((stream) => stream.detector))];
  const observationWeight = streams.reduce((total, stream) => total + stream.frames, 0) || 1;
  const busiestStream = streams.reduce((best, stream) => (stream.peakCount > best.peakCount ? stream : best), { peakCount: 0, peakFrame: 0 });
  const alerts = streams
    .flatMap((stream) => stream.alerts.map((alert) => ({ ...alert, cameraId: stream.id, cameraLabel: stream.label })))
    .sort((first, second) => (second.timestamp_seconds ?? 0) - (first.timestamp_seconds ?? 0));
  const candidates = (reidReport.candidates ?? []).slice(0, 3).map((candidate) => {
    const cropUrl = (cropPath) => {
      if (!cropPath) return null;
      const fileName = path.basename(cropPath);
      return /^[-_a-zA-Z0-9]+_track_\d+\.jpg$/.test(fileName)
        ? `/api/reid-crops/${encodeURIComponent(runId)}/${encodeURIComponent(fileName)}`
        : null;
    };
    return {
      ...candidate,
      probeImageUrl: cropUrl(candidate.probe_thumbnail),
      galleryImageUrl: cropUrl(candidate.gallery_thumbnail),
    };
  });
  return {
    id: runId,
    label: "Two Stream Demo",
    isMulti: true,
    updatedAt: statSync(summaryPath).mtime.toISOString(),
    hasVideo: streams.some((stream) => stream.previewFrames.length),
    detector: detectorLabels.length === 1 ? detectorLabels[0] : "Per-camera detector policy",
    streams,
    reid: {
      method: reidReport.summary?.method ?? "Not generated yet",
      candidateThreshold: reidReport.summary?.candidate_threshold ?? null,
      reviewCandidates: reidReport.summary?.review_candidates ?? 0,
      warning: reidReport.summary?.warning ?? "Run the updated two-stream pipeline to generate review candidates.",
      candidates,
    },
    metrics: {
      frames: summary.total_frames_processed ?? 0,
      processingFps: summary.aggregate_processing_fps ?? 0,
      duration: summary.elapsed_seconds ?? 0,
      resolution: streams.map((stream) => stream.resolution).join(" + "),
      uniqueTracks: streams.reduce((total, stream) => total + (stream.localTracks ?? 0), 0),
      averageConfidence: streams.reduce((total, stream) => total + stream.averageConfidence * stream.frames, 0) / observationWeight,
      peakCount: busiestStream.peakCount,
      peakFrame: busiestStream.peakFrame,
      alertCount: streams.reduce((total, stream) => total + stream.alertCount, 0),
      source: "Two synchronized laboratory camera views",
    },
    timeline: [],
    latencyProfile: summary.latency_profile ?? [],
    tracks: [],
    alerts,
    evaluation: null,
  };
}

export async function GET() {
  if (!existsSync(ARTIFACTS_ROOT)) {
    return Response.json({ runs: [], activeRun: null, training: readTrainingProgress(), benchmark: null });
  }

  const runs = readdirSync(ARTIFACTS_ROOT, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => {
      const summary = safeJson(path.join(ARTIFACTS_ROOT, entry.name, "summary.json"), {});
      return summary.run_type === "multi_stream" ? readMultiRun(entry.name) : readRun(entry.name);
    })
    .filter(Boolean)
    .sort((first, second) => new Date(second.updatedAt) - new Date(first.updatedAt));

  return Response.json({ runs, activeRun: runs[0] ?? null, training: readTrainingProgress(), benchmark: readLatestMotBenchmark() });
}
