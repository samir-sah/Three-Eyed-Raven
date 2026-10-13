import { readFileSync } from "node:fs";
import path from "node:path";
import { validateGroundPolicy } from "./detection-policy.mjs";

export function normalizeGroundComparison(comparison, manifest, activeProfileName) {
  const protocol = comparison.protocol ?? manifest.protocol ?? {};
  return {
    protocol: { ...protocol, limitations: comparison.limitations ?? comparison.limitation ?? protocol.limitations },
    activeProfileName,
    targetReached: comparison.target_reached === true,
    evaluations: (comparison.reports ?? comparison.evaluations).map((item) => ({
      name: item.name, checkpoint: item.checkpoint ?? item.profile?.checkpoint,
      resolution: item.resolution ?? item.profile?.imgsz, profile: item.profile,
      isActive: item.name === activeProfileName,
      detection: item.report_detection ?? item.detection,
      default: item.report_default ?? item.fixed,
      tuned: item.report_tuned ?? item.calibrated?.fixed ?? item.fixed,
      rawCounts: item.raw_counts,
      threshold: item.selected_threshold ?? item.calibrated?.threshold ?? .35,
      speed: { ...item.speed, decode_predict_fps: item.speed?.decode_predict_fps ?? item.speed?.images_per_second },
    })),
  };
}

export function readGroundBenchmark() {
  try {
    const root = path.resolve(process.cwd(), "..", "runs", "experiments");
    const pointer = JSON.parse(readFileSync(path.join(root, "ground_active.json"), "utf8"));
    if (!/^[A-Za-z0-9_]+$/.test(pointer.name ?? "")) return null;
    const folder = path.join(root, pointer.name);
    const status = JSON.parse(readFileSync(path.join(folder, "status.json"), "utf8"));
    const base = { name: pointer.name, status: status.status, updatedAt: status.updatedAt,
      images: status.images, scenes: status.scenes, activeModel: status.activeStage,
      completedImages: status.evaluationImagesDone, totalImages: status.evaluationImagesTotal };
    if (status.status !== "completed") return base;
    const comparison = JSON.parse(readFileSync(path.join(folder, "comparison.json"), "utf8"));
    const manifest = JSON.parse(readFileSync(path.join(folder, "manifest.json"), "utf8"));
    if (comparison.smoke) return null;
    let activeProfileName = "coco_yolo11s_960";
    try {
      const policy = JSON.parse(readFileSync(path.join(root, "..", "..", "ground_detection_policy.json"), "utf8"));
      activeProfileName = validateGroundPolicy(policy).name;
    } catch { /* Only a valid saved policy changes the current-profile label. */ }
    return { ...base, images: base.images ?? manifest.images.length, scenes: base.scenes ?? manifest.scenes,
      ...normalizeGroundComparison(comparison, manifest, activeProfileName) };
  } catch {
    return null;
  }
}
