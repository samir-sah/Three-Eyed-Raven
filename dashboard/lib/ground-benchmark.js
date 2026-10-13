import { readFileSync } from "node:fs";
import path from "node:path";

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
    if (comparison.smoke) return null;
    return { ...base, protocol: comparison.protocol, evaluations: comparison.evaluations.map((item) => ({
      name: item.name, checkpoint: item.checkpoint, resolution: item.resolution,
      detection: item.report_detection,
      default: item.report_default, tuned: item.report_tuned,
      threshold: item.selected_threshold, speed: item.speed,
    })) };
  } catch {
    return null;
  }
}
