import { readFileSync, existsSync } from "node:fs";
import path from "node:path";

const EXPERIMENTS = path.resolve(process.cwd(), "..", "runs", "experiments");

export function readTrainingExperiment() {
  const pointerPath = path.join(EXPERIMENTS, "active.json");
  if (!existsSync(pointerPath)) return null;
  const pointer = JSON.parse(readFileSync(pointerPath, "utf8"));
  if (!/^[A-Za-z0-9_]+$/.test(pointer.name ?? "")) throw new Error("Invalid experiment name.");
  const directory = path.join(EXPERIMENTS, pointer.name);
  const status = JSON.parse(readFileSync(path.join(directory, "status.json"), "utf8"));
  const decisionPath = path.join(directory, "decision.json");
  const decision = existsSync(decisionPath) ? JSON.parse(readFileSync(decisionPath, "utf8")) : null;
  return { ...status, decision, name: pointer.name };
}
