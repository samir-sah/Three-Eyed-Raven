"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

const numeric = (value, digits = 0) => Number.isFinite(value) ? value.toLocaleString(undefined, { maximumFractionDigits: digits }) : "—";
const percent = (value) => Number.isFinite(value) ? `${(value * 100).toFixed(2)}%` : "—";

export default function TrainingPage() {
  const [experiment, setExperiment] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    let stopped = false;
    let timer;
    const controller = new AbortController();
    async function poll() {
      try {
        const response = await fetch("/api/training", { cache: "no-store", signal: controller.signal });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || "Could not read training status.");
        if (!stopped) { setExperiment(payload.experiment); setError(""); }
      } catch (problem) {
        if (!stopped) setError(problem.message);
      } finally {
        if (!stopped) { setLoading(false); timer = setTimeout(poll, 10000); }
      }
    }
    poll();
    return () => { stopped = true; clearTimeout(timer); controller.abort(); };
  }, []);
  const finished = ["completed", "completed_with_errors", "failed"].includes(experiment?.status);
  const inferenceSearch = Number.isFinite(experiment?.profilesTotal);
  const stale = experiment && !finished && Date.now() - new Date(experiment.updatedAt).getTime() > 300000;
  const progress = experiment?.batchesTotal ? Math.min(100, (experiment.batchesDone || 0) / experiment.batchesTotal * 100) : 0;
  return <main className="training-monitor">
    <header><div><span className="eyebrow">THREE EYED RAVEN / DEVELOPMENT</span><h1>{inferenceSearch ? "Detector evaluation" : "Crowd detection training"}</h1><p>{experiment?.datasetDescription ?? "Detection experiments and aggregate crowd-count evaluation."}</p></div><Link href="/">← Dashboard</Link></header>
    {loading && <p role="status">Reading experiment progress…</p>}
    {error && <p role="alert">{error}</p>}
    {!loading && !error && !experiment && <section className="card monitor-card">No extended experiment has started yet.</section>}
    {experiment && <>
      <section className="card monitor-card" aria-live="polite">
        <div className="card-heading"><div><span className="eyebrow">CURRENT STATUS</span><h2>{experiment.status.replaceAll("_", " ")}</h2><p>{experiment.activeStage?.replaceAll("_", " ") || "Checking dataset and GPU"}</p></div><span className="subtle-pill">{finished ? "Finished" : "Updates every 10 seconds"}</span></div>
        <p>Last update: {new Date(experiment.updatedAt).toLocaleString()}</p>
        <p>Experiment: {experiment.name}{experiment.architecture ? ` · ${experiment.architecture}` : ""}</p>
        {experiment.decision && <div className="plain-result"><div><strong>{experiment.decision.summary}</strong><p>{experiment.decision.reason}</p><p>Checkpoint promoted: {experiment.decision.checkpointPromoted ? "Yes" : "No"}</p></div></div>}
        {Number.isFinite(experiment.target) && <p>Target: above {percent(experiment.target)} {experiment.targetMetric || "validation mAP50"}. {experiment.comparison ? (experiment.comparison.target_reached ? "Reached on development data; independent testing is still required." : "Not reached in this experiment.") : "Final result pending evaluation."}</p>}
        {stale && <p role="status">No update in over five minutes. Validation or slow disk access can delay updates; check the saved logs if this persists.</p>}
        {!finished && ["training", "validating"].includes(experiment.status) && <><p>Epoch {numeric(experiment.epoch)} · {numeric(experiment.batchesDone)} / {numeric(experiment.batchesTotal)} batches</p><progress value={progress} max="100" aria-label="Batches processed in current epoch" /></>}
        {experiment.status === "evaluating" && <p>Count evaluation: {numeric(experiment.evaluationImagesDone)} / {numeric(experiment.evaluationImagesTotal)} images</p>}
        {experiment.error && <details><summary>Failure details</summary><pre>{experiment.error}</pre></details>}
      </section>
      <section className="monitor-metrics">
        <div className="card monitor-card"><span>Training images</span><strong>{numeric(experiment.dataset?.train?.images)}</strong><small>Images actually used in this experiment</small></div>
        <div className="card monitor-card"><span>{inferenceSearch ? "Calibration images" : "Validation images"}</span><strong>{numeric(experiment.dataset?.val?.images)}</strong><small>Settings selection, not final test data</small></div>
        <div className="card monitor-card"><span>{inferenceSearch ? "Development report images" : "Training budget"}</span><strong>{inferenceSearch ? numeric(experiment.dataset?.report?.images) : `${numeric(experiment.budgetHours, 1)} hours`}</strong><small>{inferenceSearch ? "Frozen profile evaluation" : "Plus dataset checks and final evaluation"}</small></div>
      </section>
      {!inferenceSearch && <section className="card monitor-card"><div className="card-heading"><div><span className="eyebrow">EXPERIMENTS</span><h2>{experiment.architecture ? `${experiment.architecture} training stages` : "Training stages"}</h2></div></div>
        <p>Each stage saves best and latest checkpoints after completed epochs. Early stopping can finish a stage before its time budget.</p>
        {experiment.stages?.map(stage => <div className="monitor-stage" key={stage.name}><div><strong>{stage.imgsz} px</strong><p>{stage.status} · {numeric(stage.epoch)} completed epochs · {numeric(stage.budgetHours, 1)} hour budget</p></div><div><span>Latest validation mAP50–95</span><strong>{percent(stage.metrics?.["metrics/mAP50-95(B)"])}</strong></div></div>)}
      </section>}
      {experiment.comparison && <section className="card monitor-card"><span className="eyebrow">MEASURED RESULTS</span><h2>Model comparison</h2><p>{experiment.comparison.limitation}</p><div className="monitor-table"><table><thead><tr><th>Model</th><th>mAP50</th><th>mAP50–95</th><th>Count MAE ↓</th><th>Count RMSE ↓</th><th>Threshold</th></tr></thead><tbody>{experiment.comparison.evaluations.map(result => <tr key={result.name}><td>{result.name.replaceAll("_", " ")}</td><td>{percent(result.detection["metrics/mAP50(B)"])}</td><td>{percent(result.detection["metrics/mAP50-95(B)"])}</td><td>{numeric(result.counts.report.mae, 2)}</td><td>{numeric(result.counts.report.rmse, 2)}</td><td>{numeric(result.counts.selected_threshold, 2)}</td></tr>)}</tbody></table></div><p>MAE is the average absolute counting error per image. Lower is better. The saved report explains whether thresholds were calibrated or fixed.</p>{experiment.comparison.errors?.length > 0 && <details><summary>Stage errors ({experiment.comparison.errors.length})</summary><pre>{JSON.stringify(experiment.comparison.errors, null, 2)}</pre></details>}</section>}
      <details className="card monitor-card"><summary>Checkpoint and report locations</summary><p>{experiment.reportPath || `runs/experiments/${experiment.name}`}</p><pre>{JSON.stringify(experiment.stages?.map(({ name, bestCheckpoint, lastCheckpoint }) => ({ name, bestCheckpoint, lastCheckpoint })), null, 2)}</pre></details>
    </>}
  </main>;
}
