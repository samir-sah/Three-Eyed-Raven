"use client";

import { useState } from "react";

const labels = {
  coco_yolo11n: "Previous upload detector · COCO YOLO11n",
  coco_yolo11n_640: "Previous uploads · YOLO11n · 640px",
  coco_yolo11s_640: "YOLO11s · 640px",
  coco_yolo11n_960: "YOLO11n · 960px",
  coco_yolo11s_960: "New anonymous uploads · YOLO11s · 960px",
  historical_visdrone_50e: "Historical aerial model · 50 epochs",
  extended_visdrone_nano: "Extended aerial YOLO11n",
  experimental_visdrone_small: "Experimental aerial YOLO11s",
};
const pct = (value) => Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "—";
const decimal = (value) => Number.isFinite(value) ? value.toFixed(2) : "—";
const score = (value) => Number.isFinite(value) ? `${(value * 100).toFixed(2)}%` : "—";
function detectorName(item) {
  if (labels[item.name]) return labels[item.name];
  const filename = String(item.checkpoint ?? "").split(/[\\/]/).at(-1);
  const model = filename?.match(/^(yolo11|yolo26)([nsmxl])\.pt$/);
  const label = model ? `${model[1].toUpperCase()}${model[2]}` : item.name;
  return `${item.isActive ? "New uploads · " : "Baseline · "}${label} · ${item.resolution}px${item.profile?.augment ? " · multi-scale" : ""}`;
}

export default function GroundBenchmark({ benchmark }) {
  const [tuned, setTuned] = useState(false);
  if (!benchmark) return <section className="compact-inset"><span className="eyebrow">GROUND DETECTION</span><h2>Fresh comparison pending</h2><p className="benchmark-conclusion">The historical tracking report does not measure the current website detector. A separate detection and counting audit will appear here.</p></section>;
  if (benchmark.status !== "completed") return <section className="compact-inset"><span className="eyebrow">GROUND DETECTION AUDIT</span><h2>{benchmark.status === "failed" ? "Comparison needs attention" : "Comparison in progress"}</h2><p className="benchmark-conclusion">{benchmark.activeModel} · {benchmark.completedImages ?? 0}/{benchmark.totalImages ?? benchmark.images} sampled images. Refresh to see the latest results.</p></section>;
  const website = benchmark.evaluations.find((item) => item.isActive) ?? benchmark.evaluations.find((item) => item.name === "coco_yolo11s_960") ?? benchmark.evaluations.find((item) => item.name === "coco_yolo11n");
  const current = website?.[tuned ? "tuned" : "default"];
  return <section className="compact-inset ground-benchmark">
    <div className="compact-heading"><div><span className="eyebrow">GROUND DETECTION · DEVELOPMENT AUDIT</span><h2>{website ? `${score(website.detection.map50)} mAP50 · current upload detector` : "How the actual detectors compare"}</h2><p>{benchmark.images} sampled images across {benchmark.scenes.length} MOT17 scenes · {new Date(benchmark.updatedAt).toLocaleDateString("en-IN")}</p></div><span className="subtle-pill">Not an independent test</span></div>
    <p className="benchmark-conclusion">At {tuned ? "a threshold selected on separate calibration scenes" : "the new anonymous-upload 35% confidence setting"}, the ground detector recovered <strong>{pct(current?.recall)}</strong> of annotated pedestrians. Its average count error was <strong>{decimal(current?.count_mae)} people per image</strong> on the report scenes.</p>
    <div className="workspace-tabs" aria-label="Comparison confidence setting"><button type="button" aria-pressed={!tuned} className={!tuned ? "workspace-tab workspace-tab-active" : "workspace-tab"} onClick={() => setTuned(false)}>Website setting</button><button type="button" aria-pressed={tuned} className={tuned ? "workspace-tab workspace-tab-active" : "workspace-tab"} onClick={() => setTuned(true)}>Calibrated thresholds</button></div>
    <div className="benchmark-table-wrap"><table className="benchmark-table"><thead><tr><th>Detector</th><th>Confidence</th><th>Recall ↑</th><th>Precision ↑</th><th>Count error ↓</th><th>mAP50 ↑</th></tr></thead><tbody>{benchmark.evaluations.map((item) => {
      const measured = item[tuned ? "tuned" : "default"];
      return <tr key={item.name}><th scope="row">{detectorName(item)}</th><td>{pct(measured.threshold)}</td><td>{pct(measured.recall)}</td><td>{pct(measured.precision)}</td><td>{decimal(measured.count_mae)}</td><td>{score(item.detection.map50)}</td></tr>;
    })}</tbody></table></div>
    <p className="benchmark-conclusion">The same {current?.images} report images score both profiles; four separate scenes select the settings. Count error uses MOT17’s annotated pedestrians and neutralizes ignored regions using ground truth. New uploads use the approved profile at 35% confidence; saved videos and legacy tracking are unchanged. {website?.profile?.augment && `Accuracy-first processing is slower: about ${decimal(website.speed.decode_predict_fps)} FPS on this laptop, not live real-time inference.`}</p>
    {tuned && <p className="benchmark-conclusion">Lower thresholds also admit more false detections. A lower count error can hide missed people being offset by incorrect boxes, so compare precision and recall together before changing the website setting.</p>}
    <details className="metric-details"><summary>Understand the scores and limitations</summary><p>Recall means how many annotated people were found. Precision means how many reported boxes were correct. Count error is mean absolute error: lower is better. mAP50 summarizes detection across confidence thresholds; it is not the percentage of people detected and does not change with the operating-threshold switch.</p><p>{benchmark.protocol.limitations} This custom detection audit is not the official MOTChallenge leaderboard protocol. It does not measure your uploaded footage, aerial accuracy or cross-camera identity.</p><div className="benchmark-table-wrap"><table className="benchmark-table"><thead><tr><th>Detector</th><th>Count bias</th><th>Count RMSE</th><th>Raw count MAE at 35%</th><th>mAP50–95</th><th>Decode + predict FPS</th></tr></thead><tbody>{benchmark.evaluations.map((item) => <tr key={item.name}><th scope="row">{detectorName(item)}</th><td>{decimal(item[tuned ? "tuned" : "default"].count_bias)}</td><td>{decimal(item[tuned ? "tuned" : "default"].count_rmse)}</td><td>{decimal(item.rawCounts?.mae)}</td><td>{pct(item.detection.map5095)}</td><td>{decimal(item.speed.decode_predict_fps)}</td></tr>)}</tbody></table></div><p>Raw count error includes detections on ignored annotations; neither count metric verifies all visible humans. Negative bias indicates undercounting. Speed excludes video encoding, playback and website rendering. Checkpoints and settings are preserved in the experiment report.</p></details>
  </section>;
}
