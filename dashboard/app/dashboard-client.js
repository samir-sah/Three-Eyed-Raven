"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import GroundBenchmark from "./ground-benchmark";
import { chooseSelection, footageLabel, recordingKey, recordingsFromRuns, selectedRun as findSelectedRun } from "../lib/recordings.mjs";
import {
  Activity,
  Bell,
  CheckCircle2,
  ChevronDown,
  CircleAlert,
  Clock3,
  Cpu,
  Download,
  Eye,
  FolderKanban,
  Gauge,
  LayoutDashboard,
  Menu,
  Pause,
  Play,
  Radio,
  RefreshCw,
  Search,
  ShieldCheck,
  Upload,
  Video,
} from "lucide-react";

const navItems = [
  ["overview", LayoutDashboard, "Overview"],
  ["video-runs", Video, "Playback"],
  ["tracking", Activity, "Crowd activity"],
  ["alerts", ShieldCheck, "Alerts"],
  ["development", FolderKanban, "Project progress"],
];

const advancedNav = [["benchmarks", Gauge, "Model comparisons"], ["performance", Cpu, "Technical details"], ["review", Eye, "Appearance review"]];
const modeLabel = (mode) => [...navItems, ...advancedNav].find(([id]) => id === mode)?.[2] ?? "Overview";
const selectionStorageKey = "three-eyed-raven:selected-footage";

function clipTime(seconds) {
  const value = Math.max(0, Math.floor(seconds || 0));
  return `${Math.floor(value / 60).toString().padStart(2, "0")}:${(value % 60).toString().padStart(2, "0")}`;
}

function NativeFootagePlayer({ run }) {
  const videos = useRef({});
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [error, setError] = useState("");
  const [ready, setReady] = useState(0);
  const duration = Math.max(...run.streams.map((stream) => stream.duration || 0), 0);
  const master = run.streams.reduce((best, stream) => stream.duration > (best?.duration || 0) ? stream : best, run.streams[0])?.id;
  useEffect(() => { setTime(0); setPlaying(false); setError(""); setReady(Object.values(videos.current).filter((video) => video?.readyState >= 1).length); }, [run.id]);
  useEffect(() => {
    for (const video of Object.values(videos.current)) {
      if (!video) continue;
      if (playing && video.currentTime < video.duration - .04) video.play().catch(() => { setError("Playback could not start. Try Play again."); setPlaying(false); });
      else video.pause();
    }
  }, [playing, ready]);
  function seek(next) {
    setTime(next);
    for (const video of Object.values(videos.current)) {
      if (video && Number.isFinite(video.duration)) video.currentTime = Math.min(next, Math.max(0, video.duration - .03));
    }
  }
  function update(event) {
    const next = event.currentTarget.currentTime;
    setTime(next);
    for (const [id, video] of Object.entries(videos.current)) {
      if (id === master || !video || !Number.isFinite(video.duration)) continue;
      const position = Math.min(next, Math.max(0, video.duration - .03));
      if (Math.abs(video.currentTime - position) > .3) video.currentTime = position;
    }
  }
  return <div className="footage-player" data-recording-id={run.id}>
    <div className="footage-streams footage-streams-pair">{run.streams.map((stream) => <section className="footage-camera" key={`${run.id}:${stream.id}`}>
      <div className="footage-camera-label"><Video size={15} /><span>{stream.label}</span><button className="secondary-button" onClick={() => videos.current[stream.id]?.requestFullscreen?.()} aria-label={`Expand ${stream.label}`}>Expand</button></div>
      <video className="footage-image" ref={(video) => { videos.current[stream.id] = video; }} src={stream.videoUrl} preload="auto" muted playsInline
        onLoadedMetadata={() => setReady(Object.values(videos.current).filter((video) => video?.readyState >= 1).length)}
        onTimeUpdate={stream.id === master ? update : undefined} onEnded={stream.id === master ? () => setPlaying(false) : undefined}
        onError={() => { setError("This video could not be loaded. Refresh the page or inspect the saved encoding log."); setPlaying(false); }} />
      <p className="footage-caption">{clipTime(stream.duration)} analysed · {stream.detector.includes("VisDrone") ? "Aerial-specific detector" : "General / ground detector"}</p>
    </section>)}</div>
    <div className="frame-controls"><button className="frame-play" disabled={!duration || ready < run.streams.length} onClick={() => { if (time >= duration - .1) seek(0); setError(""); setPlaying((value) => !value); }} aria-label={playing ? "Pause footage" : "Play footage"}>{playing ? <Pause size={17} /> : <Play size={17} />}</button><span>{clipTime(time)}</span><input aria-label="Footage position" type="range" min="0" max={duration || 1} step=".1" value={Math.min(time, duration)} onChange={(event) => seek(Number(event.target.value))} /><span>{clipTime(duration)}</span></div>
    {error && <p role="alert" className="upload-message upload-message-error">{error}</p>}
    {ready < run.streams.length && !error && <p role="status" className="footage-caption">Loading videos…</p>}
    <p className="footage-caption">Full-frame annotated video · Shared elapsed clip time, not synchronized camera clocks. Shorter clips hold their final image.</p>
  </div>;
}

function FootagePlayer({ run }) {
  const streams = run.isMulti ? run.streams : [{ id: "video", label: "Selected video", previewFrames: run.previewFrames ?? [], previewFps: run.previewFps }];
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const duration = Math.max(0, ...streams.map((stream) => Math.max(0, stream.previewFrames.length - 1) / (stream.previewFps || 5)));
  useEffect(() => { setTime(0); setPlaying(false); }, [run.id]);
  useEffect(() => {
    if (!playing) return undefined;
    const timer = window.setInterval(() => setTime((current) => Math.min(duration, current + 0.1)), 100);
    return () => window.clearInterval(timer);
  }, [playing, duration]);
  useEffect(() => { if (time >= duration) setPlaying(false); }, [time, duration]);
  if (run.isMulti && run.streams.every((stream) => stream.videoUrl)) return <NativeFootagePlayer key={run.id} run={run} />;
  if (!run.isMulti && !streams[0].previewFrames.length) return <video className="video-player" controls preload="metadata" src={run.videoUrl}>Your browser does not support video playback.</video>;
  return <div className="footage-player" data-recording-id={run.id}>
    <div className={run.isMulti ? "footage-streams footage-streams-pair" : "footage-streams"}>
      {streams.map((stream) => {
        const frames = stream.previewFrames;
        const index = Math.min(Math.floor(time * (stream.previewFps || 5)), Math.max(0, frames.length - 1));
        return <section className="footage-camera" key={stream.id}>
          <div className="footage-camera-label"><Video size={15} /><span>{stream.label}</span></div>
          {frames.length ? <img className="footage-image" src={frames[index]} alt={`${stream.label} annotated video`} /> : <div className="empty-state">No preview available for this camera.</div>}
        </section>;
      })}
    </div>
    <div className="frame-controls">
      <button className="frame-play" disabled={!duration} onClick={() => { if (time >= duration) setTime(0); setPlaying((value) => !value); }} aria-label={playing ? "Pause footage" : "Play footage"}>{playing ? <Pause size={17} fill="currentColor" /> : <Play size={17} fill="currentColor" />}</button>
      <span>{clipTime(time)}</span><input aria-label="Footage position" type="range" min="0" max={duration || 1} step="0.1" value={time} disabled={!duration} onChange={(event) => setTime(Number(event.target.value))} /><span>{clipTime(duration)}</span>
    </div>
    <p className="footage-caption">Annotated preview{run.isMulti ? " · Shared position within each clip; shorter clips hold their last image." : ""}</p>
  </div>;
}

function ImportantCounts({ run }) {
  const views = run.isMulti ? run.streams : [{ id: "video", label: "Selected video", peakCount: run.metrics.peakCount }];
  return <div className="important-counts">
    {views.map((view) => <div key={view.id}><span>{view.label}</span><strong>{view.countsVerified === false ? "—" : number(view.peakCount)} <small>people</small></strong><p>{view.countsVerified === false ? "Count log unavailable" : "Highest detector estimate in this view"}</p></div>)}
    <div><span>Crowd alerts</span><strong>{run.alertsEnabled === false ? "Off" : number(run.metrics.alertCount)}</strong><p>{run.alertsEnabled === false ? "No crowd limit configured" : "Recorded threshold events"}</p></div>
  </div>;
}

function FocusedPanel({ mode, run, data, versions = [], onSelectAnalysis }) {
  if (mode === "review" && run.analysisType === "anonymous_detection") return <section className="focused-panel"><h2>Anonymous crowd analysis</h2><p>This run counts detections in each view independently. It does not identify people or produce cross-camera match suggestions.</p></section>;
  if (mode === "alerts" && run.alertsEnabled === false) return <section className="focused-panel"><h2>Crowd alerts are not configured</h2><p>This run records counts only. Zero alerts does not mean that a safety threshold was checked.</p></section>;
  if (mode === "tracking" && run.analysisType === "anonymous_detection") return <section className="focused-panel"><h2>People detected over time</h2><p>Each point is an image-level count. Counts across views are not a count of unique people.</p><div className="camera-activity-list">{run.streams.map((stream) => <section key={stream.id}><h3>{stream.label}</h3>{stream.countsVerified === false ? <p>Count log unavailable.</p> : <><p>Peak: {number(stream.peakCount)} at {clipTime(stream.peakSeconds)}</p><OccupancyChart timeline={stream.timeline} /></>}</section>)}</div></section>;
  if (mode === "video-runs") return <div className="plain-result"><CheckCircle2 size={18} /><div><strong>Playback stays on your selected footage</strong><p>Use the timeline to review the annotated clips. Switch sections freely; the recording and playback position stay the same.</p></div></div>;
  if (mode === "overview") return <div className="plain-result"><CheckCircle2 size={18} /><div><strong>Analysis complete</strong><p>{run.isMulti ? "Review each camera’s peak count separately. A person visible in both views may be counted in each." : "Review the recorded peak count and any crowd alerts."}</p></div></div>;
  if (mode === "alerts") return <section className="focused-panel"><h2>{run.metrics.alertCount ? "Recorded crowd alerts" : "No crowd alerts recorded"}</h2><p>Alerts are based on the configured limit for each camera.</p><AlertHistory alerts={run.alerts} /></section>;
  if (mode === "tracking") return <section className="focused-panel"><h2>Crowd activity</h2><p>Peak counts describe people visible at one time, rather than unique visitors over the whole recording.</p>{run.isMulti ? <div className="camera-activity-list">{run.streams.map((stream) => <div key={stream.id}><span>{stream.label}</span><strong>Up to {number(stream.peakCount)} people</strong></div>)}</div> : run.timeline?.length ? <OccupancyChart timeline={run.timeline} /> : <p>No crowd activity was recorded.</p>}</section>;
  if (mode === "development") return <section className="focused-panel"><h2>Working prototype · validation continues</h2><p>New anonymous uploads use separate aerial and ground detectors. We are testing more representative scenes before claiming final accuracy. Legacy tracking runs remain unchanged.</p><p>{data.training?.status === "completed" ? "The latest training experiment has finished; its checkpoint is not automatically deployed." : "The current experiment's progress is available below."}</p><a className="secondary-button" href="/training">Watch training progress</a><details className="metric-details"><summary>Training details</summary><TrainingProgress training={data.training} /></details></section>;
  if (mode === "benchmarks") return <section className="focused-panel"><h2>Research comparisons</h2><p>These benchmark results describe separate experiments, rather than an accuracy score for the footage playing above.</p><details className="metric-details"><summary>Open ground-detector comparison</summary><GroundBenchmark benchmark={data.groundBenchmark} /></details><FullMotBenchmark benchmark={data.benchmark} /></section>;
  if (mode === "performance") return <section className="focused-panel"><h2>Technical details</h2>{versions.length > 1 && <details className="metric-details"><summary>Saved analyses of this same footage</summary><p>Changing this version selects another saved output, not another source video. Its counts may differ because the detector or processing settings differ.</p><label className="frame-limit"><span>Analysis version</span><select aria-label="Saved analysis version" value={run.id} onChange={(event) => onSelectAnalysis(event.target.value)}>{versions.map((version) => <option key={version.id} value={version.id}>{version.analysisType === "anonymous_detection" ? "Anonymous counting" : "Legacy tracking"} · {new Date(version.updatedAt).toLocaleString("en-IN")}</option>)}</select></label></details>}<details className="metric-details"><summary>Processing information</summary><p>{run.detector} · {number(run.metrics.frames)} processed frames · {number(run.metrics.processingFps, 2)} frames/second</p><PerformanceWorkspace run={run} metrics={run.metrics} /></details></section>;
  if (mode === "review") return <section className="focused-panel"><h2>Saved appearance review</h2><p>Similar appearance does not confirm that two people are the same person.</p>{run.isMulti ? <details className="metric-details"><summary>Open existing review candidates</summary><ReIdCandidates reid={run.reid} /></details> : <p>This recording has one camera view.</p>}</section>;
  return null;
}

function number(value, digits = 0) {
  return new Intl.NumberFormat("en-IN", { maximumFractionDigits: digits }).format(value ?? 0);
}

function relativeTime(value) {
  if (!value) return "No completed runs";
  const seconds = Math.max(0, Math.round((Date.now() - new Date(value).getTime()) / 1000));
  if (seconds < 60) return "Updated just now";
  if (seconds < 3600) return `Updated ${Math.floor(seconds / 60)} min ago`;
  return `Updated ${Math.floor(seconds / 3600)} h ago`;
}

function MetricCard({ icon: Icon, label, value, note, accent = false }) {
  return (
    <section className="card metric-card">
      <div className="metric-label"><Icon size={17} /> {label}</div>
      <div className={accent ? "metric-value metric-value-accent" : "metric-value"}>{value}</div>
      <p>{note}</p>
    </section>
  );
}

function OccupancyChart({ timeline }) {
  const points = timeline.length ? timeline : [{ frame: 0, count: 0 }];
  const max = Math.max(...points.map((point) => point.count), 1);
  const width = 680;
  const height = 184;
  const position = (point) => point.seconds ?? point.frame;
  const firstPosition = position(points[0]);
  const span = Math.max(1e-6, position(points.at(-1)) - firstPosition);
  const xFor = (point) => (position(point) - firstPosition) / span * width;
  const path = points.map((point, index) => {
    const x = xFor(point);
    const y = height - (point.count / max) * (height - 26) - 12;
    return `${index === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
  return (
    <svg className="occupancy-chart" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" role="img" aria-label="People detected over processed frames">
      {[0.2, 0.5, 0.8].map((fraction) => <line key={fraction} x1="0" x2={width} y1={height * fraction} y2={height * fraction} className="grid-line" />)}
      <path d={path} className="chart-line" />
      {points.map((point, index) => {
        const x = xFor(point);
        const y = height - (point.count / max) * (height - 26) - 12;
        return <circle key={`${point.frame}-${index}`} cx={x} cy={y} r="4" className="chart-point" />;
      })}
    </svg>
  );
}

function EmptyState() {
  return (
    <div className="empty-state">
      <Video size={26} />
      <strong>No pipeline run found yet</strong>
      <span>Run the Python tracker to populate this dashboard.</span>
    </div>
  );
}

function FramePlayer({ frames, videoUrl, sourceId, previewFps = 5 }) {
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    setFrame(0);
    setPlaying(false);
  }, [sourceId]);

  useEffect(() => {
    if (!playing || frames.length < 2) return undefined;
    const timer = window.setInterval(
      () => setFrame((current) => (current + 1) % frames.length),
      Math.max(100, Math.round(1000 / previewFps)),
    );
    return () => window.clearInterval(timer);
  }, [playing, frames.length]);

  if (!frames.length) {
    return <video className="video-player" controls preload="metadata" src={videoUrl}>Your browser does not support video playback.</video>;
  }

  return (
    <div className="frame-player">
      <img className="frame-image" src={frames[frame]} alt={`Annotated tracking frame ${frame + 1}`} />
      <div className="frame-controls">
        <button className="frame-play" onClick={() => setPlaying((current) => !current)} aria-label={playing ? "Pause preview" : "Play preview"}>{playing ? <Pause size={17} fill="currentColor" /> : <Play size={17} fill="currentColor" />}</button>
        <span>{playing ? "Playing" : "Paused"}</span>
        <input aria-label="Preview timeline" type="range" min="0" max={frames.length - 1} value={frame} onChange={(event) => setFrame(Number(event.target.value))} />
        <span>{frame + 1} / {frames.length}</span>
      </div>
    </div>
  );
}

function MultiStreamPlayer({ streams, sourceId }) {
  const frameCount = Math.min(...streams.map((stream) => stream.previewFrames.length));
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  const previewFps = streams[0]?.previewFps ?? 5;

  useEffect(() => {
    setFrame(0);
    setPlaying(false);
  }, [sourceId]);

  useEffect(() => {
    if (!playing || frameCount < 2) return undefined;
    const timer = window.setInterval(
      () => setFrame((current) => (current + 1) % frameCount),
      Math.max(100, Math.round(1000 / previewFps)),
    );
    return () => window.clearInterval(timer);
  }, [playing, frameCount, previewFps]);

  if (!frameCount) return <EmptyState />;
  return (
    <div className="multi-preview-wrap">
      <div className="multi-stream-player">
        {streams.map((stream) => (
          <section className="camera-preview" key={stream.id}>
            <div className="camera-preview-heading"><strong>{stream.label}</strong><span>{number(stream.localTracks)} local IDs</span></div>
            <img className="frame-image" src={stream.previewFrames[frame]} alt={`${stream.label} annotated preview frame ${frame + 1}`} />
          </section>
        ))}
      </div>
      <div className="frame-controls multi-frame-controls">
        <button className="frame-play" onClick={() => setPlaying((current) => !current)} aria-label={playing ? "Pause synchronized preview" : "Play synchronized preview"}>{playing ? <Pause size={17} fill="currentColor" /> : <Play size={17} fill="currentColor" />}</button>
        <span>{playing ? "Playing together" : "Paused"}</span>
        <input aria-label="Synchronized preview timeline" type="range" min="0" max={frameCount - 1} value={frame} onChange={(event) => setFrame(Number(event.target.value))} />
        <span>{frame + 1} / {frameCount}</span>
      </div>
      <p className="preview-caption">Synchronized annotated preview. Each step samples five original frames; it does not assert a cross-camera identity.</p>
    </div>
  );
}

function CandidateEvidence({ src, label }) {
  return src ? <img src={src} alt={label} /> : <span className="candidate-placeholder" aria-label={`${label} unavailable`}>—</span>;
}

function ReIdCandidates({ reid }) {
  const candidates = reid?.candidates ?? [];
  return (
    <section className="reid-panel">
      <div className="reid-heading"><div><span className="eyebrow">CROSS-CAMERA REVIEW</span><h3>Selected potential matches</h3></div><span className="subtle-pill">{reid?.oneToOneSuggestions ?? 0} one-to-one links</span></div>
      <p className="reid-method">A frame is one image from the video. A local ID is a temporary track in one camera. These are the highest-scoring non-overlapping links at or above the {reid?.candidateThreshold ? `${Math.round(reid.candidateThreshold * 100)}%` : "configured"} review threshold. Method: {reid?.method ?? "Not generated"}.</p>
      {candidates.length ? <div className="candidate-list">{candidates.map((candidate, index) => <div className="candidate-row" key={`${candidate.probe_camera_id}-${candidate.probe_track_id}-${candidate.gallery_camera_id}-${candidate.gallery_track_id}-${index}`}><div className="candidate-evidence"><CandidateEvidence src={candidate.probeImageUrl} label={`${candidate.probe_camera_id} local ${candidate.probe_track_id}`} /><span>↔</span><CandidateEvidence src={candidate.galleryImageUrl} label={`${candidate.gallery_camera_id} local ${candidate.gallery_track_id}`} /></div><div><strong>{candidate.probe_camera_id}/local-{candidate.probe_track_id}</strong><span>↔ {candidate.gallery_camera_id}/local-{candidate.gallery_track_id}</span></div><div className="candidate-score candidate-review"><strong>{Math.round(candidate.similarity * 100)}%</strong><small>manual review</small></div></div>)}</div> : <div className="reid-empty">No pairs met the review threshold for this run.</div>}
      <p className="reid-warning"><CircleAlert size={14} /> {reid?.warning} These are not verified identities and must not be used to make decisions about people.</p>
    </section>
  );
}

function CrossCameraConclusion({ reid }) {
  if (!reid) return null;
  const threshold = Math.round((reid.candidateThreshold ?? 0) * 100);
  return <section className="card cross-camera-card">
    <div className="card-heading"><div><span className="eyebrow">CROSS-CAMERA CONCLUSION</span><h2>{number(reid.oneToOneSuggestions)} potential cross-view matches</h2><p>One-to-one, highest-scoring visual-similarity links selected from the two camera-local track lists.</p></div><span className="subtle-pill">Manual review required</span></div>
    <div className="cross-camera-metrics"><div><span>Potential matches</span><strong>{number(reid.oneToOneSuggestions)}</strong><small>non-overlapping track links</small></div><div><span>Candidate pairs</span><strong>{number(reid.candidatePairsAboveThreshold)}</strong><small>at or above the threshold</small></div><div><span>Similarity threshold</span><strong>{threshold}%</strong><small>{reid.method?.includes("AG-ReID") ? "trained-model review gate" : "HSV baseline review gate"}</small></div><div><span>Mean selected similarity</span><strong>{Math.round((reid.meanSimilarity ?? 0) * 100)}%</strong><small>of the potential links</small></div></div>
    <p className="cross-camera-note"><CircleAlert size={14} /> This is the project’s strongest available cross-camera conclusion: these are people who look similar enough to review, not confirmed identities. Different views, lighting, clothing similarities, and track fragmentation can create false links.</p>
  </section>;
}

function AlertHistory({ alerts }) {
  if (!alerts?.length) return <p className="muted">No zone threshold crossings in this run.</p>;
  return <div className="alert-history">{alerts.slice(0, 5).map((alert, index) => <div className="alert-history-row" key={`${alert.cameraId}-${alert.zone_id}-${alert.frame_index}-${index}`}><span className="alert-history-icon"><CircleAlert size={14} /></span><div><strong>{alert.zone_id.replaceAll("_", " ")}</strong><small>{alert.cameraLabel} · frame {number(alert.frame_index)}</small></div><span>{number(alert.count)}/{number(alert.threshold)}</span></div>)}</div>;
}

function LatencyProfile({ stages }) {
  if (!stages?.length) return <p className="muted">No stage timing profile is available for this historical run.</p>;
  return <div className="latency-list">{stages.map((stage) => <div className="latency-row" key={stage.stage}><div><strong>{stage.stage.replaceAll("_", " ")}</strong><small>{number(stage.calls)} calls · p95 {number(stage.p95_ms, 1)} ms</small></div><span>{number(stage.mean_ms, 1)} ms</span></div>)}</div>;
}

function EvaluationResults({ evaluation }) {
  if (!evaluation) return null;
  const metrics = [
    ["Precision", evaluation.precision], ["Recall", evaluation.recall], ["F1", evaluation.f1],
    ["MOTA", evaluation.mota], ["MOTP", evaluation.motp], ["IDF1", evaluation.idf1], ["HOTA", evaluation.hota], ["ID switches", evaluation.id_switches],
  ];
  return <section className="card evaluation-card"><div className="card-heading"><div><span className="eyebrow">BENCHMARK EVALUATION</span><h2>Recorded tracking metrics</h2><p>IoU threshold {Math.round((evaluation.iou_threshold ?? 0.5) * 100)}% · Ground truth: {number(evaluation.ground_truth_boxes)} boxes</p></div><span className="subtle-pill">Measured baseline</span></div><div className="evaluation-grid">{metrics.map(([label, value]) => <div key={label}><span>{label}</span><strong>{typeof value === "number" && label !== "ID switches" ? `${number(value * 100, 2)}%` : number(value)}</strong></div>)}</div></section>;
}

function TrainingProgress({ training }) {
  if (!training) return null;
  const label = training.status === "running" ? "Training in progress" : training.status === "completed" ? "Training completed" : "Training update unavailable";
  const metrics = [["Precision", training.metrics?.precision], ["Recall", training.metrics?.recall], ["mAP50", training.metrics?.map50], ["mAP50-95", training.metrics?.map5095]].filter(([, value]) => typeof value === "number");
  const headline = training.metrics?.map50;
  return <section id="development" className="card training-card"><div className="card-heading"><div><span className="eyebrow">VISDRONE MODEL DEVELOPMENT</span><h2>{label}</h2><p>{training.name} · {relativeTime(training.updatedAt)}</p></div><span className={training.status === "running" ? "training-status training-status-running" : "training-status"}>{training.status}</span></div>{typeof headline === "number" && <div className="metric-conclusion"><strong>{number(headline * 100, 2)}%</strong><span><b>mAP50 detection quality</b> on the VisDrone person validation subset.</span></div>}<div className="training-progress"><div><strong>{number(training.currentEpoch)} / {number(training.totalEpochs || 0)} epochs</strong><span>{number(training.progress)}% complete</span></div><div className="training-track"><span style={{ width: `${training.progress}%` }} /></div></div>{metrics.length > 0 && <details className="metric-details"><summary>Show supporting training metrics</summary><div className="training-metrics">{metrics.map(([name, value]) => <div key={name}><span>{name}</span><strong>{number(value * 100, 2)}%</strong></div>)}</div></details>}<p className="training-log">{training.lastLine}</p></section>;
}

function FullMotBenchmark({ benchmark }) {
  if (!benchmark) return null;
  const metrics = [
    ["Precision", benchmark.precision], ["Recall", benchmark.recall], ["F1", benchmark.f1],
    ["MOTA", benchmark.mota], ["MOTP", benchmark.motp], ["IDF1", benchmark.idf1], ["HOTA", benchmark.hota], ["ID switches", benchmark.id_switches],
  ];
  return <details className="card evaluation-card full-benchmark-card"><summary>Historical MOT17 tracking benchmark · {new Date(benchmark.updatedAt).toLocaleDateString("en-IN")}</summary><div className="card-heading"><div><span className="eyebrow">HISTORICAL CROSS-DOMAIN BENCHMARK</span><h2>Older aerial detector tested with ByteTrack</h2><p>{benchmark.label} · {number(benchmark.ground_truth_boxes)} annotated person appearances across frames</p></div><span className="subtle-pill">Historical experiment</span></div><p className="benchmark-conclusion">This saved run used an aerial-trained checkpoint plus tracking. Its {number(benchmark.recall * 100, 1)}% recall does not measure today’s upload detector. Training-domain differences and pipeline filtering are possible contributors; this report does not isolate the cause.</p><p className="benchmark-checkpoint">Checkpoint: {benchmark.model}</p><details className="metric-details"><summary>Explain and show historical metrics</summary><p>MOTA combines missed detections, false positives and ID changes; MOTP measures matched-box alignment; IDF1 measures track identity continuity; HOTA balances detection and association. HOTA is the project’s internal calculation pending official TrackEval verification. Higher is better except for ID switches.</p><div className="evaluation-grid">{metrics.map(([label, value]) => <div key={label}><span>{label}</span><strong>{typeof value === "number" && label !== "ID switches" ? `${number(value * 100, 2)}%` : number(value)}</strong></div>)}</div></details></details>;
}

function ReIdBenchmark({ benchmark }) {
  if (!benchmark) return <section className="compact-inset"><span className="eyebrow">AERIAL-TO-GROUND RE-ID</span><h2>AG-ReID training underway</h2><p className="benchmark-conclusion">The learned cross-camera encoder is training on labelled aerial and CCTV crops. A held-out C0 aerial-to-C3 ground retrieval benchmark will appear here when evaluation completes.</p></section>;
  const metrics = [["Rank-1", benchmark.rank_1], ["Rank-5", benchmark.rank_5], ["Rank-10", benchmark.rank_10], ["mAP", benchmark.mAP]];
  return <section className="compact-inset"><div className="compact-heading"><div><span className="eyebrow">AERIAL-TO-GROUND RE-ID</span><h2>Held-out AG-ReID evaluation</h2><p>{benchmark.protocol}</p></div><span className="subtle-pill">Trained model</span></div><p className="benchmark-conclusion">This measures whether an unseen aerial person crop retrieves the same labelled person from unseen ground-camera crops. It validates visual retrieval quality; dashboard links still remain manual-review candidates, not identity determinations.</p><div className="training-metrics">{metrics.map(([label, value]) => <div key={label}><span>{label}</span><strong>{number(value * 100, 2)}%</strong></div>)}</div></section>;
}

function TrackAndAlerts({ run, metrics }) {
  const tracks = run.isMulti
    ? run.streams.map((stream) => ({ id: stream.id, label: stream.label, detail: `${number(stream.frames)} frames · peak ${number(stream.peakCount)} people · ${stream.detector}`, value: number(stream.localTracks), valueLabel: "local IDs" }))
    : (run.tracks ?? []).slice(0, 5).map((track) => ({ id: track.id, label: `local-${track.id}`, detail: `Frames ${track.firstFrame}–${track.lastFrame}`, value: `${number(track.meanConfidence * 100, 0)}%`, valueLabel: "confidence" }));
  return <div className="track-alert-grid">
    <section className="compact-inset">
      <div className="compact-heading"><div><span className="eyebrow">TRACK HEALTH</span><h3>{run.isMulti ? "Independent camera summaries" : "Most persistent IDs"}</h3></div><span className="subtle-pill">{number(metrics.uniqueTracks)} total</span></div>
      <div className="track-list">{tracks.length ? tracks.map((track) => <div className="track-row" key={track.id}><span className="track-avatar">{run.isMulti ? <Video size={14} /> : track.id}</span><div><strong>{track.label}</strong><small>{track.detail}</small></div><div className="confidence"><strong>{track.value}</strong><small>{track.valueLabel}</small></div></div>) : <p className="muted">No tracked-person observations were recorded.</p>}</div>
    </section>
    <section className="compact-inset">
      <div className="compact-heading"><div><span className="eyebrow">ALERTS</span><h3>Zone activity</h3></div><span className="subtle-pill">{number(metrics.alertCount)} total</span></div>
      <div className="alert-box"><CircleAlert size={18} /><div><strong>{metrics.alertCount ? `${metrics.alertCount} crowd alert${metrics.alertCount === 1 ? "" : "s"}` : "No crowd alerts"}</strong><span>{metrics.alertCount ? "Zone threshold crossings are listed below." : "Zone threshold was not exceeded in this run."}</span></div></div>
      <AlertHistory alerts={run.alerts} />
    </section>
  </div>;
}

function PerformanceWorkspace({ run, metrics }) {
  return <div className="performance-workspace">
    {!run.isMulti && <section className="compact-inset"><div className="compact-heading"><div><span className="eyebrow">FRAME-BY-FRAME ANALYSIS</span><h3>Observed people over time</h3><p>Unique local track IDs visible in each processed frame.</p></div><div className="chart-legend"><span><i className="legend-line" /> People detected</span><span><Clock3 size={15} /> 0–{number(metrics.frames)}</span></div></div><OccupancyChart timeline={run.timeline} /><div className="chart-labels"><span>Start</span><span>Peak: {number(metrics.peakCount)} people</span><span>End</span></div></section>}
    <section className="compact-inset"><div className="compact-heading"><div><span className="eyebrow">PIPELINE PROFILING</span><h3>Stage latency breakdown</h3><p>Measured during this completed run.</p></div><span className="subtle-pill">milliseconds</span></div><LatencyProfile stages={run.latencyProfile} /></section>
    {run.evaluation && <EvaluationResults evaluation={run.evaluation} />}
  </div>;
}

function SupportingWorkspace({ run, data, metrics, refreshedAt, activePanel, onPanelChange }) {
  const panels = [
    ["review", run.isMulti ? "Match review" : "Track details"],
    ["tracking", "Camera activity"],
    ["performance", "Performance"],
    ["development", "Development"],
    ["benchmarks", "Model comparisons"],
  ];
  return <section id="supporting-workspace" className="card supporting-workspace">
    <div className="card-heading"><div><span className="eyebrow">SUPPORTING DETAILS</span><h2>Explore the run without leaving playback</h2><p>Switch views to inspect evidence, activity, performance, or project progress.</p></div></div>
    <div className="workspace-tabs" role="tablist" aria-label="Supporting dashboard details">{panels.map(([id, label]) => <button key={id} type="button" role="tab" aria-selected={activePanel === id} className={activePanel === id ? "workspace-tab workspace-tab-active" : "workspace-tab"} onClick={() => onPanelChange(id)}>{label}</button>)}</div>
    <div className="workspace-panel" role="tabpanel">
      {activePanel === "review" && (run.isMulti ? <ReIdCandidates reid={run.reid} /> : <TrackAndAlerts run={run} metrics={metrics} />)}
      {activePanel === "tracking" && <TrackAndAlerts run={run} metrics={metrics} />}
      {activePanel === "performance" && <PerformanceWorkspace run={run} metrics={metrics} />}
      {activePanel === "benchmarks" && <div className="benchmark-workspace"><GroundBenchmark benchmark={data.groundBenchmark} /><FullMotBenchmark benchmark={data.benchmark} /></div>}
      {activePanel === "development" && <div className="development-workspace"><section className="compact-inset development-card"><span className="eyebrow">DEVELOPMENT ROADMAP</span><h2>Where the project stands</h2><div className="milestones"><span className="complete"><CheckCircle2 size={16} /> Detection</span><span className="complete"><CheckCircle2 size={16} /> Local tracking</span><span className="complete"><CheckCircle2 size={16} /> Crowd analytics</span><span className="complete"><CheckCircle2 size={16} /> Re-ID review</span><span className="current"><Radio size={16} /> Field validation</span></div></section><section className="compact-inset run-details"><span className="eyebrow">RUN DETAILS</span><dl><div><dt>Input</dt><dd>{metrics.source}</dd></div><div><dt>Last refresh</dt><dd>{refreshedAt ? refreshedAt.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "—"}</dd></div></dl></section><ReIdBenchmark benchmark={data.reidBenchmark} /><TrainingProgress training={data.training} /><FullMotBenchmark benchmark={data.benchmark} /></div>}
    </div>
  </section>;
}

function UploadRunPanel({ onCompleted, initiallyOpen = false }) {
  const [aerial, setAerial] = useState(null);
  const [ground, setGround] = useState(null);
  const [maxFrames, setMaxFrames] = useState("600");
  const [aerialProfile, setAerialProfile] = useState("aerial");
  const [job, setJob] = useState(null);
  const [message, setMessage] = useState("");
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    if (!job?.runId || ["completed", "failed"].includes(job.status)) return undefined;
    let active = true;
    const poll = async () => {
      try {
        const response = await fetch(`/api/upload-run/${job.runId}`, { cache: "no-store" });
        if (!response.ok) return;
        const next = await response.json();
        if (!active) return;
        setJob(next);
        if (next.status === "completed") onCompleted(next.runId);
      } catch {
        // The next poll will retry while the local dashboard is available.
      }
    };
    poll();
    const timer = window.setInterval(poll, 2500);
    return () => { active = false; window.clearInterval(timer); };
  }, [job?.runId, job?.status, onCompleted]);

  async function submit(event) {
    event.preventDefault();
    if (!aerial || !ground) { setMessage("Choose both an aerial and a ground-surveillance video."); return; }
    setSubmitting(true);
    setMessage("");
    try {
      const form = new FormData();
      form.append("aerial", aerial);
      form.append("ground", ground);
      form.append("maxFrames", maxFrames);
      form.append("aerialProfile", aerialProfile);
      const response = await fetch("/api/upload-run", { method: "POST", body: form });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error ?? "The upload run could not start.");
      setJob(result);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "The upload run could not start.");
    } finally { setSubmitting(false); }
  }

  const status = job?.status;
  return <details className="card upload-run-panel" open={initiallyOpen}>
    <summary className="upload-summary"><span><span className="eyebrow">YOUR TWO-CAMERA RUN</span><strong>Upload and analyse another video pair</strong></span><span className="subtle-pill">Open uploader</span></summary>
    <div className="upload-run-content">
    <div className="card-heading"><div><span className="eyebrow">YOUR TWO-CAMERA RUN</span><h2>Upload aerial and ground videos</h2><p>Files remain on this computer. Separate detectors estimate crowd counts independently, without identity matching. Ground analysis now prioritizes accuracy: processing can be slower than the video's playback speed.</p></div><span className="subtle-pill">Local processing</span></div>
    <form className="upload-form" onSubmit={submit}>
      <label className={aerial ? "upload-slot upload-slot-ready" : "upload-slot"}><Upload size={19} /><span><b>Aerial / drone video</b><small>{aerial ? aerial.name : "MP4, AVI, MOV, MKV, MPG or WebM"}</small></span><input type="file" accept="video/*,.mp4,.avi,.mov,.mkv,.mpg,.mpeg,.webm" onChange={(event) => setAerial(event.target.files?.[0] ?? null)} /></label>
      <label className={ground ? "upload-slot upload-slot-ready" : "upload-slot"}><Upload size={19} /><span><b>Ground surveillance video</b><small>{ground ? ground.name : "MP4, AVI, MOV, MKV, MPG or WebM"}</small></span><input type="file" accept="video/*,.mp4,.avi,.mov,.mkv,.mpg,.mpeg,.webm" onChange={(event) => setGround(event.target.files?.[0] ?? null)} /></label>
      <label className="frame-limit"><span>Frames per camera</span><input type="number" min="1" max="20000" value={maxFrames} onChange={(event) => setMaxFrames(event.target.value)} placeholder="Full video" /><small>Use 600 for a quick demo, or clear this field for the full videos.</small></label>
      <label className="frame-limit"><span>Aerial upload viewpoint</span><select value={aerialProfile} onChange={(event) => setAerialProfile(event.target.value)}><option value="aerial">Drone / top-down · aerial detector</option><option value="general">Ground-level / elevated CCTV · general detector</option></select><small>Select the actual viewpoint. The aerial-trained model can miss people in ground-level clips.</small></label>
      <button className="primary-button upload-submit" disabled={submitting || ["queued", "processing"].includes(status)}><Upload size={17} /> {submitting ? "Uploading…" : status === "processing" || status === "queued" ? "Processing…" : "Start two-camera analysis"}</button>
    </form>
    {message && <p className="upload-message upload-message-error">{message}</p>}
    {job && <div className={status === "failed" ? "upload-message upload-message-error" : "upload-message"}><strong>{status === "completed" ? "Analysis complete" : status === "failed" ? "Analysis failed" : "Working on your run"}</strong><span>{job.error ?? job.stage}</span>{status === "completed" && <small>Your new run is selected above. Open the playback below to review it.</small>}</div>}
    <p className="upload-privacy"><CircleAlert size={14} /> Counts are estimates, not verified identities or unique visitors. Please do not start an upload while GPU training is running.</p>
    </div>
  </details>;
}

export default function DashboardClient() {
  const [data, setData] = useState({ runs: [], activeRun: null });
  const [service, setService] = useState({ available: false, runCount: 0 });
  const [selection, setSelection] = useState(null);
  const [activeSection, setActiveSection] = useState("overview");
  const [mobileOpen, setMobileOpen] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const loadDashboard = useCallback(async () => {
    setLoading(true);
    try {
      const [dashboardResult, serviceResult] = await Promise.allSettled([
        fetch("/api/dashboard", { cache: "no-store", signal: AbortSignal.timeout(15000) }),
        fetch("/api/service-health", { cache: "no-store", signal: AbortSignal.timeout(3000) }),
      ]);
      if (dashboardResult.status !== "fulfilled" || !dashboardResult.value.ok) throw new Error("Could not load your saved footage. Please try Refresh.");
      const dashboardResponse = dashboardResult.value;
      const next = await dashboardResponse.json();
      setData(next);
      if (serviceResult.status === "fulfilled" && serviceResult.value.ok) setService(await serviceResult.value.json());
      let stored = null;
      try { stored = JSON.parse(window.localStorage.getItem(selectionStorageKey)); } catch { /* Browser storage can be unavailable. */ }
      setSelection((current) => chooseSelection(next.runs, current, stored));
      setError("");
      return next;
    } catch (problem) {
      setError(problem.message || "Could not load saved footage.");
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadDashboard();
  }, [loadDashboard]);

  useEffect(() => {
    if (selection) {
      try { window.localStorage.setItem(selectionStorageKey, JSON.stringify(selection)); } catch { /* Selection still works for this visit. */ }
    }
  }, [selection]);

  const recordings = useMemo(() => recordingsFromRuns(data.runs), [data.runs]);
  const run = useMemo(() => findSelectedRun(data.runs, selection), [data.runs, selection]);
  const selectedRecording = recordings.find((item) => item.key === selection?.recordingKey);
  const goToSection = (section) => {
    setActiveSection(section);
    setMobileOpen(false);
    requestAnimationFrame(() => document.getElementById("footage-workspace")?.scrollIntoView({ behavior: "smooth", block: "start" }));
  };
  const selectUploadedRun = useCallback(async (runId) => {
    const next = await loadDashboard();
    const uploaded = next?.runs.find((item) => item.id === runId);
    if (uploaded) { setSelection({ runId, recordingKey: recordingKey(uploaded) }); setActiveSection("overview"); }
  }, [loadDashboard]);

  return (
    <main className="shell focused-shell">
      <aside className={mobileOpen ? "sidebar sidebar-open" : "sidebar"}>
        <div className="brand"><span className="brand-mark"><Eye size={20} /></span><span>Three Eyed Raven</span></div>
        <p className="sidebar-intro">Your footage. Clearer crowd insights.</p>
        <div className="side-label">Workspace</div>
        <nav>
          {navItems.map(([id, Icon, label]) => (
            <button key={label} aria-current={activeSection === id ? "page" : undefined} onClick={() => goToSection(id)} className={activeSection === id ? "nav-item nav-item-active" : "nav-item"}><Icon size={18} />{label}</button>
          ))}
        </nav>
        <details className="advanced-nav"><summary>Advanced</summary><nav>{advancedNav.map(([id, Icon, label]) => <button key={id} aria-current={activeSection === id ? "page" : undefined} onClick={() => goToSection(id)} className={activeSection === id ? "nav-item nav-item-active" : "nav-item"}><Icon size={18} />{label}</button>)}</nav></details>
        <div className="sidebar-foot card">
          <span className="eyebrow">PROJECT STATUS</span>
          <strong>Research prototype</strong>
          <p>Video analysis is available. Validation on more scenes continues.</p>
        </div>
      </aside>
      {mobileOpen && <button className="navigation-backdrop" aria-label="Close navigation" onClick={() => setMobileOpen(false)} />}

      <section className="app-area">
        <header className="topbar">
          <button className="icon-button mobile-menu" aria-label="Open menu" aria-expanded={mobileOpen} onClick={() => setMobileOpen((open) => !open)}><Menu size={20} /></button>
          <div className="project-switch"><span className={service.available ? "status-dot status-dot-online" : "status-dot"} /> Crowd Intelligence</div>
          <div className="top-actions"><span className="subtle-pill">Local workspace</span><button className="icon-button" aria-label="Show crowd alerts" onClick={() => goToSection("alerts")}><Bell size={19} /></button></div>
        </header>

          <div id="footage-workspace" className="content focused-content">
            <div className="page-heading">
            <div><span className="eyebrow">YOUR FOOTAGE</span><h1>{modeLabel(activeSection)}</h1></div>
            <div className="heading-actions">
              <button className="secondary-button" onClick={loadDashboard} disabled={loading}><RefreshCw size={16} className={loading ? "spin" : ""} /> Refresh</button>
            </div>
          </div>
          {error && <p className="upload-message upload-message-error" role="alert">{error}</p>}
          <div className="selected-footage-bar"><div><Video size={18} /><span><small>Selected footage · stays the same across all views</small><strong>{run ? footageLabel(run, selectedRecording?.uploaded) : (loading ? "Loading your footage…" : "Choose a recording")}</strong></span></div><details className="footage-picker"><summary>Change footage <ChevronDown size={14} /></summary><div className="footage-options">{recordings.map((recording) => <button key={recording.key} aria-pressed={recording.run.id === selection?.runId} onClick={(event) => { setSelection({ runId: recording.run.id, recordingKey: recording.key }); event.currentTarget.closest("details").open = false; }}>{recording.label}</button>)}{!recordings.length && <p>No saved recordings yet. Upload a pair below.</p>}</div></details></div>
          <UploadRunPanel onCompleted={selectUploadedRun} initiallyOpen={!loading && !recordings.length} />
          {run ? <>
            <section className="card video-card focused-video-card"><FootagePlayer run={run} />{!run.isMulti && <div className="video-footer"><a href={run.videoUrl} download><Download size={15} /> Download annotated video</a></div>}</section>
            {run.countWarning && <p role="alert" className="upload-message upload-message-error">{run.countWarning}</p>}
            <ImportantCounts run={run} />
            <div className="card focused-result" aria-live="polite"><FocusedPanel key={activeSection} mode={activeSection} run={run} data={data} versions={selectedRecording?.runs} onSelectAnalysis={(runId) => { if (selectedRecording?.runs.some((version) => version.id === runId)) setSelection({ runId, recordingKey: selectedRecording.key }); }} /></div>
          </> : <div className="empty-state"><Video size={26} /><strong>{loading ? "Loading your recording…" : selection ? "The selected recording is unavailable" : "Upload footage to get started"}</strong><span>{selection && !loading ? "Use Change footage to choose another recording." : "Choose aerial and ground clips in the uploader."}</span></div>}
        </div>
      </section>
    </main>
  );
}
