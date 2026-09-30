"use client";

import { useEffect, useMemo, useState } from "react";
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
  ["video-runs", Video, "Video runs"],
  ["tracking", Activity, "Tracking"],
  ["alerts", ShieldCheck, "Alerts"],
  ["development", FolderKanban, "Development"],
];

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
  const path = points.map((point, index) => {
    const x = points.length === 1 ? 0 : (index / (points.length - 1)) * width;
    const y = height - (point.count / max) * (height - 26) - 12;
    return `${index === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
  return (
    <svg className="occupancy-chart" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" role="img" aria-label="People detected over processed frames">
      {[0.2, 0.5, 0.8].map((fraction) => <line key={fraction} x1="0" x2={width} y1={height * fraction} y2={height * fraction} className="grid-line" />)}
      <path d={path} className="chart-line" />
      {points.map((point, index) => {
        const x = points.length === 1 ? 0 : (index / (points.length - 1)) * width;
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
    <div className="cross-camera-metrics"><div><span>Potential matches</span><strong>{number(reid.oneToOneSuggestions)}</strong><small>non-overlapping track links</small></div><div><span>Candidate pairs</span><strong>{number(reid.candidatePairsAboveThreshold)}</strong><small>at or above the threshold</small></div><div><span>Similarity threshold</span><strong>{threshold}%</strong><small>HSV appearance baseline</small></div><div><span>Mean selected similarity</span><strong>{Math.round((reid.meanSimilarity ?? 0) * 100)}%</strong><small>of the potential links</small></div></div>
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
  return <section className="card evaluation-card full-benchmark-card"><div className="card-heading"><div><span className="eyebrow">MOT17 CROSS-DOMAIN EVALUATION</span><h2>Strong precision, limited ground-camera recall</h2><p>{benchmark.label} · {number(benchmark.ground_truth_boxes)} ground-truth boxes · {relativeTime(benchmark.updatedAt)}</p></div><span className="subtle-pill">Measured baseline</span></div><p className="benchmark-conclusion">This is a stress test: the VisDrone aerial-trained detector and ByteTrack were evaluated on MOT17 ground-surveillance footage. It is usually right when it detects someone ({number(benchmark.precision * 100, 1)}% precision), but it misses many people ({number(benchmark.recall * 100, 1)}% recall). This is a domain gap, not production-ready CCTV performance.</p><details className="metric-details"><summary>Explain and show all MOT17 metrics</summary><p>MOTA combines missed detections, false positives and ID changes; MOTP measures box alignment; IDF1 measures track identity continuity; HOTA balances detection and association. Higher is better except for ID switches.</p><div className="evaluation-grid">{metrics.map(([label, value]) => <div key={label}><span>{label}</span><strong>{typeof value === "number" && label !== "ID switches" ? `${number(value * 100, 2)}%` : number(value)}</strong></div>)}</div></details></section>;
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
  ];
  return <section id="supporting-workspace" className="card supporting-workspace">
    <div className="card-heading"><div><span className="eyebrow">SUPPORTING DETAILS</span><h2>Explore the run without leaving playback</h2><p>Switch views to inspect evidence, activity, performance, or project progress.</p></div></div>
    <div className="workspace-tabs" role="tablist" aria-label="Supporting dashboard details">{panels.map(([id, label]) => <button key={id} type="button" role="tab" aria-selected={activePanel === id} className={activePanel === id ? "workspace-tab workspace-tab-active" : "workspace-tab"} onClick={() => onPanelChange(id)}>{label}</button>)}</div>
    <div className="workspace-panel" role="tabpanel">
      {activePanel === "review" && (run.isMulti ? <ReIdCandidates reid={run.reid} /> : <TrackAndAlerts run={run} metrics={metrics} />)}
      {activePanel === "tracking" && <TrackAndAlerts run={run} metrics={metrics} />}
      {activePanel === "performance" && <PerformanceWorkspace run={run} metrics={metrics} />}
      {activePanel === "development" && <div className="development-workspace"><section className="compact-inset development-card"><span className="eyebrow">DEVELOPMENT ROADMAP</span><h2>Where the project stands</h2><div className="milestones"><span className="complete"><CheckCircle2 size={16} /> Detection</span><span className="complete"><CheckCircle2 size={16} /> Local tracking</span><span className="complete"><CheckCircle2 size={16} /> Crowd analytics</span><span className="complete"><CheckCircle2 size={16} /> Re-ID review</span><span className="current"><Radio size={16} /> Field validation</span></div></section><section className="compact-inset run-details"><span className="eyebrow">RUN DETAILS</span><dl><div><dt>Input</dt><dd>{metrics.source}</dd></div><div><dt>Last refresh</dt><dd>{refreshedAt ? refreshedAt.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "—"}</dd></div></dl></section><TrainingProgress training={data.training} /><FullMotBenchmark benchmark={data.benchmark} /></div>}
    </div>
  </section>;
}

function UploadRunPanel({ onCompleted, initiallyOpen = false }) {
  const [aerial, setAerial] = useState(null);
  const [ground, setGround] = useState(null);
  const [maxFrames, setMaxFrames] = useState("600");
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
    <div className="card-heading"><div><span className="eyebrow">YOUR TWO-CAMERA RUN</span><h2>Upload aerial and ground videos</h2><p>Files remain on this computer. The system detects people, creates local tracks, and produces visual-similarity suggestions for manual review.</p></div><span className="subtle-pill">Local processing</span></div>
    <form className="upload-form" onSubmit={submit}>
      <label className={aerial ? "upload-slot upload-slot-ready" : "upload-slot"}><Upload size={19} /><span><b>Aerial / drone video</b><small>{aerial ? aerial.name : "MP4, AVI, MOV, MKV, MPG or WebM"}</small></span><input type="file" accept="video/*,.mp4,.avi,.mov,.mkv,.mpg,.mpeg,.webm" onChange={(event) => setAerial(event.target.files?.[0] ?? null)} /></label>
      <label className={ground ? "upload-slot upload-slot-ready" : "upload-slot"}><Upload size={19} /><span><b>Ground surveillance video</b><small>{ground ? ground.name : "MP4, AVI, MOV, MKV, MPG or WebM"}</small></span><input type="file" accept="video/*,.mp4,.avi,.mov,.mkv,.mpg,.mpeg,.webm" onChange={(event) => setGround(event.target.files?.[0] ?? null)} /></label>
      <label className="frame-limit"><span>Frames per camera</span><input type="number" min="1" max="20000" value={maxFrames} onChange={(event) => setMaxFrames(event.target.value)} placeholder="Full video" /><small>Use 600 for a quick demo, or clear this field for the full videos.</small></label>
      <button className="primary-button upload-submit" disabled={submitting || ["queued", "processing"].includes(status)}><Upload size={17} /> {submitting ? "Uploading…" : status === "processing" || status === "queued" ? "Processing…" : "Start two-camera analysis"}</button>
    </form>
    {message && <p className="upload-message upload-message-error">{message}</p>}
    {job && <div className={status === "failed" ? "upload-message upload-message-error" : "upload-message"}><strong>{status === "completed" ? "Analysis complete" : status === "failed" ? "Analysis failed" : "Working on your run"}</strong><span>{job.error ?? job.stage}</span>{status === "completed" && <small>Your new run is selected above. Open the playback below to review it.</small>}</div>}
    <p className="upload-privacy"><CircleAlert size={14} /> Similar-looking people are only placed in a manual-review queue; the project does not verify identity or make automated decisions about people.</p>
    </div>
  </details>;
}

export default function DashboardClient() {
  const [data, setData] = useState({ runs: [], activeRun: null });
  const [service, setService] = useState({ available: false, runCount: 0 });
  const [selectedRun, setSelectedRun] = useState("");
  const [activeSection, setActiveSection] = useState("overview");
  const [activeWorkspace, setActiveWorkspace] = useState("review");
  const [loading, setLoading] = useState(true);
  const [refreshedAt, setRefreshedAt] = useState(null);

  async function loadDashboard() {
    setLoading(true);
    try {
      const [dashboardResponse, serviceResponse] = await Promise.all([
        fetch("/api/dashboard", { cache: "no-store" }),
        fetch("/api/service-health", { cache: "no-store" }),
      ]);
      const next = await dashboardResponse.json();
      setData(next);
      if (serviceResponse.ok) setService(await serviceResponse.json());
      setSelectedRun((current) => current || next.activeRun?.id || "");
      setRefreshedAt(new Date());
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadDashboard();
  }, []);

  const run = useMemo(
    () => data.runs.find((item) => item.id === selectedRun) ?? data.activeRun,
    [data, selectedRun],
  );
  const metrics = run?.metrics;
  const goToSection = (section) => {
    setActiveSection(section);
    const workspaceForSection = { tracking: "tracking", alerts: "tracking", development: "development" }[section];
    if (workspaceForSection) setActiveWorkspace(workspaceForSection);
    const target = workspaceForSection ? "supporting-workspace" : section;
    requestAnimationFrame(() => document.getElementById(target)?.scrollIntoView({ behavior: "smooth", block: "start" }));
  };
  const selectUploadedRun = async (runId) => {
    await loadDashboard();
    setSelectedRun(runId);
  };

  return (
    <main className="shell">
      <aside className="sidebar">
        <div className="brand"><span className="brand-mark"><Eye size={20} /></span><span>Three Eyed Raven</span></div>
        <button className="search-box"><Search size={18} /><span>Search project...</span><kbd>⌘K</kbd></button>
        <div className="side-label">Workspace</div>
        <nav>
          {navItems.map(([id, Icon, label]) => (
            <button key={label} onClick={() => goToSection(id)} className={activeSection === id ? "nav-item nav-item-active" : "nav-item"}><Icon size={18} />{label}</button>
          ))}
        </nav>
        <div className="sidebar-foot card">
          <span className="eyebrow">PROJECT STATUS</span>
          <strong>Prototype validated</strong>
          <p>Multi-camera tracking, review-only Re-ID, alerting, API persistence, and evaluation tooling are working.</p>
          <div className="progress-track"><span className="progress-75" /></div>
          <small>Phase 3 of 4 · field validation next</small>
        </div>
      </aside>

      <section className="app-area">
        <header className="topbar">
          <button className="icon-button mobile-menu" aria-label="Open menu"><Menu size={20} /></button>
          <div className="project-switch"><span className={service.available ? "status-dot status-dot-online" : "status-dot"} /> Crowd Intelligence <ChevronDown size={16} /></div>
          <div className="top-actions"><span className="demo-mode">Panel demo</span><button className="icon-button"><Bell size={19} /><i /></button><button className="icon-button"><ShieldCheck size={19} /></button></div>
        </header>

          <div className="content">
            <div className={service.available ? "service-status service-status-online" : "service-status"}><span>{service.available ? "Run service online" : "Run service unavailable"}</span><small>{service.available ? `${number(service.runCount)} completed run${service.runCount === 1 ? "" : "s"} exposed by FastAPI` : "Dashboard is reading local artifacts directly"}</small></div>
            <div className="page-heading">
            <div><span className="eyebrow">SURVEILLANCE ANALYTICS</span><h1>Dashboard</h1><p>{relativeTime(run?.updatedAt)}</p></div>
            <div className="heading-actions">
              <label className="run-select"><Radio size={17} /><select value={run?.id ?? ""} onChange={(event) => setSelectedRun(event.target.value)}>{data.runs.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}</select><ChevronDown size={16} /></label>
              <button className="primary-button" onClick={loadDashboard} disabled={loading}><RefreshCw size={17} className={loading ? "spin" : ""} /> Refresh</button>
            </div>
          </div>

          {!run ? <EmptyState /> : <>
            <UploadRunPanel onCompleted={selectUploadedRun} initiallyOpen={!run.hasVideo} />
            <div id="overview" className="metric-grid">
              <MetricCard icon={Video} label="Processed frames" value={number(metrics.frames)} note={`${metrics.resolution} source resolution`} />
              <MetricCard icon={Activity} label="Peak visible tracks" value={number(metrics.peakCount)} note={run.isMulti ? "In one camera; not added across views" : `Observed around frame ${number(metrics.peakFrame)}`} accent />
              <MetricCard icon={Gauge} label={run.isMulti ? "Aggregate throughput" : "Processing speed"} value={`${number(metrics.processingFps, 2)} FPS`} note={run.isMulti ? "Combined speed across both feeds" : `${number(metrics.duration, 1)} seconds of compute`} />
              <MetricCard icon={Cpu} label="Local trajectory IDs" value={number(metrics.uniqueTracks)} note={run.isMulti ? "Camera-local lifetime IDs, not people" : `${number(metrics.averageConfidence * 100, 1)}% mean detection confidence`} />
            </div>

            {run.isMulti && <CrossCameraConclusion reid={run.reid} />}

            <div className="content-grid">
              <section id="video-runs" className="card video-card">
                <div className="card-heading"><div><span className="eyebrow">ANNOTATED OUTPUT</span><h2>Tracking playback</h2><p>{run.detector ?? "Detector not recorded"}</p></div><span className="live-badge"><span /> Recorded run</span></div>
                {run.hasVideo ? (run.isMulti ? <MultiStreamPlayer streams={run.streams} sourceId={run.id} /> : <FramePlayer frames={run.previewFrames} videoUrl={run.videoUrl} sourceId={run.id} previewFps={run.previewFps} />) : <EmptyState />}
                <div className="video-footer"><span><CheckCircle2 size={16} /> {run.isMulti ? "Both feeds processed with independent local trackers" : "Detection & local tracking completed"}</span>{!run.isMulti && <a href={run.videoUrl} download><Download size={16} /> Download MP4</a>}</div>
              </section>
            </div>
            <SupportingWorkspace run={run} data={data} metrics={metrics} refreshedAt={refreshedAt} activePanel={activeWorkspace} onPanelChange={setActiveWorkspace} />
          </>}
        </div>
      </section>
    </main>
  );
}
