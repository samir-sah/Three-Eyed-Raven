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
  Video,
} from "lucide-react";

const navItems = [
  [LayoutDashboard, "Overview"],
  [Video, "Video runs"],
  [Activity, "Tracking"],
  [ShieldCheck, "Alerts"],
  [FolderKanban, "Development"],
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

function FramePlayer({ frames, videoUrl }) {
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    setFrame(0);
    setPlaying(false);
  }, [frames]);

  useEffect(() => {
    if (!playing || frames.length < 2) return undefined;
    const timer = window.setInterval(() => setFrame((current) => (current + 1) % frames.length), 115);
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

function MultiStreamPlayer({ streams }) {
  return (
    <div className="multi-stream-player">
      {streams.map((stream) => (
        <section className="camera-preview" key={stream.id}>
          <div className="camera-preview-heading"><strong>{stream.label}</strong><span>{number(stream.localTracks)} local IDs</span></div>
          <FramePlayer frames={stream.previewFrames} videoUrl="" />
        </section>
      ))}
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
      <div className="reid-heading"><div><span className="eyebrow">CROSS-CAMERA RE-ID</span><h3>Appearance review candidates</h3></div><span className="subtle-pill">{reid?.reviewCandidates ?? 0} above threshold</span></div>
      <p className="reid-method">Method: {reid?.method ?? "Not generated"}{reid?.candidateThreshold ? ` · review threshold ${Math.round(reid.candidateThreshold * 100)}%` : ""}</p>
      {candidates.length ? <div className="candidate-list">{candidates.map((candidate, index) => <div className="candidate-row" key={`${candidate.probe_camera_id}-${candidate.probe_track_id}-${candidate.gallery_camera_id}-${candidate.gallery_track_id}-${index}`}><div className="candidate-evidence"><CandidateEvidence src={candidate.probeImageUrl} label={`${candidate.probe_camera_id} local ${candidate.probe_track_id}`} /><span>↔</span><CandidateEvidence src={candidate.galleryImageUrl} label={`${candidate.gallery_camera_id} local ${candidate.gallery_track_id}`} /></div><div><strong>{candidate.probe_camera_id}/local-{candidate.probe_track_id}</strong><span>↔ {candidate.gallery_camera_id}/local-{candidate.gallery_track_id}</span></div><div className={candidate.status === "review" ? "candidate-score candidate-review" : "candidate-score"}><strong>{Math.round(candidate.similarity * 100)}%</strong><small>{candidate.status === "review" ? "manual review" : "below threshold"}</small></div></div>)}</div> : <div className="reid-empty">No candidates have been generated yet. Run the updated two-stream pipeline to populate this panel.</div>}
      <p className="reid-warning"><CircleAlert size={14} /> {reid?.warning}</p>
    </section>
  );
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
  return <section className="card training-card"><div className="card-heading"><div><span className="eyebrow">MODEL DEVELOPMENT</span><h2>{label}</h2><p>{training.name} · {relativeTime(training.updatedAt)}</p></div><span className={training.status === "running" ? "training-status training-status-running" : "training-status"}>{training.status}</span></div><div className="training-progress"><div><strong>{number(training.currentEpoch)} / {number(training.totalEpochs || 0)} epochs</strong><span>{number(training.progress)}% complete</span></div><div className="training-track"><span style={{ width: `${training.progress}%` }} /></div></div>{metrics.length > 0 && <div className="training-metrics">{metrics.map(([name, value]) => <div key={name}><span>{name}</span><strong>{number(value * 100, 2)}%</strong></div>)}</div>}<p className="training-log">{training.lastLine}</p></section>;
}

export default function DashboardClient() {
  const [data, setData] = useState({ runs: [], activeRun: null });
  const [service, setService] = useState({ available: false, runCount: 0 });
  const [selectedRun, setSelectedRun] = useState("");
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
    const timer = window.setInterval(loadDashboard, 8000);
    return () => window.clearInterval(timer);
  }, []);

  const run = useMemo(
    () => data.runs.find((item) => item.id === selectedRun) ?? data.activeRun,
    [data, selectedRun],
  );
  const metrics = run?.metrics;

  return (
    <main className="shell">
      <aside className="sidebar">
        <div className="brand"><span className="brand-mark"><Eye size={20} /></span><span>Three Eyed Raven</span></div>
        <button className="search-box"><Search size={18} /><span>Search project...</span><kbd>⌘K</kbd></button>
        <div className="side-label">Workspace</div>
        <nav>
          {navItems.map(([Icon, label], index) => (
            <button key={label} className={index === 0 ? "nav-item nav-item-active" : "nav-item"}><Icon size={18} />{label}</button>
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
            <div className="metric-grid">
              <MetricCard icon={Video} label="Processed frames" value={number(metrics.frames)} note={`${metrics.resolution} source resolution`} />
              <MetricCard icon={Activity} label="Peak crowd count" value={number(metrics.peakCount)} note={`Observed around frame ${number(metrics.peakFrame)}`} accent />
              <MetricCard icon={Gauge} label="Processing speed" value={`${number(metrics.processingFps, 2)} FPS`} note={`${number(metrics.duration, 1)} seconds of compute`} />
              <MetricCard icon={Cpu} label="Local track IDs" value={number(metrics.uniqueTracks)} note={`${number(metrics.averageConfidence * 100, 1)}% mean detection confidence`} />
            </div>

            <TrainingProgress training={data.training} />

            <div className="content-grid">
              <section className="card video-card">
                <div className="card-heading"><div><span className="eyebrow">ANNOTATED OUTPUT</span><h2>Tracking playback</h2></div><span className="live-badge"><span /> Recorded run</span></div>
                {run.hasVideo ? (run.isMulti ? <MultiStreamPlayer streams={run.streams} /> : <FramePlayer frames={run.previewFrames} videoUrl={run.videoUrl} />) : <EmptyState />}
                <div className="video-footer"><span><CheckCircle2 size={16} /> {run.isMulti ? "Both feeds processed with independent local trackers" : "Detection & local tracking completed"}</span>{!run.isMulti && <a href={run.videoUrl} download><Download size={16} /> Download MP4</a>}</div>
                {run.isMulti && <ReIdCandidates reid={run.reid} />}
              </section>

              <section className="card tracks-card">
                <div className="card-heading"><div><span className="eyebrow">TRACK HEALTH</span><h2>{run.isMulti ? "Independent camera summaries" : "Most persistent IDs"}</h2></div><span className="subtle-pill">{number(metrics.uniqueTracks)} total</span></div>
                <div className="track-list">
                  {run.isMulti ? run.streams.map((stream) => <div className="track-row" key={stream.id}><span className="track-avatar"><Video size={14} /></span><div><strong>{stream.label}</strong><small>{number(stream.frames)} frames · peak {number(stream.peakCount)} people</small></div><div className="confidence"><strong>{number(stream.localTracks)}</strong><small>local IDs</small></div></div>) : run.tracks.length ? run.tracks.map((track) => <div className="track-row" key={track.id}><span className="track-avatar">{track.id}</span><div><strong>local-{track.id}</strong><small>Frames {track.firstFrame}–{track.lastFrame}</small></div><div className="confidence"><strong>{number(track.meanConfidence * 100, 0)}%</strong><small>confidence</small></div></div>) : <p className="muted">No tracked-person observations were recorded.</p>}
                </div>
                <div className="alert-box"><CircleAlert size={18} /><div><strong>{metrics.alertCount ? `${metrics.alertCount} crowd alert${metrics.alertCount === 1 ? "" : "s"}` : "No crowd alerts"}</strong><span>{metrics.alertCount ? "Zone threshold crossings are recorded below." : "Zone threshold was not exceeded in this run."}</span></div></div>
                {run.isMulti && <section className="alert-history-panel"><span className="eyebrow">ZONE EVENT LOG</span><AlertHistory alerts={run.alerts} /></section>}
              </section>
            </div>

            <section className="card chart-card">
              <div className="card-heading"><div><span className="eyebrow">FRAME-BY-FRAME ANALYSIS</span><h2>Observed people over time</h2><p>Unique local track IDs visible in each processed frame.</p></div><div className="chart-legend"><span><i className="legend-line" /> People detected</span><span><Clock3 size={15} /> Frame range: 0–{number(metrics.frames)}</span></div></div>
              <OccupancyChart timeline={run.timeline} />
              <div className="chart-labels"><span>Start</span><span>Peak: {number(metrics.peakCount)} people</span><span>End</span></div>
            </section>

            <section className="card latency-card"><div className="card-heading"><div><span className="eyebrow">PIPELINE PROFILING</span><h2>Stage latency breakdown</h2><p>Measured during this completed run; values are per stage call.</p></div><span className="subtle-pill">milliseconds</span></div><LatencyProfile stages={run.latencyProfile} /></section>

            <EvaluationResults evaluation={run.evaluation} />

            <section className="development-row">
              <article className="card development-card"><span className="eyebrow">DEVELOPMENT ROADMAP</span><h2>Where the project stands</h2><div className="milestones"><span className="complete"><CheckCircle2 size={16} /> Detection</span><span className="complete"><CheckCircle2 size={16} /> Local tracking</span><span className="complete"><CheckCircle2 size={16} /> Crowd analytics</span><span className="complete"><CheckCircle2 size={16} /> Re-ID review</span><span className="current"><Radio size={16} /> Field validation</span></div></article>
              <article className="card run-details"><span className="eyebrow">RUN DETAILS</span><dl><div><dt>Input</dt><dd>{metrics.source}</dd></div><div><dt>Last refresh</dt><dd>{refreshedAt ? refreshedAt.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "—"}</dd></div></dl></article>
            </section>
          </>}
        </div>
      </section>
    </main>
  );
}
