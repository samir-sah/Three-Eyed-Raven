"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import GroundBenchmark from "./ground-benchmark";
import IndependentFootagePlayer from "./footage-player";
import { chooseWorkspaceSelection, footageLabel, recordingKey, recordingsFromRuns, selectedRun } from "../lib/recordings.mjs";
import { Activity, Bell, CheckCircle2, CircleAlert, Cpu, Eye, FolderKanban, Gauge, LayoutDashboard, Menu, RefreshCw, ShieldCheck, Upload, Video } from "lucide-react";

const navItems = [["overview", LayoutDashboard, "Overview"], ["upload", Upload, "Upload videos"], ["video-runs", Video, "Playback"], ["tracking", Activity, "Crowd activity"], ["alerts", ShieldCheck, "Alerts"], ["development", FolderKanban, "Project progress"]];
const advancedNav = [["benchmarks", Gauge, "Model comparisons"], ["performance", Cpu, "Technical details"]];
const modeLabel = (mode) => [...navItems, ...advancedNav].find(([id]) => id === mode)?.[2] ?? "Overview";
const selectionStorageKey = "three-eyed-raven:fresh-upload-selection";
const number = (value, digits = 0) => Number.isFinite(value) ? value.toLocaleString("en-IN", { maximumFractionDigits: digits }) : "—";
const clipTime = (seconds) => { const value = Math.max(0, Math.floor(seconds || 0)); return `${Math.floor(value / 60).toString().padStart(2, "0")}:${(value % 60).toString().padStart(2, "0")}`; };

function OccupancyChart({ timeline = [], label }) {
  if (!timeline.length) return <p>No count timeline is available.</p>;
  const max = Math.max(1, ...timeline.map((point) => point.count));
  const start = timeline[0].seconds ?? timeline[0].frame;
  const end = timeline.at(-1).seconds ?? timeline.at(-1).frame;
  const span = Math.max(1e-6, end - start);
  const point = (entry) => [((entry.seconds ?? entry.frame) - start) / span * 680, 172 - entry.count / max * 152];
  const path = timeline.map((entry, index) => `${index ? "L" : "M"}${point(entry).map((value) => value.toFixed(1)).join(",")}`).join(" ");
  return <><svg className="occupancy-chart" viewBox="0 0 680 184" role="img" aria-label={`${label}: people detected over clip time`}>
    {[.25, .5, .75].map((fraction) => <line key={fraction} x1="0" x2="680" y1={184 * fraction} y2={184 * fraction} className="grid-line" />)}<path d={path} className="chart-line" />
    {timeline.map((entry, index) => <circle key={index} cx={point(entry)[0]} cy={point(entry)[1]} r="3" className="chart-point" />)}
  </svg><div className="chart-labels"><span>Clip start</span><span>Peak: {max} detected</span><span>{clipTime(end)}</span></div></>;
}

function ImportantCounts({ run }) {
  return <div className="important-counts fresh-counts">{run.streams.map((view) => <div key={view.id}><span>{view.label}</span><strong>{view.countsVerified ? number(view.peakCount) : "—"} <small>people</small></strong><p>{view.countsVerified ? `Peak detector estimate at ${clipTime(view.peakSeconds)}` : "Count log is incomplete; no verified estimate"}</p></div>)}</div>;
}
function NoAnalysis({ title = "No uploaded analysis yet", onUpload }) {
  return <section className="card fresh-empty"><Video size={30} /><h2>{title}</h2><p>Upload your own aerial and ground videos. Results appear only after that upload finishes processing.</p><button className="primary-button" onClick={onUpload}><Upload size={16} /> Upload video pair</button></section>;
}
function Overview({ run, onPlayback, onUpload }) {
  return <section className="card focused-result fresh-overview" data-testid="overview-results"><div className="compact-heading"><div><span className="eyebrow">YOUR SELECTED ANALYSIS</span><h2>Fresh footage, measured counts</h2><p>Each result belongs to the clips named here—not a sample or historical benchmark.</p></div><span className="subtle-pill"><CheckCircle2 size={14} /> Complete</span></div><ImportantCounts run={run} />
    <div className="analysis-camera-list">{run.streams.map((stream) => <div key={stream.id}><Video size={18} /><span><strong>{run.sourceNames?.[stream.id] ?? stream.label}</strong><small>{stream.label} · {number(stream.frames)} analysed frames · {clipTime(stream.duration)}</small></span></div>)}</div>
    <p className="fresh-note">Per-frame estimates, not unique people. The views can contain the same people, so their counts must not be added.</p><div className="fresh-actions"><button className="primary-button" onClick={onPlayback}><Video size={16} /> Watch annotated videos</button><button className="secondary-button" onClick={onUpload}><Upload size={16} /> Upload another pair</button></div>
  </section>;
}
function WorkspacePanel({ mode, run, data, onUpload }) {
  if (mode === "development") return <section className="card focused-result focused-panel"><span className="eyebrow">CURRENT PRODUCT</span><h2>Upload → detect → inspect</h2><p>New uploads use anonymous detection, per-camera crowd counts and annotated playback. Tracking and identity matching are not part of this workflow.</p><div className="product-capabilities"><span><CheckCircle2 size={16} /> Separate aerial / ground detectors</span><span><CheckCircle2 size={16} /> Independent playback controls</span><span><CheckCircle2 size={16} /> Counts from your uploaded frames</span><span><CircleAlert size={16} /> Independent field testing still needed</span></div><p>Ground uploads use the approved accuracy-first profile. True drone views keep the aerial-trained detector. Laptop analysis can be slower than real-time.</p><a className="secondary-button" href="/training">Open experiment report</a></section>;
  if (mode === "benchmarks") return <section className="card focused-result focused-panel"><h2>Research benchmark—not uploaded-video accuracy</h2><p>This separate development evaluation measures the detector on labelled images. Your own clip counts are in Overview and Crowd activity.</p><GroundBenchmark benchmark={data.groundBenchmark} /></section>;
  if (!run) return <NoAnalysis onUpload={onUpload} />;
  if (mode === "tracking") return <section className="card focused-result focused-panel"><h2>Crowd activity in your clips</h2><p>Detector estimates over elapsed clip time. Each camera has its own timeline.</p><div className="fresh-activity">{run.streams.map((stream) => <section key={stream.id}><h3>{stream.label}</h3><p>{stream.countsVerified ? `Peak ${number(stream.peakCount)} at ${clipTime(stream.peakSeconds)}` : "Count log unavailable"}</p>{stream.countsVerified && <OccupancyChart timeline={stream.timeline} label={stream.label} />}</section>)}</div></section>;
  if (mode === "alerts") return <section className="card focused-result focused-panel"><span className="eyebrow">ALERT STATUS</span><h2>Crowd alerts are off</h2><p>No crowd-limit threshold was configured. People were counted, but no safety limit was checked. This is not a zero-alert safety result.</p><div className="analysis-camera-list">{run.streams.map((stream) => <div key={stream.id}><ShieldCheck size={18} /><span><strong>{stream.label}</strong><small>Counts available · alert limit not configured</small></span><span className="subtle-pill">Off</span></div>)}</div></section>;
  if (mode === "performance") return <section className="card focused-result focused-panel"><h2>Settings actually used for this upload</h2><p>Saved with this analysis. Changing a benchmark does not change these results.</p><div className="technical-camera-list">{run.streams.map((stream) => <section key={stream.id}><h3>{stream.label}</h3><dl><div><dt>Uploaded file</dt><dd>{run.sourceNames?.[stream.id] ?? "Name not recorded"}</dd></div><div><dt>Detector</dt><dd>{stream.detector}</dd></div><div><dt>Source resolution</dt><dd>{stream.resolution}</dd></div><div><dt>Frames analysed</dt><dd>{number(stream.frames)}</dd></div><div><dt>Count log</dt><dd>{stream.countsVerified ? "Complete and consistent" : "Incomplete"}</dd></div></dl></section>)}</div><p>Run: {run.id} · Processing: {number(run.metrics.duration, 1)} seconds · Throughput: {number(run.metrics.processingFps, 2)} frames/second, including encoding.</p><p className="fresh-note">A consistent log is not human-verified accuracy. No person IDs or cross-camera matches were generated.</p></section>;
  return null;
}

function UploadRunPanel({ onStarted, onCompleted, onFailed }) {
  const [aerial, setAerial] = useState(null);
  const [ground, setGround] = useState(null);
  const [maxFrames, setMaxFrames] = useState("");
  const [aerialProfile, setAerialProfile] = useState("aerial");
  const [job, setJob] = useState(null);
  const [message, setMessage] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const busy = submitting || ["queued", "processing"].includes(job?.status);
  useEffect(() => {
    if (!job?.runId || ["completed", "failed"].includes(job.status)) return;
    let stopped = false;
    let timer;
    async function poll() {
      try {
        const response = await fetch(`/api/upload-run/${job.runId}`, { cache: "no-store", signal: AbortSignal.timeout(10000) });
        if (!response.ok) throw new Error("Could not read processing status. Retrying…");
        const next = await response.json();
        if (stopped) return;
        setJob(next); setMessage("");
        if (next.status === "completed") await onCompleted(next.runId);
        if (next.status === "failed") onFailed();
      } catch (problem) { if (!stopped) setMessage(problem.message); }
      finally { if (!stopped) timer = setTimeout(poll, 2500); }
    }
    poll(); return () => { stopped = true; clearTimeout(timer); };
  }, [job?.runId, job?.status, onCompleted, onFailed]);
  async function submit(event) {
    event.preventDefault(); if (busy) return;
    if (!aerial || !ground) { setMessage("Choose both aerial and ground-surveillance videos."); return; }
    setSubmitting(true); setMessage("");
    try {
      const form = new FormData(); form.append("aerial", aerial); form.append("ground", ground); form.append("maxFrames", maxFrames); form.append("aerialProfile", aerialProfile);
      const response = await fetch("/api/upload-run", { method: "POST", body: form });
      const result = await response.json(); if (!response.ok) throw new Error(result.error ?? "The upload could not start.");
      setJob(result); onStarted(result.runId);
    } catch (problem) { setMessage(problem.message || "The upload could not start."); }
    finally { setSubmitting(false); }
  }
  return <section className="card fresh-upload" data-testid="upload-panel"><div className="compact-heading"><div><span className="eyebrow">START A NEW ANALYSIS</span><h2>Upload your two camera views</h2><p>No built-in videos or old results are loaded. Your clips stay on this computer.</p></div><span className="subtle-pill">Local processing</span></div>
    <form className="fresh-upload-form" onSubmit={submit}>
      <label className={`upload-slot${aerial ? " upload-slot-ready" : ""}`}><Upload size={24} /><span><b>Aerial / drone video</b><small>{aerial?.name ?? "Choose an aerial video"}</small></span><input aria-label="Aerial / drone video" disabled={busy} type="file" accept="video/*,.mp4,.avi,.mov,.mkv,.mpg,.mpeg,.webm" onChange={(event) => setAerial(event.target.files?.[0] ?? null)} /></label>
      <label className={`upload-slot${ground ? " upload-slot-ready" : ""}`}><Upload size={24} /><span><b>Ground surveillance video</b><small>{ground?.name ?? "Choose a ground-camera video"}</small></span><input aria-label="Ground surveillance video" disabled={busy} type="file" accept="video/*,.mp4,.avi,.mov,.mkv,.mpg,.mpeg,.webm" onChange={(event) => setGround(event.target.files?.[0] ?? null)} /></label>
      <label className="frame-limit"><span>Viewpoint of the first clip</span><select disabled={busy} value={aerialProfile} onChange={(event) => setAerialProfile(event.target.value)}><option value="aerial">Drone / top-down · aerial detector</option><option value="general">Ground-level / elevated CCTV · ground detector</option></select><small>Select the actual viewpoint; elevated CCTV is not necessarily drone footage.</small></label>
      <label className="frame-limit"><span>Frame limit per camera (optional)</span><input disabled={busy} type="number" min="1" max="20000" value={maxFrames} onChange={(event) => setMaxFrames(event.target.value)} placeholder="Full video" /><small>Blank analyses the full video. Enter 600 for a shorter test; only those frames are counted.</small></label>
      <div className="fresh-upload-footer"><p>Accuracy-first analysis can take longer than clip playback. Playback starts paused when ready.</p><button className="primary-button" disabled={busy || !aerial || !ground}><Upload size={17} />{submitting ? "Uploading…" : busy ? "Analysing…" : "Analyse my videos"}</button></div>
    </form>{message && <p className="upload-message upload-message-error" role="alert">{message}</p>}{job && <div className={`upload-message${job.status === "failed" ? " upload-message-error" : ""}`} role="status"><strong>{job.status === "completed" ? "Your analysis is complete" : job.status === "failed" ? "Analysis failed" : "Processing your uploaded videos"}</strong><span>{job.error ?? job.stage ?? "Preparing analysis…"}</span></div>}<p className="fresh-note">Counts are independent in each view. This workflow does not confirm identities or match people across cameras.</p>
  </section>;
}

export default function DashboardClient() {
  const [data, setData] = useState({ runs: [], activeRun: null, workspaceId: null });
  const [service, setService] = useState({ available: false });
  const [selection, setSelection] = useState(null);
  const [activeSection, setActiveSection] = useState("overview");
  const [mobileOpen, setMobileOpen] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const pendingRun = useRef(null);
  const requests = useRef(0);
  const loadDashboard = useCallback(async () => {
    const request = ++requests.current; setLoading(true);
    try {
      const [dashboardResult, serviceResult] = await Promise.allSettled([fetch("/api/dashboard", { cache: "no-store", signal: AbortSignal.timeout(15000) }), fetch("/api/service-health", { cache: "no-store", signal: AbortSignal.timeout(3000) })]);
      if (dashboardResult.status !== "fulfilled" || !dashboardResult.value.ok) throw new Error("Could not read your uploads. Please try Refresh.");
      const next = await dashboardResult.value.json(); if (request !== requests.current) return null;
      setData(next); if (serviceResult.status === "fulfilled" && serviceResult.value.ok) setService(await serviceResult.value.json());
      let stored = null; try { stored = JSON.parse(window.localStorage.getItem(selectionStorageKey)); } catch { /* Storage may be blocked. */ }
      const completedPending = next.runs.find((item) => item.id === pendingRun.current);
      if (completedPending && next.workspaceId) {
        pendingRun.current = null;
        setSelection({ workspaceId: next.workspaceId, runId: completedPending.id, recordingKey: recordingKey(completedPending) });
      } else setSelection((current) => pendingRun.current ? null : chooseWorkspaceSelection(next.runs, current, stored, next.workspaceId));
      setError(next.workspaceError ?? ""); return next;
    } catch (problem) { if (request === requests.current) setError(problem.message); return null; }
    finally { if (request === requests.current) setLoading(false); }
  }, []);
  useEffect(() => { loadDashboard(); }, [loadDashboard]);
  useEffect(() => { if (loading) return; try { if (selection) window.localStorage.setItem(selectionStorageKey, JSON.stringify(selection)); else window.localStorage.removeItem(selectionStorageKey); } catch { /* Selection works without storage. */ } }, [selection, loading]);
  const recordings = useMemo(() => recordingsFromRuns(data.runs), [data.runs]);
  const run = useMemo(() => selectedRun(data.runs, selection), [data.runs, selection]);
  const goToSection = useCallback((section) => { setActiveSection(section); setMobileOpen(false); requestAnimationFrame(() => document.getElementById("footage-workspace")?.scrollIntoView({ block: "start" })); }, []);
  const selectUploadedRun = useCallback(async (runId) => {
    const next = await loadDashboard(); const uploaded = next?.runs.find((item) => item.id === runId);
    if (uploaded && next.workspaceId) { pendingRun.current = null; setSelection({ workspaceId: next.workspaceId, runId, recordingKey: recordingKey(uploaded) }); goToSection("video-runs"); }
    else setError("Processing finished, but this upload is unavailable in the current workspace. Refresh to retry.");
  }, [loadDashboard, goToSection]);
  const uploadStarted = useCallback((runId) => { pendingRun.current = runId; setSelection(null); goToSection("upload"); }, [goToSection]);
  const uploadFailed = useCallback(() => { pendingRun.current = null; }, []);
  const uploaderVisible = activeSection === "upload" || (activeSection === "overview" && !run);
  return <main className="shell focused-shell fresh-shell">
    <aside className={mobileOpen ? "sidebar sidebar-open" : "sidebar"}><div className="sidebar-brand-block"><div className="brand"><span className="brand-mark"><Eye size={20} /></span><span>Three Eyed Raven</span></div><p className="sidebar-intro">Your footage. Clearer crowd insights.</p></div>
      <div className="side-label">Workspace</div><nav aria-label="Workspace navigation">{navItems.map(([id, Icon, label]) => <button key={id} aria-current={activeSection === id ? "page" : undefined} onClick={() => goToSection(id)} className={activeSection === id ? "nav-item nav-item-active" : "nav-item"}><Icon size={18} /><span>{label}</span></button>)}</nav>
      <details className="advanced-nav"><summary>Research & settings</summary><nav>{advancedNav.map(([id, Icon, label]) => <button key={id} aria-current={activeSection === id ? "page" : undefined} onClick={() => goToSection(id)} className={activeSection === id ? "nav-item nav-item-active" : "nav-item"}><Icon size={18} /><span>{label}</span></button>)}</nav></details>
      <div className="sidebar-foot card"><span className="eyebrow">WORKSPACE</span><strong>Your uploads only</strong><p>Old demo outputs are hidden. New results come from the clips you choose.</p></div>
    </aside>{mobileOpen && <button className="navigation-backdrop" aria-label="Close navigation" onClick={() => setMobileOpen(false)} />}
    <section className="app-area"><header className="topbar"><button className="icon-button mobile-menu" aria-label="Open menu" aria-expanded={mobileOpen} onClick={() => setMobileOpen((value) => !value)}><Menu size={20} /></button><div className="project-switch"><span className={service.available ? "status-dot status-dot-online" : "status-dot"} /><span>Crowd Intelligence</span></div><div className="top-actions"><span className="subtle-pill">Local workspace</span><button className="icon-button" aria-label="Show crowd alerts" onClick={() => goToSection("alerts")}><Bell size={19} /></button></div></header>
      <div id="footage-workspace" className="content focused-content" data-testid={`workspace-view-${activeSection}`}><div className="page-heading"><div><span className="eyebrow">YOUR UPLOAD WORKSPACE</span><h1>{modeLabel(activeSection)}</h1></div><button className="secondary-button" onClick={loadDashboard} disabled={loading}><RefreshCw size={16} className={loading ? "spin" : ""} />Refresh</button></div>
        {error && <p className="upload-message upload-message-error" role="alert">{error}</p>}
        {run && activeSection !== "upload" && <div className="selected-footage-bar"><div><Video size={18} /><span><small>Results for these uploaded clips</small><strong>{Object.values(run.sourceNames ?? {}).join(" + ") || footageLabel(run)}</strong></span></div>{recordings.length > 1 && <label className="fresh-run-picker"><span>Selected upload</span><select aria-label="Selected upload" value={run.id} onChange={(event) => { const chosen = data.runs.find((item) => item.id === event.target.value); if (chosen) setSelection({ workspaceId: data.workspaceId, runId: chosen.id, recordingKey: recordingKey(chosen) }); }}>{recordings.map((item) => <option key={item.run.id} value={item.run.id}>{footageLabel(item.run)}</option>)}</select></label>}</div>}
        <div hidden={!uploaderVisible} className="workspace-uploader"><UploadRunPanel onStarted={uploadStarted} onCompleted={selectUploadedRun} onFailed={uploadFailed} /></div>
        {activeSection === "overview" && run && <Overview run={run} onPlayback={() => goToSection("video-runs")} onUpload={() => goToSection("upload")} />}
        <div hidden={activeSection !== "video-runs"} className="workspace-player">{run ? <section className="card focused-video-card"><IndependentFootagePlayer key={run.id} run={run} active={activeSection === "video-runs"} /></section> : <NoAnalysis title={pendingRun.current ? "Your new videos are still processing" : "No videos loaded"} onUpload={() => goToSection("upload")} />}</div>
        {!['overview', 'upload', 'video-runs'].includes(activeSection) && <WorkspacePanel mode={activeSection} run={run} data={data} onUpload={() => goToSection("upload")} />}
        {run?.countWarning && activeSection !== "upload" && <p role="alert" className="upload-message upload-message-error">{run.countWarning}</p>}
      </div>
    </section>
  </main>;
}
