"use client";

import { useEffect, useState } from "react";
import "./marker-demo.css";

const CODE = /^V(?:0[1-9]|[1-9][0-9])$/;
const codesOnly = (value) => Array.isArray(value) ? [...new Set(value.filter((code) => typeof code === "string" && CODE.test(code)))].sort() : [];
const integer = (value) => Number.isInteger(value) && value >= 0 ? value : null;
const notice = (value) => typeof value === "string" && value.trim() ? value : null;
const clipTime = (value) => {
  if (!Number.isFinite(value) || value < 0) return "Time unavailable";
  const minutes = Math.floor(value / 60);
  return `${minutes}:${(value % 60).toFixed(1).padStart(4, "0")}`;
};
const safeEvidenceUrl = (value) => typeof value === "string" && /^\/api\/marker-evidence\/upload_\d{14}_[0-9a-f]{6}\/(?:aerial|ground)\/V(?:0[1-9]|[1-9][0-9])$/.test(value) ? value : null;

function EvidenceItem({ item, labelFor, onSeek }) {
  const views = Array.isArray(item.views) ? item.views : [];
  return <details className="marker-demo-evidence-item">
    <summary><strong>{item.code}</strong><span>{views.map((view) => labelFor(view.cameraId)).join(" + ") || "View evidence unavailable"}</span><small>View tag evidence</small></summary>
    <div className="marker-demo-evidence-views">{views.map((view, index) => {
      const url = safeEvidenceUrl(view.evidenceUrl);
      const reads = Array.isArray(view.readFrames) ? view.readFrames.length : integer(view.readFrames);
      const canSeek = typeof onSeek === "function" && Number.isFinite(view.firstSeconds) && view.firstSeconds >= 0;
      return <section key={`${view.cameraId}-${index}`} className="marker-demo-evidence-view">
        <h4>{labelFor(view.cameraId)}</h4>
        {url ? <img src={url} alt={`${item.code} printed QR-tag crop from ${labelFor(view.cameraId)}`} loading="lazy" /> : <div className="marker-demo-evidence-missing">Tag crop unavailable</div>}
        <p><strong>{clipTime(view.firstSeconds)}–{clipTime(view.lastSeconds)}</strong><span>{reads === null ? "Read count unavailable" : `${reads} readable frames`}</span></p>
        <small>Source frames {integer(view.firstFrame) ?? "—"}–{integer(view.lastFrame) ?? "—"} · each clip’s own clock</small>
        {canSeek && <button type="button" className="marker-demo-seek" onClick={() => onSeek({ cameraId: view.cameraId, seconds: view.firstSeconds, frame: view.firstFrame })}>Jump to first tag read</button>}
      </section>;
    })}</div>
  </details>;
}

export default function MarkerReview({ report, streams = [], onSeek }) {
  const [filter, setFilter] = useState("");
  const cameras = Array.isArray(report?.cameras) ? report.cameras : [];
  const evidence = (Array.isArray(report?.evidence) ? report.evidence : []).filter((item) => CODE.test(String(item.code))).sort((a, b) => a.code.localeCompare(b.code));
  const evidenceCodesKey = [...new Set(evidence.map((item) => item.code))].join(",");
  useEffect(() => {
    if (filter && !evidenceCodesKey.split(",").includes(filter)) setFilter("");
  }, [evidenceCodesKey, filter]);
  const labelFor = (id) => cameras.find((camera) => camera.id === id)?.label
    ?? (Array.isArray(streams) ? streams.find((stream) => stream.id === id)?.label : null) ?? id ?? "Camera";
  if (!report) return <section className="card marker-demo-review">
    <span className="eyebrow">CONTROLLED VOLUNTEER DEMO</span><h2>Tag comparison is off for this analysis</h2>
    <p className="marker-demo-help">To compare anonymous printed codes, enable the volunteer QR-tag option on a new upload and confirm participation. Existing clips are not scanned automatically.</p>
    <p className="marker-demo-help">This demonstration checks whether the same visible tag was read in both views. It does not establish that two sightings are the same person.</p>
    <a className="marker-demo-print-link" href="/markers" target="_blank" rel="noopener noreferrer">Prepare printable QR tags ↗</a>
  </section>;
  if (report.status !== "ready" || report.readerVerified !== true || report.method !== "explicit_qr_codes" || report.scope !== "controlled_marker_demo") return <section className="card marker-demo-review">
    <span className="eyebrow">CONTROLLED VOLUNTEER DEMO</span><h2>Tag comparison unavailable</h2>
    <p className="marker-demo-help">{notice(report.reason) || notice(report.warning) || "The marker scan did not finish or its saved report is unavailable. No shared-code count can be concluded."}</p>
    {notice(report.reason) && notice(report.warning) && report.reason !== report.warning && <p className="marker-demo-help">{report.warning}</p>}
    <p className="marker-demo-help">Unavailable is not a count of zero. The detection/counting results remain separate.</p>
  </section>;

  const sharedCodes = codesOnly(report.sharedCodes);
  const sharedCount = Array.isArray(report.sharedCodes) ? sharedCodes.length : integer(report.sharedCount);
  const visible = filter ? evidence.filter((item) => item.code === filter) : evidence;
  const minReads = Number.isInteger(report.minReads) && report.minReads >= 3 ? report.minReads : 3;
  return <section className="card marker-demo-review">
    <div className="marker-demo-heading"><div><span className="eyebrow">VISIBLE TAGS · CONTROLLED DEMO</span><h2>Codes read in both views</h2></div><span className="subtle-pill">Printed markers only</span></div>
    <div className="marker-demo-kpis">
      <div className="marker-demo-kpi marker-demo-kpi-primary"><span>Shared marker codes</span><strong>{sharedCount ?? "—"}</strong><small>{sharedCodes.length ? sharedCodes.join(", ") : Array.isArray(report.sharedCodes) ? "No qualifying shared code recorded" : "Shared-code list unavailable"}</small></div>
      {cameras.map((camera) => {
        const count = Array.isArray(camera.eligibleCodes) ? codesOnly(camera.eligibleCodes).length : integer(camera.eligibleCount);
        return <div className="marker-demo-kpi" key={camera.id}><span>{labelFor(camera.id)}</span><strong>{count ?? "—"}</strong><small>Codes read in ≥{minReads} frames</small></div>;
      })}
    </div>
    <p className="marker-demo-conclusion">{sharedCount === null ? "No valid shared-code total is available." : sharedCount === 0 ? `No code met the ${minReads}-frame rule in both views. This does not prove that no participant appeared in both.` : `${sharedCount} distinct printed ${sharedCount === 1 ? "code was" : "codes were"} read in at least ${minReads} frames of each view.`} This is <strong>not a count of verified people or an identity-accuracy score</strong>.</p>
    {(notice(report.coverageWarning) || notice(report.ambiguityWarning)) && <div className="marker-demo-warnings" role="note" aria-label="Marker interpretation warnings">
      {notice(report.coverageWarning) && <p><strong>Incomplete scan coverage.</strong> {report.coverageWarning}</p>}
      {notice(report.ambiguityWarning) && <p><strong>Ambiguous tags.</strong> {report.ambiguityWarning}</p>}
    </div>}
    {evidence.length > 0 && <div className="marker-demo-evidence">
      <div className="marker-demo-evidence-heading"><h3>Tag-only evidence</h3><label><span>Code</span><select aria-label="Filter marker evidence by code" value={filter} onChange={(event) => setFilter(event.target.value)}><option value="">All recorded codes</option>{[...new Set(evidence.map((item) => item.code))].map((code) => <option key={code} value={code}>{code}</option>)}</select></label></div>
      {visible.slice(0, 10).map((item, index) => <EvidenceItem key={`${item.code}-${index}`} item={item} labelFor={labelFor} onSeek={onSeek} />)}
      {visible.length > 10 && <details className="marker-demo-more"><summary>Show remaining {visible.length - 10} recorded codes</summary>{visible.slice(10).map((item, index) => <EvidenceItem key={`${item.code}-${index}`} item={item} labelFor={labelFor} onSeek={onSeek} />)}</details>}
    </div>}
    <details className="marker-demo-limitations"><summary>What this demonstration can and cannot conclude</summary>
      <p>A shared code means the camera decoded that printed payload repeatedly. Identical or exchanged tags can produce the same code on different people. Small, blurred, occluded or turned-away tags may not be read.</p>
      <p>Times are relative to each uploaded clip, not a common wall clock. The minimum-read rule is an evidence threshold, not calibrated statistical confidence. Images show QR-tag crops only; no face or body matching is performed.</p>
      {report.warning && <p>{report.warning}</p>}
    </details>
  </section>;
}
