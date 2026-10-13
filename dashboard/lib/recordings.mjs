const normaliseSource = (source) => String(source).replaceAll("\\", "/").toLowerCase();

export function recordingKey(run) {
  const sources = run.isMulti ? (run.streams ?? []).map((stream) => stream.source) : [run.metrics?.source];
  if (!sources.length || sources.some((source) => source == null || source === "—")) return `run:${run.id}`;
  return JSON.stringify(sources.map(normaliseSource));
}

export function recordingsFromRuns(runs) {
  const groups = new Map();
  for (const run of runs.filter((item) => item.hasVideo)) {
    const key = recordingKey(run);
    const group = groups.get(key) ?? { key, runs: [], isMulti: Boolean(run.isMulti), uploaded: false };
    group.runs.push(run);
    group.uploaded ||= run.id.startsWith("upload_");
    groups.set(key, group);
  }
  return [...groups.values()].map((group) => {
    group.runs.sort((a, b) => new Date(b.updatedAt) - new Date(a.updatedAt));
    const newest = group.runs[0];
    const date = new Date(newest.updatedAt).toLocaleString("en-IN", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
    return { ...group, run: newest, label: `${group.uploaded ? "Uploaded" : "Sample"} ${group.isMulti ? "video pair" : "video"} · ${date}` };
  }).sort((a, b) => Number(b.uploaded) - Number(a.uploaded) || new Date(b.run.updatedAt) - new Date(a.run.updatedAt));
}

export function chooseSelection(runs, current, stored) {
  // Preserve both the recording and its analysis output. Refreshing or adding a
  // newer experiment must not silently replace either of them.
  for (const selection of [current, stored]) {
    if (typeof selection?.runId === "string" && selection.runId && typeof selection?.recordingKey === "string" && selection.recordingKey) return selection;
  }
  const first = recordingsFromRuns(runs)[0];
  return first ? { runId: first.run.id, recordingKey: first.key } : null;
}

export function selectedRun(runs, selection) {
  return runs.find((run) => run.id === selection?.runId && run.hasVideo && recordingKey(run) === selection.recordingKey) ?? null;
}
