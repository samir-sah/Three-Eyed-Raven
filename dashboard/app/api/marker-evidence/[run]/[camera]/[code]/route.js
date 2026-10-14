import { readFileSync } from "node:fs";
import path from "node:path";
import { readDashboardWorkspace, isCurrentWorkspaceUpload } from "../../../../../../lib/dashboard-workspace.mjs";
import { readUploadJob } from "../../../../../../lib/upload-jobs.js";
import { isMarkerCode, isMarkerUploadId, markerRunDirectory, markerSummaryPath, markerEvidencePath, readMarkerReport } from "../../../../../../lib/marker-report.mjs";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export async function GET(_request, { params }) {
  const { run, camera, code } = await params;
  if (!isMarkerUploadId(run) || !["aerial", "ground"].includes(camera) || !isMarkerCode(code)) return new Response("Not found", { status: 404 });
  try {
    const root = path.resolve(process.cwd(), "..");
    const directory = markerRunDirectory(root, run);
    const summary = JSON.parse(readFileSync(markerSummaryPath(directory), "utf8"));
    const job = await readUploadJob(run);
    if (!isCurrentWorkspaceUpload(run, summary, job, readDashboardWorkspace(root), root)) throw new Error("Not current");
    const report = readMarkerReport(directory, summary);
    if (report?.status !== "ready" || !report.cameras.find((view) => view.id === camera)?.observations.some((row) => row.code === code && row.evidenceUrl)) throw new Error("No tag evidence");
    const filename = markerEvidencePath(directory, camera, code);
    return new Response(readFileSync(filename), { headers: { "Content-Type": "image/jpeg", "Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff" } });
  } catch { return new Response("Not found", { status: 404 }); }
}
