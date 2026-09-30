import { readUploadJob } from "../../../../lib/upload-jobs";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function GET(_request, { params }) {
  const { runId } = await params;
  const job = await readUploadJob(runId);
  if (!job) return Response.json({ error: "Upload run not found." }, { status: 404 });
  return Response.json(job);
}
