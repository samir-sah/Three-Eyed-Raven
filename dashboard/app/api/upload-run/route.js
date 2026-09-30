import { startUploadRun } from "../../../lib/upload-jobs";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function POST(request) {
  try {
    const job = await startUploadRun(await request.formData());
    return Response.json(job, { status: 202 });
  } catch (error) {
    return Response.json({ error: error instanceof Error ? error.message : "Could not start the upload run." }, { status: 400 });
  }
}
