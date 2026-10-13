import path from "node:path";
import { videoResponse } from "../../../../../lib/video-response.mjs";

export const runtime = "nodejs";

const ARTIFACTS_ROOT = path.resolve(process.cwd(), "..", "artifacts");
const allowedRun = /^[a-zA-Z0-9_-]+$/;

async function serve(request, { params }, head) {
  const { run, file } = await params;
  if (!allowedRun.test(run) || file !== "annotated.mp4") {
    return new Response("Not found", { status: 404 });
  }

  const videoPath = path.join(ARTIFACTS_ROOT, run, file);
  return videoResponse(request, videoPath, head);
}
export const GET = (request, context) => serve(request, context, false);
export const HEAD = (request, context) => serve(request, context, true);
