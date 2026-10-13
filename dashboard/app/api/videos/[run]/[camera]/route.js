import path from "node:path";
import { videoResponse } from "../../../../../lib/video-response.mjs";

export const runtime = "nodejs";
const root = path.resolve(process.cwd(), "..", "artifacts");
async function serve(request, { params }, head) {
  const { run, camera } = await params;
  if (!/^[a-zA-Z0-9_-]+$/.test(run) || !["aerial", "ground"].includes(camera)) return new Response("Not found", { status: 404 });
  return videoResponse(request, path.join(root, run, camera, "browser.mp4"), head);
}
export const GET = (request, context) => serve(request, context, false);
export const HEAD = (request, context) => serve(request, context, true);
