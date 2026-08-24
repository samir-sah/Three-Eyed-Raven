import { createReadStream, existsSync } from "node:fs";
import { Readable } from "node:stream";
import path from "node:path";

export const runtime = "nodejs";

const ARTIFACTS_ROOT = path.resolve(process.cwd(), "..", "artifacts");
const allowedRun = /^[a-zA-Z0-9_-]+$/;
const allowedFrame = /^frame_\d+\.jpg$/;

export async function GET(_request, { params }) {
  const { run, frame } = await params;
  if (!allowedRun.test(run) || !allowedFrame.test(frame)) {
    return new Response("Not found", { status: 404 });
  }
  const framePath = path.join(ARTIFACTS_ROOT, run, "preview_frames", frame);
  if (!existsSync(framePath)) return new Response("Not found", { status: 404 });
  return new Response(Readable.toWeb(createReadStream(framePath)), {
    headers: { "Content-Type": "image/jpeg", "Cache-Control": "public, max-age=3600" },
  });
}
