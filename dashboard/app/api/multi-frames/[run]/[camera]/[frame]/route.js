import { createReadStream, existsSync } from "node:fs";
import { Readable } from "node:stream";
import path from "node:path";

export const runtime = "nodejs";

const ARTIFACTS_ROOT = path.resolve(process.cwd(), "..", "artifacts");
const safePart = /^[a-zA-Z0-9_-]+$/;
const safeFrame = /^frame_\d+\.jpg$/;

export async function GET(_request, { params }) {
  const { run, camera, frame } = await params;
  if (!safePart.test(run) || !safePart.test(camera) || !safeFrame.test(frame)) {
    return new Response("Not found", { status: 404 });
  }
  const framePath = path.join(ARTIFACTS_ROOT, run, camera, "preview_frames", frame);
  if (!existsSync(framePath)) return new Response("Not found", { status: 404 });
  return new Response(Readable.toWeb(createReadStream(framePath)), {
    headers: { "Content-Type": "image/jpeg", "Cache-Control": "public, max-age=3600" },
  });
}
