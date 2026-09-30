import { createReadStream, existsSync } from "node:fs";
import path from "node:path";
import { Readable } from "node:stream";

export const runtime = "nodejs";

const ARTIFACTS_ROOT = path.resolve(process.cwd(), "..", "artifacts");
const SAFE_RUN = /^[a-zA-Z0-9_-]+$/;
const SAFE_CROP = /^[a-zA-Z0-9_-]+_track_\d+\.jpg$/;

export async function GET(_request, { params }) {
  const { run, file } = await params;
  if (!SAFE_RUN.test(run) || !SAFE_CROP.test(file)) {
    return new Response("Not found", { status: 404 });
  }

  const cropPath = path.join(ARTIFACTS_ROOT, run, "reid_crops", file);
  if (!existsSync(cropPath)) return new Response("Not found", { status: 404 });

  return new Response(Readable.toWeb(createReadStream(cropPath)), {
    headers: { "Content-Type": "image/jpeg", "Cache-Control": "no-store" },
  });
}
