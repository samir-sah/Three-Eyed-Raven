import { createReadStream, existsSync, statSync } from "node:fs";
import { Readable } from "node:stream";

export function parseByteRange(header, size) {
  if (!header) return null;
  const match = /^bytes=(\d*)-(\d*)$/.exec(header);
  if (!match || !size || (!match[1] && !match[2])) return false;
  let start, end;
  if (!match[1]) {
    const suffix = Number(match[2]);
    if (!Number.isSafeInteger(suffix) || suffix <= 0) return false;
    start = Math.max(0, size - suffix);
    end = size - 1;
  } else {
    start = Number(match[1]);
    end = match[2] ? Number(match[2]) : size - 1;
    if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start >= size || end < start) return false;
    end = Math.min(end, size - 1);
  }
  return { start, end };
}

export function videoResponse(request, videoPath, head = false) {
  if (!existsSync(videoPath) || !statSync(videoPath).isFile()) return new Response("Not found", { status: 404 });
  const size = statSync(videoPath).size;
  const range = parseByteRange(request.headers.get("range"), size);
  const headers = { "Accept-Ranges": "bytes", "Content-Type": "video/mp4", "Cache-Control": "private, max-age=3600" };
  if (range === false) return new Response(null, { status: 416, headers: { ...headers, "Content-Range": `bytes */${size}` } });
  if (range) {
    headers["Content-Range"] = `bytes ${range.start}-${range.end}/${size}`;
    headers["Content-Length"] = String(range.end - range.start + 1);
  } else headers["Content-Length"] = String(size);
  return new Response(head ? null : Readable.toWeb(createReadStream(videoPath, range ?? undefined)), { status: range ? 206 : 200, headers });
}
