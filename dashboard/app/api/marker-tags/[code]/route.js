import { readFileSync } from "node:fs";
import { isMarkerCode, markerTagPath } from "../../../../lib/marker-report.mjs";

export const runtime = "nodejs";
export async function GET(request, { params }) {
  const { code } = await params;
  if (!isMarkerCode(code)) return new Response("Unknown anonymous code", { status: 404 });
  try {
    const data = readFileSync(markerTagPath(process.cwd(), code));
    const download = new URL(request.url).searchParams.get("download") === "1";
    return new Response(data, { headers: { "Content-Type": "image/png", "X-Content-Type-Options": "nosniff",
      "Cache-Control": "public, max-age=86400", ...(download ? { "Content-Disposition": `attachment; filename="${code}.png"` } : {}) } });
  } catch { return new Response("Printable tag unavailable", { status: 404 }); }
}
