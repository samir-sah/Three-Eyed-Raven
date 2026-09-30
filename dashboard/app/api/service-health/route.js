export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const SERVICE_URL = process.env.TRACKER_API_URL ?? "http://127.0.0.1:8000";

export async function GET() {
  try {
    const response = await fetch(`${SERVICE_URL}/health`, {
      cache: "no-store",
      signal: AbortSignal.timeout(2000),
    });
    if (!response.ok) throw new Error(`Service returned ${response.status}`);
    const health = await response.json();
    return Response.json({ available: health.status === "ok", runCount: health.run_count ?? 0 });
  } catch {
    return Response.json({ available: false, runCount: 0 });
  }
}
