import { readTrainingExperiment } from "../../../lib/training-status";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export function GET() {
  try {
    return Response.json({ experiment: readTrainingExperiment() }, { headers: { "Cache-Control": "no-store" } });
  } catch {
    return Response.json({ error: "Training status could not be read. Retrying will not interrupt training." }, { status: 503 });
  }
}
