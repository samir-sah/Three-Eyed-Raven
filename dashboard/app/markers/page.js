import PrintButton from "./print-button";
import "./print.css";

export default async function MarkerSheet({ searchParams }) {
  const query = await searchParams;
  const raw = typeof query.codes === "string" ? query.codes.split(",") : ["V01", "V02", "V03"];
  const valid = raw.length <= 99 && raw.length > 0 && raw.every((code) => /^V(?:0[1-9]|[1-9][0-9])$/.test(code));
  const codes = valid ? [...new Set(raw)].sort() : [];
  return <main className="marker-print-page"><header className="marker-print-instructions"><a href="/">← Dashboard</a><span className="eyebrow">CONTROLLED VOLUNTEER DEMO</span><h1>Printable anonymous QR tags</h1><p>Assign one unique code per volunteer. Make each tag large, flat and clearly readable in both cameras. Do not duplicate, swap or encode names. These codes are labels, not biometric identity verification.</p><p>Print in black and white on a white background, without trimming the white quiet zone. Aerial distance, angle, blur and occlusion can prevent reading. First test one tag at your actual camera distance.</p>{codes.length ? <PrintButton /> : <p role="alert">Choose valid anonymous codes V01–V99.</p>}</header><div className="marker-print-grid">{codes.map((code) => <section className="marker-print-card" key={code}><img src={`/api/marker-tags/${code}`} alt={`${code}: QR tag encoding TER:${code}`} /><a className="marker-print-download" href={`/api/marker-tags/${code}?download=1`} download={`${code}.png`}>Download {code}</a><span className="marker-print-payload">TER:{code}</span></section>)}</div></main>;
}
