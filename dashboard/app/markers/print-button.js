"use client";
export default function PrintButton() {
  return <button type="button" className="primary-button" onClick={() => window.print()}>Print selected tags</button>;
}
