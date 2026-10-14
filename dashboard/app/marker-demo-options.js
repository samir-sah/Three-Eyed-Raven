"use client";

import { useEffect, useId, useRef, useState } from "react";
import "./marker-demo.css";

const CODE = /^V(?:0[1-9]|[1-9][0-9])$/;
export const DEFAULT_MARKER_DEMO = {
  enabled: false,
  confirmed_volunteer_demo: false,
  allowed_codes: ["V01", "V02", "V03"],
  min_reads: 3,
};

export function parseMarkerCodes(input) {
  if (typeof input !== "string" || !input.trim()) {
    return { codes: [], error: "Enter at least one anonymous code, for example V01." };
  }
  const codes = input.split(",").map((code) => code.trim().toUpperCase());
  if (codes.length > 99 || codes.some((code) => !CODE.test(code))) {
    return { codes: [], error: "Use comma-separated codes V01 to V99 only. Do not enter names." };
  }
  if (new Set(codes).size !== codes.length) {
    return { codes: [], error: "Each code must appear once in this list." };
  }
  return { codes, error: null };
}

export function markerDemoValidation(value) {
  if (!value?.enabled) return null;
  if (value.confirmed_volunteer_demo !== true) return "Confirm that all participants volunteered to this visible-tag demo.";
  if (!Array.isArray(value.allowed_codes) || !value.allowed_codes.length || value.allowed_codes.length > 99
    || value.allowed_codes.some((code) => typeof code !== "string" || !CODE.test(code))
    || new Set(value.allowed_codes).size !== value.allowed_codes.length) {
    return "Enter unique anonymous codes V01 to V99.";
  }
  if (!Number.isInteger(value.min_reads) || value.min_reads < 3 || value.min_reads > 20000) return "The readable-frame threshold must be between 3 and 20,000 per view.";
  return null;
}

export default function MarkerDemoOptions({ value = DEFAULT_MARKER_DEMO, onChange, disabled = false }) {
  const config = { ...DEFAULT_MARKER_DEMO, ...value };
  const codeKey = Array.isArray(config.allowed_codes) ? config.allowed_codes.join(", ") : "";
  const [input, setInput] = useState(codeKey);
  const expectedCodeKey = useRef(codeKey);
  const id = useId();
  useEffect(() => {
    if (codeKey !== expectedCodeKey.current) setInput(codeKey);
    expectedCodeKey.current = codeKey;
  }, [codeKey]);
  const parsed = parseMarkerCodes(input);
  const update = (changes) => onChange?.({
    enabled: Boolean(config.enabled),
    confirmed_volunteer_demo: Boolean(config.confirmed_volunteer_demo),
    allowed_codes: Array.isArray(config.allowed_codes) ? config.allowed_codes : [],
    min_reads: config.min_reads,
    ...changes,
  });

  return <fieldset className="marker-demo-options" disabled={disabled}>
    <legend>Optional controlled demonstration</legend>
    <label className="marker-demo-check">
      <input type="checkbox" checked={Boolean(config.enabled)} onChange={(event) => update({
        enabled: event.target.checked,
        confirmed_volunteer_demo: false,
      })} />
      <span><strong>Read visible volunteer QR tags in both clips</strong><small>Off by default. This compares printed codes, not people’s appearance or identity.</small></span>
    </label>
    {config.enabled && <div className="marker-demo-setup">
      <label className="marker-demo-check marker-demo-consent">
        <input type="checkbox" checked={Boolean(config.confirmed_volunteer_demo)} onChange={(event) => update({ confirmed_volunteer_demo: event.target.checked })} />
        <span>All participants volunteered to this visible-tag demo.</span>
      </label>
      <div className="marker-demo-code-entry">
        <label htmlFor={`${id}-codes`}>Anonymous codes to read</label>
        <input id={`${id}-codes`} type="text" value={input} maxLength={600} spellCheck={false}
          autoComplete="off" placeholder="V01, V02, V03" aria-invalid={Boolean(parsed.error)}
          aria-describedby={`${id}-help${parsed.error ? ` ${id}-error` : ""}`}
          onChange={(event) => {
            setInput(event.target.value);
            const result = parseMarkerCodes(event.target.value);
            expectedCodeKey.current = result.codes.join(", ");
            update({ allowed_codes: result.codes });
          }} />
        <small id={`${id}-help`}>V01–V99 only. A V07 tag must encode <code>TER:V07</code>. Never encode a real name.</small>
        {parsed.error && <p id={`${id}-error`} className="marker-demo-error" role="alert">{parsed.error}</p>}
      </div>
      <p className="marker-demo-help">Print one distinct code per volunteer, keep it large and flat, and make it visible in both clips. A code counts only after at least {Number.isInteger(config.min_reads) && config.min_reads >= 3 ? config.min_reads : 3} readable frames in each view. The clips do not need to show the code at the same time.</p>
      {!parsed.error && <a className="marker-demo-print-link" href={`/markers?codes=${encodeURIComponent(parsed.codes.join(","))}`} target="_blank" rel="noopener noreferrer">Open printable tag sheet ↗</a>}
      {!config.confirmed_volunteer_demo && <p className="marker-demo-help">Volunteer confirmation is required before starting a tag-enabled analysis.</p>}
    </div>}
  </fieldset>;
}
