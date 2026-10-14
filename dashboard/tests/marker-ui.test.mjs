import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import Module, { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";

// Use the JSX transformer already bundled with Next; no test-only dependency,
// build, generated file or application server is needed for these UI checks.
const require = createRequire(import.meta.url);
const babel = require("next/dist/compiled/babel/core");
const React = require("react");
const { renderToStaticMarkup } = require("react-dom/server");
const appRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../app");

function loadClientComponent(filename) {
  const absolutePath = path.join(appRoot, filename);
  const transformed = babel.transformSync(readFileSync(absolutePath, "utf8"), {
    filename: absolutePath,
    presets: [[require.resolve("next/dist/compiled/babel/preset-react"), { runtime: "automatic" }]],
    plugins: [require.resolve("next/dist/compiled/babel/plugin-transform-modules-commonjs")],
    babelrc: false,
    configFile: false,
  });
  const loaded = new Module(absolutePath);
  loaded.filename = absolutePath;
  loaded.paths = Module._nodeModulePaths(path.dirname(absolutePath));
  const previousCssLoader = Module._extensions[".css"];
  Module._extensions[".css"] = (stylesheet) => { stylesheet.exports = {}; };
  try {
    loaded._compile(transformed.code, absolutePath);
  } finally {
    if (previousCssLoader) Module._extensions[".css"] = previousCssLoader;
    else delete Module._extensions[".css"];
  }
  return loaded.exports;
}

const options = loadClientComponent("marker-demo-options.js");
const MarkerReview = loadClientComponent("marker-review.js").default;
const renderOptions = (value = options.DEFAULT_MARKER_DEMO, extra = {}) => renderToStaticMarkup(React.createElement(options.default, { value, ...extra }));
const renderReview = (report, extra = {}) => renderToStaticMarkup(React.createElement(MarkerReview, { report, ...extra }));
const confirmedDemo = (changes = {}) => ({
  ...options.DEFAULT_MARKER_DEMO,
  allowed_codes: [...options.DEFAULT_MARKER_DEMO.allowed_codes],
  enabled: true,
  confirmed_volunteer_demo: true,
  ...changes,
});
const RUN = "upload_20261014010101_abcdef";
const tagUrl = (camera = "aerial", code = "V01") => `/api/marker-evidence/${RUN}/${camera}/${code}`;

function readyReport(changes = {}) {
  return {
    status: "ready",
    readerVerified: true,
    method: "explicit_qr_codes",
    scope: "controlled_marker_demo",
    minReads: 3,
    sharedCodes: ["V01"],
    sharedCount: 1,
    cameras: [
      { id: "aerial", label: "Aerial view", eligibleCodes: ["V01"], eligibleCount: 1 },
      { id: "ground", label: "Ground view", eligibleCodes: ["V01"], eligibleCount: 1 },
    ],
    evidence: [{ code: "V01", views: [
      { cameraId: "aerial", firstFrame: 0, lastFrame: 9, firstSeconds: 0, lastSeconds: 0.3, readFrames: 3, evidenceUrl: tagUrl("aerial") },
      { cameraId: "ground", firstFrame: 3, lastFrame: 12, firstSeconds: 0.1, lastSeconds: 0.4, readFrames: 4, evidenceUrl: tagUrl("ground") },
    ] }],
    ...changes,
  };
}

test("marker controls default to off and never imply volunteer consent", () => {
  assert.equal(options.DEFAULT_MARKER_DEMO.enabled, false);
  assert.equal(options.DEFAULT_MARKER_DEMO.confirmed_volunteer_demo, false);
  assert.equal(options.markerDemoValidation(options.DEFAULT_MARKER_DEMO), null);
  const html = renderOptions();
  assert.equal((html.match(/type="checkbox"/g) ?? []).length, 1);
  assert.doesNotMatch(html, /checked=""/);
  assert.doesNotMatch(html, /marker-demo-setup/);
  assert.match(html, /Off by default/);
});

test("enabled controls require an explicit boolean volunteer confirmation", () => {
  for (const value of [false, undefined, null, "true", 1]) {
    assert.ok(options.markerDemoValidation(confirmedDemo({ confirmed_volunteer_demo: value })));
  }
  assert.equal(options.markerDemoValidation(confirmedDemo()), null);
  const html = renderOptions(confirmedDemo({ confirmed_volunteer_demo: false }));
  assert.match(html, /All participants volunteered/);
  assert.match(html, /Volunteer confirmation is required/);
  assert.equal((html.match(/checked=""/g) ?? []).length, 1, "Only the opt-in checkbox is checked");
  assert.match(html, /TER:V07/);
  assert.match(html, /href="\/markers\?codes=V01%2CV02%2CV03"/);
});

test("busy controls are disabled without creating consent", () => {
  const html = renderOptions(confirmedDemo({ confirmed_volunteer_demo: false }), { disabled: true });
  assert.match(html, /<fieldset[^>]*disabled=""/);
  assert.equal((html.match(/checked=""/g) ?? []).length, 1);
});

test("anonymous code parser trims input and normalizes case without accepting names", () => {
  assert.deepEqual(options.parseMarkerCodes(" v01, V07 ,v99 "), { codes: ["V01", "V07", "V99"], error: null });
  const allCodes = Array.from({ length: 99 }, (_, index) => `V${String(index + 1).padStart(2, "0")}`);
  assert.deepEqual(options.parseMarkerCodes(allCodes.join(",")).codes, allCodes);
  for (const input of ["", " ", "V00", "V100", "V1", "Alice", "V01,Alice", "TER:V01", "V01,V01", "v01,V01", "V01,", "V01,,V02", null, 7]) {
    const parsed = options.parseMarkerCodes(input);
    assert.ok(parsed.error, `Expected rejection of ${String(input)}`);
    assert.deepEqual(parsed.codes, []);
  }
});

test("enabled configuration rejects invalid, duplicated or missing registered codes", () => {
  for (const allowed_codes of [[], ["V00"], ["V100"], ["V01", "V01"], ["Alice"], ["v01"], null, "V01"]) {
    assert.ok(options.markerDemoValidation(confirmedDemo({ allowed_codes })));
  }
  assert.equal(options.markerDemoValidation(confirmedDemo({ allowed_codes: ["V01", "V99"] })), null);
});

test("minimum readable frames accepts integer 3 through 20000 only", () => {
  for (const min_reads of [3, 4, 20000]) {
    assert.equal(options.markerDemoValidation(confirmedDemo({ min_reads })), null);
  }
  for (const min_reads of [0, 1, 2, 20001, -1, 3.5, "3", null, undefined, Number.NaN, Number.POSITIVE_INFINITY]) {
    assert.ok(options.markerDemoValidation(confirmedDemo({ min_reads })), `Expected rejection of ${String(min_reads)}`);
  }
});

test("mode-off review asks for opt-in on a new upload instead of reporting zero", () => {
  const html = renderReview(null);
  assert.match(html, /Tag comparison is off/);
  assert.match(html, /new upload/);
  assert.doesNotMatch(html, /marker-demo-kpis/);
});

test("unavailable review displays its reason and never presents a measured zero", () => {
  const html = renderReview({ status: "unavailable", reason: "Camera bounds differ from saved scan.", warning: "Saved QR report cannot be trusted." });
  assert.match(html, /Tag comparison unavailable/);
  assert.match(html, /Camera bounds differ from saved scan/);
  assert.match(html, /Saved QR report cannot be trusted/);
  assert.match(html, /Unavailable is not a count of zero/);
  assert.doesNotMatch(html, /marker-demo-kpis/);
});

test("unverified, wrong-method or wrong-scope ready claims cannot show a shared-code KPI", () => {
  for (const changes of [
    { readerVerified: false }, { readerVerified: undefined }, { readerVerified: "true" },
    { method: "appearance_similarity" }, { method: undefined },
    { scope: "person_identification" }, { scope: undefined },
  ]) {
    const html = renderReview(readyReport(changes));
    assert.match(html, /Tag comparison unavailable/);
    assert.doesNotMatch(html, /marker-demo-kpis/);
  }
});

test("ready view derives distinct valid shared codes instead of trusting a supplied total", () => {
  const html = renderReview(readyReport({ sharedCodes: ["V07", "V01", "V01", "V00", "Alice"], sharedCount: 999 }));
  assert.match(html, /Shared marker codes<\/span><strong>2<\/strong>/);
  assert.match(html, /V01, V07/);
  assert.doesNotMatch(html, /<strong>999<\/strong>/);
  assert.match(html, /not a count of verified people or an identity-accuracy score/);
});

test("a valid zero result states the limited conclusion rather than participant absence", () => {
  const html = renderReview(readyReport({ sharedCodes: [], sharedCount: 7, evidence: [] }));
  assert.match(html, /Shared marker codes<\/span><strong>0<\/strong>/);
  assert.match(html, /does not prove that no participant appeared in both/);
});

test("coverage and ambiguity warnings are visible outside collapsed limitations", () => {
  const html = renderReview(readyReport({
    coverageWarning: "Some QR scans failed in the ground clip.",
    ambiguityWarning: "Duplicate V03 tags excluded from shared evidence.",
  }));
  const limitationsIndex = html.indexOf("marker-demo-limitations");
  for (const text of ["Some QR scans failed in the ground clip.", "Duplicate V03 tags excluded from shared evidence."]) {
    const index = html.indexOf(text);
    assert.ok(index >= 0 && index < limitationsIndex, `${text} must be visible before the collapsed limitations`);
  }
  assert.match(html, /role="note"/);
  assert.match(html, /aria-label="Marker interpretation warnings"/);
  assert.doesNotMatch(renderReview(readyReport()), /marker-demo-warnings/);
});

test("valid evidence shows tag-only crops, source frames and each clip's relative times", () => {
  const html = renderReview(readyReport());
  assert.match(html, new RegExp(`src="${tagUrl("aerial")}"`));
  assert.match(html, /printed QR-tag crop from Aerial view/);
  assert.match(html, /0:00\.0–0:00\.3/);
  assert.match(html, /3 readable frames/);
  assert.match(html, /Source frames 0–9/);
  assert.match(html, /each clip’s own clock/);
  assert.doesNotMatch(html, /Jump to first tag read/);
});

test("evidence rejects foreign API routes, remote URLs and unsafe marker paths", () => {
  for (const evidenceUrl of [
    "https://example.com/tag.jpg", "//example.com/tag.jpg", "javascript:alert(1)",
    "/api/videos/recording/aerial", "/api/artifacts/recording/annotated.mp4", "/api/health",
    `/api/marker-evidence/${RUN}/aerial/V00`, `/api/marker-evidence/${RUN}/aerial/V100`,
    `/api/marker-evidence/${RUN}/aerial/V01?other=1`, `/api/marker-evidence/${RUN}/aerial/V01/extra`,
    `/api/marker-evidence/../${RUN}/aerial/V01`, `/api/marker-evidence/${RUN}/other/V01`,
    `/api/marker-evidence/${RUN}/aerial\\V01`, `/api/marker-evidence/${RUN}/aerial/V01\n`,
  ]) {
    const report = readyReport({ evidence: [{ code: "V01", views: [{ cameraId: "aerial", firstFrame: 0, lastFrame: 9, firstSeconds: 0, lastSeconds: 0.3, readFrames: 3, evidenceUrl }] }] });
    const html = renderReview(report);
    assert.match(html, /Tag crop unavailable/, `Must reject ${evidenceUrl}`);
    assert.doesNotMatch(html, /<img\b/, `Must not request ${evidenceUrl}`);
  }
});

test("warnings and camera labels remain escaped text, not executable markup", () => {
  const html = renderReview(readyReport({
    cameras: [{ id: "aerial", label: "<script>camera</script>", eligibleCodes: ["V01"] }],
    coverageWarning: "<img src=x onerror=alert(1)>",
  }));
  assert.match(html, /&lt;script&gt;camera&lt;\/script&gt;/);
  assert.match(html, /&lt;img src=x onerror=alert\(1\)&gt;/);
  assert.doesNotMatch(html, /<script>/);
});
