import test from "node:test";
import assert from "node:assert/strict";
import { GET as printableTag } from "../app/api/marker-tags/[code]/route.js";
import { GET as markerEvidence } from "../app/api/marker-evidence/[run]/[camera]/[code]/route.js";

const RUN = "upload_20261013180000_123abc";
const request = new Request("http://localhost/api/marker-test");

test("printable-tag API rejects unsafe/non-string code parameters without filesystem access", async () => {
  for (const code of ["V01\n", "../V01", "V00", "V100", null, { toString: () => "V01" }]) {
    assert.equal((await printableTag(request, { params: Promise.resolve({ code }) })).status, 404);
  }
});

test("evidence API rejects traversal, newline and invalid camera/code parameters", async () => {
  for (const change of [{ run: `../${RUN}` }, { run: `${RUN}\n` }, { run: null }, { camera: "../aerial" },
    { camera: "third" }, { code: "V01\n" }, { code: "../../secret" }, { code: null }]) {
    const params = { run: RUN, camera: "aerial", code: "V01", ...change };
    assert.equal((await markerEvidence(request, { params: Promise.resolve(params) })).status, 404);
  }
});
