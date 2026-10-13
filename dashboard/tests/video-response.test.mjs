import test from "node:test";
import assert from "node:assert/strict";
import { parseByteRange } from "../lib/video-response.mjs";

test("regular, open-ended and suffix ranges", () => {
  assert.equal(parseByteRange(null, 100), null);
  assert.deepEqual(parseByteRange("bytes=0-9", 100), { start: 0, end: 9 });
  assert.deepEqual(parseByteRange("bytes=50-", 100), { start: 50, end: 99 });
  assert.deepEqual(parseByteRange("bytes=-10", 100), { start: 90, end: 99 });
  assert.deepEqual(parseByteRange("bytes=0-999", 100), { start: 0, end: 99 });
  assert.deepEqual(parseByteRange("bytes=-999", 100), { start: 0, end: 99 });
});
test("malformed, unsatisfiable and multipart ranges rejected", () => {
  for (const value of ["bytes=100-", "bytes=90-80", "bytes=-0", "bytes=-", "bytes=-1-2", "bytes=0-9,20-30", "garbage=1-3", "bytes=1x-3", "bytes=99999999999999999999-"]) assert.equal(parseByteRange(value, 100), false);
  assert.equal(parseByteRange("bytes=0-", 0), false);
});
