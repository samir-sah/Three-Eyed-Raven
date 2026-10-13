import test from "node:test";
import assert from "node:assert/strict";
import { clipEnded, finiteDuration, formatClipTime, playClip, seekClip } from "../lib/camera-playback.mjs";

function fakeVideo(overrides = {}) {
  return {
    duration: 20, currentTime: 4, paused: true, ended: false, loop: false, autoplay: false,
    plays: 0, pauses: 0,
    async play() { this.plays += 1; this.paused = false; },
    pause() { this.pauses += 1; this.paused = true; },
    ...overrides,
  };
}

test("normal play never changes position or starts another camera", async () => {
  const first = fakeVideo();
  const second = fakeVideo({ currentTime: 11 });
  assert.equal(await playClip(first), true);
  assert.equal(first.currentTime, 4);
  assert.equal(first.plays, 1);
  assert.equal(second.plays, 0);
  assert.equal(second.currentTime, 11);
});

test("ended clips never restart through an ordinary play action", async () => {
  for (const video of [fakeVideo({ ended: true, currentTime: 20 }), fakeVideo({ currentTime: 20 })]) {
    assert.equal(await playClip(video), false);
    assert.equal(video.plays, 0);
    assert.equal(video.currentTime, 20);
    assert.equal(clipEnded(video), true);
  }
});

test("explicit replay alone resets the selected video and plays it once", async () => {
  const video = fakeVideo({ currentTime: 20, ended: true, loop: true, autoplay: true });
  assert.equal(await playClip(video, { replay: true }), true);
  assert.equal(video.currentTime, 0);
  assert.equal(video.plays, 1);
  assert.equal(video.loop, false);
  assert.equal(video.autoplay, false);
});

test("seeking is per-video and does not start paused media", () => {
  const first = fakeVideo();
  const second = fakeVideo({ currentTime: 11 });
  assert.equal(seekClip(first, 8.5), 8.5);
  assert.equal(first.currentTime, 8.5);
  assert.equal(first.plays, 0);
  assert.equal(first.paused, true);
  assert.equal(second.currentTime, 11);
});

test("seeking to the end pauses without wrapping, seeking backward stays manual", () => {
  const video = fakeVideo({ paused: false });
  assert.equal(seekClip(video, 100), 20);
  assert.equal(video.pauses, 1);
  assert.equal(video.paused, true);
  assert.equal(clipEnded(video), true);
  assert.equal(seekClip(video, 5), 5);
  assert.equal(video.plays, 0);
  assert.equal(video.paused, true);
  assert.equal(clipEnded(video), false);
  assert.equal(seekClip(video, -2), 0);
});

test("seeking unavailable metadata fails clearly instead of changing another clip", () => {
  for (const duration of [0, Infinity, NaN, undefined]) {
    const video = fakeVideo({ duration });
    assert.throws(() => seekClip(video, 5), /metadata/);
    assert.equal(video.currentTime, 4);
    assert.equal(video.plays, 0);
  }
});

test("play errors propagate to the selected camera's error message", async () => {
  const video = fakeVideo({ async play() { throw new Error("network error"); } });
  await assert.rejects(playClip(video), /network error/);
  assert.equal(video.currentTime, 4);
});

test("duration and display helpers handle unknown, short and hour-long videos", () => {
  assert.equal(finiteDuration(Infinity, 12), 12);
  assert.equal(finiteDuration(NaN), 0);
  assert.equal(finiteDuration(-1, -2), 0);
  assert.equal(formatClipTime(0), "00:00");
  assert.equal(formatClipTime(62.9), "01:02");
  assert.equal(formatClipTime(3662), "01:01:02");
  assert.equal(formatClipTime(NaN), "00:00");
});
