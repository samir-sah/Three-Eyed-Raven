/** Small media actions that operate on exactly one explicitly selected video. */
export function finiteDuration(value, fallback = 0) {
  if (Number.isFinite(value) && value > 0) return value;
  return Number.isFinite(fallback) && fallback > 0 ? fallback : 0;
}

export function formatClipTime(seconds) {
  const total = Math.max(0, Math.floor(Number.isFinite(seconds) ? seconds : 0));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor(total / 60) % 60;
  const remainder = total % 60;
  return `${hours ? `${hours.toString().padStart(2, "0")}:` : ""}${minutes.toString().padStart(2, "0")}:${remainder.toString().padStart(2, "0")}`;
}

export function clipEnded(video) {
  const duration = finiteDuration(video?.duration);
  return Boolean(video?.ended || (duration && video.currentTime >= duration));
}

export function seekClip(video, seconds) {
  const duration = finiteDuration(video?.duration);
  if (!video || !duration) throw new Error("Wait for this video's metadata before seeking.");
  const position = Math.min(duration, Math.max(0, Number.isFinite(seconds) ? seconds : 0));
  if (position >= duration) video.pause();
  video.currentTime = position;
  return position;
}

/** Only a user-initiated replay may reset an ended clip to zero. */
export async function playClip(video, { replay = false } = {}) {
  if (!video) return false;
  video.loop = false;
  video.autoplay = false;
  if (replay) video.currentTime = 0;
  else if (clipEnded(video)) return false;
  await video.play();
  return !video.paused;
}
