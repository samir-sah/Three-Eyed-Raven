"use client";

import { useEffect, useRef, useState } from "react";
import { Maximize2, Minimize2, Pause, Play, RotateCcw, Video } from "lucide-react";
import { finiteDuration, formatClipTime, playClip, seekClip } from "../lib/camera-playback.mjs";

function CameraPlayer({ stream, active }) {
  const videoRef = useRef(null);
  const cameraRef = useRef(null);
  const playRequest = useRef(0);
  const wantedPlayback = useRef(false);
  const [duration, setDuration] = useState(() => finiteDuration(stream.duration));
  const [time, setTime] = useState(0);
  const [ready, setReady] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [ended, setEnded] = useState(false);
  const [waiting, setWaiting] = useState(false);
  const [fullscreen, setFullscreen] = useState(false);
  const [error, setError] = useState("");
  const label = stream.label || stream.id || "Camera";

  useEffect(() => {
    if (!active) {
      playRequest.current += 1;
      wantedPlayback.current = false;
      videoRef.current?.pause();
      setPlaying(false);
      setWaiting(false);
    }
  }, [active]);

  useEffect(() => {
    const video = videoRef.current;
    return () => {
      playRequest.current += 1;
      wantedPlayback.current = false;
      video?.pause();
    };
  }, []);

  useEffect(() => {
    const updateFullscreen = () => setFullscreen(document.fullscreenElement === cameraRef.current);
    document.addEventListener("fullscreenchange", updateFullscreen);
    updateFullscreen();
    return () => document.removeEventListener("fullscreenchange", updateFullscreen);
  }, []);

  function metadata(event) {
    const video = event.currentTarget;
    setDuration(finiteDuration(video.duration, stream.duration));
    setTime(Number.isFinite(video.currentTime) ? video.currentTime : 0);
    setReady(video.readyState >= 1);
  }

  async function start(replay = false) {
    const video = videoRef.current;
    if (!active || !ready || !video || (ended && !replay)) return;
    const request = ++playRequest.current;
    wantedPlayback.current = true;
    setError("");
    if (replay) {
      setTime(0);
      setEnded(false);
    }
    try {
      await playClip(video, { replay });
      // Leaving Playback or pressing Pause cancels this specific play request.
      if (!wantedPlayback.current) video.pause();
    } catch (cause) {
      if (request !== playRequest.current || cause?.name === "AbortError") return;
      wantedPlayback.current = false;
      setPlaying(false);
      setWaiting(false);
      setError("Playback could not start. Press Play again or reload this clip.");
    }
  }

  function toggle() {
    const video = videoRef.current;
    if (!video || !active) return;
    if (!video.paused && !video.ended) {
      playRequest.current += 1;
      wantedPlayback.current = false;
      video.pause();
      setPlaying(false);
      setWaiting(false);
    } else {
      void start();
    }
  }

  function seek(event) {
    if (!active) return;
    try {
      const video = videoRef.current;
      const position = seekClip(video, Number(event.target.value));
      setTime(position);
      setEnded(position >= finiteDuration(video.duration));
      setError("");
    } catch (cause) {
      setError(cause.message || "This position could not be loaded.");
    }
  }

  async function expand() {
    try {
      if (document.fullscreenElement === cameraRef.current) {
        if (!document.exitFullscreen) throw new Error("Exiting fullscreen is not supported by this browser.");
        await document.exitFullscreen();
      } else {
        if (!cameraRef.current?.requestFullscreen) throw new Error("Fullscreen is not supported by this browser.");
        await cameraRef.current.requestFullscreen();
      }
      setFullscreen(document.fullscreenElement === cameraRef.current);
    } catch (cause) {
      setError(cause.message || "Fullscreen could not be changed.");
    }
  }

  function reload() {
    playRequest.current += 1;
    wantedPlayback.current = false;
    setPlaying(false);
    setEnded(false);
    setReady(false);
    setWaiting(false);
    setError("");
    setTime(0);
    videoRef.current?.pause();
    videoRef.current?.load();
  }

  const state = ended ? "Finished" : waiting ? "Buffering…" : playing ? "Playing" : "Paused";
  return <section className="footage-camera independent-camera" ref={cameraRef} data-camera-id={stream.id} data-testid={`camera-player-${stream.id}`} aria-label={`${label} playback`}>
    <div className="footage-camera-label independent-camera-header">
      <span className="independent-camera-title"><Video size={15} aria-hidden="true" /> {label}</span>
      <button type="button" className="secondary-button independent-expand" onClick={expand} disabled={!active && !fullscreen}
        aria-label={fullscreen ? `Exit fullscreen ${label}` : `Expand ${label}`}>
        {fullscreen ? <Minimize2 size={14} aria-hidden="true" /> : <Maximize2 size={14} aria-hidden="true" />}<span>{fullscreen ? "Exit" : "Expand"}</span>
      </button>
    </div>
    <video className="footage-image independent-video" ref={videoRef} src={stream.videoUrl}
      data-testid={`video-${stream.id}`} aria-label={`${label} annotated video`}
      preload="metadata" muted playsInline autoPlay={false} loop={false}
      onLoadedMetadata={metadata} onDurationChange={metadata}
      onTimeUpdate={(event) => setTime(Number.isFinite(event.currentTarget.currentTime) ? event.currentTarget.currentTime : 0)}
      onPlay={(event) => { if (!active || !wantedPlayback.current) event.currentTarget.pause(); else { setPlaying(true); setEnded(false); } }}
      onPause={(event) => { if (event.currentTarget.paused) { wantedPlayback.current = false; setPlaying(false); setWaiting(false); } }}
      onPlaying={() => setWaiting(false)}
      onWaiting={(event) => { if (!event.currentTarget.paused) setWaiting(true); }}
      onCanPlay={() => setWaiting(false)}
      onEnded={(event) => { wantedPlayback.current = false; event.currentTarget.pause(); setPlaying(false); setWaiting(false); setEnded(true); setTime(event.currentTarget.currentTime); }}
      onError={() => { playRequest.current += 1; wantedPlayback.current = false; videoRef.current?.pause(); setReady(false); setPlaying(false); setWaiting(false); setError("This video could not be loaded. Reload this clip to try again."); }}>
      Your browser does not support video playback.
    </video>
    <div className="frame-controls independent-controls">
      <button type="button" className="frame-play independent-play" disabled={!active || !ready || ended} onClick={toggle}
        aria-label={`${playing ? "Pause" : "Play"} ${label}`} title={`${playing ? "Pause" : "Play"} ${label}`}>
        {playing ? <Pause size={18} aria-hidden="true" /> : <Play size={18} aria-hidden="true" />}
      </button>
      <span className="independent-time" aria-hidden="true">{formatClipTime(time)}</span>
      <input className="independent-seek" aria-label={`Seek ${label}`} aria-valuetext={`${formatClipTime(time)} of ${formatClipTime(duration)}`}
        type="range" min="0" max={duration || 1} step="0.1" value={Math.min(time, duration || 0)}
        disabled={!active || !ready || !duration} onChange={seek} />
      <span className="independent-time" aria-hidden="true">{formatClipTime(duration)}</span>
      <button type="button" className="independent-replay" disabled={!active || !ready || !ended} onClick={() => void start(true)}
        aria-label={`Replay ${label}`} title={`Replay ${label} from the start`}><RotateCcw size={15} aria-hidden="true" /><span>Replay</span></button>
    </div>
    <p className="footage-caption independent-camera-meta"><span className={!ready && !error ? "independent-loading" : undefined}>{!ready && !error ? "Loading video…" : state}</span> · {formatClipTime(duration)} analysed</p>
    {error && <div className="upload-message upload-message-error independent-error" role="alert"><span>{error}</span><button type="button" className="secondary-button" disabled={!active} onClick={reload}>Reload {label}</button></div>}
  </section>;
}

/** Stable mounted players retain positions, but never resume automatically. */
export default function IndependentFootagePlayer({ run, active = true }) {
  return <div className="footage-player independent-footage-player" data-recording-id={run.id}>
    <div className="footage-streams footage-streams-pair independent-footage-streams">
      {(run.streams || []).map((stream) => <CameraPlayer key={`${run.id}:${stream.id}:${stream.videoUrl}`} stream={stream} active={active} />)}
    </div>
    <p className="footage-caption independent-note">Each view has its own controls. Videos do not auto-play or repeat, and leaving Playback pauses them.</p>
  </div>;
}
