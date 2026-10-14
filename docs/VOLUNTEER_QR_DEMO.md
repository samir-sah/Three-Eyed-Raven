# Controlled volunteer QR-tag demonstration

## What it does

Reads explicit visible QR payloads `TER:V01` through `TER:V99` from two uploaded clips. It compares registered **codes**, not faces, bodies, names or person identities. Detection/counting remains independent of the marker scan. There is no new appearance encoder, Re-ID training, identity association or change to the legacy tracking paths.

## Use it

1. Open the dashboard's **Upload videos** screen and enable **Read visible volunteer QR tags in both clips**.
2. Enter unique anonymous codes (for example `V07, V08, V09`), open the printable tag sheet, and print one distinct code per volunteer. A V07 tag encodes `TER:V07`, not a real name.
3. Record a controlled, volunteered demonstration with large, flat, readable tags in both views. Test readability at the actual camera distance first; a distant aerial camera may not resolve a tag.
4. Select the two clips and their actual viewpoint. Explicitly confirm volunteer participation, then start analysis. Leave the frame limit blank for the full clips or specify a deliberate partial analysis.
5. Open **Tag comparison**. The main conclusion is the number and list of qualifying shared codes. Expand a code for tag-only crops, readable-frame counts, source frame indexes and first/last clip times. Playback retains independent camera controls.

The minimum shared-code rule is three distinct readable source frames **per camera**. Repeated calls with the same source frame cannot inflate the evidence. This rule is an evidence threshold, not calibrated similarity, statistical confidence or identification accuracy.

## Interpret results correctly

- A shared code establishes repeated decoding of the same printed payload, not that the same person wore it. Duplicated or exchanged tags can mislead this demonstration.
- Multiple separated locations of a code in either camera make that code ambiguous; it is excluded from the shared total.
- Zero qualifying shared codes does not prove no participant appeared in both views. Blur, occlusion, angle, lighting and resolution can prevent decoding.
- Missing/corrupt opted-in results or a completely failed scan are **unavailable**, not a measured zero. Partial failures are disclosed.
- Frame indexes begin at zero. Times use each clip's nominal FPS and are not a synchronized camera wall clock.
- Only registered payloads are persisted. Evidence rectifies the decoded QR quadrilateral to 256 × 256 pixels; it is not a face/body crop. Printable V01–V99 cards retain a white quiet zone.
- Existing analyses without tags do not gain automatic cross-camera person matches. A new opted-in upload is required.

## Artifacts and guardrails

Each opted-in anonymous run adds `markers.json` at its artifact root. Each camera adds `markers.jsonl` and `marker_evidence/Vxx.jpg` when an unambiguous code crop is available. The normal counts and H.264 output remain separate. Duplicate-only observations can have a null crop. The dashboard recomputes shared-code intersections from validated records and ignores supplied totals.

Upload, code, file, frame and timestamp validation is enforced on the server. Evidence is served only for current-workspace, completed anonymous uploads that explicitly enabled and confirmed the tag demo. Path redirects/junctions and foreign evidence paths are rejected. A normal upload does not create marker artifacts.

Printable cards are bundled in `dashboard/public/marker-tags`. To generate cards in a **new, non-existing** folder from the project root:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m crowd_tracker.marker_cards_cli --output-dir marker_cards --codes "V07,V08,V09"
```

## Verification — 14 October 2026

- 105 Python regression tests and 78 dashboard Node tests passed; production Next.js build passed.
- A synthetic, tag-only pair was uploaded through the production UI, not seeded as a completed result. Camera A contained V07/V08, Camera B V07/V09. Each clip contained 16 frames at 2 FPS (8 seconds). This is **not a person-detection or identity benchmark**.
- Result: exactly one shared code, V07; all four expected camera/code records had 16 readable frames, source indexes 0–15 and times 0–7.5 seconds, with no scan failures or duplicate locations.
- Both QR-only evidence images loaded at 256 × 256. Printable/download PNG endpoints returned 200; invalid/absent/off-mode evidence returned 404. Video byte-range requests returned 206.
- Independent play, pause, explicit replay and end-of-clip stopping were browser-checked. Switching sidebar screens preserved video sources/positions, paused playback and did not resume it automatically.
- Synthetic QA run: `upload_20261014063049_e2142a`. Its inputs, outputs and original job metadata were preserved; only this generated test job was isolated from the normal workspace list. The user's original selected upload `upload_20261013175036_6f5ca4` and the workspace identifier remained unchanged. No user clip was rescanned.

These checks establish software behavior on clean synthetic tags, not reliable tag reading in real aerial recordings. Ground mAP50 70.96% remains a separate development result; this feature does not change detector weights or those scores.
