# Three Eyed Raven Dashboard

Presentation-facing Next.js dashboard for the crowd-tracking project. It reads the tracker artifacts from `../artifacts` and displays the annotated video, real processing metrics, local tracking IDs, crowd-count timeline, and development milestones.

## Run locally

```bat
cd /d S:\three_eyed_raven\dashboard
npm install
npm run dev
```

Open `http://localhost:3000`.

The dashboard refreshes its metrics every eight seconds. Run the Python tracker again to add/update artifacts, then select the run in the dashboard header.

## Notes

- It is a local presentation dashboard. Do not expose it to the public internet without access control.
- Only `annotated.mp4` is served by the video route. It supports HTTP range requests, so the browser can seek through the recording.
