# WWM — Workout With Me

Follow along with a demo video, and an AI coach scores your form.
WWM tracks your body through the webcam, compares your movement with the demonstration, and tells you what to fix — all inside the browser.

**▶ Try it: https://pan1624.github.io/wwm/** (works on desktop and phone; allow camera access)

<!-- Add a screenshot of the scoring page here, e.g. ![WWM scoring page](docs/screenshot.png) -->

## Features

- **11 guided exercises** across arms, back, chest and legs, each with a demonstration video and form tips
- **Live skeleton overlay** on your camera feed while you train
- **Form score (0–100)** with a breakdown of posture, range of motion and tempo, a rep count, and written feedback (in Traditional Chinese) such as *「手肘角度與示範差距較大（平均差 23°）」*
- **HIIT timer** and **weight / training trackers**
- **Installable app (PWA)** — add it to your phone's home screen or install it from Chrome
- **Private by design** — pose detection and scoring run on your device; video never leaves the browser

## How the scoring works

```
camera + demo video ──► MediaPipe Pose (33 keypoints, 3D) ──► 8 joint angles per frame
                                                                     │
           score + feedback ◄── posture 70% + range 15% + tempo 15% ◄── DTW alignment
```

1. **Pose estimation.** While the demo plays, both the camera and the demo video are sampled 10 times per second with MediaPipe Pose Landmarker. Using its 3D world coordinates, each frame becomes eight joint angles (both elbows, shoulders, hips and knees), so the result doesn't depend on how far you stand from the camera.
2. **Cleaning.** Short detection gaps are interpolated and the angle curves are lightly smoothed; joints that are rarely visible are dropped.
3. **Alignment with dynamic time warping (DTW).** People start late and move at different speeds, so frame *i* of the demo is not frame *i* of the user. DTW finds the best frame-to-frame alignment before comparing.
4. **Exercise-specific weighting.** Each exercise only scores the joints that matter (e.g. a bicep curl weighs the elbow at 70%). An angle within 10° of the demo scores full marks, falling linearly to zero at 45°. Left and right are also compared swapped, so mirrored cameras or the other arm still work.
5. **Range, tempo and reps.** Range of motion compares the 5th–95th percentile swing of the key joint; tempo measures how far the DTW path drifts from the diagonal; reps are counted with a hysteresis threshold on the angle curve.

## Project history

| Version | What changed |
|---|---|
| **v1** (Sep 2024 – Jan 2025) | Software Engineering course project with a 3-person team at National Taipei University: exercise pages, tutorials, timer and trackers in an Electron app with a Node.js/Express backend |
| **v2** (Oct 2026) | New scoring engine in Python: 3D joint angles, DTW alignment, per-exercise weights, range/tempo/rep analysis and feedback |
| **v3** (Oct 2026) | Ported the scoring engine to JavaScript so everything runs in the browser, with no server; turned the site into an installable PWA |

The JavaScript scorer was checked against the Python reference on randomized test sequences and produces identical scores, rep counts and feedback.

## Repository layout

```
index.html, manifest.json,      Web app (served by GitHub Pages)
service-worker.js, icons/
app/
  js/coach.js                   Scoring engine (JavaScript)
  js/exercise.js                Camera, live skeleton, sampling, results
  *.html, *.mp4                 Exercise pages and demonstration videos
  timer.html, track.html        HIIT timer and trackers
python/
  app.py                        Reference implementation of the scoring engine
```

## Running locally

The camera only works over `https` or on `localhost`, so serve the folder instead of opening the HTML file directly:

```bash
python -m http.server 5500
# then open http://localhost:5500
```

To score two videos with the Python reference:

```bash
pip install -r python/requirements.txt
python python/app.py my_attempt.mp4 app/shoulder_press.mp4 shoulder_press
```

## Tech stack

JavaScript (ES modules) · MediaPipe Tasks Vision (Pose Landmarker) · Python · NumPy · OpenCV · PWA (Service Worker, Web App Manifest) · GitHub Pages
