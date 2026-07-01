# Terrain-Aware Locomotion — Final Project Page

A clean, one-page GitHub Pages website for presenting a final robotics project. It is intentionally structured like an academic project page: title and abstract, one hero video, method overview, focused video cases, experiment design, and a concise conclusion.

## Project structure

```text
.
├── index.html
├── assets/
│   ├── css/style.css
│   ├── js/main.js
│   ├── images/                  # video fallback posters
│   ├── videos/                  # replace the placeholder MP4 files here
│   └── report/                  # put final-report.pdf here when ready
└── README.md
```

## Replace the bundled placeholder videos

The page already works with four small placeholder clips. Replace them with your own MP4 files **without changing their names**:

| Slot on website | Replace this file |
|---|---|
| Large hero video | `assets/videos/hero-demo.mp4` |
| Stepping-stone simulation | `assets/videos/sim-stepping-stones.mp4` |
| Gap / perturbation simulation | `assets/videos/sim-gap-perturbation.mp4` |
| Real-world Go2 demo | `assets/videos/real-go2-demo.mp4` |

Recommended export:

- Aspect ratio: **16:9**
- Video codec: **H.264 MP4**
- Resolution: 1280×720 or 1920×1080
- Duration: 10–45 seconds per clip
- Put the most persuasive complete traversal in the hero slot.
- For the real-world clip, keep the robot and terrain in the same shot. A small subtitle embedded in the video is helpful.

## Replace author / project information

Edit the following parts near the beginning of `index.html`:

- Project title
- Subtitle
- Author name
- Department / affiliation
- Abstract

The remaining method and experiment text has been drafted from the current Unitree Go2 terrain-aware locomotion project and can be edited directly in the same file.

## Add final report PDF

1. Place your PDF at `assets/report/final-report.pdf`.
2. Find the `Final report PDF` block in `index.html`.
3. Replace the explanatory placeholder with a link like:

```html
<a class="pill-link" href="./assets/report/final-report.pdf">Download PDF</a>
```

## Publish with GitHub Pages

1. Create a GitHub repository, for example `go2-terrain-project`.
2. Upload every file in this folder to the repository root.
3. Open **Settings → Pages** in GitHub.
4. Set **Build and deployment** to **Deploy from a branch**.
5. Choose branch `main` and folder `/(root)`.
6. Save. GitHub will give you the public website address once deployment finishes.

## Keep the repository public-safe

Do not publish robot IP addresses, network settings, passwords, private deployment scripts, raw sensor logs, or data that has not been approved for release. For a research-final-report site, representative clips, condensed result figures, and a public-facing PDF are normally enough.
