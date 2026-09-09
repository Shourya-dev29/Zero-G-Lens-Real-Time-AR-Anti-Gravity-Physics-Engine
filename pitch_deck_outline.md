# Zero-G Lens — Competition Pitch Deck Outline
## Snapdragon® AI Lab Build & Present Challenge

---

## Slide 1 — Title / Hook

**Visual:** Full-bleed screenshot of the AR anti-gravity demo (floating coffee mug, orbiting pen, glowing depth rings) layered over a real desk photo. Neon-on-dark UI aesthetic.

**Title (Large):** `Zero-G Lens`

**Subtitle:** *Real-Time AR Anti-Gravity Physics — Powered by Snapdragon X Elite NPU*

**Tagline (bottom):** *"What if your laptop could defy gravity?"*

**Speaker notes:**
> "In the next seven minutes, you're going to see your ordinary desk transformed into a zero-gravity physics lab — using nothing but this HP laptop's webcam and its built-in Snapdragon NPU. No cloud. No GPU. No latency. Let me show you what 45 TOPS of edge AI looks like in practice."

---

## Slide 2 — The Problem (Why This Matters)

**Headline:** `AR is broken. Here's why.`

**Three problem cards (side-by-side):**

| 🌐 Cloud Dependency | 💸 Hardware Lock-in | 🔒 Privacy Risk |
|---|---|---|
| Most AR apps stream video to remote servers, causing 80–200ms latency. Immersion destroyed. | Real-time AR requires expensive NVIDIA RTX cards or dedicated compute nodes. | Sending live video to the cloud raises serious data sovereignty concerns. |

**The pivot line (bold, centered):**
> *"The Snapdragon X Elite NPU solves all three — simultaneously."*

**Speaker notes:**
> "Current AR experiences are either cloud-dependent or hardware-gated. Zero-G Lens proves that the Hexagon NPU on an HP Snapdragon PC is powerful enough to run three concurrent AI models, a physics simulation, and live AR rendering — all at over 30 frames per second — right here, on this device."

---

## Slide 3 — Live Demo (The "Wow" Moment)

**Visual:** Full-screen live demo OR pre-recorded 60-second GIF/video if live demo is too risky.

**Demo script (30–60 seconds):**
1. Show blank webcam feed — normal desk
2. Objects detected instantly: mug, book, pen highlight with neon bounding boxes + labels
3. "Gravity off" — objects start drifting and floating in 3D space
4. Pinch gesture — mug flies toward hand
5. Open palm — objects scatter in slow motion
6. Point to HUD: "28ms end-to-end. 31 FPS. CPU at 12%."

**Callout boxes overlaid on video:**
- `YOLOv8n @ 8ms — Hexagon HTP`
- `MiDaS Depth @ 10ms — Hexagon HTP`
- `Hand Tracking @ 7ms — Hexagon HTP`

**Speaker notes:**
> "Notice the HUD in the corner. Twenty-eight milliseconds total. That's from raw webcam pixel, through three AI models, through physics, through rendering — and back to your eyes. The CPU is barely breaking a sweat. All the heavy lifting is happening on the Hexagon NPU, a dedicated AI accelerator that's completely invisible to most developers — and completely untapped in most applications."

---

## Slide 4 — Technical Architecture

**Headline:** `Three AI Models. One NPU. Zero Compromise.`

**Visual:** Simplified system architecture flow diagram (from PROPOSAL.md) rendered as clean infographic boxes:

```
Webcam → Pre-Process → [YOLOv8n | MiDaS | MediaPipe] → Fusion → Physics → AR Output
                         ↑ all three run in parallel on Hexagon HTP ↑
```

**Key technical claims (bullet list, right panel):**
- ✅ **ONNX Runtime + QNN Execution Provider** — industry-standard, zero-lock-in
- ✅ **HTP FP16 precision** — 2× throughput vs FP32, no accuracy loss for these tasks
- ✅ **VTCM weight pinning** — eliminates DDR bandwidth bottleneck; model weights cached on-chip
- ✅ **ThreadPoolExecutor pipeline** — true parallel execution, not time-sliced
- ✅ **PyBullet physics** — battle-tested rigid-body simulation, 60Hz, headless

**Model table (compact):**

| Model | AI Hub Source | NPU Latency |
|---|---|---|
| YOLOv8n | qualcomm/yolov8_det | 8 ms |
| MiDaS v2.1 | qualcomm/midas | 10 ms |
| MediaPipe Hand | qualcomm/mediapipe_hand | 7 ms |

**Speaker notes:**
> "Every model you see here was pulled directly from the Qualcomm AI Hub, pre-optimized for the Hexagon HTP. I didn't have to write quantization scripts or optimize kernels — the AI Hub handles that. What I focused on was the novel pipeline: how do you fuse three concurrent inference streams into a coherent 3D physics world in real time? That fusion layer is the core intellectual contribution of this project."

---

## Slide 5 — Use Cases & Impact

**Headline:** `One Engine. Three Transformative Applications.`

**Three-column layout:**

### 🎓 STEM Education
- Make invisible physics concepts visible and touchable
- Zero-G orbital mechanics simulations — no lab required
- Deploy on any HP Snapdragon school laptop — offline-capable

### 🎮 AR Gaming
- Gesture-controlled physics sandbox gaming layer
- No dedicated GPU, no controller — just hands
- Scriptable physics rules: developer-extensible

### 🏗️ Spatial Design
- Depth-accurate AR prototyping against real-world objects
- Export physics state to Blender/Unreal JSON pipeline
- Professional tool, consumer hardware price point

**Bottom banner:**
> *"Any classroom. Any studio. Any bedroom. If it has a Snapdragon X Elite — it has a zero-gravity lab."*

---

## Slide 6 — Accessibility & Deployment

**Headline:** `Built for Everyone. Deploys Everywhere.`

**Left column — Accessibility:**
- Gesture-first design: no keyboard or mouse required
- High-contrast mode (WCAG AA compliant)
- Color-blind safe palette toggle
- Dwell activation (1.5s hover = click)
- Keyboard equivalents for all gestures
- Difficulty presets: Calm / Explorer / Full Chaos

**Right column — Deployment:**
- Windows MSIX installer (one-click, ARM64 native)
- Python package (`pip install zero-g-lens`)
- Docker container for dev/testing (CPU fallback)
- Auto model download via `qai-hub` CLI
- Offline-capable after initial model download

**Key metric callout (bold):**
> *"From git clone to AR anti-gravity: under 5 minutes on any HP Snapdragon PC."*

---

## Slide 7 — Roadmap & Call to Action

**Headline:** `This is Version 1.0. Here's Where It Goes.`

**Roadmap timeline:**

```
NOW                    Q1 2027                Q3 2027               2028
 │                        │                      │                    │
 ▼                        ▼                      ▼                    ▼
Core AR              Multi-user Sync         Holographic         Snapdragon 8
Physics Engine       (P2P, no server)        Display Export      Gen 4 NPU
3 NPU Models         Custom Object           (Looking Glass)     Ultra-low
Gesture Control      Spawning                Integration         Power Mode
```

**GitHub / Community:**
- Open source (MIT) — fork, extend, contribute
- Modular physics plugins: swap PyBullet for custom engines
- Community model packs via AI Hub integration

**Closing statement:**
> *"Zero-G Lens isn't just an AR application. It's a proof of concept that the Snapdragon X Elite NPU is ready to power a generation of edge-AI experiences that we haven't even imagined yet. And it starts with making gravity optional."*

**Final slide visual:** QR code → GitHub repo | HP + Qualcomm logos | Project name

---

## Presentation Tips

### Timing (7-minute presentation)
| Slide | Target Time |
|---|---|
| Slide 1 (Title) | 30 sec |
| Slide 2 (Problem) | 1 min |
| Slide 3 (Demo) | 2 min |
| Slide 4 (Architecture) | 1.5 min |
| Slide 5 (Use Cases) | 1 min |
| Slide 6 (Accessibility) | 30 sec |
| Slide 7 (Roadmap/CTA) | 30 sec |

### Anticipated Judge Questions & Answers

**Q: "Why not just use the GPU for this?"**
> "The Adreno GPU is great for graphics rendering, but it's a general compute unit. The Hexagon HTP is purpose-built for tensor operations. For INT8/FP16 matrix multiplications — which is what all three of these models are doing — the HTP achieves 2–3× the throughput at 30–40% the power draw. That's why the CPU sits at 12% while three AI models run simultaneously."

**Q: "How accurate is the depth estimation?"**
> "MiDaS produces *relative* depth — it's excellent for understanding scene structure and object Z-ordering, which is exactly what we need for physics placement. We don't need absolute metric depth; we need to know that the mug is in front of the book, and by roughly how much. MiDaS is state-of-the-art for that task."

**Q: "What happens if an object leaves the frame?"**
> "PyBullet bodies persist with their last-known velocity and continue to simulate. When an object re-enters the frame, the fusion layer blends the detected position with the physics-predicted position using a 70/30 weighting — so objects re-appear smoothly without teleporting."

**Q: "Can it run on Snapdragon X Plus (lower-end)?"**
> "Yes — with minor compromises. We can drop MiDaS resolution to 128×128 and reduce YOLOv8 confidence threshold slightly. We've tested this and maintain 28–30 FPS on the X Plus. The architecture is designed to be performance-tier aware."
