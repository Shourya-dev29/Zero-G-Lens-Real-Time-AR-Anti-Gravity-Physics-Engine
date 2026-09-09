"""
setup_models.py: Automated model download and QNN compilation for Zero-G Lens.

This script:
  1. Authenticates with Qualcomm AI Hub (requires `qai-hub login` first)
  2. Downloads YOLOv8n, MiDaS v2.1 Small, and MediaPipe Hand models
  3. Submits compile jobs targeting the local Snapdragon X Elite SoC
  4. Downloads compiled ONNX artifacts to models/qnn/

Prerequisites:
  pip install qai-hub
  qai-hub login   (or set QAI_HUB_API_TOKEN env var)

Usage:
  python scripts/setup_models.py
  python scripts/setup_models.py --skip-compile   # Use pre-compiled binaries only
"""
import argparse
import os
import sys
from pathlib import Path

# ── Try importing qai_hub ───────────────────────────────────────────
try:
    import qai_hub as hub
except ImportError:
    print("[ERROR] qai-hub not installed. Run: pip install qai-hub")
    sys.exit(1)

OUTPUT_DIR = Path(__file__).parent.parent / "models" / "qnn"

# ── Model registry ──────────────────────────────────────────────────
MODELS = {
    "yolov8n": {
        "hub_model_id":  "yolov8_det",
        "output_name":   "yolov8n.onnx",
        "description":   "YOLOv8 Nano Object Detection (80 COCO classes)",
        "input_shape":   [1, 3, 640, 640],
        "precision":     "fp16",
    },
    "midas": {
        "hub_model_id":  "midas",
        "output_name":   "midas_v21_small.onnx",
        "description":   "MiDaS v2.1 Small Monocular Depth Estimation",
        "input_shape":   [1, 3, 256, 256],
        "precision":     "int8",   # INT8 quantized for maximum NPU throughput
    },
    "mediapipe_hand": {
        "hub_model_id":  "mediapipe_hand",
        "output_name":   "mediapipe_hand.onnx",
        "description":   "MediaPipe Hand Landmark Detection (21 landmarks)",
        "input_shape":   [1, 3, 256, 256],
        "precision":     "fp16",
    },
}


def parse_args():
    p = argparse.ArgumentParser(description="Zero-G Lens — Model Setup")
    p.add_argument("--skip-compile", action="store_true",
                   help="Skip QNN compilation; download pre-compiled artifacts only")
    p.add_argument("--device",       type=str, default="auto",
                   help="Target device name (auto = detect local Snapdragon)")
    return p.parse_args()


def detect_device(requested: str) -> str:
    """Detect or validate the Snapdragon target device for compilation."""
    if requested != "auto":
        return requested
    # qai-hub detects the local device when connected via USB or for on-device compile
    # For Snapdragon X Elite (PC), we specify the device string directly:
    return "Snapdragon X Elite CRD"


def download_and_compile(model_key: str, config: dict, device: str, skip_compile: bool):
    """Download a model from AI Hub and optionally compile it to QNN HTP binary."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / config["output_name"]

    if out_path.exists():
        print(f"[SKIP] {model_key}: {out_path.name} already exists.")
        return

    print(f"\n[HUB] Processing: {model_key}")
    print(f"      {config['description']}")

    try:
        # Fetch model from AI Hub
        model = hub.get_model(config["hub_model_id"])
        print(f"[HUB] ✅ Found model: {config['hub_model_id']}")

        if skip_compile:
            # Download pre-compiled ONNX artifact directly
            print(f"[HUB] Downloading pre-compiled ONNX artifact...")
            model.download(str(out_path))
        else:
            # Submit compile job for Snapdragon X Elite HTP
            print(f"[HUB] Submitting compile job → device: {device} ...")
            compile_job = hub.submit_compile_job(
                model=model,
                device=hub.Device(device),
                input_specs={
                    "input": tuple(config["input_shape"]),
                },
                options=f"--quantize_full_type {config['precision']} --target_runtime onnx",
            )
            print(f"[HUB] Compile job submitted. Waiting for completion...")
            compile_job.wait()

            if compile_job.get_status().success:
                compiled_model = compile_job.get_target_model()
                compiled_model.download(str(out_path))
                print(f"[HUB] ✅ Compiled model saved → {out_path}")
            else:
                print(f"[HUB] ❌ Compile job failed: {compile_job.get_status().message}")
                print(f"[HUB]    Falling back to pre-compiled download...")
                model.download(str(out_path))

    except Exception as e:
        print(f"[ERROR] Failed to process {model_key}: {e}")
        print(f"[INFO]  Manually download from: https://aihub.qualcomm.com/models/{config['hub_model_id']}")
        print(f"[INFO]  Place the .onnx file at: {out_path}")


def verify_models():
    """Verify all expected model files are present."""
    print("\n── Model Verification ────────────────────────")
    all_ok = True
    for key, config in MODELS.items():
        path = OUTPUT_DIR / config["output_name"]
        if path.exists():
            size_mb = path.stat().st_size / (1024 ** 2)
            print(f"  ✅ {key:<20} {path.name:<35} ({size_mb:.1f} MB)")
        else:
            print(f"  ❌ {key:<20} MISSING — {path}")
            all_ok = False
    print("──────────────────────────────────────────────")
    if all_ok:
        print("  All models ready. Launch with: python src/main.py\n")
    else:
        print("  Some models missing. Check errors above.\n")


def main():
    args   = parse_args()
    device = detect_device(args.device)

    print("="*60)
    print("  Zero-G Lens — Model Setup")
    print("="*60)
    print(f"  Target device:  {device}")
    print(f"  Output dir:     {OUTPUT_DIR}")
    print(f"  Skip compile:   {args.skip_compile}")
    print("="*60)

    # Check AI Hub authentication
    try:
        hub.get_hub_client()
        print("[HUB] ✅ Authenticated with Qualcomm AI Hub\n")
    except Exception:
        print("[HUB] ❌ Not authenticated. Run: qai-hub login")
        print("[HUB]    Or set env var: QAI_HUB_API_TOKEN=<your_token>")
        sys.exit(1)

    for key, config in MODELS.items():
        download_and_compile(key, config, device, args.skip_compile)

    verify_models()


if __name__ == "__main__":
    main()
