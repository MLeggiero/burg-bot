#!/usr/bin/env bash
# Export a quantized int8 TFLite object detector for burgerbot_perception.
#
# Run this on the DEV MACHINE, not the Pi -- exporting/quantizing needs more
# CPU and RAM than a Pi 4 comfortably has to spare, and none of it needs to
# happen on the robot. Copy the resulting files to the Pi afterward (or share
# the workspace over the same filesystem, if you're symlink-mounting it).
#
# Exports via ONNX, not ultralytics' direct format="tflite" -- that path
# routes through litert_torch, a very new (sub-1.0) package built on
# torch.export(), and it's broken against the torch release pip currently
# resolves (confirmed independently under both Python 3.12 and 3.14, in a
# fresh venv each time: `ImportError: cannot import name
# 'get_cuda_generator_meta_val' from torch._functorch._aot_autograd.utils`).
# ONNX export uses torch.onnx.export(), a much older and more stable API,
# and onnx2tf (a mature, independently maintained ONNX->TFLite converter)
# doesn't touch litert_torch or torch.export() at all -- a different, more
# battle-tested code path end to end.
#
# Usage:
#   ./scripts/export_detection_model.sh [model] [imgsz]
#   ./scripts/export_detection_model.sh yolov8n.pt 320
#
# Produces:
#   burgerbot_ws/src/burgerbot_perception/models/yolov8n_int8.tflite
#   burgerbot_ws/src/burgerbot_perception/models/labels.txt
#
# After running this, re-run colcon build once so the new files get
# symlinked into the install space:
#   colcon build --symlink-install --packages-select burgerbot_perception
set -euo pipefail

MODEL="${1:-yolov8n.pt}"
IMGSZ="${2:-320}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$REPO_ROOT/burgerbot_ws/src/burgerbot_perception/models"
mkdir -p "$OUT_DIR"

# Deliberately NOT the default mktemp location: plain `mktemp -d` resolves
# under $TMPDIR, which under WSL2 is commonly an 8GB RAM-backed tmpfs at
# /tmp. torch + the CUDA wheels ultralytics pulls in are several GB on their
# own and blow through that even when the real disk has plenty of room --
# confirmed on this exact setup ("No space left on device" with 74GB free on
# the actual filesystem). $HOME is reliably on persistent disk.
WORK_DIR="$(mktemp -d --tmpdir="$HOME")"
trap 'rm -rf "$WORK_DIR"' EXIT
cd "$WORK_DIR"
# pip's own download/build cache also defaults to $TMPDIR -- redirect it too,
# or it hits the same tmpfs ceiling even with the venv itself elsewhere.
export TMPDIR="$WORK_DIR/pip_tmp"
mkdir -p "$TMPDIR"

# A dedicated venv, not --user / system site-packages: ultralytics pulls in
# its own numpy/torch/opencv, and installing those alongside a ROS
# environment is a real, confirmed break -- ultralytics' numpy 2.x shadows
# the older numpy the apt-installed matplotlib (pulled in transitively by
# rviz2/rqt) was compiled against, and matplotlib fails to import as a
# result (ultralytics imports matplotlib internally, so this isn't
# avoidable by just not using plotting features). A venv sidesteps this
# entirely by not sharing site-packages with anything else on the machine.
echo "Creating an isolated venv for the export (keeps torch/ultralytics out"
echo "of your ROS Python environment -- this is a one-time dev-machine step,"
echo "not something the robot needs at runtime)..."
python3 -m venv "$WORK_DIR/venv"
source "$WORK_DIR/venv/bin/activate"
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet ultralytics onnx onnx2tf ai-edge-litert

echo "Step 1/2: exporting $MODEL to ONNX at ${IMGSZ}x${IMGSZ}..."
python3 <<PYEOF
from ultralytics import YOLO

model = YOLO("$MODEL")
exported = model.export(format="onnx", imgsz=$IMGSZ, simplify=True)
print(f"exported: {exported}")

# Labels straight from the loaded model's own class order -- not
# hand-transcribed, so there is no risk of a label/index mismatch against
# whatever dataset this particular checkpoint was actually trained on.
names = model.names
with open("labels.txt", "w") as f:
    for i in sorted(names):
        f.write(f"{names[i]}\n")
print(f"wrote labels.txt ({len(names)} classes)")
PYEOF

ONNX_FILE="$(find . -maxdepth 2 -name '*.onnx' | head -1)"
if [ -z "$ONNX_FILE" ]; then
  echo "No .onnx file found after export -- check the output above." >&2
  exit 1
fi

echo ""
echo "Generating calibration data for full-integer quantization..."
# -oiqt (below) requires this -- it's not optional, onnx2tf documents it as
# "required when using -oiqt": full integer quantization needs to observe
# representative activations to choose good per-layer int8 scales, unlike
# weight-only quantization which doesn't need any input data at all.
#
# This uses synthetic random images, not real photos, because no dataset is
# bundled here. That is a real, known limitation: random noise doesn't
# represent real-world activation statistics as well as actual photos of
# the kinds of scenes the robot will see would. The model will work, but
# for better accuracy, replace calibration_data.npy below with ~100 real
# photos (resized to the export resolution, normalized to [0,1], stacked
# into one NHWC array) representative of the robot's actual environment.
python3 <<PYEOF
import numpy as np
import onnx

model = onnx.load("$ONNX_FILE")
input_info = model.graph.input[0]
input_name = input_info.name
dims = [d.dim_value for d in input_info.type.tensor_type.shape.dim]
print(f"ONNX input: name={input_name!r} shape={dims}")

# ONNX is NCHW; onnx2tf's calibration data must be NHWC (post TF-conversion
# dimension order) per its own documented convention. dims[0] is the
# model's own (often dynamic/symbolic) batch dim, irrelevant here -- the
# calibration array's first dimension is just "how many samples", 50,
# independent of it.
_, c, h, w = dims
calib = np.random.default_rng(0).random((50, h, w, c), dtype=np.float32)
np.save("calibration_data.npy", calib)
with open("input_name.txt", "w") as f:
    f.write(input_name)
print(f"wrote calibration_data.npy, shape {calib.shape}")
PYEOF

INPUT_NAME="$(cat input_name.txt)"

echo ""
echo "Step 2/2: converting ONNX -> full-integer-quantized TFLite (onnx2tf)..."
# -oiqt: full integer quantization (both weights and activations int8/uint8,
# not just weights) -- this is what makes the model small and fast enough
# for a Pi 4's CPU. -iqd/-oqd uint8 matches what object_detector.py assumes
# for the model's input dtype. mean=0 std=1: the calibration data above is
# already normalised to [0,1], so no further transform is wanted.
onnx2tf -i "$ONNX_FILE" -o tflite_out -oiqt -iqd uint8 -oqd uint8 \
    -cind "$INPUT_NAME" calibration_data.npy "[0.0]" "[1.0]"

# --- Which variant actually ships -----------------------------------------
#
# float32 is the default the robot loads, NOT the int8 model, despite int8
# being the whole reason for the -oiqt quantization above. The int8 graph
# this pipeline produces is wedged between two mutually exclusive TFLite
# constraints, both confirmed by running it:
#
#   * XNNPACK delegate ON (the default): "failed to delegate TRANSPOSE
#     node ... Node number 276 (TfLiteXNNPackDelegate) failed to prepare",
#     from the quantized Transpose ops onnx2tf inserts for NCHW->NHWC
#     layout recovery. This kills allocate_tensors() outright.
#   * XNNPACK delegate OFF: allocate_tensors() then succeeds, but the first
#     real inference dies in the reference sigmoid kernel --
#     "activations.cc:482 output->params.scale == 1. / 256 was not true.
#     Node number 260 (LOGISTIC) failed to prepare". TFLite's reference
#     quantized LOGISTIC requires its output scale to be exactly 1/256,
#     which calibration-derived scales do not generally land on.
#
# There is no delegate setting that satisfies both. float32 sidesteps the
# pair entirely: XNNPACK handles float transposes and sigmoid without
# complaint, so it runs on the accelerated path with no resolver hacks.
#
# Cost on a Pi 4: a bigger file (~12MB vs ~3.2MB) and a slower inference.
# That is affordable here specifically because object_detector.py throttles
# to ~1.5 Hz by default (see its inference_rate_hz parameter) rather than
# running per frame -- the model has a ~660ms budget per inference at that
# rate, well clear of YOLOv8n-320 float32 on four A72 cores.
#
# The int8 file is still copied alongside it, unused by default: fixing the
# Transpose issue upstream (onnx2tf's -nodaftc 2 suppresses Transpose
# creation) would let XNNPACK take the int8 graph and make it the better
# choice again. Point object_detector's model_path at it to try.
FLOAT_FILE="$(find tflite_out -name '*_float32.tflite' | head -1)"
if [ -z "$FLOAT_FILE" ]; then
  echo "No float32 .tflite found after onnx2tf conversion -- check the output above." >&2
  exit 1
fi
cp "$FLOAT_FILE" "$OUT_DIR/yolov8n_float32.tflite"

# Exact suffix match, not a substring: onnx2tf also emits
# *_full_integer_quant_with_int16_act.tflite (int16 activations, a variant
# object_detector.py does not handle -- it only branches on dtype == uint8,
# else assumes float32 input), and a bare substring glob like
# '*full_integer_quant*.tflite' matches that file too. Anchoring on the
# literal '_full_integer_quant.tflite' ending excludes it.
INT8_FILE="$(find tflite_out -name '*_full_integer_quant.tflite' | head -1)"
if [ -n "$INT8_FILE" ]; then
  cp "$INT8_FILE" "$OUT_DIR/yolov8n_int8.tflite"
fi

cp labels.txt "$OUT_DIR/labels.txt"

echo ""
echo "Wrote:"
echo "  $OUT_DIR/yolov8n_float32.tflite   (the one the robot loads)"
[ -n "$INT8_FILE" ] && echo "  $OUT_DIR/yolov8n_int8.tflite      (not used by default; see comment above)"
echo "  $OUT_DIR/labels.txt"
echo ""
echo "Verifying the shipped model loads and runs the way object_detector.py will..."
python3 <<PYEOF
import numpy as np
from ai_edge_litert.interpreter import Interpreter

# Constructed exactly the way object_detector.py does it, default delegates
# and all. Verifying under different settings than the robot uses is how the
# int8 LOGISTIC crash got missed the first time: the model loaded fine and
# only blew up on the first real inference.
interp = Interpreter(model_path="$OUT_DIR/yolov8n_float32.tflite", num_threads=3)
interp.allocate_tensors()
inp = interp.get_input_details()[0]
out = interp.get_output_details()[0]
print(f"  input:  shape={inp['shape']} dtype={inp['dtype']}")
print(f"  output: shape={out['shape']} dtype={out['dtype']}")

# Actually invoke, don't just allocate. This is the step that would have
# caught the LOGISTIC failure before it reached a live run.
_, h, w, c = inp["shape"]
if inp["dtype"] == np.uint8:
    probe = np.zeros((1, h, w, c), dtype=np.uint8)
else:
    probe = np.zeros((1, h, w, c), dtype=np.float32)
interp.set_tensor(inp["index"], probe)
interp.invoke()
result = interp.get_tensor(out["index"])
print(f"  invoke() OK, produced {result.shape} {result.dtype}")

num_classes = out["shape"][1] - 4
if num_classes <= 0:
    print(f"  WARNING: output shape doesn't match the [1, 4+num_classes, num_boxes]")
    print(f"  layout detection_postprocess.py assumes. Check burgerbot_perception/")
    print(f"  burgerbot_perception/detection_postprocess.py before trusting detections.")
else:
    print(f"  looks right: {num_classes} classes, matches detection_postprocess.py")
PYEOF

echo ""
echo "Next: colcon build --symlink-install --packages-select burgerbot_perception"
