# TFOD2 VS Code Object Detection Demo

This project is a clean TensorFlow Object Detection 2 inference workspace for VS Code. It can download a TFOD2 Model Zoo model and run these tasks:

- Detect objects in one image.
- Detect objects in every image in a folder.
- Detect objects in a video file.
- Detect objects live from a webcam.
- Benchmark inference speed on one image.
- Save annotated images/videos and JSON detection results.

The code uses TFOD2 SavedModel exports directly, so you do not need to clone the full TensorFlow Models repository for inference.

## 1. Open in VS Code

Open this folder in VS Code:

```bash
cd /Users/ingledarshan/Documents/Codex/2026-04-25/create-a-perfect-code-for-tfod2
code .
```

Install the Microsoft Python extension if VS Code asks for it.

## 2. Create and activate a virtual environment

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Use Python 3.9, 3.10, or 3.11. Python 3.11 is recommended for this project.

## 3. Download a TFOD2 model

Default fast model:

```bash
PYTHONPATH=src python -m tfod2_demo setup-model
```

Available model names:

```bash
PYTHONPATH=src python -m tfod2_demo setup-model --model ssd_mobilenet_v2
PYTHONPATH=src python -m tfod2_demo setup-model --model efficientdet_d0
PYTHONPATH=src python -m tfod2_demo setup-model --model faster_rcnn_resnet50
```

The default model is `ssd_mobilenet_v2`, which is the best first choice for CPU and webcam testing.

## 4. Run object detection on an image

Add an image at:

```text
sample_images/test.jpg
```

Run:

```bash
PYTHONPATH=src python -m tfod2_demo detect-image --image sample_images/test.jpg
```

Outputs:

```text
output/detected_image.jpg
output/detected_image.json
```

You can change the confidence threshold:

```bash
PYTHONPATH=src python -m tfod2_demo detect-image --image sample_images/test.jpg --min-score 0.35
```

## 5. Run batch detection

Put multiple images in `sample_images/`, then run:

```bash
PYTHONPATH=src python -m tfod2_demo detect-batch --input sample_images
```

Outputs:

```text
output/batch/
output/batch_results.json
```

## 6. Run video detection

```bash
PYTHONPATH=src python -m tfod2_demo detect-video --video path/to/video.mp4
```

To preview while processing:

```bash
PYTHONPATH=src python -m tfod2_demo detect-video --video path/to/video.mp4 --display
```

Output:

```text
output/detected_video.mp4
```

## 7. Run webcam detection

```bash
PYTHONPATH=src python -m tfod2_demo detect-webcam --camera 0
```

Press `q` in the preview window to stop.

On macOS, VS Code or Terminal may ask for camera permission the first time.

## 8. Benchmark inference

```bash
PYTHONPATH=src python -m tfod2_demo benchmark --image sample_images/test.jpg --runs 20
```

## 9. Run from VS Code

Use **Terminal > Run Task**:

- `Create virtual environment`
- `Install dependencies`
- `Download default TFOD2 model`
- `Run image detection`
- `Run batch detection`

Use **Run and Debug** for these launch configurations:

- `TFOD2: Setup Model`
- `TFOD2: Detect Image`
- `TFOD2: Detect Batch`
- `TFOD2: Detect Webcam`

## Project structure

```text
.
├── .vscode/                 # VS Code tasks and debug configurations
├── assets/labels/           # COCO label map
├── sample_images/           # Add test images here
├── src/tfod2_demo/          # Python package
├── output/                  # Generated results
├── models/                  # Downloaded TFOD2 models
├── requirements.txt
└── README.md
```

## Troubleshooting

If `tensorflow` fails to install, check that your virtual environment uses Python 3.9, 3.10, or 3.11:

```bash
python --version
```

If `ModuleNotFoundError: No module named 'tfod2_demo'` appears, run commands with:

```bash
PYTHONPATH=src python -m tfod2_demo ...
```

If webcam detection opens a black window, try another camera index:

```bash
PYTHONPATH=src python -m tfod2_demo detect-webcam --camera 1
```

## Results

### Detection Result 1
![Result 1](screenshots/screenshot1.png)

### Detection Result 2
![Result 2](screenshots/screenshot2.png)

### Detection Result 3
![Result 3](screenshots/screenshot3.png)