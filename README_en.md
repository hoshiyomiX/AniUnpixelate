# AniRestore - Anime Image AI Restoration Tool

**English Version** | [中文版本](README.md)

![Example Demo](img/Example_1.gif)

**AniRestore (AniUnpixelate)** is an AI-powered restoration tool specifically designed for anime-style illustrations and CGI art. It leverages deep neural networks to intelligently repair and fill in pixelated or mosaic areas in images.

> [!IMPORTANT]
> **Technical Note**: The AI model used in this project is **NOT a generative model**. It does not "re-draw" or "imagine" new content from scratch. Instead, it utilizes deep convolutional algorithms to analyze the color distribution and structural patterns within pixelated blocks to **reconstruct/restore the underlying color information and original visual composition**.

## 🌟 Key Features

- **AI-Powered Unpixelating**: Uses pre-trained deep convolutional neural networks for texture synthesis in missing pixel areas.
- **Offline ONNX Inference**: Integrated with ONNX Runtime, all AI computations are performed locally on the device—no cloud uploads, ensuring maximum privacy.
- **Customizable Repair Area**: Allows users to manually select or brush over specific areas for precise local restoration.
- **Seamless Blending**: Employs Alpha blending technology to ensure naturally smooth transitions between AI-generated content and the original background.

## 🚀 Getting Started

1. **Import Image**: Click the "Import Anime Photo" button to select a local image.
2. **Select Area**: Drag or brush over the pixelated/mosaic area on the preview.
3. **Run Restoration**: Click "Unpixelate," and the AI will automatically process the image.
4. **Save Result**: Once restored, the image will be saved automatically to your gallery.

## 🛠️ Technology Stack

- **Platform**: Android
- **Language**: Java / Kotlin
- **AI Engine**: ONNX Runtime
- **Model Architecture**: big-lama (SAIC AI, Apache-2.0) ONNX export (512×512 input, image+mask dual input, FFC global context)

## 📦 Building the APK

### Option A: GitHub Actions (recommended — no local setup)

1. Fork or push this repository to your GitHub account.
2. Open the **Actions** tab — the `Build APK` workflow runs automatically on every push / PR.
3. When the run finishes, download `AniRestore-APK` from the run's **Artifacts** section (contains debug and unsigned release APKs).
4. Every successful build is also published to the repo's **Releases** page: regular pushes refresh a rolling `nightly` pre-release (fully replaced on each build; the `nightly` tag always points at the newest commit), while pushing a tag like `v1.0` creates a proper Release named after the tag. Both carry the debug and unsigned release APKs (file names embed the commit / version).

### Option B: Local build

```bash
# 1. Download the model (Apache-2.0, big-lama ONNX export, 208 MB)
#    from https://huggingface.co/gumka/lama-onnx
#    and place it at app/src/main/assets/model.onnx
#    (that path is excluded by .gitignore)
# 2. Build
./gradlew assembleDebug
# Output: app/build/outputs/apk/debug/
```

> Note: the model is not committed to the repo (see `.gitignore`); CI fetches it automatically from Hugging Face and verifies its SHA256 (see the workflow file). The model is a **big-lama** (SAIC AI, Apache-2.0) ONNX export: input 1×3×512×512 ([0,1]) + mask 1×1×512×512 (1 = restore zone), output returns at 255 scale. Without the model the APK still builds and installs, but restoration shows a "model missing" message. The release APK is unsigned — sign it with `apksigner` before distribution.


## ⚠️ Legal & Disclaimer

This project is intended for academic research on deep learning applications in image processing.

1. **Prohibited Use**: Any illegal or unauthorized usage is strictly forbidden. Users hold full responsibility for their actions.
2. **Scope of Application**: To prevent abuse, this demo is optimized **only for Anime/CGI style** images and is not suitable for photorealistic images.
3. **Privacy**: The application operates entirely offline and does not collect any user image data.

---