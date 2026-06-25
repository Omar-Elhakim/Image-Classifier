# Image Classifier

An image classifier web app built with Streamlit and PyTorch's pretrained ResNet50.

You can try the live app [here](https://photo-classifier.streamlit.app/).

## Overview

Upload an image and the app runs it through a ResNet50 model pretrained on
ImageNet, then shows the top-5 predicted classes with their confidence scores.
It's a small, self-contained demo of running a pretrained vision model behind a
simple web UI.

## Features

- Upload images in PNG, JPG, JPEG, or WebP format.
- Top-5 ImageNet predictions with confidence percentages.
- Runs entirely on a pretrained model — no training or dataset required.

## Tech stack

- **Python**
- **Streamlit** — web UI
- **PyTorch / torchvision** — ResNet50 model and image preprocessing
- **Pillow** — image loading
- **requests** — fetching the ImageNet class labels

## Getting started

Prerequisites: Python 3.

```bash
# Clone the repository
git clone https://github.com/omar-elhakim/image-classifier.git
cd image-classifier

# Install dependencies
pip install -r requirements.txt
```

## Usage

Run the app locally with Streamlit:

```bash
streamlit run main.py
```

Then open the URL shown in the terminal, upload an image, and click
**Analyse Image** to see the predictions.

## License

Released under the [MIT License](LICENSE).
