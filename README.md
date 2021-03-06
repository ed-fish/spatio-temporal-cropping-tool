## Spatio-Temporal Cropping Tool

Collect consistent spatio-temporal crops from videos with augmentation set by probability for use in contrastive learning.

---

Papers which use contrastive learning with spatio-temporal crops include:

[Spatiotemporal Contrastive Video Representation Learning](https://arxiv.org/abs/2008.03800)

[Hierarchically Decoupled Spatial-Temporal Contrast for Self-supervised Video Representation Learning](https://arxiv.org/abs/2011.11261)

---
#### Features

- You can use either a directory or a single mp4 video file.
- Select the number of frames you would like to extract as one temporal crop (referenced as a chunk in the code)
- Select the amount of augmentation (from 0.0 to 1.0)
- Supports cv2 GPU by default (if built from source)
- All transforms perfomed in cv2 for efficient computation

Augmentations include 
  - random crop
  - gaussian noise
  - blur
  - grayscale
  - flip

---

## Usage instructions

#### Install Requirements

`pip install -r requirements.txt`

Note - Open CV was compiled from source for GPU use. You may need to add your own version of opencv `pip install cv2` for the code to work. File an issue if you have problems. 

#### Set parameters either in the config.yaml or via arguments. 

`python main.py --help`








