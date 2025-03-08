# Stateful Gaussian Splatting

A Python liquid neural network project for stateful Gaussian splatting using multi-view camera images.

## Overview

This project implements a novel approach to Gaussian splatting that incorporates stateful processing through liquid neural networks. By leveraging multi-view camera images, the system can create dynamic 3D representations with temporal consistency.

## Features

- Multi-view camera image processing
- Stateful Gaussian splatting implementation
- Liquid neural network architecture
- Temporal consistency in 3D reconstruction
- Support for RGB, depth, and alpha channel processing

## Installation

```bash
# Clone the repository
git clone https://github.com/pompoi/stateful_gaussian_splatting.git
cd stateful_gaussian_splatting

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

## Data Structure

The project expects data organized in the following structure:

```
FramesByNumber/
  1/
    CameraL_000_rgb.png
    CameraL_000_depth.png
    CameraL_000_alpha.png
    CameraR_000_rgb.png
    CameraR_000_depth.png
    CameraR_000_alpha.png
    ...
  2/
    ...
  ...
  300/
    ...
```

## Usage

```python
from stateful_gaussian_splatting import StatefulGaussianSplatter

# Initialize the model
splatter = StatefulGaussianSplatter(data_path='path/to/FramesByNumber')

# Train the model
splatter.train(epochs=100)

# Generate 3D representation
splatter.generate_representation(output_path='output/model')

# Render from new viewpoint
splatter.render(viewpoint=[0, 0, 5], output_path='output/render.png')
```

## License

MIT