# CA-YOLO-IRT

**CA-YOLO-IRT**: A lightweight thermographic detector for automated subsurface concrete defect detection using infrared thermography.

---

## Overview

This repository contains the official implementation of **CA-YOLO-IRT**, a lightweight YOLOv5-based model with Coordinate Attention for subsurface concrete defect detection using infrared thermography (IRT).

The related manuscript entitled *"CA-YOLO-IRT: Lightweight coordinate-attention deep learning for robust subsurface concrete defect detection using infrared thermography"* is under review at **Computer-Aided Civil and Infrastructure Engineering**.

---

## Key Features

- 🚀 **Lightweight architecture**: ~0.39M parameters, 0.9 GFLOPs
- 🎯 **Coordinate Attention mechanism** for enhanced thermal anomaly localization
- 🔬 **Comprehensive ablation studies** and Eigen-CAM interpretability analysis
- 🏗️ **Lab-to-field validation** on real-world concrete structures

---

## Requirements

- Python 3.9+
- PyTorch 2.1.0+
- CUDA 11.8 (for GPU inference)
- See `requirements.txt` for full dependencies

---

## Installation

```bash
git clone https://github.com/quangtaita/CA-YOLO-IRT.git
cd CA-YOLO-IRT
pip install -r requirements.txt