# Deep Image Prior (DIP) for Fetal Ultrasound Enhancement in MATLAB

This repository contains the MATLAB implementation of the **Deep Image Prior (DIP)** method for solving inverse image problems such as **denoising** and **super-resolution** — specifically applied to **fetal ultrasound images**.

The project was developed as part of the [MathWorks Excellence in Innovation](https://github.com/mathworks/MathWorks-Excellence-in-Innovation/blob/main/projects/Deep%20Image%20Prior%20for%20Inverse%20Problems%20in%20Imaging) program and also as the author’s bachelor thesis at ETSEIB – UPC Barcelona.

## Project Objectives
- Adapt Python DIP code to MATLAB using native deep learning tools.
- Implement DIP from scratch without pretraining or external datasets.
- Evaluate DIP performance on ultrasound and standard test images.
- Focus on medical imaging: improving the quality of fetal ultrasound images where clarity is crucial for diagnosis.
- Explore its feasibility in low-resource environments (no GPU, no large datasets).

DIP uses an untrained convolutional neural network as an image prior. Unlike typical deep learning methods, it does **not require any dataset or training phase** — it fits a randomly initialized network to a single corrupted image.

## Tasks Implemented
- `Denoising/` --> Apply DIP to remove Gaussian and blind noise from ultrasound and standard images.
- `Super-resolution/`  --> Recover high-resolution detail from low-resolution fetal ultrasound images.
-  `Ecos/`  -->  Dataset and preprocessing of fetal ultrasound images from different sources.

## Requirements
- MATLAB R2023a or newer  
- Deep Learning Toolbox  
- Image Processing Toolbox  
- (Optional: use the `Deep Network Designer` app for visualization)

## Structure of the Repository
- MATLAB-AI-CHALLENGE/
     - Denoising/
          - denoising.m
          - utils
          - Images/
          - Results/
     - Super-resolution/
          - super_resolution.m
          - utils
          - Images/
          - Results/
     - Ecos/
          - ultrasound1.png
          - ultrasound2.png
          - ultrasound3.png
- README.md
- LICENSE

## References
- Ulyanov, Vedaldi & Lempitsky, “Deep Image Prior”, 2018.  
  [[Paper]](https://arxiv.org/abs/1711.10925)
- MathWorks DIP Challenge Repository  
  [GitHub Link](https://github.com/mathworks/MathWorks-Excellence-in-Innovation/blob/main/projects/Deep%20Image%20Prior%20for%20Inverse%20Problems%20in%20Imaging)
- Montal Morta, M. (2025). *Deep Image Prior applied to Fetal Ultrasounds*.  
  Bachelor's Thesis, ETSEIB – UPC Barcelona.

## Author: **Mariona Montal Morta**  
Bachelor in Industrial Technologies and Economic Analysis  
UPC Barcelona – July 2025  
Supervisor: Antoni Susín Sánchez

## License
This project is open-source and shared under the MIT License.
