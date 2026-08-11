# 23Na-MQC-MRI

Code for image reconstruction and post-processing of undersampled 23Na multi-quantum coherences (MQC) and 23Na double half-echo MRI using multi-dimensional low-rank and sparse models.

This repository contains implementations associated with three published works on 23Na MRI reconstruction, denoising, signal separation, and compressed sensing.

## Repository Structure

| Folder | Description | Associated Paper |
|---|---|---|
| `5D_CS/` | Multi-dimensional compressed sensing reconstruction for undersampled 23Na MQC MRI | Licht et al., Magn Reson Med, 2023 |
| `DHE_SAKE/` | Double half-echo 23Na and accelerated 23Na MQC MRI reconstruction | Licht et al., Magn Reson Med, 2024 |
| `Denoising_DMD/` | Low-rank denoising and SQ/TQ signal separation using DMD | Licht et al., Z Med Phys, 2025 |


## Publications

### 1. Multi-Dimensional Compressed Sensing for 23Na MQC MRI

Licht C, Reichert S, Guye M, Schad LR, Rapacchi S.  
**Multidimensional compressed sensing to advance 23Na multi-quantum coherences MRI.**  
*Magnetic Resonance in Medicine.* 2023; 1-16.  
doi: [10.1002/mrm.29902](https://doi.org/10.1002/mrm.29902)

Code: `5D_CS/`

This implementation is based on:

- Goldstein et al., 2009. doi: [10.1137/080725891](https://doi.org/10.1137/080725891)
- Montesinos et al., 2014. doi: [10.1002/mrm.24936](https://doi.org/10.1002/mrm.24936)  
  GitHub: https://github.com/HGGM-LIM/Split-Bregman-ST-Total-Variation-MRI
  
### 2. Double Half-Echo and Accelerated 23Na MQC MRI

Licht C, Reichert S, Bydder M, et al.  
**Low-rank reconstruction for simultaneous double half-echo 23Na and undersampled 23Na multi-quantum coherences MRI.**  
*Magnetic Resonance in Medicine.* 2024; 1-16.  
doi: [10.1002/mrm.30132](https://doi.org/10.1002/mrm.30132)

Code: `DHE_SAKE/`

This implementation is based on code by Mark Bydder:  
https://github.com/marcsous/parallel

### 3. Low-Rank Denoising and DMD-Based Signal Separation

Licht C, Ilicak E, Boada FE, Guye M, Zöllner FG, Schad LR, Rapacchi S.  
**A noise-robust post-processing pipeline for accelerated phase-cycled 23Na Multi-Quantum Coherences MRI.**  
*Zeitschrift für Medizinische Physik.* 2025;35(1):98-108.  
https://www.sciencedirect.com/science/article/pii/S093938892400117X

Code: `Denoising_DMD/`

The DMD implementation is based on code by Efe Ilicak:  
https://github.com/Computer-Assisted-Clinical-Medicine/DMD_Lung

## Requirements

The code is primarily written in MATLAB.

Recommended:
- MATLAB R20XXx or newer
- Image Processing Toolbox
- Signal Processing Toolbox
- Wavelet Toolbox, if using MATLAB built-in thresholding functions such as `wthresh`

Additional requirements may differ between subfolders. See the README or comments inside each folder for details.

## Usage

Each subfolder contains code corresponding to one reconstruction or post-processing workflow. Please refer to the scripts inside each folder for example usage.

Example:

```matlab
cd Denoising_DMD
main_Denoising_DMD
