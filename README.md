# 23Na-MQC-MRI
Contains the code for image reconstruction of Double Half-Echo 23Na and accelerated 23Na MQC MRI as presented in:
1) Licht C, Reichert S, Bydder M, et al. Low-rank reconstruction for simultaneous double half-echo 23Na and undersampled 23Na multi-quantum coherences MRI. Magn Reson Med. 2024; 1-16. doi: 10.1002/mrm.30132

The code is based on the implementations of Mark Bydder and can be found here:
https://github.com/marcsous/parallel 
   
Contains the code for low-rank based post-processing pipeline for efficient denoising as well as SQ and TQ signal separation based on Dynamic Mode Decomposition (DMD):

2) Licht, C., Ilicak, E., Boada, F. E., Guye, M., Zöllner, F. G., Schad, L. R., & Rapacchi, S. (2025). A noise-robust post-processing pipeline for accelerated phase-cycled 23Na Multi-Quantum Coherences MRI. Zeitschrift für Medizinische Physik, 35(1), 98-108. https://www.sciencedirect.com/science/article/pii/S093938892400117X

The DMD code is based on the implementations of Efe Ilicak and can be found here:
https://github.com/Computer-Assisted-Clinical-Medicine/DMD_Lung

Please see the following papers for details:

(1) Bydder M, Ali F, Ghodrati V, Hu P, Yao J, Ellingson BM. Minimizing echo and repetition times in magnetic resonance imaging using a double half-echo k-space acquisition and low-
rank reconstruction. NMR Biomed. 2021;34(4):e4458. doi:10.1002/nbm.4458.

(2) Bydder M, Ali F, Saucedo A, Ghodrati V, Samsonov A, Akhtari M, Wang C, Hagiwara A, Yao J, Ellingson BM. Low-rank off-resonance correction for double half-echo k-space        acquisitions. Magn Reson Imaging. 2022;94:43–47. doi:10.1016/j.mri.2022.08.017.

(3) Shin PJ, Larson PEZ, Ohliger MA, Elad M, Pauly JM, Vigneron DB, Lustig M. Calibrationless parallel imaging reconstruction based on structured low-rank matrix completion. Magn Reson Med. 2014;72(4):959–970. doi:10.1002/mrm.24997.

(4) Bydder M, Du J. Noise reduction in multiple-echo data sets using singular value decomposition. Magn Reson Imaging. 2006;24(7):849–856. doi:10.1016/j.mri.2006.04.003.

(5) Schmid PJ. Dynamic mode decomposition of numerical and experimental data. J Fluid Mech. 2010;656:5–28. doi:10.1017/S0022112010001217.
