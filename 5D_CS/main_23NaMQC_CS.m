%% Compressed Sensing for 23Na MQC MRI
% main
% Author:   Christian Licht
% Date:     22/11/2023
% Computer Assisted Clinical Medicine, Medical Faculty Mannheim, Heidelberg
% University, Mannheim, Germany

%% load data
close all; clear; clc
% myk0, myk90:  raw data of 23Na MQC MRI with [x,y,z,TE,pc]
%               x: phase encoding, y: frequency encoding, z: partitions,
%               TE: echo time, pc=phase-cycling
% mask_R3:      Undersampling mask for R=3 and alternated along phase-cycling
load('myk0')        
load('myk90')
load('mask_R3')

%% Reconstruction fully sampled and zero-filled
% Quadrature combination used: data_a+i*data_b
% fs: fully sampled; us: undersampled; zf: zero-filling ifft reconstruction
myk_fs = ifft3c(myk0)+i*ifft3c(myk90);
MQC_spectrum_fs = abs(fftc(myk_fs,5));  % Fourier transform computed along phase-cycling to obtain spectrum

% Perform undersampling
myk0_us = myk0.*mask_R3;
myk90_us = myk90.*mask_R3;

myk_zf = ifft3c(myk0_us)+i*ifft3c(myk90_us);
MQC_spectrum_zf = abs(fftc(myk_zf,5));

%% 5D CS reconstruction
% Sparsity trhesholds for spatial (xyz), TE and phase-cycling (pc)
beta_xyz = 0.3;
beta_TE = 0.5;
beta_pc = 1.6;
use_gpu = 1;    % 1: to compute on GPU, 0: CPU
[myk0_5DCS]=SBM_23NaMQC_CS(use_gpu, myk0_us, 1, 300,  beta_xyz, beta_TE, beta_pc);
[myk90_5DCS]=SBM_23NaMQC_CS(use_gpu, myk90_us, 1, 300,  beta_xyz, beta_TE, beta_pc);

myk_CS = myk0_5DCS+i*myk90_5DCS;
MQC_spectrum_CS = abs(fftc(myk_CS,5));

%% Plots
hfig=figure;
figwidth = 15;
figratio=.6
slice=5;
tiledlayout(2,3,'TileSpacing','Compact');

% SQ
n1=nexttile;
im1=imagesc(MQC_spectrum_fs(:,:,slice,1,3))
title('FS')
ylabel('SQ')
MakeImScNice(gca)
caxis([min(min(MQC_spectrum_fs(:,:,slice,1,3))) ...
    max(max(MQC_spectrum_fs(:,:,slice,1,3)))])

n2=nexttile;
imagesc(MQC_spectrum_zf(:,:,slice,1,3))
title('ZF')
MakeImScNice(gca)
caxis([min(min(MQC_spectrum_fs(:,:,slice,1,3))) ...
    max(max(MQC_spectrum_fs(:,:,slice,1,3)))])

n3=nexttile;
imagesc(MQC_spectrum_CS(:,:,slice,1,3))
title('5D CS')
MakeImScNice(gca)
caxis([min(min(MQC_spectrum_fs(:,:,slice,1,3))) ...
    max(max(MQC_spectrum_fs(:,:,slice,1,3)))])

% TQ
n4=nexttile;
imagesc(MQC_spectrum_fs(:,:,slice,3,1))
ylabel('TQ')
MakeImScNice(gca)
caxis([min(min(MQC_spectrum_fs(:,:,slice,3,1))) ...
    max(max(MQC_spectrum_fs(:,:,slice,3,1)))])

n5=nexttile;
imagesc(MQC_spectrum_zf(:,:,slice,3,1))
MakeImScNice(gca)
caxis([min(min(MQC_spectrum_fs(:,:,slice,3,1))) ...
    max(max(MQC_spectrum_fs(:,:,slice,3,1)))])

n6=nexttile;
imagesc(MQC_spectrum_CS(:,:,slice,3,1))
MakeImScNice(gca)
caxis([min(min(MQC_spectrum_fs(:,:,slice,3,1))) ...
    max(max(MQC_spectrum_fs(:,:,slice,3,1)))])

SetFigProps(hfig,figwidth,figratio); % sets figure properties

