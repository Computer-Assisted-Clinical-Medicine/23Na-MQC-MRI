%% MQC Denoising and DMD Signal Separation
% This script loads MQC raw k-space data, performs low-rank SVD denoising,
% applies Dynamic Mode Decomposition (DMD) across phase-cycling dimensions,
% and visualizes representative denoising and signal-separation results.
% Paper:
% Licht, C., Ilicak, E., Boada, F. E., Guye, M., Zöllner, F. G., Schad, L. R.,
% & Rapacchi, S. (2025). A noise-robust post-processing pipeline for accelerated 
% phase-cycled 23Na Multi-Quantum Coherences MRI. Zeitschrift für Medizinische Physik, 
% 35(1), 98-108. https://www.sciencedirect.com/science/article/pii/S093938892400117X

clear; clc;

%% Parameters

TE_vec = 1:10;

svdThresh = 0.07;     % Hard threshold as fraction of max singular value
svdRank   = 15;       % Number of singular vectors retained by svds

dmdRank = 4;          % DMD truncation rank
dt      = 3.3e-3;     % Approx. echo spacing / sampling interval in seconds

sliceToShow = 7;

%% Load Data

load('rawdata_MQC_Xi0.mat');
load('rawdata_MQC_Xi90.mat');

raw_Xi90 = double(rawdata_MQC_Xi90);
ksz = size(raw_Xi90);

% Convert raw k-space data to image space.
img_Xi90 = ifftc(ifftc(ifftc(raw_Xi90, 1), 2), 3);

%% Low-Rank SVD Denoising

% Reshape the data into a 2D matrix:
% second dimension is highly redundant (TE + phase-cycling)!
% rows = spatial voxels, columns = TE/phase-cycling measurements.
Xraw = reshape(raw_Xi90, [prod(ksz(1:3)), 1 * 6 * 10]);

[U, S, V] = svds(Xraw, svdRank);

% Hard-threshold singular values.
singVals = diag(S);
singValsThresh = wthresh(singVals, 'h', max(singVals) * svdThresh);

% Reconstruct denoised k-space and image-space data.
myk_SVD = reshape(U * diag(singValsThresh) * V', [ksz(1:4), 1 * 6]);

myimg_SVD = ifftc(ifftc(ifftc(myk_SVD, 1), 2), 3);

%% Signal Separation with DMD

% DMD is computed across the phase-cycling dimension for each TE.
DMDvarin = {'dt', dt, 'r', dmdRank};

Phi_all = zeros([ksz(1:3), dmdRank, numel(TE_vec)]);

for ii = 1:numel(TE_vec)
    XdmdInput = reshape( ...
        squeeze(myk_SVD(:, :, :, ii, :)), ...
        [prod(ksz(1:3)), 6] ...
    );

    [Phi, omega, lambda, b, freq, Xdmd, r] = DynamicModeDecomp(XdmdInput, DMDvarin);

    Phi_all(:, :, :, :, ii) = reshape(Phi, [ksz(1:3), dmdRank]);
end

% Convert DMD modes from k-space to image space.
Phi_final = abs(ifftc(ifftc(ifftc(permute(Phi_all, [1, 2, 3, 5, 4]), 1), 2), 3));

%% Plot Representative Results

hfig = figure('Color', 'w');
tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% Denoising comparison.
nexttile;
imagesc(abs(img_Xi90(:, :, sliceToShow, 1, 1)));
title('No denoising');
formatImageAxes;

nexttile;
imagesc(abs(myimg_SVD(:, :, sliceToShow, 3, 4)));
title('With SVD denoising');
formatImageAxes;

% DMD-separated components.
nexttile;
sqMode = sqrt( ...
    Phi_final(:, :, sliceToShow, 1, 1).^2 + ...
    Phi_final(:, :, sliceToShow, 1, 3).^2 ...
);
imagesc(sqMode);
title('SQ, TE = 1');
formatImageAxes;

nexttile;
imagesc(Phi_final(:, :, sliceToShow, 3, 4));
title('TQ, TE = 3');
formatImageAxes;

colormap gray;

%% Local Helper Function

function formatImageAxes
    axis image off;
    colorbar;
end