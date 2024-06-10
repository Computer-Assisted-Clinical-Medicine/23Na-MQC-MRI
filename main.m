%% Low-rank reconstruction for simultaneous double half-echo 23Na and undersampled 23Na multi-quantum coherences MRI

% Licht C, Reichert S, Bydder M, et al. Low-rank reconstruction for simultaneous double half-echo 23Na 
% and undersampled 23Na multi-quantum coherences MRI. Magn Reson Med. 2024; 1-16. doi: 10.1002/mrm.30132

% This script provides an example on how to reconstruct phantom double half-echo
% sodium and undersampled sodium multi-quantum coherences MRI by leveraging
% two low-rank frameworks (DHE and SAKE). Please see the following papers for details:
% (1) DHE: Bydder M, Ali F, Ghodrati V, Hu P, Yao J, Ellingson BM. Minimizing echo and repetition times in magnetic resonance imaging using a double half-echo k-space acquisition and low-rank reconstruction. NMR Biomed. 2021 Apr;34(4):e4458. doi: 10.1002/nbm.4458. Epub 2020 Dec 9. PMID: 33300182; PMCID: PMC7935763.
% (2) SAKE: Shin PJ, Larson PE, Ohliger MA, Elad M, Pauly JM, Vigneron DB, Lustig M. Calibrationless parallel imaging reconstruction based on structured low-rank matrix completion. Magn Reson Med. 2014 Oct;72(4):959-70. doi: 10.1002/mrm.24997. Epub 2013 Nov 18. PMID: 24248734; PMCID: PMC4025999.

% The used implementations can be found here:
% https://github.com/marcsous/parallel

clear
close all

%% Load data for DHE sodium MRI
% This contains the phantom data
% Conventional Cartesian sodium MRI based on the DHE technique
load('rawdata_DHE_rev')
load('rawdata_DHE_forw')
   
%% Do DHE low-rank reconstruction
imsz=size(rawdata_DHE_rev);

% Do the low-rank reconstruction slice-by-slice
% No sparsity, for sparsity please make sure you have access to the Wavelet
% toolbox
for z = 1:size(rawdata_DHE_rev,3)
[ksp_2D, basic_2D, norms_2D, opts_2D] = ...
    DHE_Reconstruction(rawdata_DHE_rev(:,:,z),rawdata_DHE_forw(:,:,z));
    ksp_2D_z(:,:,:,z) = squeeze(gather((ksp_2D)));
end

% ifft to reverse fft prior to reconstruction (reconstruction is done
% slice-wise); ifftc is a simple ifft with fftshift operations
ksp_2D_final = ifftc(permute(ksp_2D_z,[1 2 4 3]),3);

ksp_2D=reshape(squeeze(gather(squeeze(ksp_2D_final))),size(ksp_2D_final,1),imsz(2),imsz(3),2);
% SodiumImg_2Dreco=ifft3c_new((ksp_2D(:,:,:,:,:)));
SodiumImg_2Dreco=ifftc(ifftc(ifftc(ksp_2D,1),2),3);

% Combines forward and reverse halves
SodiumImg_2Dreco_combd = my_coilcombine2(cat(4,SodiumImg_2Dreco(:,:,:,1),flip(SodiumImg_2Dreco(:,:,:,2),1)),4);

%% Load data for Sodium Multi-Quantum Coherences MRI
load('rawdata_MQC_Xi90')    % phantom data Xi90
load('rawdata_MQC_Xi0')     % phantom data Xi0
load('Sampling_mask')       % Undersampling mask

%% Compute fully sampled SQ and TQ images
img_Xi90 = ifftc(ifftc(ifftc(rawdata_MQC_Xi90,1),2),3);
img_Xi0 = ifftc(ifftc(ifftc(rawdata_MQC_Xi0,1),2),3);
img_Xi_combd = fftc(img_Xi0 +i*img_Xi90,5); % compute fft along phase-cycling to obtain spectrum
SQ_FullySampled = my_coilcombine2(cat(5,img_Xi_combd(:,:,:,:,3),img_Xi_combd(:,:,:,:,5)),5);
TQ_FullySampled = img_Xi_combd(:,:,:,:,1);

%% Reconstruct undersampled data via SAKE
rawdata_MQC_Xi90_US = rawdata_MQC_Xi90.*Sampling_mask;
rawdata_MQC_Xi0_US = rawdata_MQC_Xi0.*Sampling_mask;
rawdata_US_combd = cat(5,rawdata_MQC_Xi90_US,rawdata_MQC_Xi0_US);
sz=size(rawdata_US_combd);
% Do the reconstruction leveraging SAKE
rawdata_Reconstructed_combd = gather(SAKE_Reco(reshape(rawdata_US_combd,[sz(1) sz(2) sz(3) sz(4)*sz(5)])));
rawdata_Reconstructed_final = double(reshape(rawdata_Reconstructed_combd,[sz(1),sz(2),sz(3),sz(4),sz(5)]));

%% Compute SQ and TQ of SAKE reconstructed data
img_Xi90_Reco = ifftc(ifftc(ifftc(rawdata_Reconstructed_final(:,:,:,:,1:6),1),2),3);
img_Xi0_Reco = ifftc(ifftc(ifftc(rawdata_Reconstructed_final(:,:,:,:,7:12),1),2),3);
img_Xi_Reco_combd = fftc(img_Xi0_Reco +i*img_Xi90_Reco,5); % compute fft along phase-cycling to obtain spectrum
SQ_US_Reco = my_coilcombine2(cat(5,img_Xi_Reco_combd(:,:,:,:,3),img_Xi_Reco_combd(:,:,:,:,5)),5);
TQ_US_Reco = img_Xi_Reco_combd(:,:,:,:,1);

%%
% Data structure:
% SodiumImg_2Dreco: [x,y,z,forw/rev]
% TQ/SQ: [x,y,z,TE,phase-cycle]

hfig=figure;
figwidth=25
figratio=0.7

tiledlayout(2,3,'TileSpacing','compact');

n1=nexttile
imagesc(abs(SodiumImg_2Dreco(:,:,25,1)))
colormap(gray)
set(gca,'xtick',[])
set(gca,'xticklabel',[])
set(gca,'ytick',[])
set(gca,'yticklabel',[])
title({'Sodium Image','DHE reverse half'})

n2=nexttile
imagesc(abs(SQ_FullySampled(:,:,7,1)))
colormap(gray)
set(gca,'xtick',[])
set(gca,'xticklabel',[])
set(gca,'ytick',[])
set(gca,'yticklabel',[])
title({'SQ Fully Sampled','R=1'})
caxis([min(min(abs(SQ_FullySampled(:,:,7,1)))),...
    max(max(abs(SQ_FullySampled(:,:,7,1))))])

n3=nexttile
imagesc(abs(TQ_FullySampled(:,:,7,3)))
colormap(gray)
set(gca,'xtick',[])
set(gca,'xticklabel',[])
set(gca,'ytick',[])
set(gca,'yticklabel',[])
title({'TQ Fully Sampled','R=1'})
caxis([min(min(abs(TQ_FullySampled(:,:,7,3)))),...
    max(max(abs(TQ_FullySampled(:,:,7,3))))])

n4=nexttile
imagesc(abs(SodiumImg_2Dreco(:,:,25,2)))
colormap(gray)
set(gca,'xtick',[])
set(gca,'xticklabel',[])
set(gca,'ytick',[])
set(gca,'yticklabel',[])
title({'Sodium Image','DHE forward half'})

n5=nexttile
imagesc(abs(SQ_US_Reco(:,:,7,1)))
colormap(gray)
set(gca,'xtick',[])
set(gca,'xticklabel',[])
set(gca,'ytick',[])
set(gca,'yticklabel',[])
title({'SQ SAKE reco','R=3'})
caxis([min(min(abs(SQ_FullySampled(:,:,7,1)))),...
    max(max(abs(SQ_FullySampled(:,:,7,1))))])

n6=nexttile
imagesc(abs(TQ_US_Reco(:,:,7,3)))
colormap(gray)
set(gca,'xtick',[])
set(gca,'xticklabel',[])
set(gca,'ytick',[])
set(gca,'yticklabel',[])
title({'TQ SAKE reco','R=3'})
caxis([min(min(abs(TQ_FullySampled(:,:,7,3)))),...
    max(max(abs(TQ_FullySampled(:,:,7,3))))])

% Some settings to make the figure look nicer :)
picturewidth = figwidth;
hw_ratio =  figratio;
set(findall(hfig,'-property','FontSize'),'FontSize',18)
set(findall(hfig,'-property','Box'),'Box','off')
set(findall(hfig,'-property','Box'),'DefaultFigureColor','w')
set(findall(hfig,'-property','Interpreter'),'Interpreter','latex') 
set(findall(hfig,'-property','TickLabelInterpreter'),'TickLabelInterpreter','latex')
set(hfig,'Units','centimeters','Position',[1 1 picturewidth hw_ratio*picturewidth])
pos = get(hfig,'Position');
