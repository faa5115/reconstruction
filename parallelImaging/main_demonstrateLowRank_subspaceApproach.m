% this demonstrates how a low rank subspace approach is done. 

load('whitened2DACR.mat') % whitened acr data
load('whitenedNoise4ACRData.mat') % whitened noise data
% The noise data was acquired using the same readout bandwidth as the
% image.  We use this data to identify the null space. 
rawog = rawW; clear rawW; % rawog refers to "original"/fully sampled raw data.
[Nx, Ny, Nz, Nc, Np] = size(rawog);

% prepare calibration/reference data.  called "calib"
refLines = 24;
calib = rawog(:, Ny/2 -  refLines/2 : Ny/2 + refLines/2 - 1, :, :, :);

R = 6; % acceleration factor.  we will need 2 kernel shapes.
raw = zeros(size(rawog));
raw(:, 1 : R : end, :, :) = rawog(:, 1 : R : end, :, :);


%% Show rawog images. 
imRaw = zeros(size(rawog));
imRaw6x = zeros(size(rawog));

for chiter = 1 : Nc
    imRaw(:, :, :, chiter) = ifftnc(rawog(:, :, :, chiter));
    imRaw6x(:, :, :, chiter) = ifftnc(raw(:, :, :, chiter));
end

imRawsos = func_sqSOS(imRaw, []);
imRaw6xsos = func_sqSOS(imRaw6x, []);

figure, imshow(abs([squeeze(imRawsos), sqrt(6).*squeeze(imRaw6xsos)]), [])
%% Prepare recon parameters

% we go from multichannel data (raw in this example) to a structured
% autocalibration matrix, which we call "A" in the recon code (which is
% called func_UNCLE_SAM_NMR) by convolving with a kernel.  This results 
% in A having repeated entries of raw.  This reconstruction is an iterative
% procedure which involves going from raw -> A.  After imposing that A is
% low rank, we go from A -> raw to force all repeated entries of raw in A
% are the same. This backward procedure is time consuming, so I create a
% file for each matrix called idxM (idk what my logic was behind the name
% ... i think id referred to "indices") which makes it quick to go from A
% -> raw. the creation of this matrix only requires the size of the raw
% data (Nx x Ny x Nz x Nc) and the size of the kernel (Nkx x Nky x Nkz).
% You can save it for later (which I do in the folder "idxM") to use for
% other datasets of that size. 


% size of the recon kernel.
Nkx = 3; Nky = 3; Nkz = 1; 
kSize = [Nkx, Nky, Nkz];

% if you do not have an idxM matrix for the size of raw and kernel:
idxM = func_make_idxM(raw, kSize);




%%

nIter = 1000; % max number of iterations. 
bUseGPU = 0;



[rawUpdate, f, Scalib, normValues] = ...
    func_UNCLE_SAM_outputSV_mv_conv(raw_og, kSize, ...
    nIter, calib, bUseGPU, idxM, noise);
% f shows the shrinkage function, which depends on input noise.
% Scalib gives the singular values of the calibration data ( we make the 
% matrix Acal by convolving the kenrel with the calibration data). f is applied to
% these singular valures. 
% normValues gives you the normalized Frobenius norm for each iteration to
% show convergence. 

imRawUpdate = zeros(size(rawUpdate));
for chiter = 1 : Nc
    imRawUpdate(:, :, :, chiter) = ...
        ifftnc(rawUpdate(:, :, :, chiter));
end

imRawUpdatesos = func_sqSOS(imRawUpdate, []);


figure, imshow(abs([squeeze(imRawsos), ...
    sqrt(6).*squeeze(imRaw6xsos), imRawUpdatesos]), [])

figure,
plot(Scalib, 'linewdith', 5.0)
hold on
plot(f, 'linewidth', 5.0)
title('show Acal singular values and the minimum variance shrinkage function')

figure,
plot(normValues, 'linewidth', 5.0)
title('show the convergence')
