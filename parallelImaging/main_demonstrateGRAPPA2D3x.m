%% Purpose:
% This is just to demonstrate how SENSE works. 
% This is just done for simple 2D Cartesian imaging.  therefore the
% unaliasing is just along 1D, the phase-encoding direction. 
%%
% load('brain_8ch.mat'); NcSets = 1; % number of channel sets. 
% load('brain_alias_8ch.mat'); NcSets = 2; % number of channel sets. (for
% e-spirit ... see my "simplified E-SPIRit" code.  

% load whitened raw data and whitened noise
 % raw = load('whitened2DACR.mat');
 % raw = raw.rawW;
 % noise = load('whitenedNoise4ACRData.mat');
 % noise = noise.noiseMatw;
 load('brain_8ch.mat');

[Nx, Ny, Nz, Nc] = size(raw);

R =3; % 4;
rawUs = zeros(size(raw));
rawUs(:, 1 : R : end, :, :) = raw(:, 1 : R : end, :, :);

calib =  raw(:, round(Ny/2) - 14 : round(Ny/2) + 13, :, :);
calibPad = zeros(size(raw));
calibPad(:, round(Ny/2) - 14 : round(Ny/2) + 13, :, :) = raw(:, round(Ny/2) - 14 : round(Ny/2) + 13, :, :);
%% Show the original (fully sampled) and undersampled (US) images. 
% We must coil combine them first.  I use Walsh's method.  See my walsh
% implementation in the "coilCombine" directory. 
Npatchx = 15; Npatchy = 15; Npatchz = 1; Npatchm = 1;

imRaw         = zeros(size(raw        ));
imRawUs       = zeros(size(raw        ));
imCalibPad    = zeros(size(raw        ));
imCalib       = zeros(size(calib      ));
% imDiff        = zeros(size(raw        ));
for channelIter = 1 : Nc
    imRaw        (:, :, :, channelIter) = ifftnc(raw        (:, :, :, channelIter));
    imRawUs      (:, :, :, channelIter) = ifftnc(rawUs      (:, :, :, channelIter));
    imCalib      (:, :, :, channelIter) = ifftnc(calib      (:, :, :, channelIter));
    imCalibPad   (:, :, :, channelIter) = ifftnc(calibPad   (:, :, :, channelIter));
end

[imRaw_cc     ] = func_WalshMethod(imRaw     ,        [], [Npatchx, Npatchy, Npatchz, Npatchm]);
[imRawUs_cc   ] = func_WalshMethod(imRawUs   ,        [], [Npatchx, Npatchy, Npatchz, Npatchm]);
[imCalib_cc   ] = func_WalshMethod(imCalib   ,        [], [Npatchx, Npatchy, Npatchz, Npatchm]);
[imCalibPad_cc] = func_WalshMethod(imCalibPad,        [], [Npatchx, Npatchy, Npatchz, Npatchm]);
imRaw_sos      = func_sqSOS(imRaw  , []);
imRawUs_sos    = func_sqSOS(imRawUs, []);
imCalib_sos    = func_sqSOS(imCalib, []);
imCalibPad_sos = func_sqSOS(imCalibPad, []);

mask_im   = imRaw_cc   > 0.06 * max(imRaw_cc  (:));
mask_imus = imRawUs_cc > 0.06 * max(imRawUs_cc(:));

%% Display the full sampled image, low resolution calibration image, and the fully sampled image
figure,
subplot(3, 3, 1)
imshow(abs(imRaw_cc), [])
title('original fully sampled image')
subplot(3, 3, 2)
imshow(angle(imRaw_cc), [-pi, pi])
title('original fully sampled phase')
subplot(3, 3, 3)
imshow(log(abs(raw(:, :, 1, 1))), [])
title('original fully sampled k-space')



subplot(3, 3, 4)
imshow(abs(imCalibPad_cc), [])
title('Low resolution image')
subplot(3, 3, 5)
imshow(angle(imCalibPad_cc), [-pi, pi])
title('Low resolution phase')
subplot(3, 3, 6)
imshow(log(abs(calibPad(:, :, 1, 1))), [])
title('Low resolution k-space')

subplot(3, 3, 7)
imshow(abs(imRawUs_cc), [])
title('Undersampled image')
subplot(3, 3, 8)
imshow(angle(imRawUs_cc), [-pi, pi])
title('Undersampled phase')
subplot(3, 3, 9)
imshow(log(abs(rawUs(:, :, 1, 1))), [])
title('Undersampled k-space')


%% Prepare the kernel. 
% * - refers to acquired points providing the "source"
% 0 - refers to unacquired
% x - refers to unacquired point that is the target of the kernel
% kernel shape 1: 
% * * * 
% 0 x 0
% 0 0 0
% * * *

% kernel shape 1: 
% * * * 
% 0 0 0
% 0 x 0
% * * *

Nkx = 3; % length of kernel along the kx direction.
Nky = R+1; % length of kernel along the ky direction.
Nkz = 1; % length of kernel along the kz direction.
numKernelShapes = R - 1;
% kernelShape is just a binary indicating the source.
% in this example, the kernel is 
kernelShape = zeros(Nkx, Nky, Nkz, numKernelShapes); 

% indicating that the edge along ky are the sources.
kernelShape(:, 1, 1, :) = 1;
kernelShape(:, R+1, 1, :) = 1;

% kSolveIndices tells you the indices along each direction where the
% target.  In this example, kernel shape 1 has its target at [2, 2, 1] and
% kernel shape 2 has its target at [2, 3, 1]. 
kSolveIndices = zeros(3, R-1);  % 3 : kx, ky, kz.  7 refers to 8-1 different targets.
kSolveIndices(1, 1) = 2; kSolveIndices(2, 1) = 2; kSolveIndices(3, 1) = 1;
kSolveIndices(1, 2) = 2; kSolveIndices(2, 2) = 3; kSolveIndices(3, 2) = 1;

%% GRAPPA reconstruction.
 [recon, ~] =  ...
        func_grappa_recon(rawUs(:, :, :, :), ...
                          calib(:, :, :, :), ...
                          kernelShape, kSolveIndices);
imRecon = zeros(size(recon));
for chiter = 1 : Nc
    imRecon(:, :, :, chiter) = ifftnc(recon(:, :, :, chiter));
end

imReconsos = func_sqSOS(imRecon, []);

figure, 
imshow(abs(imReconsos), [])


