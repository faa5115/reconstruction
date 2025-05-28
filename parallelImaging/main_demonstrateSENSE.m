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

R =2; % 4;
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





%%
% senseRecon = SENSE_fa1D(imRawUs, sensemap, 2);

% imCalibInput = imRaw;
imCalibInput = imCalib; % ImCalib is the image from the low resolution region of k-space.  In this example, 


senseMapsOption = 1; % this option takes the imCalib images and divide them by their square root sum of squares.  if you make it =2, I use E-SPIRiT to generate the sense maps.  I will discuss E-SPIRiT later. 
noise = []; % I do not have noise data for this example. 

Rinput = [R , 1];
% Rinput is a vector of length 2.  The first index is the acceleration factor along ky and the second dimension is the acceleration along kz.
% this is a 2D image example, so kz is fully sampled.  In this snippet of code R is either 2 or 4.  

tic
[imRawSense, senseMaps] = func_SENSE(imRawUs, imCalibInput, Rinput, noise, senseMapsOption);
toc

%%
figure,


subplot(3, 2, 1)
imshow(abs(imRaw_cc), [])
title('Recon Magnitude of fully sampled image ')

subplot(3, 2, 2)
imshow(angle(imRaw_cc), [-pi, pi])
title('Recon Phase of fully sampled image')

subplot(3, 2, 3)
imshow(abs(imRawUs_cc), [])
title(strcat(num2str(R),'x',32,'reduction recon magnitude'))

subplot(3, 2, 4)
imshow(angle(imRawUs_cc), [-pi, pi])
title(strcat(num2str(R),'x',32,'reduction recon phase'))

subplot(3, 2, 5)
imshow(abs(imRawSense), [])
title(strcat(num2str(R),'x',32,'acceleration sense recon magnitude'))

subplot(3, 2, 6)
imshow(angle(imRawSense), [-pi, pi])
title(strcat(num2str(R),'x',32,'acceleration sense recon phase'))
%%

imSenseChannels = zeros(Nx, Ny, Nz, Nc);

for chiter = 1 : Nc
    senseMapiter = senseMaps(:, :, :, chiter);
    imSenseChannels(:, :, :, chiter) = ...
        senseMapiter .* imRawSense;
end

%% show imSenseChannels

figure,
for chiter = 1 : Nc
    subplot(2, 4, chiter)
    imshow(abs(imSenseChannels(:, :, :, chiter)), [])
end

figure,
for chiter = 1 : Nc
    subplot(2, 4, chiter)
    imshow(angle(imSenseChannels(:, :, :, chiter)), [-pi, pi])
end
%% show original channels

figure,
for chiter = 1 : Nc
    subplot(2, 4, chiter)
    imshow(abs(imRaw(:, :, :, chiter)), [])
end

figure,
for chiter = 1 : Nc
    subplot(2, 4, chiter)
    imshow(angle(imRaw(:, :, :, chiter)), [-pi, pi])
end