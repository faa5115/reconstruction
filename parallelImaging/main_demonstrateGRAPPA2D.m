load('whitened2DACR.mat') % whitened acr data
raw = rawW; clear rawW;
[Nx, Ny, Nz, Nc, Np] = size(raw);

% prepare calibration/reference data.  called "calib"
refLines = 24;
calib = raw(:, Ny/2 -  refLines/2 : Ny/2 + refLines/2 - 1, :, :, :);

R = 3; % acceleration factor.  we will need 2 kernel shapes.
raw3x = zeros(size(raw));
raw3x(:, 1 : R : end, :, :) = raw(:, 1 : R : end, :, :);

%% show images of raw, calib, and raw3x
imRaw = zeros(size(raw));
imCalib = zeros(size(calib));
imRaw3x = zeros(size(raw3x));

for chiter = 1 : Nc
    imRaw(:, :, :, chiter) = ifftnc(raw(:, :, :, chiter));
    imCalib(:, :, :, chiter) = ifftnc(calib(:, :, :, chiter));
    imRaw3x(:, :, :, chiter) = ifftnc(raw3x(:, :, :, chiter));
end

imRawsos = func_sqSOS(imRaw, []);
imCalibsos = func_sqSOS(imCalib, []);
imRaw3xsos = func_sqSOS(imRaw3x, []);
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
kernelShape3x = zeros(Nkx, Nky, Nkz, numKernelShapes); 

% indicating that the edge along ky are the sources.
kernelShape3x(:, 1, 1, :) = 1;
kernelShape3x(:, R+1, 1, :) = 1;

% kSolveIndices tells you the indices along each direction where the
% target.  In this example, kernel shape 1 has its target at [2, 2, 1] and
% kernel shape 2 has its target at [2, 3, 1]. 
kSolveIndices3x = zeros(3, R-1);  % 3 : kx, ky, kz.  7 refers to 8-1 different targets.
kSolveIndices3x(1, 1) = 2; kSolveIndices3x(2, 1) = 2; kSolveIndices3x(3, 1) = 1;
kSolveIndices3x(1, 2) = 2; kSolveIndices3x(2, 2) = 3; kSolveIndices3x(3, 2) = 1;

%% GRAPPA reconstruction.
 [recon3x, ~] =  ...
        func_grappa_recon(raw3x(:, :, :, :), ...
                          calib(:, :, :, :), ...
                          kernelShape3x, kSolveIndices3x);
imRecon3x = zeros(size(recon3x));
for chiter = 1 : Nc
    imRecon3x(:, :, :, chiter) = ifftnc(recon3x(:, :, :, chiter));
end

imRecon3xsos = func_sqSOS(imRecon3x, []);

figure, 
imshow(abs(imRecon3xsos), [])


