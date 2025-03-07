%%  
load('whitened3DACR_2xky_2xkz.mat') % load raw data:  raw3Dus,  Ry = 2 Rz = 2 undersampled data
load('whitened3DCalibACR.mat') % load calibration data:  calib3D, 3d calibration data
[Nx, Ny, Nz, Nc] = size(raw3Dus);
[Nxc, Nyc, Nzc, ~] = size(calib3D);
Ry2 = 2; Rz2 = 2;
%%
imRaw3Dus = zeros(size(raw3Dus));
for chiter = 1 : Nc
    imRaw3Dus(:, :, :, chiter) = ...
        ifftnc(raw3Dus(:, :, :, chiter));
end
imRaw3Dussos = func_sqSOS(imRaw3Dus, []);
%% GRAPPA 2X 2X: Part 2 prepare kernel.
Nkx = 3; Nky = Ry2 + 1; Nkz = Rz2 + 1;
nkernshapesRy2Rz2 = Ry2 * Rz2 - 1;
kernelShapeRy2Rz2 = zeros(Nkx, Nky, Nkz, nkernshapesRy2Rz2);
kSolveIndicesRy2Rz2 = zeros(3, nkernshapesRy2Rz2);

% for Ry2 * Rz2 = 4, there should be three different kernel shapes.
% First kernel shape. vert. directoin is ky, horiz. is kz. kx goes into the screen:
% 1 0 1
% 0 x 0 "x" indicates the target (but is still a zero)
% 1 0 1
kshapeiter = 1;
kernelShapeRy2Rz2(:, 1, 1, kshapeiter) = 1;
kernelShapeRy2Rz2(:, 1, 3, kshapeiter) = 1;
kernelShapeRy2Rz2(:, 3, 1, kshapeiter) = 1;
kernelShapeRy2Rz2(:, 3, 3, kshapeiter) = 1;
kSolveIndicesRy2Rz2(:, kshapeiter) = [round(Nkx/2), round(Nky/2), round(Nkz/2)].';
% Second kernel shape. vert. directoin is ky, horiz. is kz. kx goes into the screen:
% 0 1 0
% 0 x 0 
% 0 1 0
kshapeiter = 2;
kernelShapeRy2Rz2(:, 1, 2, kshapeiter) = 1;
kernelShapeRy2Rz2(:, 3, 2, kshapeiter) = 1;
kSolveIndicesRy2Rz2(:, kshapeiter) = [round(Nkx/2), round(Nky/2), round(Nkz/2)].';
% Third kernel shape. vert. directoin is ky, horiz. is kz. kx goes into the screen:
% 0 0 0
% 1 x 1 
% 0 0 0
kshapeiter = 3;
kernelShapeRy2Rz2(:, 2, 1, kshapeiter) = 1;
kernelShapeRy2Rz2(:, 2, 3, kshapeiter) = 1;
kSolveIndicesRy2Rz2(:, kshapeiter) = [round(Nkx/2), round(Nky/2), round(Nkz/2)].';


%%
kernelShapeRy2Rz2Struct = permute(kernelShapeRy2Rz2, [1 2, 3, 5, 4]);
kSolveIndicesRy2Rz2Struct = permute(kSolveIndicesRy2Rz2, [1, 3, 2]);
[ rawRecon2R2R] = func_complete_grappa_recon(raw3Dus, calib3D, kernelShapeRy2Rz2Struct, kSolveIndicesRy2Rz2Struct);

%%
imRawRecon2R2R = zeros(size(rawRecon2R2R));

for chiter = 1 : Nc
    imRawRecon2R2R(:, :, :, chiter) = ...
        ifftnc(rawRecon2R2R(:, :, :, chiter));
end

imRawRecon2R2Rsos = func_sqSOS(imRawRecon2R2R, []); 

sliter = Nz/2;
figure,
imshow(abs([squeeze(imRawRecon2R2Rsos(:, :, sliter)),...
    squeeze(imRaw3Dussos(:, :, sliter))]), [])

