function [ rawRecon] = func_complete_grappa_recon(rawUs, calib, kernelShapeStruct, kSolveIndicesStruct)
% func_grappa_recon(rawUs, calib, kernelShape, kSolveIndices)
% This is generalized for cases where the kernels have different 3D shapes
% ...
% For example ... consider the following sampling pattern: 
% x 0 x 0 x 0 x 0 x 0 x 0
% 0 0 0 0 0 0 0 0 0 0 0 0 
% x 0 x 0 x 0 x 0 x 0 x 0
% 0 0 0 0 0 0 0 0 0 0 0 0
% x 0 x 0 x 0 x 0 x 0 x 0
% 0 0 0 0 0 0 0 0 0 0 0 0
% x 0 x 0 x 0 x 0 x 0 x 0kernelShapeStruct
% Three different ky x kz kernels will be necessary: 
% the first kernel: 
% x 0 x
% 0 T 0
% x 0 x
% the second kernel: 
% 0 x 0
% 0 T 0
% 0 x 0
% the second kernel: 
% 0 0 0
% x T x
% 0 0 0
% The first shape hasa four nonzero entries and the second just two.  my
% previous GRAPPA func_getWeights and func_grappa_recon functions works
% where the kernel shapes all have the same number of nonzero entries. We
% do that so that for a given kernel the nonzero entries can be vectorized,
% and then we can carry out the solution for the missing k-space entries by
% multiplying the kernel (in vector form) by a matrix (that consists only
% of the source elements).  We cannot do that with the kernel shapes having
% different sizes.  so we should have a separate variable, called "kernel
% structure."  
% So let's make this clear:  "shape" refers to the location
% of the target entry.  in this case, different target entries are
% estimated from the same source of a given shape. 
% such as: 
% shape number 1
% x x x
% 0 0 0
% 0 T 0
% x x x
 % shape number 2
 % x x x
 % 0 T 0
 % 0 0 0
 % x x x
% "structure" refers to the arrangement of the sources.  kernelShapeStruct
% kernelShapeStruct has size Nkx, Nky, Nkz, numKernelShapes, numKernelStructs. 
% kSolveIndices is of size 3 (kx, ky, kz) x numKernelShapes x numKernelStructs
rawUs = single(rawUs); calib = single(calib); 
[Nx, Ny, Nz, Nc] = size(rawUs);
[Nxc, Nyc, Nzc, ~] = size(calib);
[Nkx  Nky, Nkz, numKernelShapes, numKernelStructs] = size(kernelShapeStruct);

acquiredElements = find(rawUs);
rawRecon = rawUs;
for structiter = 1 : numKernelStructs
    disp(strcat('start structure',32,num2str(structiter)))
    kernelShape = kernelShapeStruct(:, :, :, :, structiter);
    kSolveIndices = kSolveIndicesStruct(:, :, structiter);

    [rawReconiter, ~] = func_grappa_recon(rawUs, calib, kernelShape, kSolveIndices);
    rawReconiterWithoutSampledEntries = zeros(Nx, Ny, Nz, Nc);
    rawReconiterWithoutSampledEntries(find(rawUs == 0)) = rawReconiter(find(rawUs == 0));
    rawReconiterWithoutSampledEntries = reshape(rawReconiterWithoutSampledEntries, [Nx, Ny, Nz, Nc]);
    rawRecon(find(rawReconiterWithoutSampledEntries)) = rawReconiterWithoutSampledEntries(find(rawReconiterWithoutSampledEntries));
    clear rawReconiter rawReconiterWithoutSampledEntries
end
rawRecon = reshape(rawRecon, [Nx, Ny, Nz, Nc]);