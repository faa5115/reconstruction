function [ kernels, kernels_ones, kernels_solve] = func_generate_kernels(kernelShape,kSolveIndices)
% Just a lines of code i repeat frequently ... so i decided to make them
% into a function.  
% 
% Something to know:  Nkx, Nky, Nkz, numKernelShapes refer to number of
% elements along kx, ky, kz, in a kernel, and the number of kernel shapes respetively. 
% kernelShape has size:  Nkx, Nky, Nkz, numKernelShapes
%
% kSolveIndices tell us where the target index in the kernel is ... 3 x numKernelElements
% element array saying the location along each kernel direction (kSolveX,
% kSolveY, kSolveZ) for each numKernelShapes.  
% kSolveIndices: 3 x numKernelShapes where each 3 entries is kSolveX, kSolveY, and kSolveZ
%
% FYI numkernelShapes refers to the different shapes when solving for a
% kernel ... Example for a 1D case (* - acquired, o for empty, x for empty target):
%  *ox* and *xo*. 
% 
% Outputs:  
% kernels (Nkx x Nky x Nkz x Nk x numKernelShapes):  
%       each (:, :, :, iter1, iter2) corresponds to a 1 at each location
%       and 0 everywhere else ... or zero everywhere if iter1 refers
%       to a location that does not have a source.  
%       This makes it easy to convolve using matlab's convolve function. 
% kernels_ones (Nkx x Nky x Nkz x Nk):  just a one for each 
%       (:, :, :, iter1) location. 
% kernels_solve (Nkx x Nky x Nkz x Nk x numKernelShapes):  same as kernels,
%       only having 1s at target locations and zeros everywhere else. 

    Nkx             = size(kernelShape, 1);
    Nky             = size(kernelShape, 2);
    Nkz             = size(kernelShape, 3);
    numKernelShapes = size(kSolveIndices, 2);
    kSize        = [Nkx, Nky, Nkz];
    Nk = prod(kSize); %you'll need this to prepare your kernel for the convolution.  


    kernelShapeOnes = ones(Nkx, Nky, Nkz); %just ones to indicate the shape. 
    kernelSolve = zeros(Nkx, Nky, Nkz, numKernelShapes); %This will just have a 1 for each numKernelShapes to indicate the target.
    

    

    kernels       = zeros(Nkx, Nky, Nkz, Nk, numKernelShapes);
    kernels_ones  = zeros(Nkx, Nky, Nkz, Nk); %doesn't need numKernelShapes because it only defines the span of the kernel.
    kernels_solve = zeros(Nkx, Nky, Nkz, Nk, numKernelShapes);

    for kernelShapeIter = 1 : numKernelShapes

        kSolveX = kSolveIndices(1, kernelShapeIter);
        kSolveY = kSolveIndices(2, kernelShapeIter);
        kSolveZ = kSolveIndices(3, kernelShapeIter);

        kernelSolve(kSolveX, kSolveY, kSolveZ, kernelShapeIter) = 1;
        for kernelIter = 1 : Nk
            kernelShapeInput = kernelShape(:, :, :, kernelShapeIter);
            kInput = zeros(Nkx, Nky, Nkz);
            kInput(kernelIter) = kernelShapeInput(kernelIter);
            kernels(:, :, :, kernelIter, kernelShapeIter) = reshape(kInput, [Nkx, Nky, Nkz]);
            
            kernelSolveInput = kernelSolve(:, :, :, kernelShapeIter);
            kSolveInput = zeros(Nkx, Nky, Nkz);
            kSolveInput(kernelIter) = kernelSolveInput(kernelIter);
            kernels_solve(:, :, :, kernelIter, kernelShapeIter) = reshape(kSolveInput, [Nkx, Nky, Nkz]);
        end
    end

    for kernelIter = 1 : Nk
        kOneInput = zeros(Nkx, Nky, Nkz);
        kOneInput(kernelIter) = kernelShapeOnes(kernelIter);
        kernels_ones(:, :, :, kernelIter) = reshape(kOneInput, [Nkx, Nky, Nkz]);
    end

end