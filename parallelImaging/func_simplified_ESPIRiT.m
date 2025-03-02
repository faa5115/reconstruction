function [eigValuesM, eigVectorsM,mask_eig, mask_im, harmonicMatrix] = func_simplified_ESPIRiT(raw, kernelShape, kSolveIndices)
    % raw is the full sampled calibration data.  Nx x Ny x Nz x Nc
    % kernelShape is the kernel's shape.  binary kernel where 1's 
    % indicate the source neighbors.  Nkx x Nky x Nkz (ignore
    % numKernelShapes).  
    % kSolveIndices indicates the target element in kernelShape. 3 x 1
    % (again ignore numKernelShapes for now). 

    [Nx, Ny, Nz, Nc] = size(raw);

    Nkx             = size(kernelShape, 1);
    Nky             = size(kernelShape, 2);
    Nkz             = size(kernelShape, 3);

%----------------------------------------------------------------------------------
% determine the channel with the max intensity.  this will be the channel's
% relative phase that we will use. 
sumEachChannel = zeros(1, Nc);
for channelIter = 1 : Nc
    rawCh = raw(:, :, :, channelIter);
    sumEachChannel(1, channelIter) = sum(rawCh(:));
end
[~,maxIntensityChannel] = max(sumEachChannel(:));
locationIndices  = 1 : Nx * Ny * Nz;
% locationIndicesM = reshape(locationIndices, Nxc, Nyc, Nzc);
%----------------------------------------------------------------------------------
%% prepare the kernels
    [ kernels, kernels_ones, kernels_solve] = func_generate_kernels(kernelShape,kSolveIndices);
%% get the weights
    [weightsKernels, targetColumnIndex, sourceColumnIndex, ...
    weightsKernels_decompressed, weightsKernels_reformatted, ...
    Ac_sourceKernels_harmonicsx_reformatted, ...
          Ac_sourceKernels_harmonicsy_reformatted, ...
          Ac_sourceKernels_harmonicsz_reformatted]  = ...
    func_getWeights_II(raw, kernelShape,kSolveIndices);
%% get the calibration image. 
    imRaw         = zeros(size(raw        ));
    % imDiff        = zeros(size(raw        ));
    for channelIter = 1 : Nc
        imRaw        (:, :, :, channelIter) = ifftnc(raw        (:, :, :, channelIter));
    end
    
    imRaw_sos = sqrt(sum(abs(imRaw).^2, 4));
    mask_im = imRaw_sos > 0.06 * max(imRaw_sos(:));
%     figure, imshow(mask_im .* abs(imRaw_sos), [])
%     figure, imshow(abs(mask_im), [])

%% Build the harmonics matrix that will be used to get the eigen values and eigen vectors of. 
%     compositeTest_sum = zeros(size(imRaw_test));
%     compositeTest_p0p0 = zeros(size(imRaw_test));
    
    kSolveX = kSolveIndices(1, 1);  kSolveY = kSolveIndices(2, 1); kSolveZ = kSolveIndices(3, 1); 
    harmonicTerm = 0; % initialize.
    
    % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    harmonicMatrix = zeros(Nx, Ny, Nz, Nc, Nc); 
    % Usually, the first channel dimension is the source and the second is the
    % target.  To make things easier for the eigen-value decomposition, we will
    % make the first channel dimension be the target, and the  second be the
    % source.  
    % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    for channelTest = 1 : Nc
    
        % for weights kenel reformatted:  the first channel dimension is the
        % source while the second is the target.
        weightsKernels_reformatted_targetIter = weightsKernels_reformatted(:,:, :, :, channelTest);
        
    
        for  xIter = 1 : Nx
        
            for yIter = 1 : Ny
                zIter = 1;
                for channelIter = 1 : Nc
                    
%                     compositeTest_p0p0(xIter,yIter, 1, channelTest) = ...
%                     compositeTest_p0p0(xIter, yIter,  1, channelTest) + ...
%                     weightsKernels_reformatted_targetIter(kSolveX, kSolveY, kSolveZ, channelIter) * ...
%                     imRaw_test(xIter, yIter, 1, channelIter);
        
                    harmonicTerm = 0; % initialize here ...
                    for xKernel = 1 : Nkx
                        for yKernel = 1 : Nky
                            for zKernel = 1 : Nkz
                                
        %                         if 1
        %                          if (xKernel ~= kSolveX) || (yKernel ~= kSolveY) || (zKernel ~= kSolveZ)
                                 if ((xKernel == kSolveX) && (yKernel == kSolveY) && (zKernel == kSolveZ)) && (channelIter == channelTest)
                                   % disp('hi')
                                    harmonicTerm = harmonicTerm + ...
                                        0*weightsKernels_reformatted_targetIter(xKernel, yKernel, zKernel, channelIter) * ... 
                                        exp(-1i * 2*pi * (  ...
                                                        Ac_sourceKernels_harmonicsx_reformatted(xKernel, yKernel, zKernel, channelIter)*xIter/Nx + ...
                                                        Ac_sourceKernels_harmonicsy_reformatted(xKernel, yKernel, zKernel, channelIter)*yIter/Ny + ...
                                                        Ac_sourceKernels_harmonicsz_reformatted(xKernel, yKernel, zKernel, channelIter)*zIter/Nz  ...
                                                        ) ...
                                            );
                                 else
                                   % disp('hi')
                                    harmonicTerm = harmonicTerm + ...
                                        weightsKernels_reformatted_targetIter(xKernel, yKernel, zKernel, channelIter) * ... 
                                        exp(-1i * 2*pi * (  ...
                                                        Ac_sourceKernels_harmonicsx_reformatted(xKernel, yKernel, zKernel, channelIter)*xIter/Nx + ...
                                                        Ac_sourceKernels_harmonicsy_reformatted(xKernel, yKernel, zKernel, channelIter)*yIter/Ny + ...
                                                        Ac_sourceKernels_harmonicsz_reformatted(xKernel, yKernel, zKernel, channelIter)*zIter/Nz  ...
                                                        ) ...
                                            );
                                     
                                end
                            end
                        end
                    end
                    harmonicMatrix(xIter, yIter, zIter, channelTest, channelIter) = harmonicTerm;  
%                     compositeTest_sum(xIter, yIter, zIter, channelTest) = compositeTest_sum(xIter, yIter, zIter, channelTest) + ...
%                                          imRaw_test(xIter, yIter, zIter, channelIter) * harmonicTerm;
                end
            end
        
        end
    end

    %%  Eigen value decomposition. 
    % There will be Nc Eigen values because each harmonicMatrixIter is Nc x Nc.
    % Each Eigen vector will have Nc elements.  
    
    % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    eigVectorsM = zeros(Nx, Ny, Nz, Nc, Nc); 
    % first channel dimension is entries of a vector and 
    % the second channel dimension is a new eigen vector. 
    eigValuesM  = zeros(Nx, Ny, Nz, Nc); 
    % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    for xIter = 1 : Nx
        for yIter = 1 : Ny
            for zIter = 1 : Nz
                harmonicMatrixIter = squeeze(harmonicMatrix(xIter, yIter, zIter, :, :));
                [V, D] = eig(harmonicMatrixIter); % [V, D, ~ ]= svd(, 'econ')
                eigVectorsM(xIter, yIter, zIter, :, :) = V      ;
                eigValuesM (xIter, yIter, zIter, :)    = diag(D);
            end
        end
    end
    
    % Now you must spacially fftShift eigValuesM and eigValuesM.
    for eigValueIter = 1 : Nc
        eigValuesM(:, :, :, eigValueIter) = fftshift(squeeze(eigValuesM(:, :, :, eigValueIter)));
        for eigEntryIter = 1 : Nc
            eigVectorsM(:, :, :, eigEntryIter, eigValueIter) = ...
                fftshift(squeeze(eigVectorsM(:, :, :, eigEntryIter, eigValueIter)));
        end
    end

    mask_eig = (eigValuesM(:, :, :, :)) > 0.95;

    % before we just chose the last channel. 
    % phases = repmat(exp(-1i .* angle(eigVectorsM(:, :, :, Nc, :))), [1, 1, 1, Nc, 1]);
    % now we use the channel with the peak intensity. 
    phases = repmat(exp(-1i .* angle(eigVectorsM(:, :, :, maxIntensityChannel, :))), [1, 1, 1, Nc, 1]);
    eigVectorsM = eigVectorsM .* phases;

end