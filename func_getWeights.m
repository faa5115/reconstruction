function [weightsKernels, targetColumnIndex, sourceColumnIndex, weightsKernels_decompressed, weightsKernels_reformatted]  = ...
    func_getWeights(calib, kernelShape,kSolveIndices)
    % gives you the weights and also gives you the weights in the null
    % space vector format. 

    disp('starting function to get the weights')
    [Nxc, Nyc, Nzc, Nc] = size(calib);
    Nkx             = size(kernelShape, 1);
    Nky             = size(kernelShape, 2);
    Nkz             = size(kernelShape, 3);
    Nk              = Nkx * Nky * Nkz;
    numKernelShapes = size(kSolveIndices, 2);
    [ kernels, kernels_ones, kernels_solve] = func_generate_kernels(kernelShape,kSolveIndices);


    singleKernelShape = kernelShape(:, :, :, 1);
    Nk_nonzeros = sum(singleKernelShape(:));%sum(kernelShape(:)) / numKernelShapes;

    [Acalib_mc, AelCalib] = func_create_A(calib, kernels_ones);
    kernelShapeIter = 1;
    
    Ac_sourceKernels_mc = zeros(size(Acalib_mc, 1), Nk_nonzeros, Nc, numKernelShapes);
    Ac_targetKernels_mc = zeros(size(Acalib_mc, 1), 1          , Nc, numKernelShapes);
    
    numKernelFits_Acalib = (Nxc - (Nkx - 1)) * (Nyc - (Nky - 1)) * (Nzc - (Nkz - 1)); %the number of times the kernel fits across the A matrix
    
    weightsKernels = zeros(Nc * Nk_nonzeros, Nc, numKernelShapes);
    targetColumnIndex = zeros(1          , numKernelShapes);
    sourceColumnIndex = zeros(Nk_nonzeros, numKernelShapes);
    disp('starting kernel shape loop')
    for kernelShapeIter = 1 : numKernelShapes
        kernel_target = kernels_solve(:, :, :, :, kernelShapeIter);
        kernel_source = kernels      (:, :, :, :, kernelShapeIter);
        
        [Ac_target_mc, Acel_target] = func_create_A(calib, kernel_target);
        [Ac_source_mc, Acel_source] = func_create_A(calib, kernel_source);
        
    %     Acel_source_compressed = Acel_source(Acel_source~=0);
    %     Acel_source_compressed = reshape(Acel_source_compressed, [size(Acel_source, 1), sum(kernelShape(:))]);
    % 
    %     Acel_target_compressed = Acel_source(Acel_target~=0);
    %     Acel_target_compressed = reshape(Acel_target_compressed, [size(Acel_target, 1), sum(kernelSolve(:))]);
        Acel_target_binary = Acel_target ~= 0;
        Acel_source_binary = Acel_source ~= 0;
        targetColumnIndex(1, kernelShapeIter) = find(Acel_target_binary(1, :));
        sourceColumnIndex(:, kernelShapeIter) = find(Acel_source_binary(1, :));
        for channelIter = 1 : Nc
            Ac_source_input = Ac_source_mc(:, :, channelIter);
            Ac_target_input = Ac_target_mc(:, :, channelIter);
    
    %         Ac_source_output = Ac_source_input(Acel_source_binary);
    %         Ac_source_output = reshape(Ac_source_output, [size(Acel_source, 1), sum(kernelShape(:))]);
    %         Ac_sourceKernels_mc(:, :, channelIter, kernelShapeIter) = Ac_source_output;
    %         
    %         Ac_target_output = Ac_target_input(Acel_target_binary); %don't need to reshape. 
    %         Ac_targetKernels_mc(:, :, channelIter, kernelShapeIter) = Ac_target_output;
    
            Ac_sourceKernels_mc(:, :, channelIter, kernelShapeIter) = ...
                            Ac_source_input(:, sourceColumnIndex(:, kernelShapeIter));
            Ac_targetKernels_mc(:, :, channelIter, kernelShapeIter) = ...
                            Ac_target_input(:, targetColumnIndex(:, kernelShapeIter));
    
        end
        
        %Next determine the weights for each kernel.  
        Ac_sourceKernelsIter_mc = Ac_sourceKernels_mc(:, :, :, kernelShapeIter);
        Ac_sourceKernelsIter    = reshape(Ac_sourceKernelsIter_mc, [size(Ac_sourceKernels_mc, 1), Nc * Nk_nonzeros]);
    
        Ac_targetKernelsIter_mc = Ac_targetKernels_mc(:, :, :, kernelShapeIter);
        Ac_targetKernelsIter    = reshape(Ac_targetKernelsIter_mc, [ size(Ac_targetKernels_mc, 1), Nc * 1]);
    
        weightsKernels(:, :, kernelShapeIter) = pinv(Ac_sourceKernelsIter, 0) * Ac_targetKernelsIter;
    end

    %% Prepare the weights in a null space vector format. 
    sourceSkippedColumnIndex(:, kernelShapeIter) = find(~Acel_source_binary(1, :));
    % The way it works is:
    % Ac_targetKernelsIter = Ac_sourceKernelsIter * weightsKernels(:, :, kernelShapeIter);
    % Lets go backwards:  Ac_sourceKernelsIter is the reshaped version of Ac_sourceKernelsIter_mc
    % and Ac_sourceKernelsIter_mc is Ac_sourceKernels_mc(:, :, :, kernelShapeIter).
    
    % recall weightsKernels is Nknonzeros * Nc x Nc x numKernelShapes...
    % so weightsKernels_mc is Nknonzeros x Nc x Nc x numKernelShapes. 
    weightsKernels_mc = reshape(weightsKernels, [Nk_nonzeros, Nc, size(weightsKernels, 2), numKernelShapes]);
    weightsKernels_decompressed_mc = zeros(Nk, Nc, Nc, numKernelShapes);
    tempKernelElementsArray = zeros(Nk, 1);
    
    for kernelShapeIter = 1 : numKernelShapes
        for channelIter = 1 : Nc % iterates over the source channels 
            for channelIter2 = 1 : Nc % iterates over the channels where the target element resides.  
                tempKernelElementsArray(sourceColumnIndex(:, kernelShapeIter) ) = ...
                    weightsKernels_mc(:, channelIter, channelIter2, kernelShapeIter);
                weightsKernels_decompressed_mc(:, channelIter, channelIter2, kernelShapeIter) = ...
                    tempKernelElementsArray;
            end
        end
    end
    
    for kernelShapeIter = 1 : numKernelShapes
    
        for channelIter2 = 1 : Nc
            weightsKernels_decompressed_mc(targetColumnIndex(1, kernelShapeIter), channelIter2, channelIter2) = -1;
        end
    end
    
    weightsKernels_decompressed = reshape(weightsKernels_decompressed_mc, [Nk * Nc, size(weightsKernels, 2), numKernelShapes]);


    %% try to change weightsKernels_decompressed to have kernelShapeOnes's format. 
    % weightsKernels_decompressed has dimensions Nk * Nc x Nc.
    % The second column indicates which channel the target being fit to
    % resides, while the first column describes the source neighbors of all channels.  
    % we want this to be arranged th[Nxc, Nyc, Nzc, Nc] = size(calib);
        % [Nx , Ny , Nz , Nc] = size(rawUs);e following way:  Nk x Nc x Nc, where the first dimensions gets broken into two. 
    
    % weightsKernels_reformatted = reshape(weightsKernels_decompressed, [Nkx, Nky, Nkz, Nc, Nc]); %zeros(Nk, Nc, Nc);
    weightsKernels_reformatted = zeros(Nkx, Nky, Nkz, Nc, Nc, numKernelShapes);
    for kernelShapeIter = 1 : numKernelShapes
        weightsKernels_reformatted(:, :, :, :, :, kernelShapeIter) = ...
            reshape(weightsKernels_decompressed(:, :, kernelShapeIter), [Nkx, Nky, Nkz, Nc, Nc]);
    end
end