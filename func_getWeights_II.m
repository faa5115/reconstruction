function [weightsKernels, targetColumnIndex, sourceColumnIndex,...
          weightsKernels_decompressed, weightsKernels_reformatted,  ...
          Ac_sourceKernels_harmonicsx_reformatted, ...
          Ac_sourceKernels_harmonicsy_reformatted, ...
          Ac_sourceKernels_harmonicsz_reformatted]  = ...
    func_getWeights_II(calib, kernelShape,kSolveIndices)
    % func_getWeights_II includes a harmonicsM output so that we 
    % can easily dot-product it with weightsKernels_reformatted when summing channel harmonics.  

    % gives you the weights and also gives you the weights in the null
    % space vector format. 

    [Nxc, Nyc, Nzc, Nc] = size(calib);
    Nkx             = size(kernelShape, 1);
    Nky             = size(kernelShape, 2);
    Nkz             = size(kernelShape, 3);
    Nk              = Nkx * Nky * Nkz;
    numKernelShapes = size(kSolveIndices, 4);
    [ kernels, kernels_ones, kernels_solve] = func_generate_kernels(kernelShape,kSolveIndices);

    harmonicsM = zeros(Nkx, Nky, Nkz,  Nc, numKernelShapes);
    for kernelIter = 1 : numKernelShapes
        target = kSolveIndices(:, kernelIter);
        for xIter = 1 : Nkx
            for yIter = 1 : Nky
                for zIter = 1 : Nkz
                    for channelIter = 1 : Nc
                        %x difference
                        harmonicsM(xIter, yIter, zIter, 1, channelIter, kernelIter) = xIter - target(1);
                        harmonicsM(xIter, yIter, zIter, 2, channelIter, kernelIter) = yIter - target(2);
                        harmonicsM(xIter, yIter, zIter, 3, channelIter, kernelIter) = zIter - target(3);
                    end
                end
            end
        end
    end
    
    Nk_nonzeros = sum(kernelShape(:));

    [Acalib_mc     , AelCalib     ] = func_create_A(calib, kernels_ones);
    
    kernelShapeIter = 1;
    
    Ac_sourceKernels_mc = zeros(size(Acalib_mc, 1), Nk_nonzeros, Nc, numKernelShapes);
    Ac_targetKernels_mc = zeros(size(Acalib_mc, 1), 1          , Nc, numKernelShapes);

    %------------------------------------------------------------------------------------
    Ac_sourceKernels_harmonicsx_mc = zeros(1, Nk_nonzeros, Nc, numKernelShapes);
    Ac_sourceKernels_harmonicsy_mc = zeros(1, Nk_nonzeros, Nc, numKernelShapes);
    Ac_sourceKernels_harmonicsz_mc = zeros(1, Nk_nonzeros, Nc, numKernelShapes);

    Ac_targetKernels_harmonicsx_mc = zeros(1,           1, Nc, numKernelShapes);
    Ac_targetKernels_harmonicsy_mc = zeros(1,           1, Nc, numKernelShapes);
    Ac_targetKernels_harmonicsz_mc = zeros(1,           1, Nc, numKernelShapes);

    Ac_sourceKernels_harmonicsx    = zeros(1,           Nc * Nk_nonzeros, numKernelShapes);
    Ac_sourceKernels_harmonicsy    = zeros(1,           Nc * Nk_nonzeros, numKernelShapes);
    Ac_sourceKernels_harmonicsz    = zeros(1,           Nc * Nk_nonzeros, numKernelShapes);

    Ac_targetKernels_harmonicsx    = zeros(1,                1 * Nc, numKernelShapes);
    Ac_targetKernels_harmonicsy    = zeros(1,                1 * Nc, numKernelShapes);
    Ac_targetKernels_harmonicsz    = zeros(1,                1 * Nc, numKernelShapes);
    %------------------------------------------------------------------------------------
    
    numKernelFits_Acalib = (Nxc - (Nkx - 1)) * (Nyc - (Nky - 1)) * (Nzc - (Nkz - 1)); %the number of times the kernel fits across the A matrix
    
    weightsKernels = zeros(Nc * Nk_nonzeros, Nc, numKernelShapes);
    targetColumnIndex = zeros(1          , numKernelShapes);
    sourceColumnIndex = zeros(Nk_nonzeros, numKernelShapes);
    for kernelShapeIter = 1 : numKernelShapes

        harmonicsMx = harmonicsM(:, :, :, 1, :, kernelShapeIter);
        harmonicsMy = harmonicsM(:, :, :, 2, :, kernelShapeIter);
        harmonicsMz = harmonicsM(:, :, :, 3, :, kernelShapeIter);

        
        [AharmonicsMx_mc, AelHarmonicsMx] = func_create_A(harmonicsMx, kernels_ones);
        [AharmonicsMy_mc, AelHarmonicsMy] = func_create_A(harmonicsMy, kernels_ones);
        [AharmonicsMz_mc, AelHarmonicsMz] = func_create_A(harmonicsMz, kernels_ones);

        kernel_target = kernels_solve(:, :, :, :, kernelShapeIter);
        kernel_source = kernels      (:, :, :, :, kernelShapeIter);
        
        [Ac_target_mc, Acel_target] = func_create_A(calib, kernel_target);
        [Ac_source_mc, Acel_source] = func_create_A(calib, kernel_source);

        %---------------------------------------------------------------------------------------------
        [AharmonicsMx_target_mc, AelHarmonicsMx_target] = func_create_A(harmonicsMx, kernel_target);
        [AharmonicsMy_target_mc, AelHarmonicsMy_target] = func_create_A(harmonicsMy, kernel_target);
        [AharmonicsMz_target_mc, AelHarmonicsMz_target] = func_create_A(harmonicsMz, kernel_target);

        [AharmonicsMx_source_mc, AelHarmonicsMx_source] = func_create_A(harmonicsMx, kernel_source);
        [AharmonicsMy_source_mc, AelHarmonicsMy_source] = func_create_A(harmonicsMy, kernel_source);
        [AharmonicsMz_source_mc, AelHarmonicsMz_source] = func_create_A(harmonicsMz, kernel_source);
        %---------------------------------------------------------------------------------------------
        
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

            %---------------------------------------------------------------------------------------------
            AharmonicsMx_source_input = AharmonicsMx_source_mc(:, :, channelIter);
            AharmonicsMy_source_input = AharmonicsMy_source_mc(:, :, channelIter);
            AharmonicsMz_source_input = AharmonicsMz_source_mc(:, :, channelIter);

            AharmonicsMx_target_input = AharmonicsMx_target_mc(:, :, channelIter);
            AharmonicsMy_target_input = AharmonicsMy_target_mc(:, :, channelIter);
            AharmonicsMz_target_input = AharmonicsMz_target_mc(:, :, channelIter);
            %---------------------------------------------------------------------------------------------
    
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

            %---------------------------------------------------------------------------------------------
            Ac_sourceKernels_harmonicsx_mc(:, :, channelIter, kernelShapeIter) = ...
                            AharmonicsMx_source_input(:, sourceColumnIndex(:, kernelShapeIter));
            Ac_sourceKernels_harmonicsy_mc(:, :, channelIter, kernelShapeIter) = ...
                            AharmonicsMy_source_input(:, sourceColumnIndex(:, kernelShapeIter));
            Ac_sourceKernels_harmonicsz_mc(:, :, channelIter, kernelShapeIter) = ...
                            AharmonicsMz_source_input(:, sourceColumnIndex(:, kernelShapeIter));

            Ac_targetKernels_harmonicsx_mc(:, :, channelIter, kernelShapeIter) = ...
                            AharmonicsMx_target_input(:, targetColumnIndex(:, kernelShapeIter));
            Ac_targetKernels_harmonicsy_mc(:, :, channelIter, kernelShapeIter) = ...
                            AharmonicsMy_target_input(:, targetColumnIndex(:, kernelShapeIter));
            Ac_targetKernels_harmonicsz_mc(:, :, channelIter, kernelShapeIter) = ...
                            AharmonicsMz_target_input(:, targetColumnIndex(:, kernelShapeIter));
            %---------------------------------------------------------------------------------------------
        end
        
        %Next determine the weights for each kernel.  
        Ac_sourceKernelsIter_mc = Ac_sourceKernels_mc(:, :, :, kernelShapeIter);
        Ac_sourceKernelsIter    = reshape(Ac_sourceKernelsIter_mc, [size(Ac_sourceKernels_mc, 1), Nc * Nk_nonzeros]);
    
        Ac_targetKernelsIter_mc = Ac_targetKernels_mc(:, :, :, kernelShapeIter);
        Ac_targetKernelsIter    = reshape(Ac_targetKernelsIter_mc, [ size(Ac_targetKernels_mc, 1), Nc * 1]);


        %---------------------------------------------------------------------------------------------
        Ac_sourceKernelsIter_harmonicsx_mc = Ac_sourceKernels_harmonicsx_mc(:, :, :, kernelShapeIter);
        Ac_sourceKernelsIter_harmonicsy_mc = Ac_sourceKernels_harmonicsy_mc(:, :, :, kernelShapeIter);
        Ac_sourceKernelsIter_harmonicsz_mc = Ac_sourceKernels_harmonicsz_mc(:, :, :, kernelShapeIter);

        %reshape
        Ac_sourceKernelsIter_harmonicsx = ...
            reshape(Ac_sourceKernelsIter_harmonicsx_mc, [size(Ac_sourceKernelsIter_harmonicsx_mc, 1), Nc * Nk_nonzeros]);
        Ac_sourceKernelsIter_harmonicsy = ...
            reshape(Ac_sourceKernelsIter_harmonicsy_mc, [size(Ac_sourceKernelsIter_harmonicsy_mc, 1), Nc * Nk_nonzeros]);
        Ac_sourceKernelsIter_harmonicsz = ...
            reshape(Ac_sourceKernelsIter_harmonicsz_mc, [size(Ac_sourceKernelsIter_harmonicsz_mc, 1), Nc * Nk_nonzeros]);

        Ac_targetKernelsIter_harmonicsx_mc = Ac_targetKernels_harmonicsx_mc(:, :, :, kernelShapeIter);
        Ac_targetKernelsIter_harmonicsy_mc = Ac_targetKernels_harmonicsy_mc(:, :, :, kernelShapeIter);
        Ac_targetKernelsIter_harmonicsz_mc = Ac_targetKernels_harmonicsz_mc(:, :, :, kernelShapeIter);

        %reshape
        Ac_targetKernelsIter_harmonicsx = ...
            reshape(Ac_targetKernelsIter_harmonicsx_mc, [size(Ac_targetKernelsIter_harmonicsx_mc, 1), Nc * 1]);
        Ac_targetKernelsIter_harmonicsy = ...
            reshape(Ac_targetKernelsIter_harmonicsy_mc, [size(Ac_targetKernelsIter_harmonicsy_mc, 1), Nc * 1]);
        Ac_targetKernelsIter_harmonicsz = ...
            reshape(Ac_targetKernelsIter_harmonicsz_mc, [size(Ac_targetKernelsIter_harmonicsz_mc, 1), Nc * 1]);

        Ac_sourceKernels_harmonicsx(:, :, kernelShapeIter) = Ac_sourceKernelsIter_harmonicsx;
        Ac_sourceKernels_harmonicsy(:, :, kernelShapeIter) = Ac_sourceKernelsIter_harmonicsy;
        Ac_sourceKernels_harmonicsz(:, :, kernelShapeIter) = Ac_sourceKernelsIter_harmonicsz;

        Ac_targetKernels_harmonicsx(:, :, kernelShapeIter) = Ac_targetKernelsIter_harmonicsx;
        Ac_targetKernels_harmonicsy(:, :, kernelShapeIter) = Ac_targetKernelsIter_harmonicsy;
        Ac_targetKernels_harmonicsz(:, :, kernelShapeIter) = Ac_targetKernelsIter_harmonicsz;
        %---------------------------------------------------------------------------------------------

    
        weightsKernels(:, :, kernelShapeIter) = pinv(Ac_sourceKernelsIter, 0) * Ac_targetKernelsIter;
        % fadilali:  weightsKernels Nk_nonzeros*Nc x Nc ( x numKernelShapes).
        % Ac_sourceKernels_harmonicsxyz is 1 x Nk_nonzeros*Nc ( x numKernelShapes).
        % Ac_sourceKernels_harmonicsxyz has the same number of columns as
        % weightsKernels does rows. 
        % The channels dimension at the end refers to the target channels.
        % 
    end

    %% Prepare the weights in a null space vector format. 
    sourceSkippedColumnIndex(:, kernelShapeIter) = find(~Acel_source_binary(1, :));
    % The way it works is:
    % Ac_targetKernelsIter = Ac_sourceKernelsIter * weightsKernels(:, :, kernelShapeIter);
    % Lets go backwards:  Ac_sourceKernelsIter is the reshaped version of Ac_sourceKernelsIter_mc
    % and Ac_sourceKernelsIter_mc is Ac_sourceKernels_mc(:, :, :, kernelShapeIter).
    
    % recall weightsKernels is Nknonzeros * Nc x Nc x numKernelShapes...
    % so weightsKernels_mc is Nknonzeros x Nc x Nc x numKernelShapes. 
    % the first channel dimension is the source and the second channel
    %dimensions is the target. 
    weightsKernels_mc = reshape(weightsKernels, [Nk_nonzeros, Nc, size(weightsKernels, 2), numKernelShapes]);
    weightsKernels_decompressed_mc = zeros(Nk, Nc, Nc, numKernelShapes);

    %--------------------------------------------------------------------------
    % These following variables have the same layout as weightsKernels_mc.
    % Ac_sourceKernels_harmonicsx_mc; % Ac_sourceKernels_harmonicsy_mc; % Ac_sourceKernels_harmonicsz_mc;
    % Ac_targetKernels_harmonicsx_mc; % Ac_targetKernels_harmonicsy_mc; % Ac_targetKernels_harmonicsz_mc;
    % so their second dimension gets broken into Nk_nonzeros*Nc.  
    Ac_sourceKernels_harmonicsx_decompressed_mc = zeros( Nk, Nc, numKernelShapes);
    Ac_sourceKernels_harmonicsy_decompressed_mc = zeros( Nk, Nc, numKernelShapes);
    Ac_sourceKernels_harmonicsz_decompressed_mc = zeros( Nk, Nc, numKernelShapes);
    %--------------------------------------------------------------------------

    tempKernelElementsArray  = zeros(Nk, 1);
    tempKernelElementsArray2 = zeros(1, Nk, 1);

    for kernelShapeIter = 1 : numKernelShapes
        for channelIter = 1 : Nc % iterates over the source channels 
            for channelIter2 = 1 : Nc % iterates over the channels where the target element resides.  
                tempKernelElementsArray(sourceColumnIndex(:, kernelShapeIter) ) = ...
                    weightsKernels_mc(:, channelIter, channelIter2, kernelShapeIter);
                weightsKernels_decompressed_mc(:, channelIter, channelIter2, kernelShapeIter) = ...
                    tempKernelElementsArray;
            end
            tempKernelElementsArray2(1, sourceColumnIndex(:, kernelShapeIter) ) = ...
                Ac_sourceKernels_harmonicsx_mc(1, :, channelIter);
            Ac_sourceKernels_harmonicsx_decompressed_mc( :, channelIter, kernelShapeIter) = tempKernelElementsArray2;

            tempKernelElementsArray2(1, sourceColumnIndex(:, kernelShapeIter) ) = ...
                Ac_sourceKernels_harmonicsy_mc(1, :, channelIter);
            Ac_sourceKernels_harmonicsy_decompressed_mc( :, channelIter, kernelShapeIter) = tempKernelElementsArray2;

            tempKernelElementsArray2(1, sourceColumnIndex(:, kernelShapeIter) ) = ...
                Ac_sourceKernels_harmonicsz_mc(1, :, channelIter);
            Ac_sourceKernels_harmonicsz_decompressed_mc( :, channelIter, kernelShapeIter) = tempKernelElementsArray2;
        end
    end
    
    for kernelShapeIter = 1 : numKernelShapes
    
        for channelIter2 = 1 : Nc
            weightsKernels_decompressed_mc(targetColumnIndex(1, kernelShapeIter), channelIter2, channelIter2) = -1;
        end
    end
    
    weightsKernels_decompressed = reshape(weightsKernels_decompressed_mc, [Nk * Nc, size(weightsKernels, 2), numKernelShapes]);

    Ac_sourceKernels_harmonicsx_decompressed = reshape(Ac_sourceKernels_harmonicsx_decompressed_mc, [Nk * Nc, 1]);
    Ac_sourceKernels_harmonicsy_decompressed = reshape(Ac_sourceKernels_harmonicsy_decompressed_mc, [Nk * Nc, 1]);
    Ac_sourceKernels_harmonicsz_decompressed = reshape(Ac_sourceKernels_harmonicsz_decompressed_mc, [Nk * Nc, 1]);

    %% try to change weightsKernels_decompressed to have kernelShapeOnes's format. 
    % weightsKernels_decompressed has dimensions Nk * Nc x Nc.
    % The second column indicates which channel the target being fit to
    % resides, while the first column describes the source neighbors of all channels.  
    % we want this to be arranged the following way:  Nk x Nc x Nc, where the first dimensions gets broken into two. 
    
    % the first channel dimension is the source and the second channel
    % dimension is the target.  
    weightsKernels_reformatted = reshape(weightsKernels_decompressed, [Nkx, Nky, Nkz, Nc, Nc]); %zeros(Nk, Nc, Nc);

    Ac_sourceKernels_harmonicsx_reformatted = reshape(Ac_sourceKernels_harmonicsx_decompressed, [Nkx, Nky, Nkz, Nc]);
    Ac_sourceKernels_harmonicsy_reformatted = reshape(Ac_sourceKernels_harmonicsy_decompressed, [Nkx, Nky, Nkz, Nc]);
    Ac_sourceKernels_harmonicsz_reformatted = reshape(Ac_sourceKernels_harmonicsz_decompressed, [Nkx, Nky, Nkz, Nc]);
end