function [ rawRecon, weightsKernels] = func_grappa_recon(rawUs, calib, kernelShape, kSolveIndices)
% Something to know:  Nkx, Nky, Nkz, numKernelShapes refer to number of
% elements along kx, ky, kz, in a kernel, and the number of kernel shapes respetively. 
% kernelShape has size:  Nkx, Nky, Nkz, numKernelShapes
%
% kSolveIndices tell us where the target index in the kernel is ... 3 x numKernelElements
% element array saying the location along each kernel direction (kSolveX,
% kSolveY, kSolveZ) for each numKernelShapes.  
% kSolveIndices: 3 x numKernelShapes where each 3 entries is kSolveX, kSolveY, and kSolveZ
disp( 'starting grapa function')
    if size(calib, 4) == size(rawUs, 4)
        [Nxc, Nyc, Nzc, Nc] = size(calib);
        [Nx , Ny , Nz , Nc] = size(rawUs);
        Nkx             = size(kernelShape, 1);
        Nky             = size(kernelShape, 2);
        Nkz             = size(kernelShape, 3);
        numKernelShapes = size(kSolveIndices, 2);

       
        % Generate the kernels. 
        [ kernels, kernels_ones, kernels_solve] = func_generate_kernels(kernelShape,kSolveIndices);
    disp('preparing the kernels')
        % Get the weights
        [weightsKernels, targetColumnIndex, sourceColumnIndex, ~, ~] = func_getWeights(calib, kernelShape,kSolveIndices);
    disp('solved for the weights')

        %% Carry out the recon for each kernel.
        %perhaps just store the indices across all k-space and repeat that for each
        %channel ...
        elRawData = single(reshape((1 : 1 : prod(size(rawUs, 1, 2, 3))), [size(rawUs, 1, 2, 3)]));

        % numKernelFits_Anopad = (Nx - (Nkx - 1)) * (Ny - (Nky - 1)) * (Nz - (Nkz - 1)); %the number of times the kernel fits across the A matrix
    
        %determine the unacquired indices ...
        channelIter = 1;
        rawUsSingleChannel = abs(rawUs(:, :, :, channelIter));
        skippedElements = find(~rawUsSingleChannel(:));
        % numOfSkippedElements = length(skippedifference between single and double precisiondElements);

%         rawUsPad     = padarray(rawUs       , [kSolveX - 1  , kSolveY - 1  , kSolveZ - 1  , 0], 'pre'  );
%         rawUsPad     = padarray(rawUsPad    , [Nkx - kSolveX, Nky - kSolveY, Nkz - kSolveZ, 0], 'post' );
%         elRawDataPad = padarray(elRawData   , [kSolveX - 1  , kSolveY - 1  , kSolveZ - 1  , 0], 'pre'  );
%         elRawDataPad = padarray(elRawDataPad, [Nkx - kSolveX, Nky - kSolveY, Nkz - kSolveZ, 0], 'post' );
% 
%         [Nxpad, Nypad, Nzpad] = size(rawUsPad, 1, 2, 3);
%         numKernelFits_Apad = (Nxpad - (Nkx - 1)) * (Nypad - (Nky - 1)) * (Nzpad - (Nkz - 1)); %the number of times the kernel fits across the A matrix
%         
%         [Apad_mc, Apadel]   = func_create_A(rawUsPad    , kernels_ones);
%         [AelPad , AelPadel] = func_create_A(elRawDataPad, kernels_ones);
%         targetColumn = zeros(numKernelShapes, size(AelPad, 1));
      

        %just to have to test your skippedElements: 
        rawRecon = rawUs;
        
        disp('going into kernelshape loop')
        for kernelShapeIter = 1 : numKernelShapes
            disp(strcat('kernel shape iter',32,num2str(kernelShapeIter)))
            kSolveX = kSolveIndices(1, kernelShapeIter);
            kSolveY = kSolveIndices(2, kernelShapeIter);
            kSolveZ = kSolveIndices(3, kernelShapeIter);


            %-----------------------------------------
            disp('pad raw data')
                rawUsPad     = padarray(rawUs       , [kSolveX - 1  , kSolveY - 1  , kSolveZ - 1  , 0], 'pre'  );
                rawUsPad     = padarray(rawUsPad    , [Nkx - kSolveX, Nky - kSolveY, Nkz - kSolveZ, 0], 'post' );
                elRawDataPad = padarray(elRawData   , [kSolveX - 1  , kSolveY - 1  , kSolveZ - 1  , 0], 'pre'  );
                elRawDataPad = padarray(elRawDataPad, [Nkx - kSolveX, Nky - kSolveY, Nkz - kSolveZ, 0], 'post' );
        
                [Nxpad, Nypad, Nzpad] = size(rawUsPad, 1, 2, 3);
                % numKernelFits_Apad = (Nxpad - (Nkx - 1)) * (Nypad - (Nky - 1)) * (Nzpad - (Nkz - 1)); %the number of times the kernel fits across the A matrix
                
                % [Apad_mc, Apadel]   = func_create_A(rawUsPad    , kernels_ones);
                [AelPad , ~] = func_create_A(elRawDataPad, kernels_ones);% [AelPad , AelPadel] = func_create_A(elRawDataPad, kernels_ones);
                targetColumn = zeros(numKernelShapes, size(AelPad, 1));
            %-----------------------------------------

            kernel_target = kernels_solve(:, :, :, :, kernelShapeIter);
            kernel_source = kernels      (:, :, :, :, kernelShapeIter);

            targetColumn(kernelShapeIter, :) = AelPad(:, targetColumnIndex(1, kernelShapeIter));
    
            disp('create target and source A matrices')
            % [Apad_target_mc, ~] = func_create_A(rawUsPad, kernel_target);% [Apad_target_mc, Apadel_target] = func_create_A(rawUsPad, kernel_target);
            [Apad_source_mc, ~] = func_create_A(rawUsPad, kernel_source);% [Apad_source_mc, Apadel_source] = func_create_A(rawUsPad, kernel_source);

            [AelPad_target , ~] = func_create_A(elRawDataPad, kernel_target);% [AelPad_target , AelPadel_target] = func_create_A(elRawDataPad, kernel_target);
            [AelPad_source , ~] = func_create_A(elRawDataPad, kernel_source);% [AelPad_source , AelPadel_source] = func_create_A(elRawDataPad, kernel_source);

            rowTargetIsSkipped   = ismember(targetColumn(kernelShapeIter, :), skippedElements); %should be size prod([Nx, Ny, Nz])
            rowTargetIsSkippedEl = find(rowTargetIsSkipped); %should be size prod([Nx, Ny, Nz])/2.  
            %figure, plot(rowTargetIsSkippedEl, 'linewidth', 2.0)

            disp('create compressed Ael (element) matrices')
            AelPad_target_compressed = AelPad_target(rowTargetIsSkippedEl, :);
            AelPad_target_compressed = AelPad_target_compressed(:, targetColumnIndex(:, kernelShapeIter));
            AelPad_source_compressed = AelPad_source(rowTargetIsSkippedEl, :);
            AelPad_source_compressed = AelPad_source_compressed(:, sourceColumnIndex(:, kernelShapeIter)); 
            clear AelPad_source AelPad_target
        %     AelPad_target_compressed(1000, :)
        %     AelPad_source_compressed(1000, :)
        %     AelPad_source_compressed(1000, sourceColumnIndex(:, 1))

            % Apad_target_mc_compressed = zeros(size(AelPad_target_compressed, 1) , ...
            %                                   size(AelPad_target_compressed, 2) , Nc);
            Apad_source_mc_compressed  = zeros(size(AelPad_source_compressed, 1) , ...
                                              size(AelPad_source_compressed, 2) , Nc);
            clear  AelPad_source_compressed
            disp('looping across channels to identify target entries and their sources')
            for channelIter = 1 : Nc
                % Apad_target_mc_iter = Apad_target_mc(:, :, channelIter);
                % Apad_target_mc_compressed(:, :, channelIter) = ... 
                %                         Apad_target_mc_iter(rowTargetIsSkippedEl, targetColumnIndex(:, kernelShapeIter));
                                        %previous mistake:  you had Apad_target_mc
                                        %above
                Apad_source_mc_iter = Apad_source_mc(:, :, channelIter);
                Apad_source_mc_compressed(:, :, channelIter) = ... 
                                        Apad_source_mc_iter(rowTargetIsSkippedEl, sourceColumnIndex(:, kernelShapeIter));
                                        %previous mistake:  you had Apad_source_mc
                                        %above
    
            end
            clear Apad_target_mc_iter Apad_source_mc_iter Apad_target_mc Apad_source_mc
            disp('finished identifying target entries and their sources.')

            %Apad_target_compressed should just be zeros.  
            % Apad_target_compressed = reshape(Apad_target_mc_compressed, ...
            %                             [size(Apad_target_mc_compressed, 1), ...
            %                             size(Apad_target_mc_compressed, 2) * Nc]);
            disp('reshaping Apad_source_compressed.')
            Apad_source_compressed = reshape(Apad_source_mc_compressed, ...
                                        [size(Apad_source_mc_compressed, 1), ...
                                        size(Apad_source_mc_compressed, 2) * Nc]);
        
            Apad_target_compressed_update = Apad_source_compressed * squeeze(weightsKernels(:, :, kernelShapeIter));
            disp(strcat('going into final loop', 32,'loop iter', 32, num2str(kernelShapeIter)));
            rawInputSingleChannel = zeros(Nx, Ny, Nz);
            for channelIter = 1 : Nc
        %         Apad_target_compressed_update_singleChannel = ...
        %             Apad_target_compressed_update(:, channelIter);
        
%                 rawUsChannelIter = rawUs(:, :, :, channelIter);
%                 rawUsChannelIter(AelPad_target_compressed) = ...
%                     Apad_target_compressed_update(:, channelIter);
%                 rawRecon(:, :, :, channelIter) =   rawUsChannelIter;
                
                rawInputSingleChannel(AelPad_target_compressed) = Apad_target_compressed_update(:, channelIter);
                rawRecon(:, :, :, channelIter) = rawRecon(:, :, :, channelIter) + rawInputSingleChannel;
            end
            disp(strcat('finished kernelshapeiter',32,num2str(kernelShapeIter)))

        end

    else
        error('calibration and raw data have different channel sizes ...');
    end
end