function [rawRecon, weightsKernels, targetColumnIndex] = ...
    func_LSrFOV_pre_sliceGrappa(raw_fFOV_og, calibCollapse, calibTargetSlice, kSize)
    % Assume the data is already whitened. 
    % Input:  raw_fFOV_og:  the whitened raw data encoding for the rFOV:  Nx x Ny x Nz x Nc.
    %         calibCollapse:  The whitened calib data properly collapsed
    %         for the given sms problem.  (as in the image of the
    %         calibration images of hte given slices were alread shifted
    %         and summed).  
    %         calibTargetSlice:  the whitened calib data that only has the
    %         target slice, where the FOV was properly shifted. 
    %         noise:  whitened noise:  Nns x Nc (ns = "noise samples").
    %         mask:  The mask.  Nxc x Nyc x Nzc.
    [Nx, Ny, Nz, Nc] = size(raw_fFOV_og);
    [Nxc, Nyc, Nzc, ~] = size(calibCollapse);
   
    
    
    % get the full fov calibration images. 
    imCalibCollapse = zeros(size(calibCollapse));
    imCalibTargetSlice = zeros(size(calibTargetSlice));
    for chiter = 1 : Nc
        imCalibCollapse(:, :, :, chiter) = ifftnc(calibCollapse(:, :, :, chiter));% circshift(ifftnc(calib(:, :, :, chiter)), -shiftValue, 2);
        imCalibTargetSlice(:, :, :, chiter) = ifftnc(calibTargetSlice(:, :, :, chiter));
        
        % shift to align with the center of the center of the main acquisition. 
    end

    imCalibCollapse_sos = func_sqSOS(imCalibCollapse, []);
    imCalibTargetSlice_sos = func_sqSOS(imCalibTargetSlice, []);
    
    
    % for debugging purposes only - - - - - - - - - - - - - - - - - - - 
    % imCalibAliased_sos = func_sqSOS(imCalibAliased, []);
    % imCalibMaskAliased_sos = func_sqSOS(imCalibMaskAliased, []);
    % 
    % figure,
    % imshow(abs(imCalibAliased_sos), [])
    % 
    % figure,
    % imshow(abs(imCalibMaskAliased_sos), [])

    % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

   
    
    

   

    % for debugging purposes only:
    % imCalibAliasedRed_sos = func_sqSOS(imCalibAliasedRed, []);
    % imCalibMaskAliasedRed_sos = func_sqSOS(imCalibMaskAliasedRed, []);
    % 
    % figure,
    % imshow(abs(imCalibAliasedRed_sos), [])
    % 
    % figure,
    % imshow(abs(imCalibMaskAliasedRed_sos), [])
    % ----------------------------------------------------------------
    % ----------------------------------------------------------------

    Nkx = kSize(1); Nky = kSize(2); Nkz = kSize(3);
    % idxMDir = '/bme/home/alif10/matlab/bloch_sim/SENSE_Tutorial/idxM/';
    % 
    % idxM = load(strcat(idxMDir,...
    %         'idxM_',num2str(Nx),'Nx',num2str(Ny),'Ny',num2str(Nz),'Nz_',...
    %         num2str(Nkx),'Nkx',num2str(Nky),'Nky',num2str(Nkz),'Nkz.mat'));
    % idxM = idxM.idxM;
    % 
    % nIter = 1000;
    bUseGPU = 1;

    % % % % % % [rawUpdate_fFOV, rawUpdate_rFOV, f, Scalib, normValues] = ...
    % % % % % % func_reducedFOVIterations(raw_fFOV_og, kSize, nIter, kCalibPadFullFOV, kCalibPadMaskFOV, bUseGPU, idxMSG);
    % % % % % 
    % % % % % % [rawUpdate_fFOV, rawUpdate_rFOV, f, Scalib, normValues] = ...
    % % % % % % func_rFOVIterations(raw_fFOV_og, kSize, nIter, calib_rFOVint, calibMask_rFOVint, noise, bUseGPU, idxM);


    % try  this instead ...
    % [rawUpdate_fFOV, rawUpdate_rFOV, f, Scalib, Vcalib, normValues, stdAndNoiseFloor] = ...
    % func_rFOVIterations(raw_fFOV_og, kSize, nIter, calibAliasedRed, calibMaskAliasedRed, noise, bUseGPU, idxM);

    % Above was for LR-rFOV. ... we will focus on LS-rFOV-------------------


    Nk = prod(kSize);
    kernelShape = ones(Nkx, Nky, Nkz, 1);
    % kernels = eye(Nk, Nk);
    % kernels = flip(kernels, 2);
    % kernels = reshape(kernels, [Nkx, Nky, Nkz, Nk]);
    % kernels = permute(kernels, [2, 1, 3, 4]);
    

    [kernels, kernels_ones, kernels_solve] = ...
        func_generate_kernels(kernelShape, [round(Nkx/2), round(Nky/2), round(Nkz/2)]');
    disp('preparing the kernels')
        % Get the weights
        [weightsKernels, targetColumnIndex, sourceColumnIndex, ~, ~] = ...
            func_getWeights_lsrFOV(calibCollapse, calibTargetSlice, kernelShape,[round(Nkx/2), round(Nky/2), round(Nkz/2)]');
    disp('solved for the weights')

    elRawData = single(reshape((1 : 1 : prod(size(raw_fFOV_og, 1, 2, 3))), [size(raw_fFOV_og, 1, 2, 3)]));
    numKernelFits_Anopad = (Nx - (Nkx - 1)) * (Ny - (Nky - 1)) * (Nz - (Nkz - 1)); %the number of times the kernel fits across the A matrix
    
    rawRecon = zeros(size(raw_fFOV_og));
    disp('going into kernelshape loop')
    numKernelShapes = 1; % don't need more than one ... but i am copying from my grappa code, which has this variable. 
    
    for kernelShapeIter = 1 : numKernelShapes
        % these were copied from my grappa code
        % kSolveX = kSolveIndices(1, kernelShapeIter);
        % kSolveY = kSolveIndices(2, kernelShapeIter);
        % kSolveZ = kSolveIndices(3, kernelShapeIter);
        % The should instead be:
        kSolveX = round(Nkx/2);
        kSolveY = round(Nky/2);
        kSolveZ = round(Nkz/2);

        raw_fFOV_ogPad     = padarray(raw_fFOV_og       , [kSolveX - 1  , kSolveY - 1  , kSolveZ - 1  , 0], 'pre'  );
        raw_fFOV_ogPad     = padarray(raw_fFOV_ogPad    , [Nkx - kSolveX, Nky - kSolveY, Nkz - kSolveZ, 0], 'post' );
        elRawDataPad = padarray(elRawData   , [kSolveX - 1  , kSolveY - 1  , kSolveZ - 1  , 0], 'pre'  );
        elRawDataPad = padarray(elRawDataPad, [Nkx - kSolveX, Nky - kSolveY, Nkz - kSolveZ, 0], 'post' );

        [Nxpad, Nypad, Nzpad] = size(raw_fFOV_ogPad, 1, 2, 3);
        numKernelFits_Apad = (Nxpad - (Nkx - 1)) * (Nypad - (Nky - 1)) * (Nzpad - (Nkz - 1)); %the number of times the kernel fits across the A matrix
                

        [Apad_mc, Apadel]   = func_create_A(raw_fFOV_ogPad    , kernels_ones);
        [AelPad , AelPadel] = func_create_A(elRawDataPad, kernels_ones);
        targetColumn = zeros(numKernelShapes, size(AelPad, 1));

        kernel_target = kernels_solve(:, :, :, :, kernelShapeIter);
        kernel_source = kernels      (:, :, :, :, kernelShapeIter);
        targetColumn(kernelShapeIter, :) = AelPad(:, targetColumnIndex(1, kernelShapeIter));


        % [Apad_target_mc, Apadel_target] = func_create_A(rawUsPad, kernel_target);
        [Apad_source_mc, Apadel_source] = func_create_A(raw_fFOV_ogPad, kernel_source);

        Apad_source = reshape(Apad_source_mc, ...
                                        [size(Apad_source_mc, 1), ...
                                        size(Apad_source_mc, 2) * Nc]);
        Apad_target_update = Apad_source * squeeze(weightsKernels(:, :, kernelShapeIter));
        disp(strcat('going into final loop', 32,'loop iter', 32, num2str(kernelShapeIter)));
        rawInputSingleChannel = zeros(Nx, Ny, Nz);
        for channelIter = 1 : Nc
            rawInputSingleChannel(targetColumn(kernelShapeIter, :)) = ...
                Apad_target_update(:, channelIter);
            rawRecon(:, :, :, channelIter) = rawRecon(:, :, :, channelIter) + rawInputSingleChannel;

        end
        
    end

end