function [denoisedImage, net, groundTruth, baselineImage] = runDenoising(imagePath, learningRate, noiseSigma, numIterations)

% ADD UTILITY PATHS FOR FUNCTIONS
addpath('Utils/Common');
addpath('Utils/Architecture');
addpath('Utils/Denoising');

% LOAD IMAGE, PREPROCESS IT AND DISPLAY

imsize = -1;            % Original image size (-1 = keep as is)
d = 32;                 % Crop size multiple (to ensure divisibility by powers of 2)
sigma_ = noiseSigma / 255;   % Normalize to [0, 1] range
[img, ~] = get_image(imagePath, imsize);

switch imagePath
    case 'eco.jpg'
    img = imresize(img,0.25);
    img_cropped = crop_image(img,d);
    [img_clean_normalized, ~] = pil_to_np(img_cropped);
    img_noisy_normalized = get_noisy_image(img_clean_normalized, sigma_);
    sizes = size(img_noisy_pil);
    case 'snail.jpg'
    img_cropped = crop_image(img,d);
    sizes = size(img_cropped);
end

% DEFINE NETWORK (U-Net like architecture)

switch imagePath
    case 'eco.jpg'
        num_iter = numIterations; 
        figsize = 4; 
        input_depth = 3;                  % Input noise depth
        NET_TYPE = 'skip';                % Use skip architecture
        pad = 'reflection';               % Padding mode
        upsample_mode = 'bilinear';       % Upsampling method
        n_channels = 3;                   % Number of image channels
        need_sigmoid = true;             % Apply sigmoid at output
        need_bias = true;                % Include bias in layers
        skip_n33d = 128;                 % Down channels
        skip_n33u = 128;                 % Up channels
        skip_n11 = 4;                    % Skip connections
        downsample_mode = 'stride';     
        act_fun = 'LeakyReLU';          
        n_scales = 5;                    % Number of scales (depth of encoder-dsnailder)
        
        num_channels_down = repmat(skip_n33d, 1, n_scales); 
        num_channels_up = repmat(skip_n33u, 1, n_scales); 
        num_channels_skip = repmat(skip_n11, 1, n_scales);
        downsample_mode = repmat({downsample_mode}, 1, n_scales);
        
        num_input_channels = [input_depth, num_channels_down];
        num_output_channels = input_depth;
        filter_size_down = 3; 
        filter_size_up = 3; 
        filter_skip_size = 1; 
        need1x1_up = true;

    case 'snail.jpg'
        num_iter = numIterations;  
        figsize = 5; 
        input_depth = 3;
        NET_TYPE = 'skip'; 
        pad = 'reflection'; 
        upsample_mode = 'bilinear';
        n_channels = 3; 
        need_sigmoid = true; 
        need_bias = true;
        downsample_mode = 'stride';
        act_fun = 'LeakyReLU';
        
        num_scales = 5; 
        num_channels_down = [8, 16, 32, 64, 128]; 
        num_channels_up = [8, 16, 32, 64, 128]; 
        num_channels_skip = [0, 0, 0, 4, 4];   
        downsample_mode = repmat({downsample_mode}, 1, num_scales);
        
        num_input_channels = [input_depth, num_channels_down]; 
        num_output_channels = input_depth; 
        filter_size_down = 3; 
        filter_size_up = 3; 
        filter_skip_size = 1; 
        need1x1_up = true;
end 

net = skip_new(sizes, num_input_channels, num_output_channels, ...
    num_channels_down, num_channels_up, num_channels_skip, ...
    filter_size_down, filter_size_up, filter_skip_size, need_sigmoid, need_bias, pad, upsample_mode, ...
    downsample_mode, act_fun, need1x1_up);

net = dlnetwork(net);
switch imagePath
    case 'eco.jpg'
        noisy_dl = dlarray(single(img_noisy_normalized), 'SSCB');       % Noisy input
        clean_dl = dlarray(single(img_clean_normalized)/255., 'SSCB');    % Ground truth
    case 'snail.jpg'
        noisy_dl = get_noise(input_depth, 'uniform', [sizes(1) sizes(2)], 0.1); % Input noise
        clean_dl = dlarray(single(img_cropped)/255., 'SSCB');                   % Ground truth
end 

fprintf('Calculating BM3D baseline...\n');
if exist('bm3d', 'file')
    baselineImage = bm3d(img_noisy_normalized, noiseSigma);
else
    fprintf('BM3D not available, using bicubic as baseline...\n');
    baselineImage = imresize(imresize(img_noisy_normalized, 0.5), 2);
end

% TRAINING LOOP
psnr_log = zeros(numIterations, 1);
ssim_log = zeros(numIterations, 1);
loss_log = zeros(numIterations, 1);
show_every = 50;

tic
for i = 1:numIterations
    
    [loss, gradients] = dlfeval(@denoisingGradients, net, noisy_dl, clean_dl);

    if any(isnan(loss)) || any(isinf(loss))
        disp('Loss is NaN or Inf, stopping training.');
        break;
    end
    
    gradients = clipGradients(gradients, 1.0);
    net.Learnables = dlupdate(@(w, g) w - learningRate * g, net.Learnables, gradients);
    
    % Calculate metrics every few iterations
    if mod(i, show_every) == 0 || i == 1
        current_output = extractdata(forward(net, noisy_dl));
        [psnr_val, ssim_val] = calculateMetrics(extractdata(clean_dl), current_output);
        psnr_log(i) = psnr_val;
        ssim_log(i) = ssim_val;
        loss_log(i) = extractdata(loss);
        
        fprintf('Iteration %d: Loss=%.4f, PSNR=%.2f dB, SSIM=%.4f\n', ...
                i, extractdata(loss), psnr_val, ssim_val);
    else
        psnr_log(i) = psnr_log(max(1, i-1));
        ssim_log(i) = ssim_log(max(1, i-1));
        loss_log(i) = extractdata(loss);
    end
end
toc

denoisedImage = extractdata(forward(net,noisy_dl));
groundTruth = extractdata(clean_dl);

% Calculate final metrics and baseline comparisons
fprintf('\n=== FINAL RESULTS ===\n');

% Our method metrics
[final_psnr, final_ssim] = calculateMetrics(groundTruth, denoisedImage);
fprintf('Our Method - PSNR: %.2f dB, SSIM: %.4f\n', final_psnr, final_ssim);

% BM3D baseline metrics
[bm3d_psnr, bm3d_ssim] = calculateMetrics(groundTruth, baselineImage);
fprintf('BM3D Baseline - PSNR: %.2f dB, SSIM: %.4f\n', bm3d_psnr, bm3d_ssim);

% Bicubic baseline
switch imagePath
    case 'eco.jpg'
        bicubic_baseline = imresize(imresize(img_noisy_normalized, 0.5), 2);
    case 'snail.jpg'
        bicubic_baseline = imresize(imresize(img_cropped, 0.5), 2);
end
[bicubic_psnr, bicubic_ssim] = calculateMetrics(groundTruth, bicubic_baseline);
fprintf('Bicubic Baseline - PSNR: %.2f dB, SSIM: %.4f\n', bicubic_psnr, bicubic_ssim);

% Plot training progress
figure('Name', 'Training Progress');
subplot(2,2,1);
plot(1:numIterations, loss_log, 'b-', 'LineWidth', 2);
title('Training Loss');
xlabel('Iteration');
ylabel('Loss');
grid on;

subplot(2,2,2);
plot(1:numIterations, psnr_log, 'r-', 'LineWidth', 2);
title('PSNR Progress');
xlabel('Iteration');
ylabel('PSNR (dB)');
grid on;

subplot(2,2,3);
plot(1:numIterations, ssim_log, 'g-', 'LineWidth', 2);
title('SSIM Progress');
xlabel('Iteration');
ylabel('SSIM');
grid on;

subplot(2,2,4);
bar([final_psnr, bm3d_psnr, bicubic_psnr]);
set(gca, 'XTickLabel', {'Our Method', 'BM3D', 'Bicubic'});
title('PSNR Comparison');
ylabel('PSNR (dB)');
grid on;

fprintf('\nTraining completed successfully!\n');
end
