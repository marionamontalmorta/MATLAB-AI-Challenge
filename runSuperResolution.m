function [superResImage, net, groundTruth, baselineImage] = runSuperResolution(imagePath, learningRate, numIterations, scaleFactor, noiseSigma)

% ADD UTILITY PATHS FOR FUNCTIONS
addpath('Utils/Common');
addpath('Utils/Architecture');
addpath('Utils/Super-resolution');

% LOAD IMAGE, PREPROCESS IT AND DISPLAY
imsize = -1;            % Original image size (-1 = keep as is)
d = 32;                 % Crop size multiple (to ensure divisibility by powers of 2)
sigma_ = noiseSigma / 255;   % Normalize to [0, 1] range
enforse_div32 = 'CROP';
[img, ~] = get_image(imagePath, imsize);

[img_orig_pil, img_orig_np,img_LR_pil,img_LR_np,img_HR_pil,img_HR_np] = load_LR_HR_imgs_sr(imagePath, imsize, scaleFactor, enforse_div32);
[img_bicubic_np, img_bic_sharp_np, img_nearest_np] = get_baselines(img_LR_pil, img_HR_pil);
sizes = size(img_HR_np);

% DEFINE NETWORK (U-Net like architecture)
input_depth = 3;                  % Input noise depth
input = 'noise';
pad = 'reflection';               % Padding mode
opt_over = 'net';
kernel_type = 'lanczos2';
tv_weight = 0.0;
optimizer = 'adam';

mode = 'normal';
img_size = size(img_HR_np);
var = 1/10;

net_input = get_noise(input_depth, mode, img_size(1,1:2), var);

figsize = 4; 
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

net = skip_new(sizes, num_input_channels, num_output_channels, ...
    num_channels_down, num_channels_up, num_channels_skip, ...
    filter_size_down, filter_size_up, filter_skip_size, need_sigmoid, need_bias, pad, upsample_mode, ...
    downsample_mode, act_fun, need1x1_up);

net = dlnetwork(net);
LR_dl = dlarray(single(img_LR_np), 'SSCB');       % Noisy input
HR_dl = dlarray(single(img_HR_np), 'SSCB');    % Ground truth

% TRAINING LOOP
psnr_log = zeros(numIterations, 1);
ssim_log = zeros(numIterations, 1);
loss_log = zeros(numIterations, 1);
show_every = 50;

n_planes = 3; kernel_type = 'lanczos2'; phase = 0; kernel_width = 4 * scaleFactor + 1; support = 2; preserve_size = false;
down = Downsampler(n_planes, scaleFactor, kernel_type, phase, kernel_width, support, noiseSigma, preserve_size);

tic
for i = 1:numIterations
    
    [loss, gradients] = dlfeval(@superResGradients, net, net_input, HR_dl);
   
    if any(isnan(loss)) || any(isinf(loss))
        disp('Loss is NaN or Inf, stopping training.');
        break;
    end

    gradients = clipGradients(gradients, 1.0);
    net.Learnables = dlupdate(@(w, g) w - learningRate * g, net.Learnables, gradients);
    
    % Calculate metrics every few iterations
    if mod(i, show_every) == 0 || i == 1
        current_output = extractdata(forward(net, net_input));
        [psnr_val, ssim_val] = calculateMetrics(extractdata(HR_dl), current_output);
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

superResImage = extractdata(forward(net, net_input));
groundTruth = extractdata(HR_dl);
baselineImage = img_bicubic_np;

% Calculate final metrics and baseline comparisons
fprintf('\n=== FINAL RESULTS ===\n');

% Our method metrics
[final_psnr, final_ssim] = calculateMetrics(groundTruth, superResImage);
fprintf('Our Method - PSNR: %.2f dB, SSIM: %.4f\n', final_psnr, final_ssim);

% Bicubic baseline metrics
[bicubic_psnr, bicubic_ssim] = calculateMetrics(groundTruth, baselineImage);
fprintf('Bicubic Baseline - PSNR: %.2f dB, SSIM: %.4f\n', bicubic_psnr, bicubic_ssim);

% Nearest neighbor baseline
nearest_baseline = imresize(img_LR_np, scaleFactor, 'nearest');
[nearest_psnr, nearest_ssim] = calculateMetrics(groundTruth, nearest_baseline);
fprintf('Nearest Neighbor Baseline - PSNR: %.2f dB, SSIM: %.4f\n', nearest_psnr, nearest_ssim);

% Plot training progress
figure('Name', 'Super-Resolution Training Progress');
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
bar([final_psnr, bicubic_psnr, nearest_psnr]);
set(gca, 'XTickLabel', {'Our Method', 'Bicubic', 'Nearest'});
title('PSNR Comparison');
ylabel('PSNR (dB)');
grid on;

fprintf('\nSuper-resolution training completed successfully!\n');
end