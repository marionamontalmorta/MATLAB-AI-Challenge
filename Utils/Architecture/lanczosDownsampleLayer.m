function layer = lanczosDownsampleLayer(numChannels, stride, kernel_type)
    % Custom Lanczos downsampling layer for MATLAB compatibility
    % This creates a custom layer that applies Lanczos downsampling
    
    % Create a custom layer class
    layer = lanczosDownsampleCustomLayer(numChannels, stride, kernel_type);
end

classdef lanczosDownsampleCustomLayer < nnet.layer.Layer
    % Custom layer for Lanczos downsampling
    
    properties (Learnable)
        % No learnable parameters needed for Lanczos
    end
    
    properties
        % Layer properties
        NumChannels
        Stride
        KernelType
        Kernel
    end
    
    methods
        function layer = lanczosDownsampleCustomLayer(numChannels, stride, kernel_type)
            % Constructor
            layer.Name = 'lanczosDownsample';
            layer.Description = 'Lanczos downsampling layer';
            layer.Type = 'Custom Layer';
            
            layer.NumChannels = numChannels;
            layer.Stride = stride;
            layer.KernelType = kernel_type;
            
            % Generate Lanczos kernel
            layer.Kernel = layer.generateLanczosKernel(stride, kernel_type);
        end
        
        function kernel = generateLanczosKernel(layer, factor, kernel_type)
            % Generate Lanczos kernel based on factor and type
            if strcmp(kernel_type, 'lanczos2')
                support = 2;
                kernel_width = 4 * factor + 1;
            elseif strcmp(kernel_type, 'lanczos3')
                support = 3;
                kernel_width = 6 * factor + 1;
            else
                support = 2;
                kernel_width = 4 * factor + 1;
            end
            
            kernel = zeros(kernel_width);
            center = (kernel_width + 1) / 2;
            
            for i = 1:kernel_width
                for j = 1:kernel_width
                    di = abs(i - center) / factor;
                    dj = abs(j - center) / factor;
                    
                    val = 1;
                    if di ~= 0 && di < support
                        val = val * support * sin(pi * di) * sin(pi * di / support) / (pi^2 * di^2);
                    elseif di >= support
                        val = 0;
                    end
                    
                    if dj ~= 0 && dj < support
                        val = val * support * sin(pi * dj) * sin(pi * dj / support) / (pi^2 * dj^2);
                    elseif dj >= support
                        val = 0;
                    end
                    
                    kernel(i,j) = val;
                end
            end
            
            kernel = kernel / sum(kernel(:)); % Normalize
        end
        
        function Z = predict(layer, X)
            % Forward pass - apply Lanczos downsampling
            Z = layer.forward(X);
        end
        
        function Z = forward(layer, X)
            % Apply Lanczos downsampling
            [H, W, C, N] = size(X);
            
            % Apply convolution with Lanczos kernel for each channel
            Z = zeros(ceil(H/layer.Stride), ceil(W/layer.Stride), C, N, 'like', X);
            
            for n = 1:N
                for c = 1:C
                    % Apply 2D convolution with Lanczos kernel
                    conv_result = conv2(X(:,:,c,n), layer.Kernel, 'same');
                    % Downsample by stride
                    Z(:,:,c,n) = conv_result(1:layer.Stride:end, 1:layer.Stride:end);
                end
            end
        end
    end
end
