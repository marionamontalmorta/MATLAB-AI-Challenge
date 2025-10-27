function layer = conv(in_f, out_f, kernel_size, stride, bias, pad, downsample_mode, name)

if nargin < 4
    stride = 1;
end
if nargin < 5
    bias = true;
end
if nargin < 6
    pad = 'zero';
end
if nargin < 7
    downsample_mode = 'stride';
end
if nargin < 8
    name = 'convblock'; % default name prefix if none given
end

downsampler = [];
if stride ~= 1 && ~strcmp(downsample_mode, 'stride')
    switch downsample_mode
        case 'avg'
            downsampler = averagePooling2dLayer(stride, 'Stride', stride, 'Name', [name]);
        case 'max'
            downsampler = maxPooling2dLayer(stride, 'Stride', stride, 'Name', [name]);
        otherwise
            if any(strcmp(downsample_mode, {'lanczos2', 'lanczos3'}))
                downsampler = lanczosDownsampleLayer(out_f, stride, downsample_mode); 
                downsampler.Name = name;
            else
                error('Unsupported downsample_mode: %s', downsample_mode);
            end
    end
    stride = 1;
end

% Handle padding
padder = [];
to_pad = floor((kernel_size - 1) / 2);
if strcmp(pad, 'reflection')
    padder = ReflectionPad2dLayer([to_pad to_pad], [name '_pad']);
    to_pad = 0;
end

% Convolution layer
convLayer = convolution2dLayer(kernel_size, out_f, 'Stride', stride, 'Padding', to_pad, 'Name', [name '_conv']);

% Combine all non-empty layers into array
layers = {};
if ~isempty(padder)
    layers{end + 1} = padder;
end
layers{end + 1} = convLayer;
if ~isempty(downsampler)
    layers{end + 1} = downsampler;
end

layer = [layers{:}];
end