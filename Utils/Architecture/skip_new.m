function model = skip_new(inputsize, num_input_channels, num_output_channels, num_channels_down, num_channels_up, num_channels_skip, filter_size_down, filter_size_up, filter_skip_size, need_sigmoid, need_bias, pad, upsample_mode, downsample_mode, act_fun, need1x1_up)

    % ---------- Handle optional/default arguments ----------
    if nargin < 17 || isempty(need1x1_up),     need1x1_up = true;         end
    if nargin < 16 || isempty(act_fun),        act_fun = 'LeakyReLU';     end
    if nargin < 15 || isempty(downsample_mode),downsample_mode = 'stride';end
    if nargin < 14 || isempty(upsample_mode),  upsample_mode = 'nearest'; end
    if nargin < 13 || isempty(pad),            pad = 'zero';              end
    if nargin < 12 || isempty(need_bias),      need_bias = true;          end
    if nargin < 11 || isempty(need_sigmoid),   need_sigmoid = true;       end
    if nargin < 10 || isempty(filter_skip_size),filter_skip_size = 1;     end
    if nargin < 9  || isempty(filter_size_up), filter_size_up = 3;        end
    if nargin < 8  || isempty(filter_size_down),filter_size_down = 3;     end
    if nargin < 7  || isempty(num_channels_skip),num_channels_skip = [4 4 4 4 4]; end
    if nargin < 6  || isempty(num_channels_up), num_channels_up = [16 32 64 128 128]; end
    if nargin < 5  || isempty(num_channels_down),num_channels_down = [16 32 64 128 128]; end
    if nargin < 4  || isempty(num_output_channels),num_output_channels = 3; end
    if nargin < 3  || isempty(num_input_channels),num_input_channels = 2; end

    % Number of scales (levels of encoder-decoder)
    n_scales = length(num_channels_down);
    
    % ---------- Expand scalars to lists if needed ----------
    if ~iscell(upsample_mode)
        upsample_mode = repmat({upsample_mode}, 1, n_scales);
    end
    if ~iscell(downsample_mode)
        downsample_mode = repmat({downsample_mode}, 1, n_scales);
    end
    if isscalar(filter_size_down)
        filter_size_down = repmat(filter_size_down, 1, n_scales);
    end
    if isscalar(filter_size_up)
        filter_size_up = repmat(filter_size_up, 1, n_scales);
    end
    if isscalar(num_channels_skip)
        num_channels_skip = repmat(num_channels_skip, 1, n_scales);
    end

    % Ensure all configuration vectors are same length
    if ~(length(num_channels_down) == length(num_channels_up) && length(num_channels_up) == length(num_channels_skip))
        error('num_channels_down, num_channels_up and num_channels_skip must have the same length.');
    end

    % Initialize main layer graph
    lgraph = layerGraph();
    inputLayer = imageInputLayer([inputsize(1) inputsize(2) num_input_channels], 'Name', 'Input');
    lgraph = addLayers(lgraph, inputLayer);

    skipConnections = cell(1, n_scales); % To store skip connections
    currentInput = 'Input'; % The first input is the 'Input' layer
    
    for i = 1:n_scales
        blockName = ['enc' num2str(i)];
        [lgraph, currentOutput, skipConnections{i}] = addEncoderBlock(...
            lgraph, currentInput, blockName, num_channels_down(i));
        
        currentInput = currentOutput; % The output of this block is the input of the next one
    end

    % Use addDecoderBlock (because it only has Conv-Conv)
    [lgraph, bottleneckOutput] = addDecoderBlock(...
        lgraph, currentInput, 'bottleneck', num_channels_down(end));
        
    currentInput = bottleneckOutput;

    for i = n_scales:-1:1 % Now we go backwards to go up
        
        % Upsampling (using TransposedConv)
        upName = ['dec' num2str(i) '_upsample'];
        upLayer = transposedConv2dLayer(2, num_channels_up(i), 'Stride', 2, 'Name', upName, 'BiasLearnRateFactor', 0);
        lgraph = addLayers(lgraph, upLayer);
        lgraph = connectLayers(lgraph, currentInput, upName);
        
        currentDecoderInput = upName;

        if num_channels_skip(i) > 0
            [lgraph, processedSkip] = addSkipConnection(...
                lgraph, skipConnections{i}, ['skip' num2str(i)], num_channels_skip(i));
            
            % Concatenate (using ConcatCropLayer)
            concatName = ['dec' num2str(i) '_concat'];
            concatLayer = createConcatLayer(3, concatName, 2);
            lgraph = addLayers(lgraph, concatLayer); 
            lgraph = connectLayers(lgraph, upName, [concatName '/in1']);
            lgraph = connectLayers(lgraph, processedSkip, [concatName '/in2']);
            
            currentDecoderInput = concatName; % The input to the block will be the concatenation
        end

        [lgraph, decoderOutput] = addDecoderBlock(...
            lgraph, currentDecoderInput, ['dec' num2str(i)], num_channels_up(i));
            
        currentInput = decoderOutput; % The output of this level is the input of the next one
    end

    convfinal = conv(num_channels_up(1), num_output_channels, 1, 1, need_bias, pad, 'stride', 'Final');

    if need_sigmoid
        sigmoid = sigmoidLayer('Name', 'Final_sigmoid');         
        lgraph = addLayers(lgraph, [convfinal, sigmoid]);
    else
        lgraph = addLayers(lgraph, convfinal);
    end

    lgraph = connectLayers(lgraph, currentInput, convfinal(1).Name);

    model = lgraph;
end
