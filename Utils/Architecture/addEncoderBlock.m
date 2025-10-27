function [lgraph, outputName, skipConnectionName] = addEncoderBlock(lgraph, inputName, blockName, numFilters)
    % addEncoderBlock - Adds a complete encoder block to the layer graph
    %
    % Inputs:
    %   lgraph - Current layer graph
    %   inputName - Name of the input layer to connect to
    %   blockName - Base name for this block (e.g., 'Deep_1')
    %   numFilters - Number of filters for convolutions
    %
    % Outputs:
    %   lgraph - Updated layer graph
    %   outputName - Name of the output layer (after pooling)
    %   skipConnectionName - Name of the layer for skip connection (before pooling)
    
    % Bloc de dues convolucions
    layers = [
        convolution2dLayer(3, numFilters, 'Padding', 'same', 'Name', [blockName '_conv1'])
        reluLayer('Name', [blockName '_relu1'])
        convolution2dLayer(3, numFilters, 'Padding', 'same', 'Name', [blockName '_conv2'])
        reluLayer('Name', [blockName '_relu2'])
    ];
    lgraph = addLayers(lgraph, layers);
    lgraph = connectLayers(lgraph, inputName, [blockName '_conv1']);

    % The output for the 'skip connection' is from this block
    skipConnectionName = [blockName '_relu2'];

    % Pooling block to go down one level
    poolLayer = maxPooling2dLayer(2, 'Stride', 2, 'Name', [blockName '_pool']);
    lgraph = addLayers(lgraph, poolLayer);
    lgraph = connectLayers(lgraph, skipConnectionName, [blockName '_pool']);

    % The output of this block is the pooling layer, which will be the input of the next one
    outputName = [blockName '_pool'];
end
