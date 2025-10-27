function [lgraph, outputName] = addDecoderBlock(lgraph, inputName, blockName, numFilters)
    % addDecoderBlock - Adds a complete decoder block to the layer graph
    %
    % Inputs:
    %   lgraph - Current layer graph
    %   inputName - Name of the input layer to connect to
    %   blockName - Base name for this block (e.g., 'Tail_1')
    %   numFilters - Number of filters for convolutions
    %
    % Outputs:
    %   lgraph - Updated layer graph
    %   outputName - Name of the output layer
    
    % Bloc de dues convolucions per al decoder
    layers = [
        convolution2dLayer(3, numFilters, 'Padding', 'same', 'Name', [blockName '_conv1'])
        reluLayer('Name', [blockName '_relu1'])
        convolution2dLayer(3, numFilters, 'Padding', 'same', 'Name', [blockName '_conv2'])
        reluLayer('Name', [blockName '_relu2'])
    ];
    lgraph = addLayers(lgraph, layers);
    lgraph = connectLayers(lgraph, inputName, [blockName '_conv1']);

    % The output of this block is the last activation
    outputName = [blockName '_relu2'];
end
