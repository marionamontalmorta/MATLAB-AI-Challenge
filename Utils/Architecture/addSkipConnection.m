function [lgraph, outputName] = addSkipConnection(lgraph, inputName, blockName, numChannels)
    % addSkipConnection - Adds a skip connection block to the layer graph
    %
    % Inputs:
    %   lgraph - Current layer graph
    %   inputName - Name of the input layer to connect to
    %   blockName - Base name for this block (e.g., 'Skip_1')
    %   numChannels - Number of output channels
    %
    % Outputs:
    %   lgraph - Updated layer graph
    %   outputName - Name of the output layer
    
    % 1x1 convolution block for the skip connection
    layers = [
        convolution2dLayer(1, numChannels, 'Padding', 'same', 'Name', [blockName '_conv'])
        reluLayer('Name', [blockName '_relu'])
    ];
    lgraph = addLayers(lgraph, layers);
    lgraph = connectLayers(lgraph, inputName, [blockName '_conv']);

    % The output of this block is the last activation
    outputName = [blockName '_relu'];
end
