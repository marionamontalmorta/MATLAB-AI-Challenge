function concatLayer = createConcatLayer(dim, name, numInputs)
    % createConcatLayer - Helper function to create a ConcatCropLayer
    %
    % Inputs:
    %   dim - Dimension to concatenate along (typically 3 for channels)
    %   name - Name for the layer
    %   numInputs - Number of inputs (optional, defaults to 2)
    %
    % Outputs:
    %   concatLayer - ConcatCropLayer object
    
    if nargin < 3
        numInputs = 2;
    end
    
    concatLayer = ConcatCropLayer(dim, name, numInputs);
end
