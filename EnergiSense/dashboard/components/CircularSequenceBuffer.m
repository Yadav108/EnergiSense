classdef CircularSequenceBuffer < handle
    % CIRCULARSEQUENCEBUFFER Circular buffer for storing sequence data
    %
    % This class provides efficient storage and retrieval of sequential data
    % for RNN processing with constant memory usage.
    %
    % Properties:
    %   - Fixed memory usage regardless of data length
    %   - Automatic overwrite of oldest data when full
    %   - Multi-feature support for time series data
    %
    % Usage:
    %   buffer = CircularSequenceBuffer(sequenceLength, numFeatures)
    %   buffer.add(newDataPoint)
    %   sequence = buffer.getSequence()
    
    properties (Access = private)
        Data
        Size
        Features
        Index
        IsFull
    end
    
    methods
        function obj = CircularSequenceBuffer(sequenceLength, numFeatures)
            % Constructor - initialize circular buffer
            obj.Size = sequenceLength;
            obj.Features = numFeatures;
            obj.Data = zeros(sequenceLength, numFeatures);
            obj.Index = 1;
            obj.IsFull = false;
        end
        
        function add(obj, newData)
            % Add new data point to buffer
            if length(newData) ~= obj.Features
                newData = newData(1:min(length(newData), obj.Features));
                if length(newData) < obj.Features
                    newData(end+1:obj.Features) = 0;
                end
            end
            
            obj.Data(obj.Index, :) = newData;
            obj.Index = obj.Index + 1;
            
            if obj.Index > obj.Size
                obj.Index = 1;
                obj.IsFull = true;
            end
        end
        
        function sequence = getSequence(obj)
            % Get complete sequence in chronological order
            if obj.IsFull
                sequence = [obj.Data(obj.Index:end, :); obj.Data(1:obj.Index-1, :)];
            else
                sequence = obj.Data(1:obj.Index-1, :);
            end
        end
        
        function count = getTotalPoints(obj)
            % Get total number of data points stored
            if obj.IsFull
                count = obj.Size;
            else
                count = obj.Index - 1;
            end
        end
    end
end