classdef CircularBuffer < handle
    % CIRCULARBUFFER Circular buffer for efficient data storage
    %
    % This class provides a fixed-size buffer that overwrites the oldest
    % data when full, maintaining constant memory usage.
    %
    % Properties:
    %   - Fixed memory footprint
    %   - Automatic overwrite behavior
    %   - Chronological data retrieval
    %   - Memory efficient for continuous data streams
    %
    % Usage:
    %   buffer = CircularBuffer(size)
    %   buffer.add(value)
    %   data = buffer.getData()
    
    properties (Access = private)
        Data
        Size
        Index
        IsFull
    end
    
    methods
        function obj = CircularBuffer(size)
            % Constructor - initialize circular buffer
            obj.Size = size;
            obj.Data = zeros(size, 1);
            obj.Index = 1;
            obj.IsFull = false;
        end
        
        function add(obj, value)
            % Add new value to buffer
            obj.Data(obj.Index) = value;
            obj.Index = obj.Index + 1;
            
            if obj.Index > obj.Size
                obj.Index = 1;
                obj.IsFull = true;
            end
        end
        
        function data = getData(obj)
            % Get all data in chronological order
            if obj.IsFull
                data = [obj.Data(obj.Index:end); obj.Data(1:obj.Index-1)];
            else
                data = obj.Data(1:obj.Index-1);
            end
        end
        
        function count = getCount(obj)
            % Get number of elements currently stored
            if obj.IsFull
                count = obj.Size;
            else
                count = obj.Index - 1;
            end
        end
        
        function reset(obj)
            % Reset buffer to empty state
            obj.Data(:) = 0;
            obj.Index = 1;
            obj.IsFull = false;
        end
    end
end