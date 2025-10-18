function data = readNPY_simple(filename)
    
    fid = fopen(filename, 'rb'); %file is opened in binary mode
    if fid < 0
        error('Cannot open file: %s', filename);
    end
    
    magic = fread(fid, 6, '*uint8'); %read magic string that is NUMPY
    
    ver = fread(fid, 2, '*uint8');
    
    if ver(1) == 1
        hdr_len = fread(fid, 1, '*uint16');
    else
        hdr_len = fread(fid, 1, '*uint32');
    end
    
    hdr = char(fread(fid, hdr_len, '*uint8')');
    
    shape_match = regexp(hdr, 'shape.*?\(([^\)]+)\)', 'tokens');
    if isempty(shape_match)
        fclose(fid);
        error('Cannot find shape in header: %s', hdr);
    end
    
    shape_str = strtrim(shape_match{1}{1});
    shape_str = regexprep(shape_str, ',\s*$', '');  
    %Convert shape string to number array
    if isempty(shape_str)
        shape = [1, 1];
    else
        shape_parts = strsplit(shape_str, ',');
        shape = [];
        for i = 1:length(shape_parts)
            num = str2double(strtrim(shape_parts{i}));
            if ~isnan(num)
                shape = [shape, num];
            end
        end
    end

    % for np to matlab dtype mapping
    if contains(hdr, '<f4') || contains(hdr, 'f4')
        dtype = '*float32';
    elseif contains(hdr, '<f8') || contains(hdr, 'f8')
        dtype = '*float64';
    elseif contains(hdr, '<i4') || contains(hdr, 'i4')
        dtype = '*int32';
    elseif contains(hdr, '<i8') || contains(hdr, 'i8')
        dtype = '*int64';
    elseif contains(hdr, '<i2') || contains(hdr, 'i2')
        dtype = '*int16';
    elseif contains(hdr, '<i1') || contains(hdr, 'i1')
        dtype = '*int8';
    else
        dtype = '*float32';
    end
    
    data = fread(fid, inf, dtype);
    fclose(fid);
    if length(shape) == 1
        data = reshape(data, [shape(1), 1]);
    elseif length(shape) == 2
        data = reshape(data, [shape(2), shape(1)])';
    else
        data = reshape(data, fliplr(shape));
        data = permute(data, length(shape):-1:1);
    end
end