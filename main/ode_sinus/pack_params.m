function w = pack_params(net)
% Ekstrak semua parameter dlnetwork menjadi flat row vector double.
    vals = net.Learnables.Value;
    w = [];
    for i = 1:numel(vals)
        w = [w, double(extractdata(vals{i}(:))')]; %#ok<AGROW>
    end
end
