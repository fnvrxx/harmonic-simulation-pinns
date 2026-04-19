function net = unpack_params(net, w)
% Masukkan flat row vector double ke dalam Learnables dlnetwork.
    vals   = net.Learnables.Value;
    offset = 1;
    for i = 1:numel(vals)
        sz  = size(vals{i});
        n   = prod(sz);
        net.Learnables.Value{i} = dlarray(reshape(w(offset:offset+n-1), sz));
        offset = offset + n;
    end
end
