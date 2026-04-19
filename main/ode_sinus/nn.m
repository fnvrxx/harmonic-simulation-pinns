function net = nn(hidden_size)
% Arsitektur: 1 -> hidden -> hidden -> 1  (2 hidden layers, tanh)
% Inisialisasi: Xavier uniform (weights dan bias)
    layers = [
        featureInputLayer(1, 'Normalization', 'none', 'Name', 'input')
        fullyConnectedLayer(hidden_size, 'Name', 'fc1')
        tanhLayer('Name', 'tanh1')
        fullyConnectedLayer(hidden_size, 'Name', 'fc2')
        tanhLayer('Name', 'tanh2')
        fullyConnectedLayer(1, 'Name', 'fc_out')
    ];
    net = dlnetwork(layers);

    configs = [
        1,           hidden_size;   % fc1
        hidden_size, hidden_size;   % fc2
        hidden_size, 1             % fc_out
    ];
    layerNames = ["fc1", "fc2", "fc_out"];

    for i = 1:3
        fan_in  = configs(i, 1);
        fan_out = configs(i, 2);
        bound   = sqrt(6 / (fan_in + fan_out));

        wIdx = net.Learnables.Layer == layerNames(i) & net.Learnables.Parameter == "Weights";
        bIdx = net.Learnables.Layer == layerNames(i) & net.Learnables.Parameter == "Bias";

        wSize = size(net.Learnables.Value{wIdx});
        bSize = size(net.Learnables.Value{bIdx});

        net.Learnables.Value{wIdx} = dlarray((rand(wSize)*2 - 1) * bound);
        net.Learnables.Value{bIdx} = dlarray((rand(bSize)*2 - 1) * bound);
    end
end
