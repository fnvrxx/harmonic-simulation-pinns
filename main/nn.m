%% Arsitektur Neural Network sebagai dlnetwork
% Menggunakan dlnetwork agar mendukung automatic differentiation (dlfeval)
% Arsitektur: 1 -> hidden_size -> hidden_size -> 1
%
% Output: net (dlnetwork, siap untuk training manual / PINN)
function net = nn(hidden_size)
    layers = [
        featureInputLayer(1, 'Normalization', 'none', 'Name', 'input')
        fullyConnectedLayer(hidden_size, 'Name', 'fc1')
        tanhLayer('Name', 'tanh1')
        fullyConnectedLayer(hidden_size, 'Name', 'fc2')
        tanhLayer('Name', 'tanh2')
        fullyConnectedLayer(1, 'Name', 'fc_out')
    ];
    net = dlnetwork(layers);
end