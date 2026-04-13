%% PINN - Damped Harmonic Oscillator
% Persamaan: mu*u'' + k*u' + u = 0  (dalam notasi: m=1, mu=2d, k=w0^2)
% Kondisi awal: u(0) = 1, u'(0) = 0

clc; 
clear; 
close all;

%% 1. Parameter Sistem
d    = 2;     % koefisien redaman
w0   = 20;    % frekuensi alami
mu   = 2*d;   % setara torch: mu = 2*d
k    = w0^2;  % setara torch: k  = w0^2
t_max = 1;

%% 2. Solusi Analitik (ground truth)
N_test = 300;
[u_exact, t_test] = exact_sol(d, w0, t_max, N_test);
% t_test  : 1 x 300  vektor waktu
% u_exact : 1 x 300  solusi eksakta

%% 3. Titik Training

% -- Boundary: t=0 (kondisi awal) --
% setara: t_boundary = torch.tensor(0.).view(-1,1).requires_grad_(True)
t_boundary = dlarray(0, 'BC');   % 1x1, format BC = batch x channel

% -- Physics: 30 titik merata di [0,1] --
% setara: t_physics = torch.linspace(0,1,30).view(-1,1).requires_grad_(True)
N_phys     = 30;
t_phys_vec = linspace(0, t_max, N_phys)';   % 30x1 (kolom = 30 batch)
t_physics  = dlarray(t_phys_vec, 'BC');     % 30x1 dlarray, tiap baris = 1 sample

%% 4. Inisialisasi Network
net = nn(32);   % hidden layer size = 32

%% 5. Hyperparameter Training
lr         = 1e-3;    % learning rate
n_epochs   = 10000;   % jumlah iterasi
lambda_bc  = 1e4;     % bobot boundary loss (lebih besar agar kondisi awal dipenuhi)

loss_history = zeros(1, n_epochs);

%% 6. Training Loop
for epoch = 1:n_epochs

    % --- Hitung loss + gradien via dlfeval ---
    [loss, grads] = dlfeval(@computeLoss, net, ...
        t_boundary, t_physics, mu, k, lambda_bc);

    % --- Update parameter (Adam) ---
    [net, ~] = adamupdate(net, grads, [], [], epoch, lr);

    % --- Catat loss ---
    loss_history(epoch) = extractdata(loss);

    % --- Print tiap 1000 epoch ---
    if mod(epoch, 1000) == 0
        fprintf('Epoch %5d | Loss = %.6f\n', epoch, loss_history(epoch));
    end
end

%% 7. Prediksi Akhir
t_dl   = dlarray(t_test', 'BC');             % 300x1 kolom
u_pred = extractdata(forward(net, t_dl))';  % transpose balik ke 1x300

%% 8. Visualisasi
figure;

subplot(2,1,1);
plot(t_test, u_exact, 'b-',  'LineWidth', 2, 'DisplayName', 'Solusi Eksakta'); hold on;
plot(t_test, u_pred,  'r--', 'LineWidth', 2, 'DisplayName', 'Prediksi PINN');
xlabel('t (s)'); ylabel('u(t)');
title('PINN vs Solusi Analitik - Osilator Harmonik Teredam');
legend; grid on;

subplot(2,1,2);
semilogy(1:n_epochs, loss_history, 'k-', 'LineWidth', 1.5);
xlabel('Epoch'); ylabel('Loss (log scale)');
title('Kurva Loss Training');
grid on;

