%% Defines the analytical solution to the under-damped harmonic oscillator problem
% input  : d (damping), w0 (natural frequency), t_max (end time), N (jumlah titik)
% output : u (solusi), time (vektor waktu)
function [u, time] = exact_sol(d_input, w0_input, t_max, N)
    if nargin < 4
        N = 300;  % default 300 titik
    end
    d  = d_input;
    w0 = w0_input;
    time = linspace(0, t_max, N);  % vektor waktu seragam

    % solusi analitik
    w   = sqrt(w0^2 - d^2);
    phi = atan(-d/w);
    A   = 1 / (2*cos(phi));
    u   = exp(-d*time) .* 2*A .* cos(phi + w*time);
end
