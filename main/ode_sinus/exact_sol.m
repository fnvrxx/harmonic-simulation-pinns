function [u, x] = exact_sol(N)
% Solusi analitik ODE sinus: u'' + u^3 = -sin(x) + sin^3(x)
% Solusi eksak: u(x) = sin(x),  x in [0, pi]
    if nargin < 1
        N = 500;
    end
    x = linspace(0, pi, N)';
    u = sin(x);
end
