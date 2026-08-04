clc; clear; close all;

%% Global
S = (-2:0.01:2)';
n = numel(S);

% Random initial distribution
% x0 = (1/n) .* ones(n,1); 

% Custom initial distribution
a = -1;
m0 = a * ones(n,1);

sigma0 = 0.5;

dist2 = (S - m0).^2;

x0 = exp(-dist2/(2*sigma0^2));
x0 = x0/sum(x0);

% Time
tspan = [0,30];

%% Objective minimization
g = @(s) (3/2) * s.^4 - (1/4) * s.^3 - 3 * s.^2 + (3/4) * s + 1;

F = @(x) x' * g(S) - g(S);              % Vector payoff

f = @(t,x) x .* (F(x) - x' * F(x));     % Replicator dynamics

[tf,xf] = ode45(f, tspan, x0);          % Simulate dynamics
xf = xf';

%% Approximation minimization
q = @(s) (27/4) * (s + 1).^2 - 1;

Fq = @(x) x' * q(S) - q(S);             % Vector payoff

fq = @(t,x) x .* (Fq(x) - x' * Fq(x));  % Replicator dynamics

[tq,xq] = ode45(fq, tspan, x0);         % Simulate dynamics
xq = xq';

%% Plots
% Distribution Evolution
xfplot = xf ./ sum(xf,1);
xqplot = xq ./ sum(xq,1);

dS = S(2)-S(1);
xfdensity = xfplot / dS;
xqdensity = xqplot / dS;

figure;

subplot(1,2,1)
surf(tf,S,xfdensity,'EdgeColor','none');
xlabel('Time t');
ylabel('Strategy s');
zlabel('Distribution density');
title('Replicator dynamics for objective');
grid on;

subplot(1,2,2)
surf(tq,S,xqdensity,'EdgeColor','none');
xlabel('Time t');
ylabel('Strategy s');
zlabel('Distribution density');
title('Replicator dynamics for approximation');
grid on;

% Mean Trajectories
mf = S' * xfplot;           % mean trajectory for f
mq = S' * xqplot;           % mean trajectory for q

figure;

plot(tf,mf,'LineWidth',1.5);
hold on;
plot(tq,mq,'LineWidth',1.5);
yline(-1,'w--','LineWidth',1.2);
hold off;

xlabel('Time t');
ylabel('Mean strategy m(t)');
title('Mean trajectories for f and q');
legend('m(t) for f','m_q(t) for q','Global minimizer s = -1');
grid on;