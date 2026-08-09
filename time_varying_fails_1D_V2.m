clc; clear; close all;

%% Global
S = (-1.875:0.01:2.125)';
n = numel(S);

%% Version 2: Stuck at local maxima
% Uniform initial distribution
x0 = (1/n) .* ones(n,1); 

% % Custom initial distribution
% a = 1;
% m0 = a * ones(n,1);
% 
% sigma0 = 0.2;
% 
% dist2 = (S - m0).^2;
% 
% x0 = exp(-dist2/(2*sigma0^2));
% x0 = x0/sum(x0);

% Time
tspan = [0,10];

%  Objective minimization and derivatives
g = @(s) (3/2) * s.^4 - (1/4) * s.^3 - 3 * s.^2 + (3/4) * s + 1;
dg = @(s) 6 * s.^3 - (3/4) * s.^2 - 6 * s + 3/4;
ddg = @(s) 18 * s.^2 - (3/2) * s - 6;

F = @(x) x' * g(S) - g(S);              % Vector payoff

f = @(t,x) x .* (F(x) - x' * F(x));     % Replicator dynamics

[tg,xg] = ode45(f, tspan, x0);          % Simulate dynamics
xg = xg';

%% Approximation minimization centered at global minima
qg = @(s) (27/4) * (s + 1).^2 - 1;

Fqg = @(x) x' * qg(S) - qg(S);             % Vector payoff

fqg = @(t,x) x .* (Fqg(x) - x' * Fqg(x));  % Replicator dynamics

[tqg,xqg] = ode45(fqg, tspan, x0);         % Simulate dynamics
xqg = xqg';

%% Approximation minimization with time varying center
m = @(x) x' * S;

q = @(m,s) g(m) + ...
    dg(m) * (s - m) + ...
    0.5 * ddg(m) * (s - m).^2;

Fq = @(x) x' * q(m(x),S) - q(m(x),S);   % Vector payoff

fq = @(t,x) x .* (Fq(x) - x' * Fq(x));  % Replicator dynamics

[tq,xq] = ode45(fq, tspan, x0);         % Simulate dynamics
xq = xq';

%% Plots
% Distribution Evolution
dS = S(2)-S(1);
xgDensity = xg / dS;
xqgDensity = xqg / dS;
xqDensity = xq / dS;

figure;

subplot(2,2,1)
surf(tg,S,xgDensity,'EdgeColor','none');
xlabel('Time t');
ylabel('Strategy s');
zlabel('Distribution density');
title('Objective');
grid on;

subplot(2,2,2)
surf(tqg,S,xqgDensity,'EdgeColor','none');
xlabel('Time t');
ylabel('Strategy s');
zlabel('Distribution density');
title('Approximation at Optimizer');
grid on;

subplot(2,2,[3 4])
surf(tq,S,xqDensity,'EdgeColor','none');
xlabel('Time t');
ylabel('Strategy s');
zlabel('Distribution density');
title('Time Varying Approximation');
ylim([-1.875, 2.125])
grid on;

sgtitle('Strategy Distribution Updates via Replicator Dynamics')

% Mean Trajectories
mg = S' * xg;
mqg = S' * xqg;
mq = S' * xq;

figure;

plot(tg,mg,'LineWidth',2,'Color','#1D2DCF');
hold on;
plot(tqg,mqg,'LineWidth',2,'Color','#CF1DB1');
plot(tq,mq,'LineWidth',2,'Color','#AE1DCF');
yline(-1,'k--','LineWidth',1.5);
hold off;

xlabel('Time t');
ylabel('Mean strategy m(t)');
title('Mean trajectories');
legend('m(t) for g','m_q(t) for q centered at optimizer', ...
    'm_q(t) for time varying q','Global minimizer s = -1');
grid on;