clc; clear; close all;

%% Global
S = (-2:0.01:2)';
n = numel(S);

% Uniform initial distribution
x0 = (1/n) .* ones(n,1); 

% % Custom initial distribution
% a = 1;
% m0 = a * ones(n,1);
% 
% sigma0 = 0.3;
% 
% dist2 = (S - m0).^2;
% 
% x0 = exp(-dist2/(2*sigma0^2));
% x0 = x0/sum(x0);

% Time
tspan = [0,60];

%% Objective minimization and derivatives
g = @(s) (3/2) * s.^4 - (1/4) * s.^3 - 3 * s.^2 + (3/4) * s + 1;
dg = @(s) 6 * s.^3 - (3/4) * s.^2 - 6 * s + 3/4;
ddg = @(s) 18 * s.^2 - (3/2) * s - 6;

F = @(x) x' * g(S) - g(S);              % Vector payoff

f = @(t,x) x .* (F(x) - x' * F(x));     % Replicator dynamics

[tg,xg] = ode45(f, tspan, x0);          % Simulate dynamics
xg = xg';

%% Global quadratic approximation 
sStar = -1;         % Global minimizer

qg = @(s) (27/4) * (s + 1).^2 - 1;

Fqg = @(x) x' * qg(S) - qg(S);

fqg = @(t,x) x .* (Fqg(x) - x' * Fqg(x));

[tqg,xqg] = ode45(fqg, tspan, x0);
xqg = xqg';

%% Time-varying quadratic approximation
m = @(x) x' * S;

q = @(m,s) g(m) + ...
    dg(m) * (s - m) + ...
    0.5 * ddg(m) * (s - m).^2;

Fq = @(x) x' * q(m(x),S) - q(m(x),S);

fq = @(t,x) x .* (Fq(x) - x' * Fq(x));

[tq,xq] = ode45(fq, tspan, x0);
xq = xq';

%% Plots
% Set default interpreter as LaTeX
set(0,'defaultTextInterpreter','latex');

%% Plot 1
% Distribution Evolution
dS = S(2)-S(1);
xgDensity = xg / dS;
xqgDensity = xqg / dS;
xqDensity = xq / dS;

figure;

subplot(2,2,1)
surf(tg,S,xgDensity,'EdgeColor','none');
xlabel('Time $t$');
ylabel('Strategy $s$');
zlabel('Distribution density');
title('Objective');
grid on;

subplot(2,2,2)
surf(tqg,S,xqgDensity,'EdgeColor','none');
xlabel('Time $t$');
ylabel('Strategy $s$');
zlabel('Distribution density');
title('Approximation at Optimizer');
grid on;

subplot(2,2,[3 4])
surf(tq,S,xqDensity,'EdgeColor','none');
xlabel('Time $t$');
ylabel('Strategy $s$');
zlabel('Distribution density');
title('Time Varying Approximation');
grid on;

sgtitle('\textbf{Strategy Distribution Updates via Replicator Dynamics}')

%% Plot 2
% Mean convergence rates
mg = S' * xg;
mqg = S' * xqg;
mq = S' * xq;

err_g  = abs(mg(:)  - sStar);
err_qg = abs(mqg(:) - sStar);
err_q  = abs(mq(:)  - sStar);

epsilon = max(err_g) * 2e-2;    % 2% settling time

idx_g  = settlingIndex(err_g,  epsilon);
idx_qg = settlingIndex(err_qg, epsilon);
idx_q  = settlingIndex(err_q,  epsilon);

% End plot after final convergence
if isnan(idx_q)
    T = ceil(max([tg(idx_g),tqg(idx_qg)]));
else 
    T = ceil(max([tg(idx_g),tqg(idx_qg),tq(idx_q)]));
end

figure;

% Convergence region
cRegion = fill([0,0,tspan(2),tspan(2)],[sStar-epsilon,sStar+epsilon,...
    sStar+epsilon,sStar-epsilon],'g');
hold on;

% Mean trajectories
mgTraj = plot(tg,mg,'k','LineWidth',3);
mqgTraj = plot(tqg,mqg,'--k','LineWidth',3);
mqTraj = plot(tq,mq,':k','LineWidth',3);

% Region entry points
if ~isnan(idx_g)
    mgPoint = plot(tg(idx_g),mg(idx_g),'o', ...
        'MarkerSize',10,'LineWidth',2,'Color','k');
else 
    mgPoint = plot(NaN,NaN,'o', ...
        'MarkerSize',10,'LineWidth',2,'Color','k');
end

if ~isnan(idx_qg)
    mgqPoint = plot(tqg(idx_qg),mqg(idx_qg),'s', ...
        'MarkerSize',10,'LineWidth',2,'Color','k');
else 
    mgqPoint = plot(NaN,NaN,'s', ...
        'MarkerSize',10,'LineWidth',2,'Color','k');
end

if ~isnan(idx_q)
    mqPoint = plot(tq(idx_q),mq(idx_q),'^', ...
        'MarkerSize',10,'LineWidth',2,'Color','k');
else 
    mqPoint = plot(NaN,NaN,'^', ...
        'MarkerSize',10,'LineWidth',2,'Color','k');
end

xlabel('Time $t$');
xlim([0,T]);
ylabel('Mean strategy $m(t)$');
title('$\textbf{Mean trajectories and convergence points}$','FontSize',15);
grid on;
hold off;

legend([mgTraj,mqgTraj,mqTraj,cRegion,mgPoint,mgqPoint,mqPoint], ...
    '$m(t)$ for g', ...
    '$m_{qg}(t)$ centered at optimizer', ...
    '$m_q(t)$ for time-varying q', ...
    'Convergence region', ...
    'Convergence time for $m(t)$', ...
    'Convergence time for $m_{qg}(t)$ centered at optimizer', ...
    'Convergence time for $m_q(t)$ time-varying');

%% Functions
function idx = settlingIndex(err, epsilon)
err = err(:);

% Last recorded point outside the epsilon-neighborhood
lastOutside = find(err > epsilon, 1, 'last');

if isempty(lastOutside)
    % Already inside the neighborhood at the initial time
    idx = 1;
elseif lastOutside == numel(err)
    % Did not settle during the simulation
    idx = NaN;
else
    % First point after the final tolerance violation
    idx = lastOutside + 1;
end
end