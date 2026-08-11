clc; clear; close all;

%% Global
S = (-2:0.05:2)';
[X,Y] = meshgrid(S);
strategies = [X(:),Y(:)];
n = numel(strategies)/2;

% Initial distribution
m0 = [1.5; -1.75];

sigma0 = 0.5;

dist2 = (strategies(:,1) - m0(1)).^2 ...
      + (strategies(:,2) - m0(2)).^2;

x0 = exp(-dist2/(2*sigma0^2));
x0 = x0/sum(x0);

% Time
tspan = [0,240];

%% Minimization --- Function: 2D Three-Hump Camel
% Objective 
g = @(s1,s2) 2 * s1.^2 - 1.05 * s1.^4 + (1/6) * s1.^6 + s1 .* s2 + s2.^2;
gradG = @(s1,s2) [4 * s1 - 4.2 * s1.^3 + s1.^5 + s2;...
                  s1 + 2 * s2];
hessianG = @(s1,s2) [4 - 12.6 * s1.^2 + 5 * s1.^4,1;...
                     1,2];

% Global quadratic approximation 
sStar = [0;0];                      % Global minimizer
min1 = [-1.74755;0.873776];         % Local min 1
min2 = [1.74755;-0.873776];         % Local min 2

qg = @(s1,s2) 2 * s1.^2 + s1 .* s2 + s2.^2;

% Time-varying quadratic approximation
m = @(x) x' * strategies;

q = @(m,S) ...
    g(m(1),m(2)) ...
    + (S - m) * gradG(m(1),m(2)) ...
    + 0.5 * sum(((S - m) * hessianG(m(1),m(2))) .* (S - m), 2);

% Values on strategy grid for dynamics
gValues = g(strategies(:,1),strategies(:,2));
qgValues = qg(strategies(:,1),strategies(:,2));

% Objective dynamics
F = @(x) x' * gValues - gValues;        % Vector payoff
f = @(t,x) x .* (F(x) - x' * F(x));     % Replicator dynamics

[tg,xg] = ode45(f, tspan, x0);          % Simulate dynamics
xg = xg';

% Quadratic dynamics at optimizer
Fqg = @(x) x' * qgValues - qgValues;
fqg = @(t,x) x .* (Fqg(x) - x' * Fqg(x));

[tqg,xqg] = ode45(fqg, tspan, x0);
xqg = xqg';

% Quadratic dynamics for time-varying
Fq = @(x) x' * q(m(x),strategies) - q(m(x),strategies);
fq = @(t,x) x .* (Fq(x) - x' * Fq(x));

[tq,xq] = ode45(fq, tspan, x0);
xq = xq';

%% Plots
% Set default interpreter as LaTeX
set(0,'defaultTextInterpreter','latex');

%% Plot 1
% Mean trajectories on contour
dS = S(2) - S(1);
dA = dS^2;
gGrid = reshape(gValues, size(X));

mg = strategies' * xg;
mqg = strategies' * xqg;
mq = strategies' * xq;

figure;

contourf(X,Y,gGrid,30);
hold on;

% Mean trajectories
gTraj = plot(mg(1,:),mg(2,:),'k','LineWidth',3);
qgTraj = plot(mqg(1,:),mqg(2,:),'--k','LineWidth',3);
qTraj = plot(mq(1,:),mq(2,:),':k','LineWidth',3);

globalMin = plot(sStar(1),sStar(2),'g.','MarkerSize',25,'Color','#FF0000');
localMin1 = plot(min1(1),min1(2),'r.','MarkerSize',25,'Color','#FF8C00');
loaclMin2 = plot(min2(1),min2(2),'r.','MarkerSize',25,'Color','#FF8C00');

% % Uncomment for: Initial covariance
% r = 3 * sigma0;
% cx = m0(1);
% cy = m0(2);
% 
% theta = linspace(0,2*pi,100);
% x = r * cos(theta) + cx;
% y = r * sin(theta) + cy;
% 
% initalCov = plot(x,y,'--','LineWidth',2,'Color','#FF0000');
hold off;

xlabel('$m_1(t)$');
ylabel('$m_2(t)$');
grid on;
axis equal;
xlim([-2, 2]);
ylim([-2, 2]);
title('\textbf{Mean trajectories on objective contours}','FontSize',15);
subtitle(['Inital mean: ', mat2str(m0')]);

% led1 = 'Inital distribution w/ sigma_0=';
% led2 = num2str(sigma0);

legend([gTraj, qgTraj, qTraj, globalMin, localMin1], ...
    {'$m(t)$','$m_{qg}(t)$','$m_q(t)$','Global minimizer',...
    'Local minima'},...
    'Location','southoutside',...
    'Orientation','horizontal');

%% Plot 2
% Mean convergence rates
% Distance of mean from global minimizer
ng = vecnorm(mg - sStar,2,1);
nqg = vecnorm(mqg - sStar,2,1);
nq = vecnorm(mq - sStar,2,1);

epsilon = max(ng) * 2e-2;   % 2% settling time

idx_g  = settlingIndex(ng,  epsilon);
idx_qg = settlingIndex(nqg, epsilon);
idx_q  = settlingIndex(nq,  epsilon);

% End plot after final convergence
if isnan(idx_q)
    T = ceil(max([tg(idx_g),tqg(idx_qg)]));
else 
    T = ceil(max([tg(idx_g),tqg(idx_qg),tq(idx_q)]));
end

figure;

% Convergence region
cRegion = fill([0,0,tspan(2),tspan(2)],[-epsilon,epsilon,...
    epsilon,-epsilon],'g');
hold on;

% Mean trajectories
mgTraj = plot(tg,ng,'k','LineWidth',3);
mqgTraj = plot(tqg,nqg,'--k','LineWidth',3);
mqTraj = plot(tq,nq,':k','LineWidth',3);

% Region entry points
if ~isnan(idx_g)
    mgPoint = plot(tg(idx_g),ng(idx_g),'o', ...
        'MarkerSize',10,'LineWidth',2,'Color','k');
else 
    mgPoint = plot(NaN,NaN,'o', ...
        'MarkerSize',10,'LineWidth',2,'Color','k');
end

if ~isnan(idx_qg)
    mgqPoint = plot(tqg(idx_qg),nqg(idx_qg),'s', ...
        'MarkerSize',10,'LineWidth',2,'Color','k');
else 
    mgqPoint = plot(NaN,NaN,'s', ...
        'MarkerSize',10,'LineWidth',2,'Color','k');
end

if ~isnan(idx_q)
    mqPoint = plot(tq(idx_q),nq(idx_q),'^', ...
        'MarkerSize',10,'LineWidth',2,'Color','k');
else 
    mqPoint = plot(NaN,NaN,'^', ...
        'MarkerSize',10,'LineWidth',2,'Color','k');
end

xlabel('Time $t$');
xlim([0,T]);
ylabel('$\|m(t) - m^\star \|_2$');
title('\textbf{Mean trajectories and convergence points}','FontSize',15);
subtitle(['Starting point ', mat2str(m0')]);
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

%% Plot 3
% Evolution of covariance matrix max eigenvalue
Cg = zeros(2,2,length(tg));
Cqg = zeros(2,2,length(tqg));
Cq = zeros(2,2,length(tq));

for j = 1:length(tg)
    xj = xg(:,j);
    mj = mg(:,j);

    Cg(:,:,j) = strategies' * (strategies .* xj) - mj*mj';
end

for j = 1:length(tqg)
    xj = xqg(:,j);
    mj = mqg(:,j);

    Cqg(:,:,j) = strategies' * (strategies .* xj) - mj*mj';
end

for j = 1:length(tq)
    xj = xq(:,j);
    mj = mq(:,j);

    Cq(:,:,j) = strategies' * (strategies .* xj) - mj*mj';
end

maxEvaluesG = zeros(1, length(tg));
maxEvaluesQg = zeros(1, length(tqg));
maxEvaluesQ = zeros(1, length(tq));

for j = 1:length(tg)
    maxEvaluesG(j) = max(eig(Cg(:,:,j)));
end

for j = 1:length(tqg)
    maxEvaluesQg(j) = max(eig(Cqg(:,:,j)));
end

for j = 1:length(tq)
    maxEvaluesQ(j) = max(eig(Cq(:,:,j)));
end

% Find largest eigenvalue across all trajectories
maxG = max(maxEvaluesG);
maxQg = max(maxEvaluesQg);
maxQ = max(maxEvaluesQ);
globalMax = max([maxG,maxQg,maxQ]);

epsilon = globalMax * 2e-2;    % 2% settling time

idx_g  = settlingIndex(maxEvaluesG,  epsilon);
idx_qg = settlingIndex(maxEvaluesQg, epsilon);
idx_q  = settlingIndex(maxEvaluesQ,  epsilon);

% End plot after covariance has settled
T = ceil(max([tg(idx_g),tqg(idx_qg),tq(idx_q)]));

figure;

plot(tg,maxEvaluesG,'k','LineWidth',3);
hold on;
plot(tqg,maxEvaluesQg,'--k','LineWidth',3);
plot(tq,maxEvaluesQ,':k','LineWidth',3);
hold off;

xlabel('Time $t$');
xlim([0,T]);
ylabel('Max eigenvalue');
title('\textbf{Covariance max eigenvalue evolution}','FontSize',15);
subtitle(['Starting point ', mat2str(m0')]);
legend('$\lambda_{\max}(C(t))$','$\lambda_{\max}(C_{qg}(t))$',...
       '$\lambda_{\max}(C_q(t))$');
grid on;

%% Functions
function idx = settlingIndex(err, epsilon)

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