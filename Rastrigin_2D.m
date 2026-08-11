clc; clear; close all;

%% Global
S = (-2:0.1:2)';
[X,Y] = meshgrid(S);
strategies = [X(:),Y(:)];

% Initial distributions
m0 = [-2.5,1.5];

sigma0 = 0.5;

dist2 = (strategies(:,1) - m0(1)).^2 ...
      + (strategies(:,2) - m0(2)).^2;

x0 = exp(-dist2/(2*sigma0^2));
x0 = x0/sum(x0);

% Time
tspan = [0,30];

% Plotting options
axis_label_font_size = 16;
title_font_size = 20;

%% Minimization --- Function: 2D Rastrigin
% Objective
g = @(s1,s2) 20 + s1.^2 - 10 * cos(2 * pi .* s1) +...
                s2.^2 - 10 * cos(2 * pi .* s2);
gradG = @(s1,s2) [2 * s1 + 20 * pi * sin(2 * pi .* s1);...
                  2 * s2 + 20 * pi * sin(2 * pi .* s2)];
hessianG = @(s1,s2) [2 + 40 * pi^2 * cos(2 * pi .* s1), 0;...
                     0, 2 + 40 * pi^2 * cos(2 * pi .* s2)];

% Global quadratic approximation 
sStar = [0;0];                      % Global minimizer

qg = @(s1,s2) (1 + 20 * pi^2) * s1.^2 + (1 + 20 * pi^2) * s2.^2;

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

figure; box on; hold on; grid on;
title('\textbf{Mean Trajectory for Rastrigin Objective}','FontSize',title_font_size);

contourf(X,Y,gGrid,30);

% Mean trajectories
gTraj = plot(mg(1,:),mg(2,:),'k','LineWidth',3);
qgTraj = plot(mqg(1,:),mqg(2,:),'--k','LineWidth',3);
qTraj = plot(mq(1,:),mq(2,:),':k','LineWidth',3);

globalMin = plot(sStar(1),sStar(2),'g.','MarkerSize',25,'Color','#FF0000');

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

xlabel('$s_1$', 'fontsize', axis_label_font_size);
ylabel('$s_2$', 'fontsize', axis_label_font_size);
axis equal;
xlim([-2, 2]);
ylim([-2, 2]);

xtickformat('$%g$');
ytickformat('$%g$');
set(gca,'ticklabelinterpreter','latex','fontsize',axis_label_font_size);


% led1 = 'Inital covariance w/ $\sigma_0$=';
% led2 = num2str(sigma0);

legend([gTraj, qgTraj, qTraj, globalMin], ...
    {'Nonconvex','Optimal Approximation','Adaptive Approximation','Global Minimizer'}, ...
    'Orientation','vertical',...
    'interpreter','latex','FontSize',axis_label_font_size,'backgroundalpha',0.9);

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

figure; box on; hold on; grid on;
title('\textbf{Distance to Optimizer for Rastrigin Objective}','FontSize',title_font_size);

% Convergence region
cRegion = fill([0,0,tspan(2),tspan(2)],[-epsilon,epsilon,...
    epsilon,-epsilon],'g', 'linestyle', 'none');

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

xlabel('Time', 'fontsize', axis_label_font_size);
xlim([0,T]);
ylabel('$\|m(t) - m^\star \|_2$', 'fontsize', axis_label_font_size);

xtickformat('$%g$');
ytickformat('$%g$');
ztickformat('$%g$');
set(gca,'ticklabelinterpreter','latex','fontsize',axis_label_font_size);

legend([mgTraj,mqgTraj,mqTraj,mgPoint,mgqPoint,mqPoint,cRegion], ...
    'Nonconvex', ...
    'Optimal Approximation', ...
    'Adaptive Approximation', ...
    'Settling Time, Nonconvex', ...
    'Settling Time, Optimal Approximation', ...
    'Settling Time, Adaptive Approximation', ...
    '$2\%$ Settling Time Tube', ...
    'interpreter','latex','fontsize',axis_label_font_size,'backgroundalpha',0.9);

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

figure; box on; hold on; grid on;
title('\textbf{Covariance Trajectory for Rastrigin Objective}','FontSize',title_font_size);

plot(tg,maxEvaluesG,'k','LineWidth',3);
plot(tqg,maxEvaluesQg,'--k','LineWidth',3);
plot(tq,maxEvaluesQ,':k','LineWidth',3);

xtickformat('$%g$');
ytickformat('$%g$');
ztickformat('$%g$');
set(gca,'ticklabelinterpreter','latex','fontsize',axis_label_font_size);

xlabel('Time', 'fontsize', axis_label_font_size);
xlim([0,T]);
ylabel('Maximum Eigenvalue', 'fontsize', axis_label_font_size);
legend('Nonconvex','Optimal Approximation',...
       'Adaptive Approximation','interpreter','latex','fontsize',axis_label_font_size,'backgroundalpha',0.9);

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