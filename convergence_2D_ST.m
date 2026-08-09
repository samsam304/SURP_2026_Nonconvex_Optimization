clc; clear; close all;

%% Global
S = (-4:0.1:4)';
[X,Y] = meshgrid(S);
strategies = [X(:),Y(:)];
n = numel(strategies)/2;

% Initial distribution
m0 = [-1; 0];

sigma0 = 1;

dist2 = (strategies(:,1) - m0(1)).^2 ...
      + (strategies(:,2) - m0(2)).^2;

x0 = exp(-dist2/(2*sigma0^2));
x0 = x0/sum(x0);

% Time
tspan = [0,10];

%% Minimization --- Function: 2D Styblinski-Tang
% Objective and derivatives
g = @(s1,s2) (1/2) * (s1.^4 - 16 * s1.^2 + 5 * s1 ...
                + s2.^4 - 16 * s2.^2 + 5 * s2) + 80;
gradG = @(s1,s2) 1/2 * [5 - 32 * s1 + 4 * s1.^3;...
                        5 - 32 * s2 + 4 * s2.^3];
hessianG = @(s1,s2) 1/2 * ...
                    [-32 + 12 * s1.^2, 0;...
                     0 ,-32 + 12 * s2.^2];

% Global quadratic approximation 
sStar = [-2.903534;-2.903534];      % Global minimizer

qg = @(s1,s2) -78.3323 + ...
    9.60412*10^(-7) * (2.90353 + s1) + 17.2915 * (2.90353 + s1).^2 + ...
    9.60412*10^(-7) * (2.90353 + s2) + 17.2915 * (2.90353 + s2).^2;

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
Fg = @(x) x' * gValues - gValues;        % Vector payoff
fg = @(t,x) x .* (Fg(x) - x' * Fg(x));   % Replicator dynamics

[tg,xg] = ode45(fg, tspan, x0);          % Simulate dynamics
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

%% Plotting 
dS = S(2) - S(1);
dA = dS^2;
gGrid = reshape(gValues, size(X));

% Mean trajectories on contour
figure;

mg = strategies' * xg;
mqg = strategies' * xqg;
mq = strategies' * xq;

[~,h1] = contourf(X,Y,gGrid,30);
hold on;
colormap(gray);
gTraj = plot(mg(1,:),mg(2,:),'LineWidth',2,'Color','#1D2DCF');
qgTraj = plot(mqg(1,:),mqg(2,:),'LineWidth',2,'Color','#CF1DB1');
qTraj = plot(mq(1,:),mq(2,:),'LineWidth',2,'Color','#AE1DCF');

globalMin = plot(sStar(1),sStar(2),'g.','MarkerSize',25,'Color','#FF0000');

% Initial distribution plot
r = 3 * sigma0;
cx = m0(1);
cy = m0(2);

theta = linspace(0,2*pi,100);
x = r * cos(theta) + cx;
y = r * sin(theta) + cy;

initalDist = plot(x,y,'--','LineWidth',2,'Color','#FF0000');
hold off;

xlabel('m_1(t)');
ylabel('m_2(t)');
grid on;
axis equal;
xlim([-4, 4]);
ylim([-4, 4]);
title('Mean trajectories on objective contours');
subtitle(['Inital mean: ', mat2str(m0')]);

led1 = 'Inital distribution w/ sigma_0=';
led2 = num2str(sigma0);

legend([gTraj, qgTraj, qTraj, globalMin, initalDist], ...
    {'m(t)','m_{qg}(t)', 'm_q(t)', 'Global minimizer', ...
    strcat(led1,led2)}, ...
    'Position', [0.35 0.01 0.3 0.05], ...
    'Orientation', 'horizontal');

% Mean Convergence Rate Comparison
figure;

mg = strategies' * xg;
mqg = strategies' * xqg;
mq = strategies' * xq;

ng = vecnorm(mg - sStar,2,1);
nqg = vecnorm(mqg - sStar,2,1);
nq = vecnorm(mq - sStar,2,1);

epsilon = max(ng) * 2e-2;   % 2% settling time

idx_g  = settlingIndex(ng,  epsilon);
idx_qg = settlingIndex(nqg, epsilon);
idx_q  = settlingIndex(nq,  epsilon);

plot(tg,ng,'LineWidth',2,'Color','#1D2DCF');
hold on;
plot(tqg,nqg,'LineWidth',2,'Color','#CF1DB1');
plot(tq,nq,'LineWidth',2,'Color','#AE1DCF');

yline(0,'k--','LineWidth',1.5);
yline(epsilon,'k:','LineWidth',1.5);
yline(- epsilon,'k:','LineWidth',1.5);

if ~isnan(idx_g)
    plot(tg(idx_g),ng(idx_g),'o', ...
        'MarkerSize',8,'LineWidth',1.5,'Color','#1DCF2C');
end

if ~isnan(idx_qg)
    plot(tqg(idx_qg),nqg(idx_qg),'s', ...
        'MarkerSize',8,'LineWidth',1.5,'Color','#1D85CF');
end

if ~isnan(idx_q)
    plot(tq(idx_q),nq(idx_q),'^', ...
        'MarkerSize',8,'LineWidth',1.5,'Color','#1DCCCF');
end

xlabel('Time t');
ylabel('||m(t) - m^*||_2');
title('Mean trajectories and convergence points');
subtitle(['Starting point ', mat2str(m0')]);
grid on;
hold off;

legend('m(t) for g', ...
    'm_{qg}(t) centered at optimizer', ...
    'm_q(t) for time-varying q', ...
    'Global minimizer s = -1', ...
    '\epsilon-neighborhood', ...
    '', ...
    'Convergence time for m(t)', ...
    'Convergence time for m_q(t) centered at optimizer', ...
    'Convergence time for m_q(t) for time-varying');

% Evolution of Max EVal from Covariance Matrix
figure;

mg = strategies' * xg;
mqg = strategies' * xqg;
mq = strategies' * xq;

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

plot(tg,maxEvaluesG,'LineWidth',1.5,'Color','#1D2DCF');
hold on;
plot(tqg,maxEvaluesQg,'LineWidth',1.5,'Color','#CF1DB1');
plot(tq,maxEvaluesQ,'LineWidth',1.5,'Color','#AE1DCF');
hold off;

xlabel('Time t');
ylabel('Max eigenvalue');
title('Covariance max eigenvalue evolution');
subtitle(['Starting point ', mat2str(m0')]);
legend('lambda_{max}(C(t))','lambda_{max}(C_{qg}(t))',...
       'lambda_{max}(C_q(t))');
grid on;

%% Functions
% 1
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