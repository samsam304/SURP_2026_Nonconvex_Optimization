clc; clear; close all;

%% Global
S = (-2:0.05:2)';
[X,Y] = meshgrid(S);
strategies = [X(:),Y(:)];           % Each row is a strategy (s1,s2)

% Inital distributions 
startingPoints = {[-2; 2],[-1.5; -1.5]};
p = numel(startingPoints);
distributions = {};

for k = 1:p
    m0 = startingPoints{k};

    sigma0 = 0.5;
    
    dist2 = (strategies(:,1) - m0(1)).^2 ...
          + (strategies(:,2) - m0(2)).^2;
    
    x0 = exp(-dist2/(2*sigma0^2));
    x0 = x0/sum(x0);

    distributions{end+1} = {x0,sigma0};
end

% Time
tspan = [0,30];

%% Minimization
% Objective minimization (Three-Hump Camel) 
g = @(s1,s2) 2 * s1.^2 - 1.05 * s1.^4 + (1/6) * s1.^6 + s1 .* s2 + s2.^2;

% Approximation minimization
sStar = [0;0];                      % Global minimizer
min1 = [-1.74755;0.873776];         % Local min 1
min2 = [1.74755;-0.873776];         % Local min 2

q = @(s1,s2) 2 * s1.^2 + s1 .* s2 + s2.^2;

% Values on strategy grid for dynamics
gValues = g(strategies(:,1),strategies(:,2));
qValues = q(strategies(:,1),strategies(:,2));

solutions = {};

for k = 1:p
    x0 = distributions{k}{1};
    
    F = @(x) x' * gValues - gValues;        % Vector payoff
    f = @(t,x) x .* (F(x) - x' * F(x));     % Replicator dynamics
    
    options = odeset("RelTol",1e-10);
    [tg,xg] = ode45(f, tspan, x0);          % Simulate dynamics
    xg = xg';
    
    Fq = @(x) x' * qValues - qValues;       % Vector payoff
    fq = @(t,x) x .* (Fq(x) - x' * Fq(x));  % Replicator dynamics
    
    [tq,xq] = ode45(fq, tspan, x0);         % Simulate dynamics
    xq = xq';

    solutions{end+1} = {xg,xq,tg,tq};
end 

%% Plotting 
dS = S(2) - S(1);
dA = dS^2;

gGrid = reshape(gValues, size(X));
qGrid = reshape(qValues, size(X));

% % Objective and Approximation
% figure;
% 
% subplot(1,2,1)
% surf(X,Y,gGrid,'EdgeColor','none');
% xlabel('s_1');
% ylabel('s_2');
% zlabel('f(s_1,s_2)');
% title('Three-Hump Camel objective');
% grid on;
% view(45,30);
% 
% subplot(1,2,2)
% surf(X,Y,qGrid,'EdgeColor','none');
% xlabel('s_1');
% ylabel('s_2');
% zlabel('q(s_1,s_2)');
% title('Quadratic approximation');
% grid on;
% view(45,30);

% Mean Trajectories on Contour Plots
figure;

for k = 1:p
    xg = solutions{k}{1};
    xq = solutions{k}{2};

    xgplot = xg ./ sum(xg,1);
    xqplot = xq ./ sum(xq,1);

    mg = strategies' * xgplot;
    mq = strategies' * xqplot;

    subplot(1,2,k);

    [~,h1] = contour(X,Y,gGrid,30);
    hold on;

    h2 = plot(mg(1,:),mg(2,:),'LineWidth',1.5);
    h3 = plot(mq(1,:),mq(2,:),'LineWidth',1.5);
    h4 = plot(sStar(1),sStar(2),'g.','MarkerSize',25);  % Global min
    h6 = plot(min1(1),min1(2),'r.','MarkerSize',25);    % Local min 1
    h7 = plot(min2(1),min2(2),'r.','MarkerSize',25);    % Local min 2

    % Initial distribution plot
    r = 3 * distributions{k}{2};
    cx = startingPoints{k}(1);
    cy = startingPoints{k}(2);

    theta = linspace(0,2*pi,100);
    x = r * cos(theta) + cx;
    y = r * sin(theta) + cy;

    h5 = plot(x,y,'--','LineWidth',2);
    hold off;

    xlabel('m_1(t)');
    ylabel('m_2(t)');
    grid on;
    axis equal;
    xlim([-2, 2]);
    ylim([-2, 2]);
    title(['Inital mean: ', mat2str(startingPoints{k}')]);
end

led1 = 'Inital distribution w/ sigma_0=';
led2 = num2str(sigma0);

sgtitle('Mean trajectories on objective contours');
legend([h1, h2, h3, h4, h6, h5], ...
    {'Objective contours', 'm(t)', 'm_q(t)', 'Global minimizer', ...
    'Local minima', strcat(led1,led2)}, ...
    'Position', [0.35 0.01 0.3 0.05], ...
    'Orientation', 'horizontal');

% Mean Convergence Rate Comparison
figure;

for k = 1:p
    xg = solutions{k}{1};
    xq = solutions{k}{2};

    tg = solutions{k}{3};
    tq = solutions{k}{4};

    xgplot = xg ./ sum(xg,1);
    xqplot = xq ./ sum(xq,1);

    mg = strategies' * xgplot;
    mq = strategies' * xqplot;

    ng = vecnorm(mg - sStar,2,1);
    nq = vecnorm(mq - sStar,2,1);

    subplot(1,2,k);
    plot(tg,ng,'LineWidth',1.5);
    hold on;
    plot(tq,nq,'LineWidth',1.5);
    hold off;
    
    xlabel('Time t');
    ylabel('||m(t) - m^*||_2');
    ylim([0,10]);
    title(['Starting point ', mat2str(startingPoints{k}')]);
    legend('m(t)','m_q(t)');
    grid on;
end

sgtitle('Mean convergence to global minimizer');

% Evolution of Max EVal from Covariance Matrix
figure;

for k = 1:p
    xg = solutions{k}{1};
    xq = solutions{k}{2};

    tg = solutions{k}{3};
    tq = solutions{k}{4};

    xgplot = xg ./ sum(xg,1);
    xqplot = xq ./ sum(xq,1);

    mg = strategies' * xgplot;
    mq = strategies' * xqplot;

    Cg = zeros(2,2,length(tg));
    Cq = zeros(2,2,length(tq));

    for j = 1:length(tg)
        xj = xgplot(:,j);
        mj = mg(:,j);

        Cg(:,:,j) = strategies' * (strategies .* xj) - mj*mj';
    end

    for j = 1:length(tq)
        xj = xqplot(:,j);
        mj = mq(:,j);

        Cq(:,:,j) = strategies' * (strategies .* xj) - mj*mj';
    end

    maxEvaluesG = zeros(1, length(tg));
    maxEvaluesQ = zeros(1, length(tq));

    for j = 1:length(tg)
        maxEvaluesG(j) = max(eig(Cg(:,:,j)));
    end

    for j = 1:length(tq)
        maxEvaluesQ(j) = max(eig(Cq(:,:,j)));
    end

    subplot(1,2,k);
    plot(tg,maxEvaluesG,'LineWidth',1.5);
    hold on;
    plot(tq,maxEvaluesQ,'LineWidth',1.5);
    hold off;

    xlabel('Time t');
    ylabel('Max eigenvalue');
    title(['Starting point ', mat2str(startingPoints{k}')]);
    legend('lambda_{max}(C(t))','lambda_{max}(C_q(t))');
    grid on;
end

sgtitle('Covariance max eigenvalue evolution');