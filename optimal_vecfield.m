%% Optimal control vector field reconstruction
% From the paper "Optimal reconstruction of vector fields from data for 
% prediction and uncertainty quantification"
% This code demonstrates model-free prediction via reconstructing vector
% fields from noisy and limited data of the Lorenz 63 system in both 
% partially and fully observed settings
% Sean McGowan 13/6/24


% Simulate Lorenz 63 system
% System parameters
sigma = 10;
rho = 28;
beta = 8/3;
dxdt = @(t,x) [sigma.*(x(2)-x(1)); x(1).*(rho-x(3))-x(2);x(1).*x(2)-beta*x(3)];

% Simulation parameters
dt = 0.01;
Tstart = 0;
NT = 100; % Number of Lyapunov times of data to simulate
T = NT*1.104;
tspan = Tstart:dt:T;
% Randomise initial condition
x0 = unifrnd(-1,1); y0 = unifrnd(-1,1) ; z0 = 12;

% Solver parameters
options = odeset('reltol',1e-6,'abstol',1e-6);
ode = @(dx,T,x0) ode15s(dx,T,x0,options);

% Simulate L63 and discard start of data to ensure trajectory is on
% system attractor
[~, xfull] = ode(dxdt, tspan, [x0, y0 z0]);
N = 20; % Number of Lyapunov times of data to retain for training
tspan = tspan(floor((NT-N)*1.104)/dt+1:end);
xfull = xfull(floor((NT-N)*1.104)/dt+1:end,:);

% Add measurement noise
variance = 0.1;
noise = [zeros(1,3); sqrt(variance).*(randn(length(tspan)-1,3))];
xfull = xfull+noise;

% Optimal control parameters
mu = 1000; % Energy parameter, dependent on noise level


%% PARTIALLY OBSERVED EXAMPLE

% Partial state observation - only observe x
xhat = xfull(:, 1);

% Embedding parameters
M = 3; % Embedding dimension - can be found by False Nearest Neighbours
% Increased dimension increases run time
tau = 11; % Delay - can be found by mutual information method

% Solve optimal control problem for smoothed trajectory x and derivative
% estimation u
[x,u] = diff_oc(tspan,xhat,mu);

% Define new tspan and remove turnpike phenomena from end of smoothed
% trajectory x and derivative estimation u
tspan_embed = tspan(1+(M-1)*tau:end);
Ttp = 0.1; % Turnpike interval length
Itp = floor(Ttp/dt); % Index of turnpike interval
u = u(1:end-Itp);
x = x(1:end-Itp);

% Define Delaunay triangulation of vector field
% May be replaced with nearest neighbour interpolation to speed up code for
% higher dimensions
[DT,Xtri,dXtri] = dt_embed(x,u,M,tau);

% Find prediction point Tp at a time of minimum velocity
Tbound = 0.5;
[~,min_i] = min(dXtri(floor(end-Tbound/dt):end));
Ip = length(tspan_embed)-floor(Tbound/dt)-Itp+min_i;
Tp = tspan_embed(Ip);
                             
% Predict true system and reconstructed system 5 time units past training
Tfinal = T+5;
x0new = [x0,y0,z0];
[~,xtrue] = ode(dxdt, Tstart:dt:Tfinal, x0new);
xtrue = xtrue(floor((NT-N)*1.104)/dt+1:end,:);
[~,xpred] = ode(@(t,X) dxdt_interp(t,X,Xtri,DT,dXtri),Tp:dt:Tfinal,Xtri(Ip,:));

% Plot prediction
figure
subplot(2,1,1);
plot((Tp:dt:Tfinal)-T,xtrue(Ip+(M-1)*tau:end,1),'k')
hold on
plot((Tp:dt:Tfinal)-T,xpred(:,1),'r')
xlabel('$ t $','Interpreter','Latex','FontSize',12);
ylabel('$ x $','Interpreter','Latex','FontSize',12);
xlim([0 Tfinal-T])
legend('True trajectory','Embedded vector field prediction')
subplot(2,1,2);
E = abs(xtrue(Ip+(M-1)*tau:end,1)-xpred(:,1));
plot((Tp:dt:Tfinal)-T,E,'k')
xlabel('$ t $','Interpreter','Latex','FontSize',12);
ylabel('$|x_{\textrm{true}}-x_{\textrm{predict}}|$ ','Interpreter','Latex','FontSize',12);
xlim([0 Tfinal-T])
sgtitle(['Partially observed with $M=',num2str(M),'$'],'interpreter','latex')


%% FULLY OBSERVED EXAMPLE

% Solve optimal control problem for smoothed trajectory x and derivative
% estimation u
[x,u] = diff_oc(tspan,xfull,mu);

% Remove turnpike phenomena from end of smoothed trajectory x and 
% derivative estimation u
Ttp = 0.1; % Turnpike interval length
Itp = floor(Ttp/dt); % Index of turnpike interval
u = u(1:end-Itp,:);
x = x(1:end-Itp,:);

% Define Delaunay triangulation of vector field
% May be replaced with nearest neighbour interpolation to speed up code for
% higher dimensions
[DT,Xtri,dXtri] = dt_full(x,u);

% Find prediction point Tp at a time of minimum velocity
Tbound = 0.5;
[~,min_i] = min(dXtri(floor(end-Tbound/dt):end));
Ip = length(tspan)-floor(Tbound/dt)-Itp+min_i;
Tp = tspan(Ip);
                             
% Predict true system and reconstructed system 5 time units past training
Tfinal = T+5;
x0new = [x0,y0,z0];
[~,xtrue] = ode(dxdt, Tstart:dt:Tfinal, x0new);
xtrue = xtrue(floor((NT-N)*1.104)/dt+1:end,:);
[~,xpred] = ode(@(t,X) dxdt_interp(t,X,Xtri,DT,dXtri),Tp:dt:Tfinal,Xtri(Ip,:));

% Plot prediction
figure
subplot(2,1,1);
plot((Tp:dt:Tfinal)-T,xtrue(Ip:end,1),'k')
hold on
plot((Tp:dt:Tfinal)-T,xpred(:,1),'r')
xlabel('$ t $','Interpreter','Latex','FontSize',12);
ylabel('$ x $','Interpreter','Latex','FontSize',12);
xlim([0 Tfinal-T])
legend('True trajectory','Vector field prediction')
subplot(2,1,2);
E = abs(xtrue(Ip:end,1)-xpred(:,1));
plot((Tp:dt:Tfinal)-T,E,'k')
xlabel('$ t $','Interpreter','Latex','FontSize',12);
ylabel('$|x_{\textrm{true}}-x_{\textrm{predict}}|$ ','Interpreter','Latex','FontSize',12);
xlim([0 Tfinal-T])
sgtitle('Fully observed','interpreter','latex')


%% FUNCTIONS

% Solve optimal control problem
function [x,u] = diff_oc(tspan,xhat,mu)
% Derived quantities from parameters
N = size(xhat,2);
dt = tspan(2)-tspan(1);

% Track a spline interpolant of the simulated (or measured) trajectory 
% with the same initial condition 
tracking = @(t) interp1(tspan, xhat, t, 'spline');

% Hamiltonian equations
xdot = @(tt,xx,pp) -pp./2;
pdot = @(tt,xx,pp) 2*mu.*(tracking(tt)-xx').';

% Estimate trajectory derivative at final time
xhat_smooth = smoothdata(xhat,1,'sgolay');
dxT = (xhat_smooth(end,:)-4/3*xhat_smooth(end-1,:)...
    +1/3*xhat_smooth(end-2,:))/(2/3*dt);

% Set up functions for bvp4c
Ydot = @(tt,YY) [xdot(tt,YY(1:N),YY(1+N:2*N)); ...
    pdot(tt,YY(1:N),YY(1+N:2*N))];
% Boundary conditions
BC = @(y0, yT) [y0(1:N)-xhat(1,1:N)'; yT(1+N:2*N)+2*dxT(1:N)'];

% Solution grid and initial conditions and guesses
solinit = bvpinit(tspan, [xhat(1,1:N) zeros(1,N)]');

% Solve boundary value problem
sol = bvp4c(Ydot,BC,solinit);
Y = deval(sol, tspan).';
x = Y(:,1:N);
p = Y(:,1+N:2*N);

% Control output is the estimate of trajectory derivative
u = -p/2;
end



% Create Delaunay triangulation of embedded vector field
function [DT,X,dX] = dt_embed(x,u,M,tau)

X = zeros(length(x)-(M-1)*tau+2^M,M);
dX = zeros(length(x)-(M-1)*tau+2^M,M);

% Magnitude of auxiliary point cube
R = max(abs(x))*2;

for mm = 1:M
    % 2^M auxiliary points define a cube around embedded attractor
    % These points may take artificial vector field values of 0, or values 
    % pointing towards attractor

    Xaux = R*repmat([ones(1, 2^(M-mm)), -ones(1, 2^(M-mm))], 1, 2^(mm-1))';
    X(:,mm) = [x(1+(M-mm)*tau:end-(mm-1)*tau); Xaux];
    
    dX(:,mm) = [u(1+(M-mm)*tau:end-(mm-1)*tau); zeros(2^M,1)];
end

% Construct Delaunay triangulation on convex hull of attractor defined by
% trajectory and auxiliary points
if M<=3
    DT = delaunayTriangulation(X);
else
    DT = delaunayn(X);
end
end



% Create Delaunay triangulation of full vector field
function [DT,X,dX] = dt_full(x,u)

M = size(x,2);
X = zeros(length(x)+2^M,M);
dX = zeros(length(x)+2^M,M);

% Magnitude of auxiliary point cube
R = max(abs(x))*2;

for mm = 1:M
    % 2^M auxiliary points define a cube around embedded attractor
    % These points may take artificial vector field values of 0, or values 
    % pointing towards attractor

    Xaux = R(mm)*repmat([ones(1, 2^(M-mm)), -ones(1, 2^(M-mm))], 1, 2^(mm-1))';
    X(:,mm) = [x(:,mm); Xaux];
    
    dX(:,mm) = [u(:,mm); zeros(2^M,1)];
end

% Construct Delaunay triangulation on convex hull of attractor defined by
% trajectory and auxiliary points
if M<=3
    DT = delaunayTriangulation(X);
else
    DT = delaunayn(X);
end
end



% Discovered velocity field interpolated on Delaunay triangulation
function dXdt = dxdt_interp(t,X,Xtri,DT,dXtri)

M = size(dXtri,2);
dXdt = zeros(M,1);

% Find index and coordinates of simplex containing X
if M <= 3
    [ti,bc] = pointLocation(DT,X');
else
    [ti,bc] = tsearchn(Xtri,DT,X');
end

for mm = 1:M
    % Interpolate vector field at X
    triVals = dXtri(DT(ti,:),mm);
    dXdt(mm) = bc*triVals;
end

end

