clear;
close all;
clc;

%%

restoredefaultpath;
n = 2;
system_name = sprintf('manipulator%ddof', n);
save_dir = 'data';
addpath('general');
addpath(strcat('systems/', system_name));
load(strcat('systems/', system_name, '/sys.mat'), 'sys');

sys.X_DIMS = 2*sys.n; % [thi, ... dthi, ...]
sys.U_DIMS = sys.n;   % [taui]
if (n==2)
    sys.m = [2.5; 0.5]/2; % kg
    sys.l = [0.5; 0.25]/2; % m
    Izz = sys.m.*((sys.l));
    sys.Q = diag([8, 8, 0.6, 0.6])/5;
    sys.R = diag(0.003*(Izz(1)./Izz).^2);
    sys.lims = 5*[-Izz/Izz(1), Izz/Izz(1)]; % action limits
    
    % Define decompositions to test
    u_x = [];
    
    % Cascaded
    p = [0, 1;1, 1];
    s = [1, 0, 1, 0;0, 1, 0, 1];
    u_x = [u_x; reshape(p, 1, 2*sys.U_DIMS), reshape(s, 1, sys.U_DIMS*sys.X_DIMS)];
    
    p = [0, 1;1, 1];
    s = [0, 1, 0, 1;1, 0, 1, 0];
    u_x = [u_x; reshape(p, 1, 2*sys.U_DIMS), reshape(s, 1, sys.U_DIMS*sys.X_DIMS)];
    
    p = [0, 1;1, 1];
    s = [0, 0, 0, 0;1, 1, 1, 1];
    u_x = [u_x; reshape(p, 1, 2*sys.U_DIMS), reshape(s, 1, sys.U_DIMS*sys.X_DIMS)];
    
    p = [2, 1;0, 1];
    s = [1, 0, 1, 0;0, 1, 0, 1];
    u_x = [u_x; reshape(p, 1, 2*sys.U_DIMS), reshape(s, 1, sys.U_DIMS*sys.X_DIMS)];
    
    p = [2, 1;0, 1];
    s = [0, 1, 0, 1;1, 0, 1, 0];
    u_x = [u_x; reshape(p, 1, 2*sys.U_DIMS), reshape(s, 1, sys.U_DIMS*sys.X_DIMS)];
    
    p = [2, 1;0, 1];
    s = [1, 1, 1, 1;0, 0, 0, 0];
    u_x = [u_x; reshape(p, 1, 2*sys.U_DIMS), reshape(s, 1, sys.U_DIMS*sys.X_DIMS)];
    
    % Decoupled
    p = [0, 1;0, 2];
    s = [1, 0, 1, 0;0, 1, 0, 1];
    u_x = [u_x; reshape(p, 1, 2*sys.U_DIMS), reshape(s, 1, sys.U_DIMS*sys.X_DIMS)];
    
    p = [0, 1;0, 2];
    s = [0, 1, 0, 1;1, 0, 1, 0];
    u_x = [u_x; reshape(p, 1, 2*sys.U_DIMS), reshape(s, 1, sys.U_DIMS*sys.X_DIMS)];
    
elseif (n==3)
    sys.m = [2.5; 0.5; 0.1] * 1.1; % kg
    sys.l = [0.5; 0.25; 0.125]; % m
    Izz = sys.m.*((sys.l));
    sys.Q = diag([8*ones(1,3), 0.6*ones(1,3)])/5;
    sys.R = diag(0.004*(Izz(1)./Izz));
    sys.lims = [-16, 16; -7.5, 7.5; -1, 1]; % action limits
    
    % Define decompositions to test
    load('data/manipulator3dof/manipulator3dof_paretofront.mat');
    u_x = u_xp;

elseif (n==4)
    sys.m = [5.4; 1.8; 0.6; 0.2]; % kg
    sys.l = [0.2; 0.5; 0.25; 0.125]; % m
    Izz = sys.m.*((sys.l));
    sys.Q = diag([8*ones(1,4), 0.2*ones(1,4)])/2;
    sys.R = diag([0.002; 0.004*(Izz(2)./Izz(2:end))]);
    sys.lims = [-24, 24; -15, 15; -7.5, 7.5; -1, 1]; % action limits

    % Define decompositions to test
    load('data/manipulator4dof/manipulator4dof_paretofront.mat');
    u_x = u_xp;
end
sys.g = 9.81; % m/s^2
sys.T = 4;
sys.dt = 0.001;

sys.l_point = zeros(sys.X_DIMS, 1);
sys.l_point(1) = pi;
sys.goal = sys.l_point;
sys.u0 = zeros(sys.U_DIMS, 1);
sys.fxfu_func = @(x, u) [dynx(sys, x, u), dynu(sys, x, u)];

sys.gamma_ = 0.997;
sys.lambda_ = (1 - sys.gamma_)/sys.dt;
sys.cxmod = zeros(sys.X_DIMS,1);
sys.full_DDP = false;
sys.u0init = false;

% Optimization parameters
Op.lims  = sys.lims;
Op.maxIter = 500;
Op.gamma_ = sys.gamma_;
Op.print = 0;
Op.save_dir = save_dir;
Op.reuse_policy = false;
% Op.Alpha = [1];

theta_starts(:,:,1) = [2*pi/3, 2*pi/3, 4*pi/3, 4*pi/3;
                       -1,   1,      -1,    1];
theta_starts(:,:,2:n) = repmat([-pi/3, -pi/3, pi/3, pi/3;
                                -1,    1,     -1,  1], [1,1,n-1]);
theta_starts(2,:,:) = 0.5*theta_starts(2,:,:);

starts = zeros(sys.X_DIMS, size(theta_starts, 2)^n);
for count = 1:1:size(starts,2)
    count_ = count;
    for jj=1:1:n
        index = 1 + mod(count_-1, size(theta_starts, 2));
        starts([jj; n+jj], count) = theta_starts(:, index, jj);
        count_ = (count_ - index)/size(theta_starts, 2);
    end
end

poolobj = gcp('nocreate');
delete(poolobj);
parpool('local', 16);

%% Test Decompositions

Xd = zeros(sys.X_DIMS, round(sys.T / sys.dt)+1, size(starts, 2), size(u_x, 1));
Ud = zeros(sys.U_DIMS, round(sys.T / sys.dt), size(starts, 2), size(u_x, 1));
cd = zeros(1, size(starts, 2), size(u_x, 1));
for d=1:1:size(u_x, 1)
    
    fprintf('Decomposition : %d / %d\n', d, size(u_x, 1));
    sys.decomposition_id = d;
    p = reshape(u_x(d, 1:(2*sys.U_DIMS)), sys.U_DIMS, 2);
    s = reshape(u_x(d, (1+2*sys.U_DIMS):end), sys.U_DIMS, sys.X_DIMS);
    [Xd(:,:,:,d), Ud(:,:,:,d), cd(:,:,d)] = ilqg_decomposition_multitraj(sys, Op, p, s, starts);
    
end

%% Test Joint Optimization

p_joint = [zeros(n,1), ones(n,1)];
s_joint = ones(sys.U_DIMS, sys.X_DIMS);
sys.decomposition_id = 0;
[Xjoint, Ujoint, cjoint] = ilqg_decomposition_multitraj(sys, Op, p_joint, s_joint, starts);

err_ddp = mean(abs(reshape(cd, size(starts, 2), size(u_x, 1)) - cjoint'), 1);

save(strcat(save_dir, '/', system_name, '/summary.mat'), 'sys', 'u_x', 'Xd', 'Ud', 'cd', 'Xjoint', 'Ujoint', 'cjoint', 'err_ddp');
