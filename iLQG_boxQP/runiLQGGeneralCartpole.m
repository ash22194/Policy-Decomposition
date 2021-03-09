clear;
close all;
clc;

%%

restoredefaultpath;
system_name = 'cartpole';
addpath('general');
addpath(strcat('systems/', system_name));
load(strcat('systems/', system_name, '/sys.mat'), 'sys');

sys.mc = 5;
sys.mp = 1;
sys.l = 0.9;
sys.g = 9.81;
sys.T = 4;
sys.dt = 0.001;
sys.X_DIMS = 4; % x, x_dot, th, th_dot
sys.U_DIMS = 2; % F, tau 

sys.l_point = [0;0;pi;0];
sys.goal = sys.l_point;
sys.u0 = [0;0];
sys.fxfu_func = @(x, u) [dynx(sys, x, u), dynu(sys, x, u)];
sys.lims = 9*[-ones(2,1), ones(2,1)];

sys.gamma_ = 0.997;
sys.lambda_ = (1 - sys.gamma_)/sys.dt;
sys.Q = diag([25, 0.02, 25, 0.02]);
sys.R = 0.001*eye(2);
sys.cxmod = zeros(4,1);
sys.full_DDP = false;
sys.u0init = true;

% Optimization parameters
Op.lims  = sys.lims;
Op.maxIter = 500;
Op.gamma_ = sys.gamma_;
Op.save_dir = 'data';
Op.print = 0;
% Op.Alpha = [0.1];

cart_starts = [-0.5, -0.5, 0.5, 0.5;
               -1,  1, -1, 1];
cart_starts(2,:) = 0.5 * cart_starts(2,:);
pole_starts = [2*pi/3, 2*pi/3, 4*pi/3, 4*pi/3;
               -1,    1,     -1,   1];
pole_starts(2,:) = 0.5 * pole_starts(2,:);
starts = zeros(sys.X_DIMS, size(cart_starts, 2) ...
                           *size(pole_starts, 2));
count = 0;
for carts=1:1:size(cart_starts, 2)
    for poles=1:1:size(pole_starts, 2)
        count = count + 1;
        starts(1:2, count) = cart_starts(:, carts);
        starts(3:4, count) = pole_starts(:, poles);
    end
end

%% Test Decompositions
X_DIMENSIONS = linspace(1,4,4);
u_x = [];
Cu = cell(0, 2);

% F first
p = [2, 1;0, 1];
p = reshape(p, 1, 4);
for jj=1:1:4
    s1 = nchoosek(X_DIMENSIONS, jj);
    for ss=1:1:size(s1, 1)
        s = [zeros(1, 4); ones(1, 4)];
        s(1,s1(ss,:)) = 1;
        s(2,s1(ss,:)) = 0;
        u_x = [u_x; [p, reshape(s, 1, 8)]];
        Cu = cat(1, Cu, {s1(ss,:), 1});
    end
end

% T first
p = [0, 1;1, 1];
p = reshape(p, 1, 4);
for jj=1:1:4
    s1 = nchoosek(X_DIMENSIONS, jj);
    for ss=1:1:size(s1, 1)
        s = [ones(1, 4); zeros(1, 4)];
        s(1,s1(ss,:)) = 0;
        s(2,s1(ss,:)) = 1;
        u_x = [u_x; [p, reshape(s, 1, 8)]];
        Cu = cat(1, Cu, {s1(ss,:), 2});
    end
end

% Decoupled
p = [0, 1;0, 2];
p = reshape(p, 1, 4);
for jj=1:1:3
    s1 = nchoosek(X_DIMENSIONS, jj);
    for ss=1:1:size(s1, 1)
        s = [zeros(1, 4); ones(1, 4)];
        s(1,s1(ss,:)) = 1;
        s(2,s1(ss,:)) = 0;
        u_x = [u_x; [p, reshape(s, 1, 8)]];
        Cu = cat(1, Cu, {s1(ss,:), 3});
    end
end

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

sys.u0init = false;
sys.name = 'cartpole';
p_joint = [zeros(sys.U_DIMS,1), ones(sys.U_DIMS,1)];
s_joint = ones(sys.U_DIMS, sys.X_DIMS);
sys.decomposition_id = 0;
[Xjoint, Ujoint, cjoint] = ilqg_decomposition_multitraj(sys, Op, p_joint, s_joint, starts);

err_ddp = mean(abs(reshape(cd, size(starts, 2), size(u_x, 1)) - cjoint'), 1);

save(strcat(Op.save_dir, '/', system_name, '/summary.mat'), 'sys', 'u_x', 'Xd', 'Ud', 'cd', 'Xjoint', 'Ujoint', 'cjoint', 'err_ddp', 'Cu');