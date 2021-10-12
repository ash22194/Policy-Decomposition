clear;
close all;
clc;

%%

restoredefaultpath;
system_name = 'cartpole';
addpath(strcat('systems/', system_name));
addpath('systems');
load(strcat('data/',system_name,'System.mat'));

assert(isfield(sys, 'name') && strcmp(sys.name, system_name), 'Check loaded system!');

sys.X_DIMS = 4;
sys.U_DIMS = 2;
sys.mc = 5;
sys.mp = 1;
sys.l = 0.9;
sys.g = 9.81; % m/s^2
sys.Q = diag([25, 0.02, 25, 0.02]);
sys.R = diag([0.001, 0.001]);
sys.gamma_ = 0.997;
sys.dt = 0.001;
sys.limits = [-1.5, 1.5;
              -3, 3;
              0, 2*pi;
              -3, 3];
sys.lims = [-9, 9;
            -9, 9];

sys.l_point = [0; 0; pi; 0];
sys.goal = sys.l_point;
sys.u0 = [0; 0];

Op.num_points = [31, 31, 31, 31];
Op.num_action_samples = [12, 12];
Op.max_iter = 1000;
Op.max_policy_iter = 100;
Op.u_mean_tol = (sys.lims(:,2) - sys.lims(:,1))*2e-6;
Op.u_max_tol = (sys.lims(:,2) - sys.lims(:,1))/12;
Op.gtol = 0.000002;
Op.save_dir = 'data';
Op.reuse_policy = false;

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

policies = cell(size(u_x,1), 1);
value = cell(size(u_x,1), 1);
info = cell(size(u_x,1), 1);

for dd = 1:1:size(u_x,1)
    p = reshape(u_x(dd, 1:(2*sys.U_DIMS)), sys.U_DIMS, 2);
    s = reshape(u_x(dd, (1+2*sys.U_DIMS):end), sys.U_DIMS, sys.X_DIMS);
    
    fprintf('Decomposition %d/%d\n', dd, size(u_x,1));
    sys.decomposition_id = dd;
    [policies{dd,1}, value{dd,1}, info{dd,1}] = dp_decomposition(sys, Op, p, s);
end

%% Joint

p_joint = [zeros(sys.U_DIMS,1), ones(sys.U_DIMS,1)];
s_joint = ones(sys.U_DIMS, sys.X_DIMS);
sys.decomposition_id = 0;

disp('Joint');
[policies_joint, value_joint, info_joint] = dp_decomposition(sys, Op, p_joint, s_joint);