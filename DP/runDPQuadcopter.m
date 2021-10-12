clear;
close all;
clc;

try 
    num_gpus = gpuDeviceCount();
    gpu_id = 0;
    max_avail_memory = 0;
    for gg=1:1:num_gpus
        g = gpuDevice(gg);
        if (g.AvailableMemory > max_avail_memory)
            gpu_id = gg;
            max_avail_memory = g.AvailableMemory;
        end
    end
    g = gpuDevice(gpu_id);
    reset(g);
    fprintf('Using GPU : %d\n', gpu_id);
    use_gpu = true;
catch MEGPU
    use_gpu = false;
end

%%

restoredefaultpath;
system_name = 'quadcopter';
addpath(strcat('systems/', system_name));
addpath('systems');
load(strcat('data/',system_name,'System.mat'));
if ((use_gpu) && (isfile(strcat('systems/', system_name, '/dyn_mex_finite.cu'))))
    mexcuda(strcat('systems/', system_name, '/dyn_mex_finite.cu'), '-R2018a', '-output', strcat('systems/', system_name, '/dyn_mex_finite'));
else
    use_gpu = false;
end

assert(isfield(sys, 'name') && strcmp(sys.name, system_name), 'Check loaded system!');

sys.X_DIMS = 10; % z, ro, pi, ya, vx, vy, vz, vr0, vpi, vya
sys.U_DIMS = 4;
sys.m = 0.5;
sys.I = diag([4.86*1e-3; 4.86*1e-3; 8.8*1e-3]);
sys.l = 0.225;
sys.g = 9.81;
sys.bk = 1.14*1e-7/(2.98*1e-6); % tau/f
sys.dt = 0.00025;
sys.Q = diag([5, 0.001, 0.001, 5, 0.5, 0.5, 0.05, 0.075, 0.075, 0.05]);
sys.R = diag([0.002, 0.01, 0.01, 0.004]);
sys.gamma_ = 0.99975;
sys.limits = [0.5, 1.5;
              -0.7, 0.7;
              -0.7, 0.7;
              -pi, pi;
              -2, 2;
              -2, 2;
              -1.5, 1.5;
              -6, 6;
              -6, 6;
              -2.5, 2.5];
sys.lims = [0, 2*sys.m*sys.g;
            -0.25*sys.m*sys.g, 0.25*sys.m*sys.g;
            -0.25*sys.m*sys.g, 0.25*sys.m*sys.g;
            -0.125*sys.m*sys.g, 0.125*sys.m*sys.g];
sys.l_point = [1; zeros(9,1)];
sys.goal = sys.l_point;
sys.u0 = [sys.m*sys.g; 0; 0; 0];

Op.num_points = [7,7,7,35,7,7,7,11,11,35];
Op.num_action_samples = [10,10,10,10];
Op.max_iter = 5000;
Op.max_policy_iter = 800;
Op.u_mean_tol = (sys.lims(:,2) - sys.lims(:,1))*2e-6/5;
Op.u_max_tol = (sys.lims(:,2) - sys.lims(:,1))/12/5;
Op.gtol = 0.00000002*0;
Op.save_dir = 'data';
Op.reuse_policy = true;

u_x = [];
% GA : [T, F_roll] - (z, vz, r, vy, vr, vp), F_pitch - (z, vz, r, vy, vr, p, vx, vp, T, F_roll), F_yaw - (ya, vya) 
p = [3, 1;3, 1;0, 1;0, 2];
s = [1, 1, 0, 0, 0, 1, 1, 1, 1, 0;
     1, 1, 0, 0, 0, 1, 1, 1, 1, 0;
     0, 0, 1, 0, 1, 0, 0, 0, 0, 0;
     0, 0, 0, 1, 0, 0, 0, 0, 0, 1];
u_x = [u_x; reshape(p, 1,2*sys.U_DIMS), reshape(s, 1,sys.U_DIMS*sys.X_DIMS)];

% MCTS : [T, F_roll, F_pitch] - (z, vz, r, vy, vr, p, vx, vp), F_yaw - (ya, vya) 
p = [0, 2;0, 2;0, 2;0, 1];
s = [1, 1, 1, 0, 1, 1, 1, 1, 1, 0;
     1, 1, 1, 0, 1, 1, 1, 1, 1, 0;
     1, 1, 1, 0, 1, 1, 1, 1, 1, 0;
     0, 0, 0, 1, 0, 0, 0, 0, 0, 1];
u_x = [u_x; reshape(p, 1,2*sys.U_DIMS), reshape(s, 1,sys.U_DIMS*sys.X_DIMS)];

% Pareto : T - (z, vz), F_roll - (r, vr, vy, z, vz, T), F_pitch - (p, vx, vp, r, vr, vy, z, vz, F_roll, T), F_yaw - (ya, vya)
p = [2, 2; 3, 1; 0, 3; 0, 4];
s = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0;
     0, 1, 0, 0, 0, 1, 0, 1, 0, 0;
     0, 0, 1, 0, 1, 0, 0, 0, 1, 0;
     0, 0, 0, 1, 0, 0, 0, 0, 0, 1];
u_x = [u_x; reshape(p, 1,2*sys.U_DIMS), reshape(s, 1,sys.U_DIMS*sys.X_DIMS)];

% Self created : T - (z, vz), F_roll - (r, vy, vr), F_pitch - (p, vx, vp), F_yaw - (ya, vya)
p = [0, 1; 0, 2; 0, 3; 0, 4];
s = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0;
     0, 1, 0, 0, 0, 1, 0, 1, 0, 0;
     0, 0, 1, 0, 1, 0, 0, 0, 1, 0;
     0, 0, 0, 1, 0, 0, 0, 0, 0, 1];
u_x = [u_x; reshape(p, 1,2*sys.U_DIMS), reshape(s, 1,sys.U_DIMS*sys.X_DIMS)];

% Self created : F_roll - (r, vy, vr), F_pitch - (p, vx, vp), F_yaw - (ya, vya), T - (z, vz, r, vy, vr, p, vx, vp, F_roll, F_pitch)
p = [0, 1; 1, 1; 1, 2; 0, 2];
s = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0;
     0, 1, 0, 0, 0, 1, 0, 1, 0, 0;
     0, 0, 1, 0, 1, 0, 0, 0, 1, 0;
     0, 0, 0, 1, 0, 0, 0, 0, 0, 1];
u_x = [u_x; reshape(p, 1,2*sys.U_DIMS), reshape(s, 1,sys.U_DIMS*sys.X_DIMS)];

policies = cell(size(u_x,1), 1);
value = cell(size(u_x,1), 1);
info = cell(size(u_x,1), 1);
for dd=1:1:size(u_x, 1)
    p = reshape(u_x(dd, 1:(2*sys.U_DIMS)), sys.U_DIMS, 2);
    s = reshape(u_x(dd, (1+2*sys.U_DIMS):end), sys.U_DIMS, sys.X_DIMS);

    fprintf('Decomposition %d/%d\n', dd, size(u_x,1));
    sys.decomposition_id = dd;
    if (use_gpu)
        [policies{dd,1}, value{dd,1}, info{dd,1}] = dp_decomposition_gpu(sys, Op, p, s);
    else
        [policies{dd,1}, value{dd,1}, info{dd,1}] = dp_decomposition(sys, Op, p, s);
    end
end

%% Joint - (Requires too much memory!)

% p_joint = [zeros(sys.U_DIMS,1), ones(sys.U_DIMS,1)];
% s_joint = ones(sys.U_DIMS, sys.X_DIMS);
% sys.decomposition_id = 0;
% 
% disp('Joint');
% if (use_gpu)
%     [policies_joint, value_joint, info_joint] = dp_decomposition_gpu(sys, Op, p_joint, s_joint);
%     policies_joint = cellfun(@(x) gather(x), policies_joint, 'UniformOutput', false);
%     value_joint = gather(value_joint);
%     info_joint.state_grid = cellfun(@(x) gather(x), info_joint.state_grid, 'UniformOutput', false);
% else
%     [policies_joint, value_joint, info_joint] = dp_decomposition(sys, Op, p_joint, s_joint);
% end
