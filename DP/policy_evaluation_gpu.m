function [value, info] = policy_evaluation_gpu(sys, Op, sub_policies)
%% Parameters
    X_DIMS            = sys.X_DIMS;
    X_DIMS_FREE       = sys.X_DIMS_FREE;
    X_DIMS_FIXED      = sys.X_DIMS_FIXED;
    U_DIMS            = sys.U_DIMS;
    U_DIMS_FREE       = sys.U_DIMS_FREE;
    U_DIMS_FIXED      = sys.U_DIMS_FIXED;
    U_DIMS_CONTROLLED = sys.U_DIMS_CONTROLLED;
    limits            = sys.limits; % state limits
    lims              = sys.lims;   % action limits
    goal              = sys.goal;
    u0                = sys.u0;
    Q                 = sys.Q;
    R                 = sys.R;
    gamma_            = sys.gamma_;
    dt                = sys.dt;

    max_iter          = Op.max_iter;
    gtol              = Op.gtol;
    num_points        = Op.num_points;

%% Initialize 
    % Create state grids
    x = cell(X_DIMS, 1);
    x(X_DIMS_FIXED) = num2cell(gpuArray(goal(X_DIMS_FIXED)));
    grid_indices = cell(length(X_DIMS_FREE), 1);
    for xxi = 1:1:length(X_DIMS_FREE)
        xx = X_DIMS_FREE(xxi);
        grid_indices{xxi} = gpuArray(linspace(limits(xx,1), limits(xx,2), num_points(xx)));
    end
    [x{X_DIMS_FREE}] = ndgrid(grid_indices{:});
    
    % Initialize policy grids - fixed, free and controlled
    u = cell(U_DIMS, 1);
    u(U_DIMS_FIXED) = num2cell(zeros(length(U_DIMS_FIXED), 1));
    for uui = 1:1:length(U_DIMS_FREE)
        uu = U_DIMS_FREE(uui);
        u{uu} = lims(uu, 1) + (lims(uu, 2) - lims(uu, 1)) * rand(num_points(X_DIMS_FREE), 'gpuArray');
    end
    for jj=1:1:size(sub_policies, 1)
        U_SUBDIM = sub_policies{jj,1};
        X_SUBDIM = sub_policies{jj,2};
        [X_SUBDIM, ~] = find(X_DIMS_FREE == X_SUBDIM');
        X_SUBDIM_BAR = 1:length(X_DIMS_FREE);
        X_SUBDIM_BAR(X_SUBDIM) = [];
        
        subpolicy_size = num_points(X_DIMS_FREE);
        subpolicy_size(X_SUBDIM_BAR) = 1;
        subpolicy_newsize = num_points(X_DIMS_FREE);
        subpolicy_newsize(X_SUBDIM) = 1;
        
        u(U_SUBDIM) = cellfun(@(x) gpuArray(repmat(reshape(x, subpolicy_size), subpolicy_newsize)), sub_policies{jj,3}, 'UniformOutput', false);
    end
    
    sys.active_actions = zeros(U_DIMS,1);
    sys.active_actions(U_DIMS_FREE) = 1;
    sys.active_actions(U_DIMS_CONTROLLED) = 1;
    sys.active_actions = int32(sys.active_actions);
    
    sys.grid_size = ones(X_DIMS,1);
    sys.grid_size(X_DIMS_FREE) = size(x{X_DIMS_FREE(1)});
    sys.grid_size = int32(sys.grid_size);
    
    % Initialize cost functions
    % State cost assuming diagonal Q matrix
    cost_state = 0;
    for xxi = 1:1:length(X_DIMS_FREE)
        xx = X_DIMS_FREE(xxi);
        cost_state = cost_state + dt * Q(xx,xx) * (x{xx} - goal(xx)).^2;
    end
    
    % Controlled action cost assuming diagonal R matrix
    cost_action_controlled = 0;
    for uui=1:1:length(U_DIMS_CONTROLLED)
        uu = U_DIMS_CONTROLLED(uui);
        cost_action_controlled = cost_action_controlled + dt * R(uu, uu) * (u{uu} - u0(uu)).^2;
    end
    cost_total = cost_state + cost_action_controlled;
    
    goal_grid = gpuArray(zeros(length(X_DIMS_FREE), 1));
    for xxi=1:1:length(X_DIMS_FREE)
        xx = X_DIMS_FREE(xxi);
        [~, goal_grid(xxi)] = min(abs(grid_indices{xxi} - goal(xx)));
    end
    goal_grid = num2cell(goal_grid);
    
    G_ = gpuArray(zeros(num_points(X_DIMS_FREE)));
    G = gpuArray(ones(size(G_)));
    
    [kernel_interp, kernel_inputs] = generate_interp_kernel(x{X_DIMS_FREE});
    policy_iter = 0;
    time_total = 0;
    
    time_start = tic;
    x_ = dyn_finite_rk4_mex(sys, x, u, dt); % Next state after a step
    while ((max(abs(G_ - G), [], 'all') > gtol) && (policy_iter < max_iter))
        % Iterate to estimate value function
        tic;
        policy_iter = policy_iter + 1;
        G = G_;
        Gnext = feval(kernel_interp, x_{X_DIMS_FREE}, G_, kernel_inputs{:});
        Gnext(goal_grid{:}) = 0;

        % Update value function
        G_ = cost_total + gamma_*Gnext;
        G_(goal_grid{:}) = 0;
        
        time_iter = toc;
%         disp(strcat('Policy evaluation iter :', num2str(policy_iter), ', time : ', num2str(time_iter)));
    end
    time_total = toc(time_start);
    
    value = G_;
    info.state_grid = x(X_DIMS_FREE);
    info.policy_iter = policy_iter;
    info.time_total = time_total;
end

function [k, inputs] = generate_interp_kernel(varargin)
% x1, ..., xn generated from ndgrid

n = nargin;
grid_sizes = cell2mat(cellfun(@(x) size(x), varargin(:), 'UniformOutput', false));
size_compare = ones(n ,1) * grid_sizes(1,:) == grid_sizes;
assert(all(size_compare, 'all'), "All xi must have the same dimensions");

Nx = grid_sizes(1,:)';
x1 = cellfun(@(y) y(1), varargin(:)); % Min value in each dimension
dx = cellfun(@(y) y(end) - y(1), varargin(:)) ./ (Nx(1:n) - 1); % Discretizatio in each dimension

% Generate corners of the n dimensional hyper-cube
corners = 0:1:(2^n - 1);
cb = zeros(n, 2^n);
for ii=1:1:n
    r = floor(corners / 2);
    cb(ii, :) = corners - 2*r;
    corners = r;
end
cb = num2cell(flip(cb + 1, 1), 2);
corners_index = sub2ind(Nx, cb{:}) - 1;

grid_size = prod(Nx(1:n));
Nx = num2cell(Nx(1:n));
x1 = num2cell(x1);
dx = num2cell(dx);
inputs = cat(1, {grid_size}, Nx, dx, x1, {corners_index});

nlinear = sprintf('calc_average%d',n);
k = parallel.gpu.CUDAKernel(strcat('cuda/', nlinear, '.ptx'), ...
                            strcat('cuda/', nlinear, '.cu'), ...
                            nlinear);
k.ThreadBlockSize = 896;
k.GridSize = 1024;
end