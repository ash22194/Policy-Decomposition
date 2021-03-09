function [policy, info] = get_dp_solution_gpu(sys, Op, sub_policies)
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
    
    max_iter           = Op.max_iter;
    max_policy_iter    = Op.max_policy_iter;
    gtol               = Op.gtol;
    u_max_tol          = Op.u_max_tol;
    u_mean_tol         = Op.u_mean_tol;
    num_points         = Op.num_points;
    num_action_samples = prod(Op.num_action_samples(U_DIMS_FREE));
    
    % Logging
    save_dir       = Op.save_dir;
    subsystem_id   = strcat('U', sprintf('%d', sys.U_DIMS_FREE), '_X', sprintf('%d', sys.X_DIMS_FREE));
    save_file      = strcat(save_dir, '/', subsystem_id, '_interm.mat');
    if (isfield(Op, 'save_every')) 
        save_every = Op.save_every; 
    else 
        save_every = max_iter / 10; 
    end
    reuse_policy   = isfield(Op, 'reuse_policy') && (Op.reuse_policy) && (isfile(save_file));
    
%% Initialize 
    % Create state grids
    % Initialize policy grids - fixed, free and controlled
    x = cell(X_DIMS, 1);
    u = cell(U_DIMS, 1);
    x(X_DIMS_FIXED) = num2cell(gpuArray(goal(X_DIMS_FIXED)));
    u(U_DIMS_FIXED) = num2cell(gpuArray(zeros(length(U_DIMS_FIXED), 1)));
    grid_indices = cell(length(X_DIMS_FREE), 1);
    for xxi = 1:1:length(X_DIMS_FREE)
        xx = X_DIMS_FREE(xxi);
        grid_indices{xxi} = gpuArray(linspace(limits(xx,1), limits(xx,2), num_points(xx)));
    end

    if (reuse_policy)
        decomposition  = load(save_file);
        u(U_DIMS_FREE) = decomposition.policy;
        info           = decomposition.info;
        x(X_DIMS_FREE) = info.state_grid;
        G_             = info.value;
    else
        for uui = 1:1:length(U_DIMS_FREE)
            uu = U_DIMS_FREE(uui);
            u{uu} = lims(uu, 1) + (lims(uu, 2) - lims(uu, 1)) * rand(num_points(X_DIMS_FREE), 'gpuArray');
        end
        [x{X_DIMS_FREE}] = ndgrid(grid_indices{:});
        G_ = gpuArray(zeros(num_points(X_DIMS_FREE)));
    
        info.iter = 0;
        info.time_total = 0;
        info.time_policy_eval = 0;
        info.time_policy_update = 0;
    end
    
    % Resize sub_policies
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
    cost_fixed = cost_state + cost_action_controlled;
    
    % Identify goal location in grid co-ordinates
    goal_grid = gpuArray(zeros(length(X_DIMS_FREE), 1));
    for xxi=1:1:length(X_DIMS_FREE)
        xx = X_DIMS_FREE(xxi);
        [~, goal_grid(xxi)] = min(abs(grid_indices{xxi} - goal(xx)));
    end
    goal_grid = num2cell(goal_grid);
    
    % Initialize policy iteration
    [kernel_interp, kernel_inputs] = generate_interp_kernel(x{X_DIMS_FREE});
    constant_policy_count = 0;
    iter = info.iter;
    time_total = info.time_total;
    time_policy_eval = info.time_policy_eval;
    time_policy_update = info.time_policy_update;
    uerror_max = gpuArray(zeros(length(U_DIMS_FREE), 1));
    uerror_mean = gpuArray(zeros(length(U_DIMS_FREE), 1));
    
%% Policy Iteration
    time_start = tic;
    while (iter < max_iter)
        iter = iter + 1;
        if (mod(iter, save_every)==0)
            policy = u(U_DIMS_FREE);
            info.state_grid = x(X_DIMS_FREE);
            info.value = G_;
            info.iter = iter;
            info.time_total = time_total;
            info.time_policy_eval = time_policy_eval;
            info.time_policy_update = time_policy_update;
            sys_ = sys;
            save(save_file, 'policy', 'info', 'sys_', '-v7.3', '-nocompression');
        end
        
        G = gpuArray(ones(size(G_)));
        policy_iter = 0;
        cost_action = 0;
        x_ = dyn_finite_rk4_mex(sys, x, u, dt);
        for uui = 1:1:length(U_DIMS_FREE)
            uu = U_DIMS_FREE(uui);
            cost_action = cost_action + dt * R(uu, uu) * (u{uu} - u0(uu)).^2;
        end
        
        cost_total = cost_fixed + cost_action;
        time_policy_eval_start = tic;
        % Compute value function for current policy
         while ((max(abs(G_ - G), [], 'all') > gtol) && (policy_iter < max_policy_iter))
            % Iterate to estimate value function
            G = G_;
            policy_iter = policy_iter + 1;
            time_interp_start = tic;
            Gnext = feval(kernel_interp, x_{X_DIMS_FREE}, G_, kernel_inputs{:});
            time_interp = toc(time_interp_start);
            Gnext(goal_grid{:}) = 0;
            
            % Update value function
            G_ = cost_total + gamma_*Gnext;
            G_(goal_grid{:}) = 0;
%             fprintf('Policy Iter : %d, Time Interp : %.4f\n', policy_iter, time_interp);
        end
        time_policy_eval_curr = toc(time_policy_eval_start);
        time_policy_eval = time_policy_eval + time_policy_eval_curr;
        clear 'G' 'cost_total' 'cost_action';

        % Update policy
        % Compute Q(x,u) values g(x,u) + G(x+udt) for different actions when value
        % function G has converged and greedily update the policy i.e. choose
        % action u* at state x such that u* = argmin Q(x,u)
        minG = G_;
        unew = u;
        u_ = u;
        time_policy_update_start = tic;
        for i = 1:1:num_action_samples
            cost_action_ = 0;
            for uui=1:1:length(U_DIMS_FREE)
                uu = U_DIMS_FREE(uui);
                u_{uu} = lims(uu, 1) + (lims(uu, 2) - lims(uu, 1)) * rand(size(G_), 'gpuArray');
                cost_action_ = cost_action_ + dt * R(uu, uu) * (u_{uu} - u0(uu)).^2;
            end
            
            time_dyn_start = tic;
            x__ = dyn_finite_rk4_mex(sys, x, u_, dt);
            time_dyn = toc(time_dyn_start);
            
            time_interp_start = tic;
            Gnext_ = feval(kernel_interp, x__{X_DIMS_FREE}, G_, kernel_inputs{:});
            time_interp = toc(time_interp_start);
            Gnext_(goal_grid{:}) = 0;
            Gnext_ = cost_fixed + cost_action_ + gamma_*Gnext_;
            Gnext_(goal_grid{:}) = 0;
            
            actionUpdate = Gnext_ < minG;
            actionUpdateBar = ~actionUpdate;
            minG = (Gnext_ .* actionUpdate) + (minG .* actionUpdateBar);
            for uui = 1:1:length(U_DIMS_FREE)
                uu = U_DIMS_FREE(uui);
                unew{uu} = (u_{uu} .* actionUpdate) + (unew{uu} .* actionUpdateBar);
                unew{uu}(goal_grid{:}) = u0(uu);
            end
%             fprintf('Policy Update : %d, Time Interp : %.4f, Time Dyn : %.4f\n', i, time_interp, time_dyn);
        end
        
        for uui = 1:1:length(U_DIMS_FREE)
            uu = U_DIMS_FREE(uui);
            uerror = abs(unew{uu} - u{uu});
            uerror_max(uui) = max(uerror, [], 'all');
            uerror_mean(uui) = sqrt(sum(uerror.^2, 'all')) / numel(G_);
        end
        u = unew;
        time_policy_update_curr = toc(time_policy_update_start);
        time_policy_update = time_policy_update + time_policy_update_curr;
        clear 'x__' 'u_' 'minG' 'cost_action_' 'Gnext_' 'actionUpdate' 'actionUpdateBar' 'unew';
        
        if (all(uerror_max < u_max_tol(U_DIMS_FREE)) ...
            && all(uerror_mean < u_mean_tol(U_DIMS_FREE)))
            constant_policy_count = constant_policy_count + 1;
            if (constant_policy_count==50)
                fprintf('Policy converged!\n')
                break;
            end
            disp(strcat('Constant Policy count : ', num2str(constant_policy_count)));
        else
            disp(strcat(sprintf('Iteration : %d,', iter), ...
                        ' Max Err :', sprintf(' %.3f', uerror_max), ...
                        ', Mean Err :', sprintf(' %.5f', uerror_mean), ...
                        ', Time Eval : ', num2str(time_policy_eval_curr), ...
                        ', Time Update : ', num2str(time_policy_update_curr)));
            constant_policy_count = 0;
        end
        
    end
    time_total = toc(time_start);
    
%% Return
    policy = u(U_DIMS_FREE);
    info.state_grid = x(X_DIMS_FREE);
    info.value = G_;
    info.iter = iter;
    info.time_total = time_total;
    info.time_policy_eval = time_policy_eval;
    info.time_policy_update = time_policy_update;

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
