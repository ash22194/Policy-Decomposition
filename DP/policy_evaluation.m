function [value, info] = policy_evaluation(sys, Op, sub_policies)
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
    x(X_DIMS_FIXED) = num2cell(goal(X_DIMS_FIXED));
    grid_indices = cell(length(X_DIMS_FREE), 1);
    for xxi = 1:1:length(X_DIMS_FREE)
        xx = X_DIMS_FREE(xxi);
        grid_indices{xxi} = linspace(limits(xx,1), limits(xx,2), num_points(xx));
    end
    [x{X_DIMS_FREE}] = ndgrid(grid_indices{:});
    
    % Initialize policy grids - fixed, free and controlled
    u = cell(U_DIMS, 1);
    u(U_DIMS_FIXED) = num2cell(zeros(length(U_DIMS_FIXED), 1));
    for uui = 1:1:length(U_DIMS_FREE)
        uu = U_DIMS_FREE(uui);
        u{uu} = lims(uu, 1) + (lims(uu, 2) - lims(uu, 1)) * rand(num_points(X_DIMS_FREE));
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
        
        u(U_SUBDIM) = cellfun(@(x) repmat(reshape(x, subpolicy_size), subpolicy_newsize), sub_policies{jj,3}, 'UniformOutput', false);
    end
    
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
    
    goal_grid = zeros(length(X_DIMS_FREE), 1);
    for xxi=1:1:length(X_DIMS_FREE)
        xx = X_DIMS_FREE(xxi);
        [~, goal_grid(xxi)] = min(abs(grid_indices{xxi} - goal(xx)));
    end
    goal_grid = num2cell(goal_grid);
    
    G_ = zeros(num_points(X_DIMS_FREE));
    G = ones(size(G_));
    F = griddedInterpolant(x{X_DIMS_FREE}, G_);
    policy_iter = 0;
    time_total = 0;
    
    time_start = tic;
    x_ = dyn_finite_rk4(sys, x, u, dt); % Next state after a step
    while ((max(abs(G_ - G), [], 'all') > gtol) && (policy_iter < max_iter))
        % Iterate to estimate value function
        tic;
        policy_iter = policy_iter + 1;

        G = G_;
        Gnext = F(x_{X_DIMS_FREE});
        Gnext(goal_grid{:}) = 0;

        % Update value function
        G_ = cost_total + gamma_*Gnext;
        G_(goal_grid{:}) = 0;
        F.Values = G_;
        
        time_iter = toc;
%         disp(strcat('Policy evaluation iter :', num2str(policy_iter), ', time : ', num2str(time_iter)));
    end
    time_total = toc(time_start);
    
    value = G_;
    info.state_grid = x(X_DIMS_FREE);
    info.policy_iter = policy_iter;
    info.time_total = time_total;
end
