function out = dyn_subs_finite2(sys, x, u, sub_policies, dt)
    %% Inputs
    % subpolicies is a cell array of size (m_subs x 5) - {U_SUBDIM, X_SUBDIM, k, K, xn}
    % k(:, ii)  - (U_SUBDIM x 1)
    % K(:, ii)  - (U_SUBDIM x X_SUBDIM)
    % xn(:, ii) - (X_SUBDIM x 1)
    % U_SUBDIM are indices of U_DIM_CONTROLLED
    % X_SUBDIM are indices X_DIMS_FREE
    
    %% Dynamics calculations
    
    U_DIMS_FREE = sys.U_DIMS_FREE;
    U_DIMS_FIXED = sys.U_DIMS_FIXED;
    U_DIMS_CONTROLLED = sys.U_DIMS_CONTROLLED;
    
    X_DIMS_FREE = sys.X_DIMS_FREE;
    X_DIMS_FIXED = sys.X_DIMS_FIXED;
    TRAJ_LENGTH = size(x,2);
    
    lims = sys.lims;
    l_point = sys.l_point;
    
    X = zeros(length(X_DIMS_FREE) + length(X_DIMS_FIXED), TRAJ_LENGTH);
%     X(X_DIMS_FIXED, :) = repmat(l_point(X_DIMS_FIXED), [1, TRAJ_LENGTH]);
    X(X_DIMS_FIXED, :) = l_point(X_DIMS_FIXED)*ones(1, TRAJ_LENGTH);
    X(X_DIMS_FREE, :) = x;
    
    U = zeros(length(U_DIMS_FREE) + length(U_DIMS_FIXED) + length(U_DIMS_CONTROLLED), TRAJ_LENGTH);
    % Free actions
    U(U_DIMS_FREE, :) = u;
    % Controlled actions
    for jj = 1:1:size(sub_policies, 1)
        U_SUBDIM = U_DIMS_CONTROLLED(sub_policies{jj, 1});
        X_SUBDIM = sub_policies{jj, 2};
        
        x_ = x(X_SUBDIM, :) - sub_policies{jj, 5}(:, 1:TRAJ_LENGTH);
        x_ = reshape(x_, 1, length(X_SUBDIM), TRAJ_LENGTH);
        x_ = repmat(x_, [length(U_SUBDIM), 1, 1]);
        Kx = sum((sub_policies{jj, 4}(:,:, 1:TRAJ_LENGTH)).*x_, 2);
        U(U_SUBDIM, :) = sub_policies{jj, 3}(:, 1:TRAJ_LENGTH) + reshape(Kx(:,1,:), size(Kx, 1), size(Kx, 3));
    end
    % Fixed actions are zero 
    U = max(lims(:,1)*ones(1, TRAJ_LENGTH), ...
            min(lims(:,2)*ones(1, TRAJ_LENGTH), U));
    
    k1 = dyn(sys, X, U) * dt;
    X1 = X; X1(X_DIMS_FREE,:) = X1(X_DIMS_FREE,:) + 0.5 * k1(X_DIMS_FREE,:);
    
    k2 = dyn(sys, X1, U) * dt;
    X2 = X; X2(X_DIMS_FREE,:) = X2(X_DIMS_FREE,:) + 0.5 * k2(X_DIMS_FREE,:);
    
    k3 = dyn(sys, X2, U) * dt;
    X3 = X; X3(X_DIMS_FREE,:) = X3(X_DIMS_FREE,:) + k3(X_DIMS_FREE,:);
    
    k4 = dyn(sys, X3, U) * dt;
    
    out = X + (1.0/6.0)*(k1 + 2.0 * k2 + 2.0 * k3 + k4);
    out = out(X_DIMS_FREE, :);
end