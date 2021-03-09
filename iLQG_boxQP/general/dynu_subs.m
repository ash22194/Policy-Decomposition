function out = dynu_subs(sys, x, u, sub_policies)
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
    
    l_point = sys.l_point;
    
    X = zeros(length(X_DIMS_FREE) + length(X_DIMS_FIXED), TRAJ_LENGTH);
    X(X_DIMS_FIXED, :) = repmat(l_point(X_DIMS_FIXED), [1, TRAJ_LENGTH]);
    X(X_DIMS_FREE, :) = x;
    
    % System is control affine, fu does not depend on U!! 
    U = zeros(length(U_DIMS_FREE) + length(U_DIMS_FIXED) + length(U_DIMS_CONTROLLED), TRAJ_LENGTH);
    
    out = dynu(sys, X, U);
    out = out(X_DIMS_FREE, U_DIMS_FREE, :);

end