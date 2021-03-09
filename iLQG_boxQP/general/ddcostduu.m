function ddc = ddcostduu(sys, x, u, sub_policies)
    %% Inputs
    % subpolicies is a cell array of size (m_subs x 5) - {U_SUBDIM, X_SUBDIM, k, K, xn}
    % k(:, ii)  - (U_SUBDIM x 1)
    % K(:, ii)  - (U_SUBDIM x X_SUBDIM)
    % xn(:, ii) - (X_SUBDIM x 1)
    % U_SUBDIM are indices of U_DIM_CONTROLLED
    % X_SUBDIM are indices X_DIMS_FREE
    
    %% Derivative calculations

    U_DIMS_FREE = sys.U_DIMS_FREE;
    TRAJ_LENGTH = size(x,2);
    
    ddc = 2 * repmat(sys.R(U_DIMS_FREE, U_DIMS_FREE), 1, 1, TRAJ_LENGTH);
end