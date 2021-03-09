function ddc = ddcostdxx(sys, x, u, sub_policies)
    %% Inputs
    % subpolicies is a cell array of size (m_subs x 5) - {U_SUBDIM, X_SUBDIM, k, K, xn}
    % k(:, ii)  - (U_SUBDIM x 1)
    % K(:, ii)  - (U_SUBDIM x X_SUBDIM)
    % xn(:, ii) - (X_SUBDIM x 1)
    % U_SUBDIM are indices of U_DIM_CONTROLLED
    % X_SUBDIM are indices X_DIMS_FREE
    
    %% Derivative calculations

    U_DIMS_CONTROLLED = sys.U_DIMS_CONTROLLED;
    X_DIMS_FREE = sys.X_DIMS_FREE;
    TRAJ_LENGTH = size(x,2);
    
    R = 2 * sys.R(U_DIMS_CONTROLLED, U_DIMS_CONTROLLED);
    K = zeros(length(U_DIMS_CONTROLLED), length(X_DIMS_FREE), TRAJ_LENGTH);
    for jj=1:1:size(sub_policies, 1)
        U_SUBDIM = sub_policies{jj, 1};
        X_SUBDIM = sub_policies{jj, 2};
        
        K(U_SUBDIM, X_SUBDIM, :) = sub_policies{jj, 4};
    end
    
    ddc = 2 * repmat(sys.Q(X_DIMS_FREE, X_DIMS_FREE), 1, 1, TRAJ_LENGTH);
    if (~isempty(U_DIMS_CONTROLLED))
        for ii=1:1:TRAJ_LENGTH
            ddc(:,:, ii) = ddc(:,:, ii) + K(:,:,ii)'*R*K(:,:,ii);
        end
    end
end