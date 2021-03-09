function dc = dcostdx(sys, x, u, sub_policies)
    %% Inputs
    % subpolicies is a cell array of size (m_subs x 5) - {U_SUBDIM, X_SUBDIM, k, K, xn}
    % k(:, ii)  - (U_SUBDIM x 1)
    % K(:, ii)  - (U_SUBDIM x X_SUBDIM)
    % xn(:, ii) - (X_SUBDIM x 1)
    % U_SUBDIM are indices of U_DIM_CONTROLLED
    % X_SUBDIM are indices X_DIMS_FREE
    
    %% Derivative calculations
    
    U_DIMS_FREE = sys.U_DIMS_FREE;
    U_DIMS_CONTROLLED = sys.U_DIMS_CONTROLLED;
    X_DIMS_FREE = sys.X_DIMS_FREE;
    TRAJ_LENGTH = size(x,2);
    
    lims = sys.lims;
    l_point = sys.l_point;
    u0 = sys.u0;
    
    U = zeros(length(U_DIMS_CONTROLLED), TRAJ_LENGTH);
    K = zeros(length(U_DIMS_CONTROLLED), length(X_DIMS_FREE), TRAJ_LENGTH);
    
    % Controlled actions
    for jj = 1:1:size(sub_policies, 1)
        U_SUBDIM = sub_policies{jj, 1};
        X_SUBDIM = sub_policies{jj, 2};
        
        x_ = x(X_SUBDIM, :) - sub_policies{jj, 5};
        x_ = reshape(x_, 1, length(X_SUBDIM), TRAJ_LENGTH);
        x_ = repmat(x_, length(U_SUBDIM), 1, 1);
        K(U_SUBDIM, X_SUBDIM, :) = sub_policies{jj, 4};
        Kx = sum(sub_policies{jj, 4}.*x_, 2);
        
        U(U_SUBDIM, :) = sub_policies{jj, 3} + reshape(Kx(:,1,:), size(Kx, 1), size(Kx, 3));
    end
    
    R_CC = sys.R(U_DIMS_CONTROLLED, U_DIMS_CONTROLLED);
    R_CF = sys.R(U_DIMS_CONTROLLED, U_DIMS_FREE);
    R_FC = sys.R(U_DIMS_FREE, U_DIMS_CONTROLLED);
    
    x_bar = sign(x - l_point(X_DIMS_FREE)).*mod(abs(x - l_point(X_DIMS_FREE)), sys.cxmod(X_DIMS_FREE));
    dc = 2 * sys.Q(X_DIMS_FREE, X_DIMS_FREE) * x_bar;
    if (~isempty(U_DIMS_CONTROLLED))
        for ii=1:1:TRAJ_LENGTH
            dc(:, ii) = dc(:, ii) + K(:,:, ii)' * (R_CF + R_FC') * (max(lims(U_DIMS_FREE,1),...
                                                            min(lims(U_DIMS_FREE,2), ...
                                                                u(:, ii))) - u0(U_DIMS_FREE)) ...
                        + 2 * K(:,:, ii)' * R_CC * (max(lims(U_DIMS_CONTROLLED, 1), ...
                                                        min(lims(U_DIMS_CONTROLLED,2), ...
                                                             U(:, ii))) - u0(U_DIMS_CONTROLLED));
        end
    end
end