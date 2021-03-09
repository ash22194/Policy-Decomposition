function dc = dcostdu(sys, x, u, sub_policies)
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
    TRAJ_LENGTH = size(x,2);
    
    lims = sys.lims;
    u0 = sys.u0;
    
    U = zeros(length(U_DIMS_CONTROLLED), TRAJ_LENGTH);
    % Controlled actions
    for jj = 1:1:size(sub_policies, 1)
        U_SUBDIM = sub_policies{jj, 1};
        X_SUBDIM = sub_policies{jj, 2};
        
        x_ = x(X_SUBDIM, :) - sub_policies{jj, 5};
        x_ = reshape(x_, 1, length(X_SUBDIM), TRAJ_LENGTH);
        x_ = repmat(x_, length(U_SUBDIM), 1, 1);
        Kx = sum(sub_policies{jj, 4}.*x_, 2);
        U(U_SUBDIM, :) = sub_policies{jj, 3} + reshape(Kx(:,1,:), size(Kx, 1), size(Kx, 3));
    end
    % Fixed actions are zero
    
    R_FF = sys.R(U_DIMS_FREE, U_DIMS_FREE);
    R_FC = sys.R(U_DIMS_FREE, U_DIMS_CONTROLLED);
    R_CF = sys.R(U_DIMS_CONTROLLED, U_DIMS_FREE);

    dc = 2 * R_FF * (max(lims(U_DIMS_FREE,1)*ones(1, TRAJ_LENGTH),...
                         min(lims(U_DIMS_FREE,2)*ones(1, TRAJ_LENGTH), u)) ...
                     - u0(U_DIMS_FREE));
    if (~isempty(U_DIMS_CONTROLLED))
        dc = dc + (R_CF' + R_FC) ...
                   * (max(lims(U_DIMS_CONTROLLED, 1)*ones(1, TRAJ_LENGTH), ...
                           min(lims(U_DIMS_CONTROLLED,2)*ones(1, TRAJ_LENGTH), U)) ...
                      - u0(U_DIMS_CONTROLLED)); 
%         for ii = 1:1:TRAJ_LENGTH
%             dc(:, ii) = dc(:, ii) + (R_CF' + R_FC) * (max(lims(U_DIMS_CONTROLLED, 1), ...
%                                                           min(lims(U_DIMS_CONTROLLED,2), ...
%                                                               U(:, ii))) - u0(U_DIMS_CONTROLLED)); 
%         end
    end
end