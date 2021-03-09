function c = cost_subs(sys, x, u, sub_policies)
    %% Inputs
    % subpolicies is a cell array of size (m_subs x 5) - {U_SUBDIM, X_SUBDIM, k, K, xn}
    % k(:, ii)  - (U_SUBDIM x 1)
    % K(:, ii)  - (U_SUBDIM x X_SUBDIM)
    % xn(:, ii) - (X_SUBDIM x 1)
    % U_SUBDIM are indices of U_DIM_CONTROLLED
    % X_SUBDIM are indices X_DIMS_FREE
    
    %% Cost calculations

    U_DIMS_FREE = sys.U_DIMS_FREE;
    U_DIMS_CONTROLLED = sys.U_DIMS_CONTROLLED;
    U_DIMS_FC = [U_DIMS_FREE; U_DIMS_CONTROLLED];
    X_DIMS_FREE = sys.X_DIMS_FREE;
    TRAJ_LENGTH = size(x,2);
    
    lims = sys.lims;
    l_point = sys.l_point;
    u0 = sys.u0;
    
    U = zeros(length(U_DIMS_FC), TRAJ_LENGTH);
    % Free actions
    if (~isempty(U_DIMS_FREE))
        [~, U_DIMS_FREE] = find(U_DIMS_FREE == U_DIMS_FC');
        U(U_DIMS_FREE, :) = u;
    end
    % Controlled actions
    for jj = 1:1:size(sub_policies, 1)
        U_SUBDIM = U_DIMS_CONTROLLED(sub_policies{jj, 1});
        [~, U_SUBDIM] = find(U_SUBDIM == U_DIMS_FC');
        X_SUBDIM = sub_policies{jj, 2};
        
        x_ = x(X_SUBDIM, :) - sub_policies{jj, 5};
        x_ = reshape(x_, 1, length(X_SUBDIM), TRAJ_LENGTH);
        x_ = repmat(x_, length(U_SUBDIM), 1, 1);
        Kx = sum(sub_policies{jj, 4}.*x_, 2);
        U(U_SUBDIM, :) = sub_policies{jj, 3} + reshape(Kx(:,1,:), size(Kx, 1), size(Kx, 3));
    end
    
    U = max(lims(U_DIMS_FC,1)*ones(1, TRAJ_LENGTH), ...
            min(lims(U_DIMS_FC,2)*ones(1, TRAJ_LENGTH), U));
    
    R = sys.R(U_DIMS_FC, U_DIMS_FC);
    U0 = u0(U_DIMS_FC);
    
    Q = sys.Q(X_DIMS_FREE, X_DIMS_FREE);
    X0 = l_point(X_DIMS_FREE);
    
    x_bar = sign(x - X0).*mod(abs(x - X0), sys.cxmod(X_DIMS_FREE));
    c = diag(x_bar'*Q*x_bar + (U - U0)'*R*(U - U0))';

end