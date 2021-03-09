function out = dyn_finite(sys, x, u, dt)
    
    if (~isfield(sys, 'active_actions'))
        active_actions = zeros(sys.U_DIMS,1);
        active_actions(sys.U_DIMS_FREE) = 1;
        active_actions(sys.U_DIMS_CONTROLLED) = 1;
        active_actions = int32(active_actions);
    else
        active_actions = int32(sys.active_actions);
    end
    
    if (~isfield(sys, 'grid_size'))
        grid_size = ones(sys.X_DIMS,1);
        grid_size(X_DIMS_FREE) = size(x{X_DIMS_FREE(1)});
        grid_size = int32(grid_size);
    else
        grid_size = int32(sys.grid_size);
    end
    
    limits = sys.limits;
    
    out = cell(6,1);
    [out{:}] = dyn_mex_finite(x{:}, u{:}, sys.m, sys.I, ...
                              sys.d, sys.df, sys.l0, sys.g, dt, ...
                              grid_size, active_actions, limits, sys.X_DIMS_FREE(:));
end