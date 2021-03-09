function out = dyn_finite_rk4(sys, x, u, dt)

X_DIMS_FREE = sys.X_DIMS_FREE;
limits      = sys.limits;

k1 = dyn(sys, x, u);
q = x;
for xxi=1:1:length(X_DIMS_FREE)
    xx = X_DIMS_FREE(xxi);
    q{xx} = q{xx} + 0.5 * dt * k1{xxi};
end

k2 = dyn(sys, q, u);
q = x;
for xxi=1:1:length(X_DIMS_FREE)
    xx = X_DIMS_FREE(xxi);
    q{xx} = q{xx} + 0.5 * dt * k2{xxi};
end

k3 = dyn(sys, q, u);
q = x;
for xxi=1:1:length(X_DIMS_FREE)
    xx = X_DIMS_FREE(xxi);
    q{xx} = q{xx} + dt * k3{xxi};
end

k4 = dyn(sys, q, u);
out = x;

for xxi=1:1:length(X_DIMS_FREE)
    xx = X_DIMS_FREE(xxi);
    out{xx} = out{xx} + dt / 6.0 * (k1{xxi} + 2.0 * k2{xxi} + 2.0 * k3{xxi} + k4{xxi});
    
    % Enforce state bounds!!
    out{xx} = max(limits(xx, 1), min(limits(xx, 2), out{xx}));
end

end
