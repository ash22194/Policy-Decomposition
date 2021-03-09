function [f,c,fx,fu,fxx,fxu,fuu,cx,cu,cxx,cxu,cuu] = system_cst(sys, x, u, sub_policies, full_DDP)

    u0FREE = sys.u0(sys.U_DIMS_FREE, 1);
    final = isnan(u(1,:));
    u(:,final) = repmat(u0FREE, [1, sum(final)]);
    if (nargout == 2)
        f = dyn_subs_finite(sys, x, u, sub_policies, sys.dt);
        c = cost_subs(sys, x, u, sub_policies) * sys.dt;
    
    else
        f = [];
        X_DIM = size(x, 1);
        fx = repmat(eye(X_DIM), [1,1,size(x,2)]) ...
              + dynx_subs(sys, x, u, sub_policies) * sys.dt;
        fu = dynu_subs(sys, x, u, sub_policies) * sys.dt;
        if (full_DDP)
            fxx = dynxx_subs(sys, x, u, sub_policies) * sys.dt;
            fxu = dynxu_subs(sys, x, u, sub_policies) * sys.dt;
            fuu = dynuu_subs(sys, x, u, sub_policies) * sys.dt;
        else
            [fxx,fxu,fuu] = deal([]);
        end
        
        c = [];
        cx = dcostdx(sys, x, u, sub_policies) * sys.dt;
        cu = dcostdu(sys, x, u, sub_policies) * sys.dt;
        cxx = ddcostdxx(sys, x, u, sub_policies) * sys.dt;
        cuu = ddcostduu(sys, x, u, sub_policies) * sys.dt;
        cxu = permute(ddcostdux(sys, x, u, sub_policies), [2, 1, 3]) * sys.dt;
    end

end