function [XFinal, UFinal, X, k, K, sub_trajectories_close, ...
          ilqg_cost, ilqg_trace, ilqg_time, ...
          Xinit, Uinit, Costinit] = get_ilqg_trajectory_multitraj_parallel(sys, Op, starts, K_LQR, ...
                                                        sub_policies_LQR, sub_policies_DDP)
%%  Initialize
    
    NUM_CTRL = round(sys.T / sys.dt);
    u0 = sys.u0(sys.U_DIMS_FREE);
    lims = sys.lims(sys.U_DIMS_FREE, :);
    l_point = sys.l_point(sys.X_DIMS_FREE);
    
    Xinit = nan(length(sys.X_DIMS_FREE), NUM_CTRL+1, size(starts, 2));
    Uinit = nan(length(sys.U_DIMS_FREE), NUM_CTRL+1, size(starts, 2));
    Costinit = zeros(1, size(starts, 2));
    
    Mdl = cellfun(@(x) KDTreeSearcher(x'), sub_policies_DDP(:,6), 'UniformOutput', false);
    sub_policies_DDP_ = cell(size(sub_policies_DDP));
    sub_policies_DDP_(:,1:2) = sub_policies_DDP(:,1:2);
    
    if (isfield(sys, 'u0init') && (sys.u0init))
        for kk=1:1:size(starts,2)
            Xinit(:,1,kk) = starts(:,kk);
            discount = 1;
            for ii=1:1:NUM_CTRL
                Uinit(:,ii,kk) = u0;
                
                for jj=1:1:size(sub_policies_DDP, 1)
                    closest_x = knnsearch(Mdl{jj}, Xinit(sub_policies_DDP{jj,2},ii,kk)');
                    sub_policies_DDP_{jj, 3} = sub_policies_DDP{jj, 3}(:, closest_x);
                    sub_policies_DDP_{jj, 4} = sub_policies_DDP{jj, 4}(:,:, closest_x);
                    sub_policies_DDP_{jj, 5} = sub_policies_DDP{jj, 5}(:, closest_x);
                end
                Xinit(:,ii+1,kk) = dyn_subs_finite(sys, Xinit(:,ii,kk), Uinit(:,ii,kk), sub_policies_DDP_, sys.dt);
                if (any(isnan(Uinit(:,ii,kk))) || any(isnan(Xinit(:,ii,kk))))
                    disp('Check X, U FFull');
                end
                
                Costinit(kk) = Costinit(kk) + discount*cost_subs(sys, Xinit(:,ii,kk), Uinit(:,ii,kk), sub_policies_DDP_)*sys.dt;
                discount = discount*sys.gamma_;
            end
            
            for jj=1:1:size(sub_policies_DDP, 1)
                closest_x = knnsearch(Mdl{jj}, Xinit(sub_policies_DDP{jj,2},NUM_CTRL+1,kk)');
                sub_policies_DDP_{jj, 3} = sub_policies_DDP{jj, 3}(:, closest_x);
                sub_policies_DDP_{jj, 4} = sub_policies_DDP{jj, 4}(:,:, closest_x);
                sub_policies_DDP_{jj, 5} = sub_policies_DDP{jj, 5}(:, closest_x);
            end
            Costinit(kk) = Costinit(kk) + discount*cost_subs(sys, Xinit(:,NUM_CTRL+1,kk), u0, sub_policies_DDP_)*sys.dt;
        end
    else
        for kk=1:1:size(starts,2)
            Xinit(:,1,kk) = starts(:,kk);
            discount = 1;
            for ii=1:1:NUM_CTRL
                Uinit(:,ii,kk) = min(max(u0 + K_LQR*(Xinit(:,ii,kk) - l_point), ...
                                      lims(:,1)), ...
                                  lims(:,2));
                
                for jj=1:1:size(sub_policies_DDP, 1)
                    closest_x = knnsearch(Mdl{jj}, Xinit(sub_policies_DDP{jj,2},ii,kk)');
                    sub_policies_DDP_{jj, 3} = sub_policies_DDP{jj, 3}(:, closest_x);
                    sub_policies_DDP_{jj, 4} = sub_policies_DDP{jj, 4}(:,:, closest_x);
                    sub_policies_DDP_{jj, 5} = sub_policies_DDP{jj, 5}(:, closest_x);
                end
                Xinit(:,ii+1,kk) = dyn_subs_finite(sys, Xinit(:,ii,kk), Uinit(:,ii,kk), sub_policies_DDP_, sys.dt);
                if (any(isnan(Uinit(:,ii,kk))) || any(isnan(Xinit(:,ii,kk))))
                    disp('Check X, U FFull');
                end
                
                Costinit(kk) = Costinit(kk) + discount*cost_subs(sys, Xinit(:,ii,kk), Uinit(:,ii,kk), sub_policies_DDP_)*sys.dt;
                discount = discount*sys.gamma_;
            end
            
            for jj=1:1:size(sub_policies_DDP, 1)
                closest_x = knnsearch(Mdl{jj}, Xinit(sub_policies_DDP{jj,2},NUM_CTRL+1,kk)');
                sub_policies_DDP_{jj, 3} = sub_policies_DDP{jj, 3}(:, closest_x);
                sub_policies_DDP_{jj, 4} = sub_policies_DDP{jj, 4}(:,:, closest_x);
                sub_policies_DDP_{jj, 5} = sub_policies_DDP{jj, 5}(:, closest_x);
            end
            Costinit(kk) = Costinit(kk) + discount*cost_subs(sys, Xinit(:,NUM_CTRL+1,kk), u0, sub_policies_DDP_)*sys.dt;
        end
    end
    
%%  Compute DDP Trajectory
    
    ilqg_system = @(x, u, sub_policies_, i) ...
                    system_cst(sys, x, u, sub_policies_, sys.full_DDP);
    ilqg_time = zeros(1, size(starts,2));
    ilqg_cost = zeros(1, NUM_CTRL+1, size(starts,2));
    ilqg_trace = cell(size(starts,2), 1);
    X = zeros(length(sys.X_DIMS_FREE), NUM_CTRL+1, size(starts, 2));
    XFinal = zeros(length(sys.X_DIMS_FREE), NUM_CTRL+1, size(starts, 2));
    k = zeros(length(sys.U_DIMS_FREE), NUM_CTRL, size(starts, 2));
    UFinal = zeros(length(sys.U_DIMS_FREE), NUM_CTRL, size(starts, 2));
    K = zeros(length(sys.U_DIMS_FREE), length(sys.X_DIMS_FREE), NUM_CTRL, size(starts, 2));
    sub_trajectories_close = cell(size(sub_policies_DDP));
    sub_trajectories_close(:, 1:2) = sub_policies_DDP(:, 1:2);
    sub_trajectories_close_ = cell(size(starts, 2), 1);
    
    parfor kk=1:1:size(starts, 2)
        
%         Op.cost = Costinit(kk);
        tic;
        [XFinal_, X(:,:,kk), UFinal(:,:,kk), k(:,:,kk), K(:,:,:,kk), ...
         sub_trajectories_close_{kk, 1}, ~, ~, ilqg_cost(:,:,kk), ilqg_trace{kk}] = iLQGGeneralKDTree(ilqg_system, ...
                                                                            Xinit(:,1,kk), ... % Don't pass the state trajectory in initialization
                                                                            Uinit(:,1:NUM_CTRL,kk), ...
                                                                            sub_policies_DDP, ...
                                                                            Op);
        ilqg_time(kk) = toc;
        XFinal(:,:,kk) = XFinal_;
        disp(strcat(num2str(kk),') Final init point : ', sprintf('%.4f ', Xinit(:, end, kk))));
        disp(strcat(num2str(kk),') Final DDP point : ', sprintf('%.4f ', XFinal_(:, end)), sprintf('\n')));
    end
    
    k = cat(2, k, repmat(u0, [1, 1, size(starts, 2)]));
    UFinal = cat(2, UFinal, repmat(u0, [1, 1, size(starts, 2)]));
    K = cat(3, K, zeros(length(sys.U_DIMS_FREE), length(sys.X_DIMS_FREE), 1, size(starts, 2)));
    for kk=1:1:size(starts, 2)
        
        sub_trajectories_close(:,3) = cellfun(@(x, y) cat(3, x, y), ...
                                              sub_trajectories_close(:,3), sub_trajectories_close_{kk,1}(:,3), ...
                                              'UniformOutput', false);
        sub_trajectories_close(:,4) = cellfun(@(x, y) cat(4, x, y), ...
                                              sub_trajectories_close(:,4), sub_trajectories_close_{kk,1}(:,4), ...
                                              'UniformOutput', false);
        sub_trajectories_close(:,5) = cellfun(@(x, y) cat(3, x, y), ...
                                              sub_trajectories_close(:,5), sub_trajectories_close_{kk,1}(:,5), ...
                                              'UniformOutput', false);
        sub_trajectories_close(:,6) = cellfun(@(x, y) cat(3, x, y), ...
                                              sub_trajectories_close(:,6), sub_trajectories_close_{kk,1}(:,6), ...
                                              'UniformOutput', false);
    end
    
    X = reshape(X, size(X,1), size(X,2)*size(X,3));
    XFinal = reshape(XFinal, size(XFinal,1), size(XFinal,2)*size(XFinal,3));
    k = reshape(k, size(k,1), size(k,2)*size(k,3));
    UFinal = reshape(UFinal, size(UFinal,1), size(UFinal,2)*size(UFinal,3));
    K = reshape(K, size(K,1), size(K,2), size(K,3)*size(K,4));
    
end