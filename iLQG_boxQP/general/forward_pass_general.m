function [xnew,unew,ulast,cnew,sub_trajectories_close] = forward_pass_general(x0,u,L,x,du,...
                                                                        sub_trajectories,...
                                                                        Alpha,gamma_,DYNCST,lims,diff)
% parallel forward-pass (rollout)
% internally time is on the 3rd dimension, 
% to facillitate vectorized dynamics calls

n        = size(x0,1);
K        = length(Alpha);
K1       = ones(1,K); % useful for expansion
m        = size(u,1);
N        = size(u,2);

xnew        = zeros(n,K,N);
xnew(:,:,1) = x0(:,ones(1,K));
unew        = zeros(m,K,N);
ulast       = zeros(m,K,N);
cnew        = zeros(1,K,N+1);
sub_trajectories_close = cell(size(sub_trajectories));
sub_trajectories_close(:, 1:2) = sub_trajectories(:, 1:2);
for jj = 1:1:size(sub_trajectories,1)
    sub_trajectories_close{jj, 3} = zeros(length(sub_trajectories{jj, 1}),K,N+1);
    sub_trajectories_close{jj, 4} = zeros(length(sub_trajectories{jj, 1}), ...
                                          length(sub_trajectories{jj, 2}),K,N+1);
    sub_trajectories_close{jj, 5} = zeros(length(sub_trajectories{jj, 2}),K,N+1);
    sub_trajectories_close{jj, 6} = zeros(length(sub_trajectories{jj, 2}),K,N+1);
end
sub_trajectories_close_i = cell(size(sub_trajectories, 1), size(sub_trajectories, 2)-1);
sub_trajectories_close_i(:, 1:2) = sub_trajectories(:, 1:2);

discount = 1;
for i = 1:N
    unew(:,:,i) = u(:,i*K1);
    ulast(:,:,i) = u(:,i*K1);
    
    if ~isempty(du)
        unew(:,:,i) = unew(:,:,i) + du(:,i)*Alpha;
        ulast(:,:,i) = ulast(:,:,i) + du(:,i)*Alpha;
    end    
    
    if ~isempty(L)
        if ~isempty(diff)
            dx = diff(xnew(:,:,i), x(:,i*K1));
        else
            dx          = xnew(:,:,i) - x(:,i*K1);
        end
        unew(:,:,i) = unew(:,:,i) + L(:,:,i)*dx;
    end
    
    if ~isempty(lims)
        unew(:,:,i) = min(lims(:,2*K1), max(lims(:,1*K1), unew(:,:,i)));
    end
    
    for jj=1:1:size(sub_trajectories, 1)
        distref = pdist2(sub_trajectories{jj, 6}', xnew(sub_trajectories{jj, 2},:,i)');
        [~, closest_xn2] = min(distref);
        
        sub_trajectories_close_i{jj, 3} = sub_trajectories{jj, 3}(:, closest_xn2);
        sub_trajectories_close_i{jj, 4} = sub_trajectories{jj, 4}(:,:, closest_xn2);
        sub_trajectories_close_i{jj, 5} = sub_trajectories{jj, 5}(:, closest_xn2);
        
        sub_trajectories_close{jj, 3}(:,:, i) = sub_trajectories{jj, 3}(:, closest_xn2);
        sub_trajectories_close{jj, 4}(:,:,:, i) = sub_trajectories{jj, 4}(:,:, closest_xn2);
        sub_trajectories_close{jj, 5}(:,:, i) = sub_trajectories{jj, 5}(:, closest_xn2);
        sub_trajectories_close{jj, 6}(:,:, i) = sub_trajectories{jj, 6}(:, closest_xn2);
    end

    [xnew(:,:,i+1), cnew(:,:,i)]  = DYNCST(xnew(:,:,i), unew(:,:,i), ...
                                           sub_trajectories_close_i, ...
                                           i*K1);
    cnew(:,:,i) = discount*cnew(:,:,i);
    discount = discount*gamma_;
end
for jj=1:1:size(sub_trajectories, 1)
    distref = pdist2(sub_trajectories{jj, 6}', xnew(sub_trajectories{jj, 2},:,N+1)');
    [~, closest_xn2] = min(distref);
    
    sub_trajectories_close_i{jj, 3} = sub_trajectories{jj, 3}(:, closest_xn2);
    sub_trajectories_close_i{jj, 4} = sub_trajectories{jj, 4}(:,:, closest_xn2);
    sub_trajectories_close_i{jj, 5} = sub_trajectories{jj, 5}(:, closest_xn2);

    sub_trajectories_close{jj, 3}(:,:, N+1) = sub_trajectories{jj, 3}(:, closest_xn2);
    sub_trajectories_close{jj, 3} = permute(sub_trajectories_close{jj, 3}, [1 3 2]);
    
    sub_trajectories_close{jj, 4}(:,:,:, N+1) = sub_trajectories{jj, 4}(:,:, closest_xn2);
    sub_trajectories_close{jj, 4} = permute(sub_trajectories_close{jj, 4}, [1 2 4 3]);
    
    sub_trajectories_close{jj, 5}(:,:, N+1) = sub_trajectories{jj, 5}(:, closest_xn2);
    sub_trajectories_close{jj, 5} = permute(sub_trajectories_close{jj, 5}, [1 3 2]);
    
    sub_trajectories_close{jj, 6}(:,:, N+1) = sub_trajectories{jj, 6}(:, closest_xn2);
    sub_trajectories_close{jj, 6} = permute(sub_trajectories_close{jj, 6}, [1 3 2]);
end

[~, cnew(:,:,N+1)] = DYNCST(xnew(:,:,N+1),nan(m,K,1),sub_trajectories_close_i,i);
cnew(:,:,N+1) = discount*cnew(:,:,N+1);
% put the time dimension in the columns
xnew = permute(xnew, [1 3 2]);
unew = permute(unew, [1 3 2]);
ulast = permute(ulast, [1 3 2]);
cnew = permute(cnew, [1 3 2]);
end % forward pass
