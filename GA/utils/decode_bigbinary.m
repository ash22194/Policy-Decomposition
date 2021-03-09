function [p, s] = decode_bigbinary(sys, encoding)

    X_DIMS = sys.X_DIMS;
    U_DIMS = sys.U_DIMS;
    
    assert(size(encoding, 2)==(2*U_DIMS^2 + U_DIMS*X_DIMS), 'Check encoding length!');
    p = zeros(size(encoding, 1), 2*U_DIMS);
    s = zeros(size(encoding, 1), U_DIMS*X_DIMS);
    for ii=1:1:size(encoding,1)
        
        p_ = zeros(U_DIMS, 2);
        action_coupling = reshape(encoding(ii, 1:U_DIMS^2), U_DIMS, U_DIMS);
        action_dependence = reshape(encoding(ii, (1+U_DIMS^2):(2*U_DIMS^2)), U_DIMS, U_DIMS);
        
        actions = linspace(1, U_DIMS, U_DIMS);
        child_count = zeros(U_DIMS+1,1);
        while (~isempty(actions))
            acoupled = find(action_coupling(actions(1),:));
            parent = find(action_dependence(:,acoupled(1)));
            if (isempty(parent))
                parent = 0;
            end
            p_(acoupled, 1) = parent(1);
            child_count(parent(1)+1) = child_count(parent(1)+1) + 1;
            p_(acoupled, 2) = child_count(parent(1)+1);
            
            actions(any(acoupled' == actions, 1)) = [];
        end
        p(ii,:) = reshape(p_, 1, 2*U_DIMS);
        s(ii,:) = encoding(ii, (2*U_DIMS^2+1):end);
    end
end