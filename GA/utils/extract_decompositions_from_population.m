function [unique_population, unique_population_id] = extract_decompositions_from_population(sys, population)
    
    population = round(population);
    population_ = nan(size(population, 1), sys.U_DIMS-1+sys.U_DIMS*sys.X_DIMS);
    dummy_max_value = linspace(0, sys.U_DIMS, sys.U_DIMS+1);
    dummy_max_value = sum(dummy_max_value.*((sys.U_DIMS + 1).^dummy_max_value));
    
    for ii=1:1:size(population, 1)
        % Construct a tree
        action_tree = {};
        queue = {{0; -1}};
        p_ = [linspace(1, sys.U_DIMS, sys.U_DIMS)', ...
              reshape(population(ii, (1:2*sys.U_DIMS)), sys.U_DIMS, 2)];
              
        while(~isempty(queue))
            curr_node = queue{1}; % pop the top element
            curr_parent = curr_node{2};
            curr_node_ = curr_node{1};
            curr_node{1} = [];
            while (curr_node_ > 0)
                curr_node{1} = [curr_node{1}; mod(curr_node_, (sys.U_DIMS+1))];
                curr_node_ = round((curr_node_ - curr_node{1}(end))/(sys.U_DIMS+1));
            end
            queue = queue(2:end);
            if (isempty(curr_node{1}))
                curr_node = curr_node_;
            else
                curr_node = curr_node{1};
            end
            
            children = p_(any(p_(:,2)==curr_node', 2), [1, 3]); % Find inputs that are children to curr_parent
            childID = unique(children(:,2));                    % Find unique childIDs, inputs are coupled if child IDs are same
            
            curr_node = sum(sort(curr_node).*((sys.U_DIMS+1).^linspace(0,length(curr_node)-1,length(curr_node)))');
            curr_children = zeros(length(childID), 1);
            for jj=1:1:length(childID)
                curr_children_ = sort(children(children(:,2)==childID(jj), 1));
                curr_children_ = curr_children_ ...
                                 .*((sys.U_DIMS+1).^(linspace(0, length(curr_children_)-1, length(curr_children_))))';
                curr_children(jj) = sum(curr_children_);
                queue{end+1} = {curr_children(jj); curr_node};
            end

            action_tree(end+1, :) = {curr_node, curr_parent, curr_children};
        end
        
        try 
            % Convert action tree to prufer code
            node_list = cell2mat(action_tree(:,1));
            prufer_seq = [];
            while (size(action_tree) > 2)
                leaf_node_ids = cellfun(@(x) isempty(x), action_tree(:,end));
                leaf_nodes = cell2mat(action_tree(leaf_node_ids, 1));
                leaf_node_parents = cell2mat(action_tree(leaf_node_ids, 2));
                leaf_node_ids = find(leaf_node_ids);
                
                [smallest_leaf_node, smallest_leaf_node_id] = min(leaf_nodes);
                smallest_leaf_node_parent = leaf_node_parents(smallest_leaf_node_id);
                smallest_leaf_node_parent_id = find(cellfun(@(x) smallest_leaf_node_parent==x, action_tree(:,1)));
                
                prufer_seq = [prufer_seq, smallest_leaf_node_parent];
                children = action_tree{smallest_leaf_node_parent_id, end};
                children_to_remove = children==smallest_leaf_node;
                children(children_to_remove) = [];
                action_tree{smallest_leaf_node_parent_id, end} = children;
                action_tree(leaf_node_ids(smallest_leaf_node_id),:) = [];
            end
            prufer_seq = [prufer_seq, ...
                          dummy_max_value*ones(1, sys.U_DIMS - length(prufer_seq) - 1)];
            if (length(prufer_seq) > (sys.U_DIMS - 1))
                disp('check sequence length');
            end
            population_(ii, 1:(sys.U_DIMS - 1)) = prufer_seq;
            population_(ii, sys.U_DIMS:end) = population(ii, (2*sys.U_DIMS+1):end);
        catch ME
            disp('some bug');
        end
    end
    
    [unique_population, unique_population_id] = unique(population_, 'rows', 'stable');
end