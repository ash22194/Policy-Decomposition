function all_possible_children = create_possible_children(de)
            
    X_DIMS = de.sys.X_DIMS;
    all_possible_children = cell(de.childnode_maxcount, 1);
    children_count = 0;
    for jj=1:1:size(de.action_tree,1)
        inputs = de.action_tree{jj, 1};
        is_leaf = isempty(de.action_tree{jj, end});
        num_inputs = length(inputs);
        if ((num_inputs==1) || (~is_leaf))
            continue;
        end
        % Possible non-empty 2-groupings of the inputs
        input_2groups = cell(round(distr_n_into_r(num_inputs, 2) / 2), 2);
        num_input_groups = 0;
        for ii=1:1:floor(num_inputs/2)
            size_ii_groups = nchoosek(inputs, ii);
            if (ii == ceil(num_inputs/2))
                size_ii_groups = size_ii_groups(1:(size(size_ii_groups, 1)/2),:);
            end
            num_ii_groups  = size(size_ii_groups, 1);
            size_ii_groups = cellfun(@(x) x', mat2cell(size_ii_groups, ones(num_ii_groups, 1), ii), 'UniformOutput', false);

            input_2groups((num_input_groups+1):(num_input_groups+num_ii_groups), 1) = size_ii_groups;
            input_2groups((num_input_groups+1):(num_input_groups+num_ii_groups), 2) ...
                = cellfun(@(x) setdiff(inputs, x), size_ii_groups, 'UniformOutput', false);

            num_input_groups = num_input_groups + num_ii_groups;
        end

        states = de.action_tree{jj, 2};
        num_states = sum(states);
        states = find(states);
        % Possible 2-groupings of the states
        state_2groups = cell((2^(num_states)), 2);
        num_state_groups = 0;
        for ss=0:1:num_states
            if (num_states ~= 1)
                size_ss_groups = nchoosek(states, ss);
            else
                if (ss==0)
                    size_ss_groups = zeros(1,0);
                else
                    size_ss_groups = states;
                end
            end
            num_ss_groups  = size(size_ss_groups, 1);
            size_ss_groups = mat2cell(size_ss_groups, ones(num_ss_groups, 1), ss);

            state_2groups((num_state_groups+1):(num_state_groups+num_ss_groups), 1) = size_ss_groups;
            state_2groups((num_state_groups+1):(num_state_groups+num_ss_groups), 2) ...
                = cellfun(@(x) setdiff(states, x), size_ss_groups, 'UniformOutput', false);

            num_state_groups = num_state_groups + num_ss_groups;
        end
        
        num_children = length(de.action_tree{jj, end});
        children = linspace(1, num_children, num_children);
        % Possible 2-groupings of the children
        children_2groups = cell((2^(num_children)), 2);
        num_children_groups = 0;
        for ch=0:1:num_children
            if (num_children ~= 1)
                size_ch_groups = nchoosek(children, ch);
            else
                if (ch==0)
                    size_ch_groups = zeros(1,0);
                else
                    size_ch_groups = ones(1,1);
                end
            end
            num_ch_groups  = size(size_ch_groups, 1);
            size_ch_groups = cellfun(@(x) x', mat2cell(size_ch_groups, ones(num_ch_groups, 1), ch), 'UniformOutput', false);
            
            children_2groups((num_children_groups+1):(num_children_groups+num_ch_groups), 1) ...
                = cellfun(@(x) de.action_tree{jj, end}(x), size_ch_groups, 'UniformOutput', false);
            children_2groups((num_children_groups+1):(num_children_groups+num_ch_groups), 2) ...
                = cellfun(@(x) de.action_tree{jj, end}(setdiff(children, x)'), size_ch_groups, 'UniformOutput', false);

            num_children_groups = num_children_groups + num_ch_groups;
        end
        
        action_tree_ = cat(1, de.action_tree(1:(jj-1), :), ...
                              cell(2, 4), ...
                              de.action_tree((jj+1):end, :));
        parent_node = de.action_tree{jj, 3};
        parent_node = cellfun(@(x) fastsetequal(x,parent_node), action_tree_(:,1));
        curr_child_node = cellfun(@(x) fastsetequal(x, inputs), action_tree_{parent_node, end});

        assert(sum(curr_child_node)==1, 'Child not found!');
        action_tree_{parent_node, end}(curr_child_node,:) = [];

        for ni=1:1:num_input_groups
            curr_input_group = input_2groups(ni, :);
            action_tree_{jj,  1} = curr_input_group{1};
            action_tree_{jj,end} = cell(0,1);
            action_tree_{jj+1,1} = curr_input_group{2};
            action_tree_{jj+1,end} = cell(0,1);
            
            for nc=1:1:num_children_groups
                % Distribute the child nodes
                curr_children_group = children_2groups(nc, :);

                action_tree_{jj,  end} = cell(0,1);
                num_ch1 = size(curr_children_group{1},1);
                for ch1=1:1:num_ch1
                    child_node1 = cellfun(@(x) isempty(setdiff(curr_children_group{1}{ch1}, x)) ...
                                           && isempty(setdiff(x, curr_children_group{1}{ch1})), action_tree_(:,1));
                    action_tree_{child_node1, 3} = curr_input_group{1};
                    action_tree_{jj,end} = cat(1, action_tree_{jj,end}, action_tree_{child_node1, 1});
                end

                action_tree_{jj+1,end} = cell(0,1);
                num_ch2 = size(curr_children_group{2},1);
                for ch2=1:1:num_ch2
                    child_node2 = cellfun(@(x) isempty(setdiff(curr_children_group{2}{ch2}, x)) ...
                                           && isempty(setdiff(x, curr_children_group{2}{ch2})), action_tree_(:,1));
                    action_tree_{child_node2, 3} = curr_input_group{2};
                    action_tree_{jj+1, end} = cat(1, action_tree_{jj+1,end}, action_tree_{child_node2, 1});
                end

                for ns=1:1:num_state_groups
                    curr_state_group = state_2groups(ns, :);

                    if ~(((num_ch1==0) && (isempty(curr_state_group{1}))) ...
                        || ((num_ch2==0) && (isempty(curr_state_group{2}))))
                        action_tree_{jj,  3} = de.action_tree{jj, 3}; % Decoupled
                        action_tree_{jj+1,3} = de.action_tree{jj, 3};
                        action_tree_{parent_node, end} = cat(1, action_tree_{parent_node, end}, curr_input_group{1}, curr_input_group{2});

                        action_tree_{jj,  2} = false(1, X_DIMS);
                        action_tree_{jj,  2}(curr_state_group{1}) = true;
                        action_tree_{jj+1,2} = false(1, X_DIMS);
                        action_tree_{jj+1,2}(curr_state_group{2}) = true;

                        action_tree_encoding = fastint2str(encode_binary_adj(action_tree_));
                        if (de.nodelist.isKey(action_tree_encoding))
                            all_possible_children{children_count+1} = de.nodelist(action_tree_encoding);
                        else
                            all_possible_children{children_count+1} = action_tree_;
                        end
                        children_count = children_count + 1;
                        action_tree_{parent_node, end} = action_tree_{parent_node, end}(1:(end-2),:);
                    end

                    if ~((num_ch2==0) && isempty(curr_state_group{2}))
                        action_tree_{jj,  3} = de.action_tree{jj, 3}; % Cascaded 1 after 2
                        action_tree_{parent_node, end} = cat(1, action_tree_{parent_node, end}, curr_input_group{1});
                        action_tree_{jj+1,3} = action_tree_{jj, 1};
                        action_tree_{jj,end} = cat(1, action_tree_{jj, end}, action_tree_{jj+1, 1});

                        action_tree_{jj,  2} = false(1, X_DIMS);
                        action_tree_{jj,  2}(curr_state_group{1}) = true;
                        action_tree_{jj+1,2} = false(1, X_DIMS);
                        action_tree_{jj+1,2}(curr_state_group{2}) = true;

                        action_tree_encoding = fastint2str(encode_binary_adj(action_tree_));
                        if (de.nodelist.isKey(action_tree_encoding))
                            all_possible_children{children_count+1} = de.nodelist(action_tree_encoding);
                        else
                            all_possible_children{children_count+1} = action_tree_;
                        end
                        children_count = children_count + 1;
                        action_tree_{jj,end} = action_tree_{jj,end}(1:(end-1),:);
                        action_tree_{parent_node, end} = action_tree_{parent_node, end}(1:(end-1),:);
                    end

                    if ~((num_ch1==0) && isempty(curr_state_group{1}))
                        action_tree_{jj+1,3} = de.action_tree{jj, 3}; % Cascaded 2 after 1
                        action_tree_{parent_node, end} = cat(1, action_tree_{parent_node, end}, curr_input_group{2});
                        action_tree_{jj,  3} = action_tree_{jj+1, 1};
                        action_tree_{jj+1, end} = cat(1, action_tree_{jj+1, end}, action_tree_{jj, 1});

                        action_tree_{jj,  2} = false(1, X_DIMS);
                        action_tree_{jj,  2}(curr_state_group{1}) = true;
                        action_tree_{jj+1,2} = false(1, X_DIMS);
                        action_tree_{jj+1,2}(curr_state_group{2}) = true;

                        action_tree_encoding = fastint2str(encode_binary_adj(action_tree_));
                        if (de.nodelist.isKey(action_tree_encoding))
                            all_possible_children{children_count+1} = de.nodelist(action_tree_encoding);
                        else
                            all_possible_children{children_count+1} = action_tree_;
                        end
                        children_count = children_count + 1;
                        action_tree_{jj+1,end} = action_tree_{jj+1,end}(1:(end-1),:);
                        action_tree_{parent_node, end} = action_tree_{parent_node, end}(1:(end-1),:);
                    end
                end
            end
        end
    end
    
    assert(children_count==de.childnode_maxcount, 'Mismatch in the number of expected and enumerated child nodes');
    de.childnodes = all_possible_children;
end