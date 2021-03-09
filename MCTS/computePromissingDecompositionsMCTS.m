clear;
close all;
clc;

%% 

restoredefaultpath();
system_name = 'biped2d';
addpath(strcat('../iLQG_boxQP/systems/', system_name));
addpath('utils');
load(strcat('data/', system_name, 'System.mat'), 'sys');
sys.measure_func = @(err_lqr, err_compute) (1 - exp(-err_lqr)) .* err_compute;
% sys.measure_func = @(err_lqr, err_compute) (min(20.0, err_lqr) / 20.0) .* err_compute;

max_mcts_iter = inf;
max_mcts_time = 100;
num_runs = 1;
deterministic = false;

%%

mctree = cell(num_runs, 2);
time_mcts = zeros(num_runs, 1);
for rr=1:1:num_runs
    mctree{rr, 1} = policy_decomposition_leafexpand(sys, ...
                                              [zeros(sys.U_DIMS,1), ones(sys.U_DIMS,1)], ...
                                              ones(sys.U_DIMS, sys.X_DIMS), ...
                                              {});
    tic;
    [~, mctree{rr, 2}] = run_mcts(mctree{rr, 1}, max_mcts_iter, max_mcts_time, deterministic);
    time_mcts(rr) = toc;
end

%% Extract Promissing Decompositions

num_to_extract = 10;
alg_stats = zeros(num_runs, 8);
alg_best_decompositions = cell(num_runs, 3);
for rr=1:1:num_runs
    root = mctree{rr, 1};
    info = mctree{rr, 2};

    children = root.childnodes(cellfun(@(x) isa(x, class(root)), root.childnodes));
    children_measure = cellfun(@(x) x.measure, children);
    [~, children_measure_order] = sort(children_measure);
    children = children(children_measure_order, :);

    num_extracted = 0;
    best_children = cell(0, 1);
    while (num_extracted < num_to_extract && ~isempty(children))
        best_children = cat(1, best_children, children(1,:));
        while (abs(best_children{end, 1}.measure ...
                   - sys.measure_func(best_children{end, 1}.lqr_measure, ...
                                      best_children{end, 1}.compute_fraction)) > eps)
             best_children{end, 1} = best_children{end, 1}.childnode_best;
        end
        assert(best_children{end, 1}.measure <= children{1, 1}.measure, 'Check the best child lookup');

        if (any(cellfun(@(x) strcmp(x.decomposition_key, ...
                                    best_children{end, 1}.decomposition_key), ...
                        best_children(1:(end-1),:))))
            best_children = best_children(1:(end-1), :);
        else
            num_extracted = num_extracted + 1;
        end
        children = children(2:end,:);
    end

    % Remove decompositions that are repeated
    best_children_decomposition_id = cellfun(@(x) [reshape(x.p, 1, 2*sys.U_DIMS), reshape(x.s, 1, sys.U_DIMS*sys.X_DIMS)], ...
                                              best_children, 'UniformOutput', false);
    best_children_decomposition_id = cell2mat(best_children_decomposition_id);

    best_children_lqr_measure = cellfun(@(x) x.lqr_measure, best_children);
    best_children_compute_fraction = cellfun(@(x) x.compute_fraction, best_children);

    assert(all(abs(cellfun(@(x) x.measure, best_children)...
                   - sys.measure_func(best_children_lqr_measure, best_children_compute_fraction))...
                < eps), 'Check measure calculation!');
    
    alg_stats(rr, 1) = info.best_measure;
    alg_stats(rr, 2) = mean(cellfun(@(x) x.measure, best_children));
    alg_stats(rr, 3) = std(cellfun(@(x) x.measure, best_children));
    
    alg_stats(rr, 4) = min(best_children_lqr_measure);
    alg_stats(rr, 5) = mean(best_children_lqr_measure);
    alg_stats(rr, 6) = std(best_children_lqr_measure);
    
    alg_stats(rr, 7) = info.time_to_find_best;
    alg_stats(rr, 8) = info.num_nodes_explored;
    
    alg_best_decompositions{rr, 1} = best_children_decomposition_id;
    alg_best_decompositions{rr, 2} = best_children_lqr_measure;
    alg_best_decompositions{rr, 3} = best_children_compute_fraction;
end

save(strcat('data/', system_name, '_MCTS_explqrobj', num2str(max_mcts_time), '.mat'), 'sys', ...
     'alg_stats', 'alg_best_decompositions');
