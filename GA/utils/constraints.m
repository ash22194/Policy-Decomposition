function [c, c_eq] = constraints(p, s)
%% 
% p is m x 2 matrix where pi1 denotes the parent to input i
%                         pi2 denotes the child ID for input i
% If two inputs have the same non-zero parent and the same child ID, then they are coupled
% s is m x n matrix where sij = 1 if input i is dependent on state j

p = round(p);
s = logical(round(s));
m = size(s, 1);
n = size(s, 2);

%% Constraints

c = [];
c_eq = [];

% No input must be it's own parent
input_ids = (1:1:m)';
parent_check = (p(:,1) ~= input_ids) - 1;
c = [c; parent_check; -parent_check];

% At least one input must have a zero parent
c = [c; 1 - sum(p(:,1)==0)];

% No cycles in input tree (how to constraint?)  
for ii=1:1:m
    inputi = ii;
    loop_count = 0;
    while(inputi~=0 && p(inputi,1)~=ii)
        loop_count = loop_count + 1;
        if (loop_count > m)
            disp('Cycle constraint check loop');
            break;
        end
        inputi = p(inputi,1);
    end
%     c_eq = [c_eq; inputi];
    c = [c; inputi; -inputi]; % inputi == 0 for no cycles
end

% Each state variable must be assigned to at least one input
c = [c; 1 - sum(s, 1)'];

coupled_inputs = nchoosek(input_ids, 2);
for ii=1:1:size(coupled_inputs, 1)
    % Coupled inputs must have the same state assignment
    input1 = coupled_inputs(ii,1);
    input2 = coupled_inputs(ii,2);
    are_coupled = false;
    loop_count = 0;
    while (input1~=0) && (input2~=0) && (p(input1,2)==p(input2,2))
        loop_count = loop_count + 1;
        if (p(input1,1)==p(input2,1) || (loop_count > (m+1)))
            are_coupled = true;
            break;
        end
        input1 = p(input1,1);
        input2 = p(input2,1);
    end
    complete_state_overlap = min(1, sum(abs(s(coupled_inputs(ii,1),:) - s(coupled_inputs(ii,2),:))));
    
    % Inputs that are not coupled must have disjont state assignment
    no_state_overlap = min(1, sum(s(coupled_inputs(ii,1),:).*s(coupled_inputs(ii,2),:)));
    
    c = [c; are_coupled*complete_state_overlap; -are_coupled*complete_state_overlap;
             (1-are_coupled)*no_state_overlap; (are_coupled-1)*no_state_overlap];
end

% Constraint to avoid jointly optimizing the inputs
c = [c; sum(s, 'all') - (m*n-1)];

end