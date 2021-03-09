function [xoverKids] = crossoverfunction(sys, parents, options, nvars, FitnessFcn, ...
                                         unused, thisPopulation)
     
     % Possible crossovers 
     % Action tree of one and state assignment of the other
     % Valid crossover requires same input couplings in both parents
     
     sparents = reshape(thisPopulation(parents, (2*sys.U_DIMS + 1):end)', ...
                        sys.U_DIMS, sys.X_DIMS, length(parents));
     MAX_ACTIONSET_VALUE = ((sys.U_DIMS + 1).^linspace(0, sys.U_DIMS-1, sys.U_DIMS))*linspace(1, sys.U_DIMS, sys.U_DIMS)' + 1;
     parentID = zeros(length(parents), 1);
     for ii=1:1:length(parents)
         action = ones(sys.U_DIMS, 1);
         action_coupled = [];
         while(sum(action) > 0)
             a1 = find(action, 1);
             a1coupled = find(all(sparents(a1,:,ii) == sparents(:,:,ii), 2));
             action(a1coupled) = 0;
             a1coupled = ((sys.U_DIMS+1).^linspace(0, length(a1coupled)-1, length(a1coupled)))*a1coupled;
             action_coupled = [action_coupled; a1coupled];
         end
         action_coupled = sort(action_coupled);
         parentID(ii) = (MAX_ACTIONSET_VALUE.^linspace(0, length(action_coupled)-1, length(action_coupled)))*action_coupled;
     end
                    
     % Match parents based on the ID that indicates input coupling
     xoverKids = [];
     for ii=1:1:length(parents)
         ps1 = thisPopulation(parents(ii), :);
         ps2 = find(parentID(ii)==parentID, 2);
         if (length(ps2)<2)
             continue;
         end

         ps2 = thisPopulation(parents(ps2(2)), :);
         parentID(ii) = -1;
     
         p1 = reshape(ps1(1:2*sys.U_DIMS), sys.U_DIMS, 2);
         s1 = reshape(ps1((2*sys.U_DIMS+1):end), sys.U_DIMS, sys.X_DIMS);
         p2 = reshape(ps2(1:2*sys.U_DIMS), sys.U_DIMS, 2);
         s2 = reshape(ps2((2*sys.U_DIMS+1):end), sys.U_DIMS, sys.X_DIMS);
         
     % Check if crossover is compatible with input couplings
         s1_ = s1;
         s2_ = s2;
%          action = ones(sys.U_DIMS, 1);
%          are_compatible = true;
%          while (sum(action) > 0)
%              a1 = find(action, 1);
%              a1coupled = all(s1(a1,:) == s1, 2);
%              a2coupled = all(s2(a1,:) == s2, 2);
%              if ((sum(a1coupled.*(~a2coupled)) > 0) || (sum(a2coupled.*(~a1coupled)) > 0))
%                  are_compatible = false;
%                  disp('Incompatible!!');
%                  break;
%              end
%              action(a1coupled) = 0;
%          end

%          while (sum(action) > 0)
%              a1 = find(action, 1);
%              a1coupled = all(s1(a1,:) == s1, 2);
%              a2coupled = all(s2(a1,:) == s2, 2);
%              a1minusa2 = a1coupled & (~a2coupled);
%              a2minusa1 = a2coupled & (~a1coupled);
%              
%              if ((sum(a1minusa2) > 0) && (sum(a2minusa1) == 0))
%                  if (sum(a1coupled) > sum(s1(a1, :), 2))
%                      are_compatible = false;
%                      break;
%                  end
%                  s2_(a1coupled,:) = repmat(sum(s2(a1coupled, :), 1), sum(a1coupled), 1);
%                  
%                  sa1coupled = find(s1(a1, :));
%                  
%                  sa1decoupled = zeros(sys.X_DIMS, 1);
%                  sa1decoupled(sa1coupled) = randi([1, 2], length(sa1coupled), 1);
%                  assignment_check = sa1decoupled==[1, 2];
%                  while(~all(any(assignment_check, 1)) || (sum(assignment_check(:, 2)) < sum(a1minusa2)))
%                     sa1decoupled(sa1coupled) = randi([1, 2], length(sa1coupled), 1);
%                     assignment_check = sa1decoupled==[1, 2];
%                  end
%                  assignment_check = assignment_check';
%                  s1_(a2coupled, :) = repmat(assignment_check(1, :), sum(a2coupled), 1);
%                  s1_(a1minusa2, :) = repmat(assignment_check(2, :), sum(a1minusa2), 1);
%                  
%              elseif ((sum(a2minusa1) > 0) && (sum(a1minusa2) == 0))
%                  if (sum(a2coupled) > sum(s2(a1, :), 2))
%                      are_compatible = false;
%                      break;
%                  end
%                  s1_(a2coupled,:) = repmat(sum(s1(a2coupled, :), 1), sum(a2coupled), 1);
%                  
%                  sa2coupled = find(s2(a1, :));
%                  
%                  sa2decoupled = zeros(sys.X_DIMS, 1);
%                  sa2decoupled(sa2coupled) = randi([1, 2], length(sa2coupled), 1);
%                  assignment_check = sa2decoupled==[1, 2];
%                  while(~all(any(assignment_check, 1)) || (sum(assignment_check(:, 2)) < sum(a2minusa1)))
%                     sa2decoupled(sa2coupled) = randi([1, 2], length(sa2coupled), 1);
%                     assignment_check = sa2decoupled==[1, 2];
%                  end
%                  assignment_check = assignment_check';
%                  s2_(a1coupled, :) = repmat(assignment_check(1, :), sum(a1coupled), 1);
%                  s2_(a2minusa1, :) = repmat(assignment_check(2, :), sum(a2minusa1), 1);
%                  
%              elseif ((sum(a1minusa2) > 0) && (sum(a2minusa1) > 0))
%                  are_compatible = false;
%                  break;
%              end
%              
%              action(a1) = 0;
%          end
         
%          if (are_compatible)
             [c1, ceq1] = constraints(p1, s2_);
             [c2, ceq2] = constraints(p2, s1_);
             if (~(all(c1 <= 0) && all(ceq1==0)))
                 disp('Problem in 1');
             end
             if (~(all(c2 <= 0) && all(ceq2==0)))
                 disp('Problem in 2');
             end
              xoverKids = [xoverKids; 
                           reshape(p1, 1, 2*sys.U_DIMS), reshape(s2_, 1, sys.U_DIMS*sys.X_DIMS);
                           reshape(p2, 1, 2*sys.U_DIMS), reshape(s1_, 1, sys.U_DIMS*sys.X_DIMS)];
%          end
        if (2*size(xoverKids, 1) >= length(parents))
            break;
        end
     end
     xoverKids = xoverKids(1:round(0.5*length(parents)), :);
end
