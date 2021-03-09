clear;
close all;
clc;

%% 

sys.n = 4;
sys.name = strcat('manipulator', num2str(sys.n), 'dof');
sys.m = sym('m', [sys.n, 1]);
sys.l = sym('l', [sys.n, 1]);
sys.g = sym('g');
sys.th = sym('th', [sys.n ,1]);
sys.dth = sym('dth', [sys.n ,1]);

sys.X_DIMS = 2*sys.n;
sys.U_DIMS = sys.n;

% Zero configuration is pointing downwards at -ve y axis, 
% z axis is coming out of the plane
zero_config = [-pi/2; zeros(sys.n-1,1)];

% Forward Kinematics
joint_loc = zeros(2,1);
com_loc = [];
for j=1:1:(sys.n-1)
    joint_loc_rel = [sys.l(j)*cos(sum(sys.th(1:j) + zero_config(1:j)));
                     sys.l(j)*sin(sum(sys.th(1:j) + zero_config(1:j)))];
    com_loc = [com_loc, joint_loc(:,end) + 0.5*joint_loc_rel];
    joint_loc = [joint_loc, joint_loc(:,end) + joint_loc_rel];
end
ee_loc = joint_loc(:,end) + [sys.l(sys.n)*cos(sum(sys.th(1:sys.n) + zero_config(1:sys.n)));
                             sys.l(sys.n)*sin(sum(sys.th(1:sys.n) + zero_config(1:sys.n)))];
com_loc = [com_loc, joint_loc(:,end) ...
                    + 0.5*[sys.l(sys.n)*cos(sum(sys.th(1:sys.n) + zero_config(1:sys.n)));
                           sys.l(sys.n)*sin(sum(sys.th(1:sys.n) + zero_config(1:sys.n)))]];

% Calculate jacobian to a point
p = sym('p', [2,1]);
Jtop = [];
for j=1:1:sys.n
    joint_loc_rel_p = p - joint_loc(:,j);
    Jtop = [Jtop, [-joint_loc_rel_p(2); joint_loc_rel_p(1); 1]];
end

% Compute mass matrix
sys.M = zeros(sys.n, sys.n);
for j=1:1:sys.n
    M_com = [sys.m(j)*eye(2), zeros(2,1); zeros(1,2), sys.m(j)*sys.l(j)^2/12];
    Jtocom = subs(Jtop, p, com_loc(:,j));
    Jtocom(:,(j+1):sys.n) = zeros(3, sys.n-j);
    sys.M = sys.M + Jtocom.'*M_com*Jtocom;
end
sys.M = simplify(sys.M);

% Compute corriolis matrix
delM = reshape(jacobian(sys.M(:), sys.th), sys.n*ones(1,3));
sys.C = sym(zeros(sys.n, sys.n));
for i=1:1:sys.n
    for j=1:1:sys.n
        sys.C(i,j) = sum(reshape(delM(i,j,:), sys.n, 1).*sys.dth) ...
                     + sum(reshape(delM(i,:,j), sys.n, 1).*sys.dth) ...
                     - sum(delM(:,j,i).*sys.dth);
    end
end
sys.C = simplify(0.5.*sys.C);

% Compute gravity vector
V = sum(sys.m.*(com_loc(2,:).')*sys.g);
sys.N = jacobian(V, sys.th).';
sys.N = simplify(sys.N);

sys.x = [sys.th; sys.dth];
sys.u = sym('u', [sys.U_DIMS, 1]);

sys.f = [sys.dth; sys.M\(sys.u - sys.C*sys.dth - sys.N)];
sys.fxfu = jacobian(sys.f, [sys.x; sys.u]);
sys.fx = sys.fxfu(:, 1:sys.X_DIMS);
sys.fu = sys.fxfu(:, (sys.X_DIMS+1):end);

% Create Dynamics Files
% mkdir(sys.name);
% matlabFunction(simplify(sys.f, 'IgnoreAnalyticConstraints', true), 'File', strcat(sys.name, '/dyn'));
% matlabFunction(simplify(sys.fx, 'IgnoreAnalyticConstraints', true), 'File', strcat(sys.name, '/dynx'));
% matlabFunction(simplify(sys.fu, 'IgnoreAnalyticConstraints', true), 'File', strcat(sys.name, '/dynu'));
