clear;
close all;
clc;

%% Define system

% Parameters
sys.name = 'quadcopter';
sys.m = sym('m');
sys.g = sym('g');
sys.l = sym('l');
sys.bk = sym('bk'); % tau/f
sys.I = diag(sym('I', [3,1]));

% Continuous time dynamics
sys.X_DIMS = 10; % z, ro, pi, ya, vx, vy, vz, vro, vpi, vya
sys.U_DIMS = 4;
sys.x = sym('x', [sys.X_DIMS, 1]);
sys.u = sym('u', [sys.U_DIMS, 1]);

zero_dyn = [zeros(4,1); 0; 0; -sys.g; zeros(3, 1)];
Wn = [1, 0,          -sin(sys.x(2));
      0, cos(sys.x(3)),  cos(sys.x(2))*sin(sys.x(3));
      0, -sin(sys.x(3)), cos(sys.x(2))*cos(sys.x(3))];
nu = Wn*[sys.x(8); 
         sys.x(9); 
         sys.x(10)];
tau = [sys.l*(sys.u(4) - sys.u(2));
       sys.l*(sys.u(3) - sys.u(1));
       sys.bk*(sys.u(1) - sys.u(2) + sys.u(3) - sys.u(4))];
nudot = sys.I\(tau - cross(nu, sys.I*nu));

Wninvdot = [0, ...
            sys.x(9)*cos(sys.x(3))*tan(sys.x(2))+sys.x(8)*sin(sys.x(3))/(cos(sys.x(2))^2), ...
            -sys.x(9)*sin(sys.x(3))*cos(sys.x(2))+sys.x(8)*cos(sys.x(3))/(cos(sys.x(2))^2);
            0, -sys.x(9)*sin(sys.x(3)), -sys.x(9)*cos(sys.x(3));
            0, ...
            sys.x(9)*cos(sys.x(3))/cos(sys.x(2))+sys.x(9)*sin(sys.x(3))*tan(sys.x(2))/cos(sys.x(2)), ...
            -sys.x(9)*sin(sys.x(3))/cos(sys.x(2))+sys.x(8)*cos(sys.x(3))*tan(sys.x(2))/cos(sys.x(2))];
state_act_dyn = [sys.x(7:10);
                 (sys.u(1)+sys.u(2)+sys.u(3)+sys.u(4))/sys.m*([cos(sys.x(3))*sin(sys.x(2))*cos(sys.x(4)) + sin(sys.x(3))*sin(sys.x(4));
                                               cos(sys.x(3))*sin(sys.x(2))*sin(sys.x(4)) - sin(sys.x(3))*cos(sys.x(4));
                                               cos(sys.x(2))*cos(sys.x(3))]);
                 Wninvdot*nu + Wn\nudot];
sys.f = zero_dyn + state_act_dyn;
sys.fxu = jacobian(sys.f, [sys.x; sys.u]);
sys.fx = sys.fxu(:, 1:sys.X_DIMS);
sys.fu = sys.fxu(:, (sys.X_DIMS+1):end);

% Create Dynamics Files
% mkdir(sys.name);
% matlabFunction(simplify(sys.f), 'File', strcat(sys.name, '/dyn'));
% matlabFunction(simplify(sys.fx), 'File', strcat(sys.name, '/dynx'));
% matlabFunction(simplify(sys.fu), 'File', strcat(sys.name, '/dynu'));
