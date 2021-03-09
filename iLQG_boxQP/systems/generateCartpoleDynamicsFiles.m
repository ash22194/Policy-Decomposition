clear;
close all;
clc;

%% Define system

% Parameters
sys.name = 'cartpole';
sys.mp = sym('mp');
sys.mc = sym('mc');
sys.l = sym('l');
sys.g = sym('g');

% Continuous time dynamics
sys.X_DIMS = 4; % x, x_dot, theta, theta_dot
sys.U_DIMS = 2;
sys.x = sym('x', [sys.X_DIMS, 1]);
sys.u = sym('u', [sys.U_DIMS, 1]);

zero_dyn = [sys.x(2);
            (sys.mp*sys.l*(sys.x(4)^2)*sin(sys.x(3)) + 0.5*sys.mp*sys.g*sin(2*sys.x(3)))/(sys.mc + sys.mp*sin(sys.x(3))^2);
            sys.x(4);
            ((-0.5*sys.mp*sys.l*sys.x(4)^2)*sin(2*sys.x(3)) - (sys.mc+sys.mp)*sys.g*sin(sys.x(3)))/(sys.l*(sys.mc + sys.mp*sin(sys.x(3))^2))];
state_act_dyn = [0;
                 (sys.u(1) - sys.u(2)*cos(sys.x(3))/sys.l)/(sys.mc + sys.mp*sin(sys.x(3))^2);
                 0;
                 (sys.u(2)/sys.l*(sys.mc/sys.mp+1) - sys.u(1)*cos(sys.x(3)))/(sys.l*(sys.mc + sys.mp*sin(sys.x(3))^2))];
sys.f = zero_dyn + state_act_dyn;
sys.fxu = jacobian(sys.f, [sys.x; sys.u]);
sys.fx = sys.fxu(:, 1:sys.X_DIMS);
sys.fu = sys.fxu(:, (sys.X_DIMS+1):end);

% Create Dynamics Files
% mkdir(sys.name);
% matlabFunction(simplify(sys.f), 'File', strcat(sys.name, '/dyn'));
% matlabFunction(simplify(sys.fx), 'File', strcat(sys.name, '/dynx'));
% matlabFunction(simplify(sys.fu), 'File', strcat(sys.name, '/dynu'));
