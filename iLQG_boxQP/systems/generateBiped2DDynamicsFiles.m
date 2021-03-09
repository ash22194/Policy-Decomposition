clear;
close all;
clc;

%% Define system

% Parameters
sys.name = 'biped2d';
sys.m = sym('m');
sys.I = sym('I');
sys.l0 = sym('l0');
sys.d = sym('d');
sys.df = sym('df');
sys.g = sym('g');

% Continuous time dynamics
sys.X_DIMS = 6; % l, alpha, x_dot, z_dot, theta, theta_dot
sys.U_DIMS = 4;
sys.x = sym('x', [sys.X_DIMS, 1]);
sys.u = sym('u', [sys.U_DIMS, 1]);

ca1       = cos(sys.x(2));
sa1       = sin(sys.x(2));
x_hip     = sys.x(1)*ca1;
z_hip     = sys.x(1)*sa1;
l2        = sqrt((x_hip + sys.df)^2 + z_hip^2);
a2        = acos((sys.df + x_hip)/l2);
ca2       = cos(a2);
sa2       = sin(a2);
% contact1  = (sys.x(1)<=sys.l0);
% contact2  = (l2<=sys.l0);
F1  = sys.u(1); % *contact1;
F2  = sys.u(2); % *contact2;
Fo1 = sys.u(3)/sys.x(1); % *contact1/sys.x(1);
Fo2 = sys.u(4)/l2; % *contact2/l2;

x_hip_dot = sys.x(3) + sys.d*cos(sys.x(5))*sys.x(6);
z_hip_dot = sys.x(4) + sys.d*sin(sys.x(5))*sys.x(6);

zero_dyn = [0; 0; 0; -sys.g; 0; 0];
state_act_dyn = [x_hip_dot*ca1 + z_hip_dot*sa1;
                 (-x_hip_dot*sa1 + z_hip_dot*ca1)/sys.x(1);
                 (F1*ca1 + Fo1*sa1 + F2*ca2 + Fo2*sa2)/sys.m;
                 ((F1*sa1 - Fo1*ca1 + F2*sa2 - Fo2*ca2)/sys.m);
                 sys.x(6);
                 (Fo1*(sys.x(1) + sys.d*sin(sys.x(2) - sys.x(5))) ...
                  + Fo2*(l2 + sys.d*sin(a2 - sys.x(5))) ...
                  + F1*(sys.d*cos(sys.x(2) - sys.x(5))) ...
                  + F2*(sys.d*cos(a2 - sys.x(5))))/sys.I];

sys.f = zero_dyn + state_act_dyn;
sys.fxu = jacobian(sys.f, [sys.x; sys.u]);
sys.fx = sys.fxu(:, 1:sys.X_DIMS);
sys.fu = sys.fxu(:, (sys.X_DIMS+1):end);

% Create Dynamics Files
% mkdir(sys.name);
% matlabFunction(simplify(sys.f), 'File', strcat(sys.name, '/dyn'));
% matlabFunction(simplify(sys.fx), 'File', strcat(sys.name, '/dynx'));
% matlabFunction(simplify(sys.fu), 'File', strcat(sys.name, '/dynu'));
