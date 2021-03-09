function out = dynx(sys, x, u)
%DYNX
%    OUT = DYNX(SYS, X, U)

%    This function was generated by the Symbolic Math Toolbox version 8.3.
%    21-Sep-2020 10:14:12

m1 = sys.m(1);
m2 = sys.m(2);
l1 = sys.l(1);
l2 = sys.l(2);
g = sys.g;

th1 = x(1,:);
th2 = x(2,:);
dth1 = x(3,:);
dth2 = x(4,:);

u1 = u(1,:);
u2 = u(2,:);

t2 = cos(th1);
t3 = cos(th2);
t4 = sin(th1);
t5 = sin(th2);
t6 = dth1+dth2;
t7 = dth1.^2;
t8 = dth2.^2;
t9 = l1.^2;
t10 = l1.^3;
t11 = l2.^2;
t12 = l2.^3;
t13 = m1.*4.0;
t14 = m2.^2;
t15 = th2.*2.0;
t19 = 1.0./l1;
t21 = 1.0./l2;
t22 = m1.*8.0;
t23 = m2.*1.2e+1;
t24 = m2.*1.5e+1;
t16 = cos(t15);
t17 = t3.^2;
t18 = sin(t15);
t20 = 1.0./t9;
t25 = t15+th1;
t26 = cos(t25);
t27 = m2.*t16.*9.0;
t28 = m2.*t17.*9.0;
t29 = -t27;
t30 = -t28;
t31 = t13+t23+t30;
t32 = t22+t24+t29;
t33 = 1.0./t31;
t35 = 1.0./t32;
t34 = t33.^2;

out = zeros(size(x,1), size(x,1), size(x,2));
out(3,1,:) = g.*t19.*t35.*(m2.*t2.*5.0-m2.*t26.*3.0+t2.*t13).*-3.0;
out(4,1,:) = g.*t19.*t21.*t33.*(l2.*m1.*t2.*2.0+l2.*m2.*t2.*4.0+l1.*m1.*t2.*t3+l1.*m1.*t4.*t5.*2.0+l1.*m2.*t4.*t5.*6.0-l2.*m2.*t2.*t17.*3.0+l2.*m2.*t3.*t4.*t5.*3.0).*3.0;

out(3,2,:) = t19.*t21.*t35.*(t5.*u2.*6.0+g.*l2.*m2.*t26.*3.0+m2.*t3.*t7.*t11.*2.0+m2.*t3.*t8.*t11.*2.0+dth1.*dth2.*m2.*t3.*t11.*4.0+l1.*l2.*m2.*t7.*t16.*3.0).*6.0-m2.*t3.*t5.*t20.*t21.*t34.*(l2.*u1.*4.0-l2.*u2.*4.0-l1.*t3.*u2.*6.0-g.*l1.*l2.*m1.*t4.*2.0-g.*l1.*l2.*m2.*t4.*(5.0./2.0)+l1.*m2.*t5.*t7.*t11.*2.0+l1.*m2.*t5.*t8.*t11.*2.0+l2.*m2.*t7.*t9.*t18.*(3.0./2.0)+g.*l1.*l2.*m2.*sin(t25).*(3.0./2.0)+dth1.*dth2.*l1.*m2.*t5.*t11.*4.0).*5.4e+1;
out(4,2,:) = -t19.*t21.*t35.*(t5.*u1.*-3.6e+1+t5.*u2.*7.2e+1+m1.*t3.*t7.*t9.*1.2e+1+m2.*t3.*t7.*t9.*3.6e+1+t3.*t7.*t11.*t23+t3.*t8.*t11.*t23+dth1.*dth2.*m2.*t3.*t11.*2.4e+1+g.*l1.*m1.*t2.*t3.*1.2e+1+g.*l1.*m2.*t2.*t3.*3.6e+1+g.*l1.*m1.*t4.*t5.*6.0+g.*l2.*m2.*t2.*t16.*1.8e+1-g.*l2.*m2.*t4.*t18.*1.8e+1+l1.*l2.*m2.*t7.*t16.*3.6e+1+l1.*l2.*m2.*t8.*t16.*1.8e+1+dth1.*dth2.*l1.*l2.*m2.*t16.*3.6e+1)+(t3.*t5.*t20.*t34.*(m1.*t9.*u2.*-4.0-m2.*t9.*u2.*1.2e+1+m2.*t11.*u1.*4.0-m2.*t11.*u2.*4.0+l1.*l2.*m2.*t3.*u1.*6.0-l1.*l2.*m2.*t3.*u2.*1.2e+1-g.*l1.*t4.*t11.*t14.*(5.0./2.0)+l2.*t5.*t7.*t10.*t14.*6.0+l1.*t5.*t7.*t12.*t14.*2.0+l1.*t5.*t8.*t12.*t14.*2.0+t7.*t9.*t11.*t14.*t18.*3.0+t8.*t9.*t11.*t14.*t18.*(3.0./2.0)+l2.*m1.*m2.*t5.*t7.*t10.*2.0+g.*l2.*t2.*t5.*t9.*t14.*6.0+g.*l1.*t2.*t11.*t14.*t18.*(3.0./2.0)+g.*l1.*t4.*t11.*t14.*t16.*(3.0./2.0)+dth1.*dth2.*l1.*t5.*t12.*t14.*4.0-g.*l1.*m1.*m2.*t4.*t11.*2.0+dth1.*dth2.*t9.*t11.*t14.*t18.*3.0+g.*l2.*m1.*m2.*t2.*t5.*t9.*2.0-g.*l2.*m1.*m2.*t3.*t4.*t9).*5.4e+1)./t11;

out(1,3,:) = ones(1,1,size(x,2));
out(3,3,:) = m2.*t19.*t33.*(dth1.*l2.*t5.*4.0+dth2.*l2.*t5.*4.0+dth1.*l1.*t3.*t5.*6.0).*3.0;
out(4,3,:) = -t19.*t21.*t35.*(dth1.*m1.*t5.*t9.*2.4e+1+dth1.*m2.*t5.*t9.*7.2e+1+dth1.*m2.*t5.*t11.*2.4e+1+dth2.*m2.*t5.*t11.*2.4e+1+dth1.*l1.*l2.*m2.*t18.*3.6e+1+dth2.*l1.*l2.*m2.*t18.*1.8e+1);

out(2,4,:) = ones(1,1,size(x,2));
out(3,4,:) = (l2.*t5.*t6.*t19.*t23)./(m2.*3.0+t13+m2.*t5.^2.*9.0);
out(4,4,:) = m2.*t6.*t19.*t33.*(l2.*t5.*2.0+l1.*t3.*t5.*3.0).*-6.0;

end