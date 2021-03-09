/*
 * Example of how to use the mxGPUArray API in a MEX file.  This example shows
 * how to write a MEX function that takes a gpuArray input and returns a
 * gpuArray output, e.g. B=mexFunction(A).
 *
 * Copyright 2012 The MathWorks, Inc.
 */

#include "mex.h"
#include "matrix.h"
#include "gpu/mxGPUArray.h"
#include <iostream>

 /*
  * Device code
  */

__global__ void dyn4_mex_continuous(double* const d_dx1, // outputs
    double* const x1, double* const x2, double* const x3,
    double* const dx1, double* const dx2, double* const dx3, // input states
    double const* const in1, double const* const in2, double const* const in3, // input actions
    const double m1, const double m2, const double m3,
    const double l1, const double l2, const double l3, const double g, 
    int32_t const* const grid_size, int32_t const* const active_actions)
{
    int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x * blockDim.y
        + (threadIdx.x + threadIdx.y * blockDim.x);
    int num_threads = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
    int num_elements = grid_size[0] * grid_size[1] * grid_size[2] * grid_size[3]
                        * grid_size[4] * grid_size[5];
    double th1, th2, th3, dth1, dth2, dth3, u1, u2, u3;

    int index1 = (grid_size[0] > 1) ? index : 0;
    int index2 = (grid_size[1] > 1) ? index : 0;
    int index3 = (grid_size[2] > 1) ? index : 0;
    int index4 = (grid_size[3] > 1) ? index : 0;
    int index5 = (grid_size[4] > 1) ? index : 0;
    int index6 = (grid_size[5] > 1) ? index : 0;

    int uindex1 = (active_actions[0] == 1) ? index : 0;
    int uindex2 = (active_actions[1] == 1) ? index : 0;
    int uindex3 = (active_actions[2] == 1) ? index : 0;

    while (index < num_elements)
    {
        th1 = x1[index1];
        th2 = x2[index2];
        th3 = x3[index3];

        dth1 = dx1[index4];
        dth2 = dx2[index5];
        dth3 = dx3[index6];

        u1 = in1[uindex1];
        u2 = in2[uindex2];
        u3 = in3[uindex3];

        double t2 = cos(th2);
        double t3 = cos(th3);
        double t4 = sin(th1);
        double t5 = sin(th2);
        double t6 = sin(th3);
        double t7 = th1+th2;
        double t8 = th2+th3;
        double t9 = dth1 * dth1;
        double t10 = dth2 * dth2;
        double t11 = dth3 * dth3;
        double t12 = l1 * l1;
//         double t13 = l1 * t12;
        double t14 = l2 * l2;
//         double t15 = l2 * t14;
        double t16 = l3 * l3;
//         double t17 = l3 * t16;
        double t18 = m2 * m2;
        double t19 = m3 * m3;
//         double t20 = m3 * t19;
        double t21 = th2*2.0;
        double t22 = th3*2.0;
        double t33 = 1.0 / l3;
        double t34 = -th1;
        double t35 = -th2;
        double t36 = -th3;
        double t39 = m1*m2*1.6*10.0;
        double t40 = m1*m3*3.0*10.0;
        double t54 = m2*m3*7.5*10.0;
        double t23 = cos(t21);
        double t24 = cos(t22);
        double t25 = sin(t21);
        double t26 = sin(t22);
        double t27 = cos(t8);
        double t28 = sin(t7);
        double t29 = sin(t8);
        double t30 = t7+th3;
        double t31 = 1.0 / t12;
//         double t32 = 1.0 / t14;
        double t37 = -t22;
        double t41 = t7+th2;
        double t42 = t22+th1;
        double t43 = t8+th3;
        double t44 = t8+th2;
        double t51 = t7+t22;
        double t52 = t19*1.8*10.0;
        double t53 = t18*3.0*10.0;
        double t55 = t35+th1;
        double t57 = t36+th2;
        double t58 = t8*2.0;
        double t63 = t7+t36;
        double t65 = t8+t34;
        double t38 = sin(t30);
        double t45 = cos(t43);
        double t46 = cos(t44);
        double t47 = sin(t41);
        double t48 = sin(t42);
        double t49 = sin(t43);
        double t50 = sin(t44);
        double t56 = t37+th1;
        double t59 = cos(t57);
        double t60 = sin(t55);
        double t62 = sin(t57);
        double t64 = t55+th3;
        double t66 = sin(t51);
        double t67 = cos(t58);
        double t68 = sin(t58);
        double t69 = t8+t30;
        double t70 = sin(t63);
        double t72 = sin(t65);
        double t73 = m1*m3*t24*1.8*10.0;
        double t74 = m2*m3*t24*2.7*10.0;
        double t75 = m2*m3*t23*4.5*10.0;
        double t76 = t34+t43;
        double t78 = t18*t23*1.8*10.0;
//         double t79 = t23*t52;
        double t85 = -t19*t23*1.8*10.0;
        double t61 = sin(t56);
        double t71 = sin(t64);
        double t77 = sin(t69);
        double t80 = -t73;
        double t81 = -t74;
        double t82 = -t75;
        double t83 = sin(t76);
        double t84 = -t78;
        double t86 = m2*m3*t67*9.0;
        double t87 = t39+t40+t52+t53+t54+t80+t81+t82+t84+t85+t86;
        double t88 = 1.0 / t87;

        d_dx1[index] = (t31*t33*t88*(l2*l3*m2*u1*8.0-l2*l3*m2*u2*8.0+l2*l3*m3*u1*1.5*10.0-l2*l3*m3*u2*1.5*10.0-l1*l3*m2*t2*u2*1.2*10.0+l1*l3*m2*t2*u3*1.2*10.0-l1*l3*m3*t2*u2*1.5*10.0+l1*l3*m3*t2*u3*1.5*10.0-l2*l3*m3*t24*u1*9.0+l2*l3*m3*t24*u2*9.0-l1*l2*m2*t27*u3*3.0-l1*l2*m3*t27*u3*1.8*10.0+l1*l3*m3*t45*u2*9.0-l1*l3*m3*t45*u3*9.0+l1*l2*m2*t59*u3*9.0+l1*l2*m3*t59*u3*1.8*10.0+l1*l3*t5*t9*t14*t18*4.0+l1*l3*t5*t9*t14*t19*6.0+l1*l3*t5*t10*t14*t18*4.0+l1*l3*t5*t10*t14*t19*6.0+l2*l3*t9*t12*t18*t25*3.0+l2*l3*t9*t12*t19*t25*3.0+l1*l2*t9*t16*t19*t29*(3.0/2.0)+l1*l2*t10*t16*t19*t29*(3.0/2.0)+l1*l2*t11*t16*t19*t29*(3.0/2.0)+l1*l2*t9*t16*t19*t62*(3.0/2.0)+l1*l2*t10*t16*t19*t62*(3.0/2.0)+l1*l2*t11*t16*t19*t62*(3.0/2.0)-g*l1*l2*l3*t4*t18*5.0-g*l1*l2*l3*t4*t19*3.0+g*l1*l2*l3*t18*t47*3.0+g*l1*l2*l3*t19*t47*3.0-g*l1*l2*l3*m1*m2*t4*4.0-g*l1*l2*l3*m1*m3*t4*(1.5*10.0/2.0)-g*l1*l2*l3*m2*m3*t4*(2.5*10.0/2.0)+g*l1*l2*l3*m1*m3*t48*(9.0/4.0)+g*l1*l2*l3*m2*m3*t47*(1.5*10.0/2.0)+g*l1*l2*l3*m2*m3*t48*(9.0/4.0)+g*l1*l2*l3*m1*m3*t61*(9.0/4.0)+g*l1*l2*l3*m2*m3*t61*(9.0/4.0)-g*l1*l2*l3*m2*m3*t77*(3.0/2.0)+dth1*dth2*l1*l3*t5*t14*t18*8.0+dth1*dth2*l1*l3*t5*t14*t19*1.2*10.0+dth1*dth2*l1*l2*t16*t19*t29*3.0+dth1*dth3*l1*l2*t16*t19*t29*3.0+dth2*dth3*l1*l2*t16*t19*t29*3.0+dth1*dth2*l1*l2*t16*t19*t62*3.0+dth1*dth3*l1*l2*t16*t19*t62*3.0+dth2*dth3*l1*l2*t16*t19*t62*3.0+l1*l3*m2*m3*t5*t9*t14*(2.5*10.0/2.0)+l1*l3*m2*m3*t5*t10*t14*(2.5*10.0/2.0)+l2*l3*m2*m3*t9*t12*t25*(1.5*10.0/2.0)+l1*l2*m2*m3*t9*t16*t29+l1*l2*m2*m3*t10*t16*t29+l1*l2*m2*m3*t11*t16*t29-l1*l3*m2*m3*t9*t14*t49*(3.0/2.0)-l1*l3*m2*m3*t10*t14*t49*(3.0/2.0)+l1*l2*m2*m3*t9*t16*t62*3.0+l1*l2*m2*m3*t10*t16*t62*3.0+l1*l2*m2*m3*t11*t16*t62*3.0-l2*l3*m2*m3*t9*t12*t68*(3.0/2.0)+dth1*dth2*l1*l3*m2*m3*t5*t14*2.5*10.0+dth1*dth2*l1*l2*m2*m3*t16*t29*2.0+dth1*dth3*l1*l2*m2*m3*t16*t29*2.0+dth2*dth3*l1*l2*m2*m3*t16*t29*2.0-dth1*dth2*l1*l3*m2*m3*t14*t49*3.0+dth1*dth2*l1*l2*m2*m3*t16*t62*6.0+dth1*dth3*l1*l2*m2*m3*t16*t62*6.0+dth2*dth3*l1*l2*m2*m3*t16*t62*6.0)*6.0)/l2;

        index = index + num_threads;

        index1 = (grid_size[0] > 1) ? index : 0;
        index2 = (grid_size[1] > 1) ? index : 0;
        index3 = (grid_size[2] > 1) ? index : 0;
        index4 = (grid_size[3] > 1) ? index : 0;
        index5 = (grid_size[4] > 1) ? index : 0;
        index6 = (grid_size[5] > 1) ? index : 0;

        uindex1 = (active_actions[0] == 1) ? index : 0;
        uindex2 = (active_actions[1] == 1) ? index : 0;
        uindex3 = (active_actions[2] == 1) ? index : 0;
    }
}

__global__ void dyn5_mex_continuous(double* const d_dx2, // outputs
    double* const x1, double* const x2, double* const x3,
    double* const dx1, double* const dx2, double* const dx3, // input states
    double const* const in1, double const* const in2, double const* const in3, // input actions
    const double m1, const double m2, const double m3,
    const double l1, const double l2, const double l3, const double g, 
    int32_t const* const grid_size, int32_t const* const active_actions)
{
    int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x * blockDim.y
        + (threadIdx.x + threadIdx.y * blockDim.x);
    int num_threads = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
    int num_elements = grid_size[0] * grid_size[1] * grid_size[2] * grid_size[3]
                        * grid_size[4] * grid_size[5];
    double th1, th2, th3, dth1, dth2, dth3, u1, u2, u3;

    int index1 = (grid_size[0] > 1) ? index : 0;
    int index2 = (grid_size[1] > 1) ? index : 0;
    int index3 = (grid_size[2] > 1) ? index : 0;
    int index4 = (grid_size[3] > 1) ? index : 0;
    int index5 = (grid_size[4] > 1) ? index : 0;
    int index6 = (grid_size[5] > 1) ? index : 0;

    int uindex1 = (active_actions[0] == 1) ? index : 0;
    int uindex2 = (active_actions[1] == 1) ? index : 0;
    int uindex3 = (active_actions[2] == 1) ? index : 0;

    while (index < num_elements)
    {
        th1 = x1[index1];
        th2 = x2[index2];
        th3 = x3[index3];

        dth1 = dx1[index4];
        dth2 = dx2[index5];
        dth3 = dx3[index6];

        u1 = in1[uindex1];
        u2 = in2[uindex2];
        u3 = in3[uindex3];

        double t2 = cos(th2);
        double t3 = cos(th3);
        double t4 = sin(th1);
        double t5 = sin(th2);
        double t6 = sin(th3);
        double t7 = th1+th2;
        double t8 = th2+th3;
        double t9 = dth1 * dth1;
        double t10 = dth2 * dth2;
        double t11 = dth3 * dth3;
        double t12 = l1 * l1;
        double t13 = l1 * t12;
        double t14 = l2 * l2;
        double t15 = l2 * t14;
        double t16 = l3 * l3;
//         double t17 = l3 * t16;
        double t18 = m2 * m2;
        double t19 = m3 * m3;
//         double t20 = m3 * t19;
        double t21 = th2*2.0;
        double t22 = th3*2.0;
        double t33 = 1.0 / l3;
        double t34 = -th1;
        double t35 = -th2;
        double t36 = -th3;
        double t39 = m1*m2*1.6*10.0;
        double t40 = m1*m3*3.0*10.0;
        double t54 = m2*m3*7.5*10.0;
        double t23 = cos(t21);
        double t24 = cos(t22);
        double t25 = sin(t21);
        double t26 = sin(t22);
        double t27 = cos(t8);
        double t28 = sin(t7);
        double t29 = sin(t8);
        double t30 = t7+th3;
        double t31 = 1.0 / t12;
        double t32 = 1.0 / t14;
        double t37 = -t22;
        double t41 = t7+th2;
        double t42 = t22+th1;
        double t43 = t8+th3;
        double t44 = t8+th2;
        double t51 = t7+t22;
        double t52 = t19*1.8*10.0;
        double t53 = t18*3.0*10.0;
        double t55 = t35+th1;
        double t57 = t36+th2;
        double t58 = t8*2.0;
        double t63 = t7+t36;
        double t65 = t8+t34;
        double t38 = sin(t30);
        double t45 = cos(t43);
        double t46 = cos(t44);
        double t47 = sin(t41);
        double t48 = sin(t42);
        double t49 = sin(t43);
        double t50 = sin(t44);
        double t56 = t37+th1;
        double t59 = cos(t57);
        double t60 = sin(t55);
        double t62 = sin(t57);
        double t64 = t55+th3;
        double t66 = sin(t51);
        double t67 = cos(t58);
        double t68 = sin(t58);
        double t69 = t8+t30;
        double t70 = sin(t63);
        double t72 = sin(t65);
        double t73 = m1*m3*t24*1.8*10.0;
        double t74 = m2*m3*t24*2.7*10.0;
        double t75 = m2*m3*t23*4.5*10.0;
        double t76 = t34+t43;
        double t78 = t18*t23*1.8*10.0;
//         double t79 = t23*t52;
        double t85 = -t19*t23*1.8*10.0;
        double t61 = sin(t56);
        double t71 = sin(t64);
        double t77 = sin(t69);
        double t80 = -t73;
        double t81 = -t74;
        double t82 = -t75;
        double t83 = sin(t76);
        double t84 = -t78;
        double t86 = m2*m3*t67*9.0;
        double t87 = t39+t40+t52+t53+t54+t80+t81+t82+t84+t85+t86;
        double t88 = 1.0 / t87;

        d_dx2[index] = t31*t32*t33*t88*(l3*m1*t12*u2*(-8.0)+l3*m1*t12*u3*8.0-l3*m2*t12*u2*2.4*10.0+l3*m2*t12*u3*2.4*10.0+l3*m2*t14*u1*8.0-l3*m3*t12*u2*1.5*10.0-l3*m2*t14*u2*8.0+l3*m3*t12*u3*1.5*10.0+l3*m3*t14*u1*1.5*10.0-l3*m3*t14*u2*1.5*10.0+l2*m1*t3*t12*u3*1.2*10.0+l2*m2*t3*t12*u3*2.7*10.0+l2*m3*t3*t12*u3*1.8*10.0-l3*m3*t14*t24*u1*9.0+l3*m3*t14*t24*u2*9.0-l1*m2*t14*t27*u3*3.0-l1*m3*t14*t27*u3*1.8*10.0-l2*m2*t12*t46*u3*9.0-l2*m3*t12*t46*u3*1.8*10.0+l1*m2*t14*t59*u3*9.0+l1*m3*t14*t59*u3*1.8*10.0+l3*m3*t12*t67*u2*9.0-l3*m3*t12*t67*u3*9.0+l2*l3*t5*t9*t13*t18*1.2*10.0+l1*l3*t5*t9*t15*t18*4.0+l2*l3*t5*t9*t13*t19*6.0+l1*l3*t5*t9*t15*t19*6.0+l1*l3*t5*t10*t15*t18*4.0+l1*l3*t5*t10*t15*t19*6.0-l2*t6*t9*t12*t16*t19*(3.0/2.0)-l2*t6*t10*t12*t16*t19*(3.0/2.0)-l2*t6*t11*t12*t16*t19*(3.0/2.0)+l3*t9*t12*t14*t18*t25*6.0+l3*t9*t12*t14*t19*t25*6.0+l3*t10*t12*t14*t18*t25*3.0+l3*t10*t12*t14*t19*t25*3.0+l1*t9*t14*t16*t19*t29*(3.0/2.0)+l1*t10*t14*t16*t19*t29*(3.0/2.0)+l1*t11*t14*t16*t19*t29*(3.0/2.0)+l2*t9*t12*t16*t19*t50*(3.0/2.0)+l2*t10*t12*t16*t19*t50*(3.0/2.0)+l2*t11*t12*t16*t19*t50*(3.0/2.0)+l1*t9*t14*t16*t19*t62*(3.0/2.0)+l1*t10*t14*t16*t19*t62*(3.0/2.0)+l1*t11*t14*t16*t19*t62*(3.0/2.0)+l1*l2*l3*m2*t2*u1*1.2*10.0-l1*l2*l3*m2*t2*u2*2.4*10.0+l1*l2*l3*m3*t2*u1*1.5*10.0+l1*l2*l3*m2*t2*u3*1.2*10.0-l1*l2*l3*m3*t2*u2*3.0*10.0+l1*l2*l3*m3*t2*u3*1.5*10.0-l1*l2*l3*m3*t45*u1*9.0+l1*l2*l3*m3*t45*u2*1.8*10.0-l1*l2*l3*m3*t45*u3*9.0-g*l1*l3*t4*t14*t18*5.0-g*l1*l3*t4*t14*t19*3.0+g*l2*l3*t12*t18*t28*6.0+g*l2*l3*t12*t19*t28*3.0+g*l1*l3*t14*t18*t47*3.0+g*l1*l3*t14*t19*t47*3.0-g*l2*l3*t12*t18*t60*6.0-g*l2*l3*t12*t19*t60*3.0+dth1*dth2*l1*l3*t5*t15*t18*8.0+dth1*dth2*l1*l3*t5*t15*t19*1.2*10.0-g*l1*l3*m1*m2*t4*t14*4.0-g*l1*l3*m1*m3*t4*t14*(1.5*10.0/2.0)-g*l1*l3*m2*m3*t4*t14*(2.5*10.0/2.0)+g*l2*l3*m1*m2*t12*t28+g*l2*l3*m1*m3*t12*t28*(5.0/4.0)+g*l2*l3*m2*m3*t12*t28*(4.5*10.0/4.0)+g*l1*l3*m1*m3*t14*t48*(9.0/4.0)+g*l1*l3*m2*m3*t14*t47*(1.5*10.0/2.0)+g*l1*l3*m2*m3*t14*t48*(9.0/4.0)-g*l2*l3*m1*m2*t12*t60*3.0-g*l2*l3*m1*m3*t12*t60*(1.5*10.0/4.0)-g*l2*l3*m2*m3*t12*t60*(4.5*10.0/4.0)+g*l1*l3*m1*m3*t14*t61*(9.0/4.0)+g*l1*l3*m2*m3*t14*t61*(9.0/4.0)-g*l2*l3*m1*m3*t12*t66*(3.0/4.0)-g*l2*l3*m2*m3*t12*t66*(9.0/4.0)-g*l1*l3*m2*m3*t14*t77*(3.0/2.0)-g*l2*l3*m1*m3*t12*t83*(9.0/4.0)-g*l2*l3*m2*m3*t12*t83*(9.0/4.0)-dth1*dth2*l2*t6*t12*t16*t19*3.0-dth1*dth3*l2*t6*t12*t16*t19*3.0-dth2*dth3*l2*t6*t12*t16*t19*3.0+dth1*dth2*l3*t12*t14*t18*t25*6.0+dth1*dth2*l3*t12*t14*t19*t25*6.0+dth1*dth2*l1*t14*t16*t19*t29*3.0+dth1*dth3*l1*t14*t16*t19*t29*3.0+dth2*dth3*l1*t14*t16*t19*t29*3.0+dth1*dth2*l2*t12*t16*t19*t50*3.0+dth1*dth3*l2*t12*t16*t19*t50*3.0+dth2*dth3*l2*t12*t16*t19*t50*3.0+dth1*dth2*l1*t14*t16*t19*t62*3.0+dth1*dth3*l1*t14*t16*t19*t62*3.0+dth2*dth3*l1*t14*t16*t19*t62*3.0+l2*l3*m1*m2*t5*t9*t13*4.0+l2*l3*m1*m3*t5*t9*t13*5.0+l2*l3*m2*m3*t5*t9*t13*(4.5*10.0/2.0)+l1*l3*m2*m3*t5*t9*t15*(2.5*10.0/2.0)+l1*l3*m2*m3*t5*t10*t15*(2.5*10.0/2.0)-l2*l3*m1*m3*t9*t13*t49*3.0-l2*l3*m2*m3*t9*t13*t49*(9.0/2.0)-l1*l3*m2*m3*t9*t15*t49*(3.0/2.0)-l1*l3*m2*m3*t10*t15*t49*(3.0/2.0)-l2*m1*m3*t6*t9*t12*t16*4.0-l2*m1*m3*t6*t10*t12*t16*4.0-l2*m2*m3*t6*t9*t12*t16*9.0-l2*m1*m3*t6*t11*t12*t16*4.0-l2*m2*m3*t6*t10*t12*t16*9.0-l2*m2*m3*t6*t11*t12*t16*9.0-l3*m1*m3*t9*t12*t14*t26*3.0+l3*m2*m3*t9*t12*t14*t25*1.5*10.0-l3*m1*m3*t10*t12*t14*t26*3.0-l3*m2*m3*t9*t12*t14*t26*(9.0/2.0)+l3*m2*m3*t10*t12*t14*t25*(1.5*10.0/2.0)-l3*m2*m3*t10*t12*t14*t26*(9.0/2.0)+l1*m2*m3*t9*t14*t16*t29+l1*m2*m3*t10*t14*t16*t29+l1*m2*m3*t11*t14*t16*t29+l2*m2*m3*t9*t12*t16*t50*3.0+l2*m2*m3*t10*t12*t16*t50*3.0+l2*m2*m3*t11*t12*t16*t50*3.0+l1*m2*m3*t9*t14*t16*t62*3.0+l1*m2*m3*t10*t14*t16*t62*3.0+l1*m2*m3*t11*t14*t16*t62*3.0-l3*m2*m3*t9*t12*t14*t68*(3.0/2.0)+dth1*dth2*l1*l3*m2*m3*t5*t15*2.5*10.0-dth1*dth2*l1*l3*m2*m3*t15*t49*3.0-dth1*dth2*l2*m1*m3*t6*t12*t16*8.0-dth1*dth2*l2*m2*m3*t6*t12*t16*1.8*10.0-dth1*dth3*l2*m1*m3*t6*t12*t16*8.0-dth1*dth3*l2*m2*m3*t6*t12*t16*1.8*10.0-dth2*dth3*l2*m1*m3*t6*t12*t16*8.0-dth2*dth3*l2*m2*m3*t6*t12*t16*1.8*10.0-dth1*dth2*l3*m1*m3*t12*t14*t26*6.0+dth1*dth2*l3*m2*m3*t12*t14*t25*1.5*10.0-dth1*dth2*l3*m2*m3*t12*t14*t26*9.0+dth1*dth2*l1*m2*m3*t14*t16*t29*2.0+dth1*dth3*l1*m2*m3*t14*t16*t29*2.0+dth2*dth3*l1*m2*m3*t14*t16*t29*2.0+dth1*dth2*l2*m2*m3*t12*t16*t50*6.0+dth1*dth3*l2*m2*m3*t12*t16*t50*6.0+dth2*dth3*l2*m2*m3*t12*t16*t50*6.0+dth1*dth2*l1*m2*m3*t14*t16*t62*6.0+dth1*dth3*l1*m2*m3*t14*t16*t62*6.0+dth2*dth3*l1*m2*m3*t14*t16*t62*6.0)*(-6.0);

        index = index + num_threads;

        index1 = (grid_size[0] > 1) ? index : 0;
        index2 = (grid_size[1] > 1) ? index : 0;
        index3 = (grid_size[2] > 1) ? index : 0;
        index4 = (grid_size[3] > 1) ? index : 0;
        index5 = (grid_size[4] > 1) ? index : 0;
        index6 = (grid_size[5] > 1) ? index : 0;

        uindex1 = (active_actions[0] == 1) ? index : 0;
        uindex2 = (active_actions[1] == 1) ? index : 0;
        uindex3 = (active_actions[2] == 1) ? index : 0;
    }
}

__global__ void dyn6_mex_continuous(double* const d_dx3, // outputs
    double* const x1, double* const x2, double* const x3,
    double* const dx1, double* const dx2, double* const dx3, // input states
    double const* const in1, double const* const in2, double const* const in3, // input actions
    const double m1, const double m2, const double m3,
    const double l1, const double l2, const double l3, const double g, 
    int32_t const* const grid_size, int32_t const* const active_actions)
{
    int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x * blockDim.y
        + (threadIdx.x + threadIdx.y * blockDim.x);
    int num_threads = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
    int num_elements = grid_size[0] * grid_size[1] * grid_size[2] * grid_size[3]
                        * grid_size[4] * grid_size[5];
    double th1, th2, th3, dth1, dth2, dth3, u1, u2, u3;

    int index1 = (grid_size[0] > 1) ? index : 0;
    int index2 = (grid_size[1] > 1) ? index : 0;
    int index3 = (grid_size[2] > 1) ? index : 0;
    int index4 = (grid_size[3] > 1) ? index : 0;
    int index5 = (grid_size[4] > 1) ? index : 0;
    int index6 = (grid_size[5] > 1) ? index : 0;

    int uindex1 = (active_actions[0] == 1) ? index : 0;
    int uindex2 = (active_actions[1] == 1) ? index : 0;
    int uindex3 = (active_actions[2] == 1) ? index : 0;
    
    while (index < num_elements)
    {
        th1 = x1[index1];
        th2 = x2[index2];
        th3 = x3[index3];

        dth1 = dx1[index4];
        dth2 = dx2[index5];
        dth3 = dx3[index6];

        u1 = in1[uindex1];
        u2 = in2[uindex2];
        u3 = in3[uindex3];

        double t2 = cos(th2);
        double t3 = cos(th3);
        double t4 = sin(th1);
        double t5 = sin(th2);
        double t6 = sin(th3);
        double t7 = th1+th2;
        double t8 = th2+th3;
        double t9 = dth1 * dth1;
        double t10 = dth2 * dth2;
        double t11 = dth3 * dth3;
        double t12 = l1 * l1;
//         double t13 = l1 * t12;
        double t14 = l2 * l2;
        double t15 = l2 * t14;
        double t16 = l3 * l3;
        double t17 = l3 * t16;
        double t18 = m2 * m2;
        double t19 = m3 * m3;
        double t20 = m3 * t19;
        double t21 = th2*2.0;
        double t22 = th3*2.0;
//         double t33 = 1.0 / l3;
        double t34 = -th1;
        double t35 = -th2;
        double t36 = -th3;
        double t39 = m1*m2*1.6*10.0;
        double t40 = m1*m3*3.0*10.0;
        double t54 = m2*m3*7.5*10.0;
        double t23 = cos(t21);
        double t24 = cos(t22);
        double t25 = sin(t21);
        double t26 = sin(t22);
        double t27 = cos(t8);
        double t28 = sin(t7);
        double t29 = sin(t8);
        double t30 = t7+th3;
//         double t31 = 1.0 / t12;
        double t32 = 1.0 / t14;
        double t37 = -t22;
        double t41 = t7+th2;
        double t42 = t22+th1;
        double t43 = t8+th3;
        double t44 = t8+th2;
        double t51 = t7+t22;
        double t52 = t19*1.8*10.0;
        double t53 = t18*3.0*10.0;
        double t55 = t35+th1;
        double t57 = t36+th2;
        double t58 = t8*2.0;
        double t63 = t7+t36;
        double t65 = t8+t34;
        double t38 = sin(t30);
        double t45 = cos(t43);
        double t46 = cos(t44);
        double t47 = sin(t41);
        double t48 = sin(t42);
        double t49 = sin(t43);
        double t50 = sin(t44);
        double t56 = t37+th1;
        double t59 = cos(t57);
        double t60 = sin(t55);
        double t62 = sin(t57);
        double t64 = t55+th3;
        double t66 = sin(t51);
        double t67 = cos(t58);
        double t68 = sin(t58);
        double t69 = t8+t30;
        double t70 = sin(t63);
        double t72 = sin(t65);
        double t73 = m1*m3*t24*1.8*10.0;
        double t74 = m2*m3*t24*2.7*10.0;
        double t75 = m2*m3*t23*4.5*10.0;
        double t76 = t34+t43;
        double t78 = t18*t23*1.8*10.0;
//         double t79 = t23*t52;
        double t85 = -t19*t23*1.8*10.0;
        double t61 = sin(t56);
        double t71 = sin(t64);
        double t77 = sin(t69);
        double t80 = -t73;
        double t81 = -t74;
        double t82 = -t75;
        double t83 = sin(t76);
        double t84 = -t78;
        double t86 = m2*m3*t67*9.0;
        double t87 = t39+t40+t52+t53+t54+t80+t81+t82+t84+t85+t86;
        double t88 = 1.0 / t87;

        d_dx3[index] = (t32*t88*(l1*t14*t18*u3*(-1.5)*10.0-l1*t14*t19*u3*3.6*10.0+l1*t16*t19*u2*1.5*10.0-l1*t16*t19*u3*1.5*10.0-l1*m1*m2*t14*u3*8.0-l1*m1*m3*t14*u3*2.4*10.0+l1*m1*m3*t16*u2*8.0-l1*m2*m3*t14*u3*6.0*10.0-l1*m1*m3*t16*u3*8.0+l1*m2*m3*t16*u2*2.4*10.0-l1*m2*m3*t16*u3*2.4*10.0-l2*t2*t16*t19*u1*1.5*10.0+l2*t2*t16*t19*u2*1.5*10.0+l1*t14*t18*t23*u3*9.0+l1*t14*t19*t23*u3*3.6*10.0-l3*t14*t19*t27*u2*1.8*10.0+l2*t16*t19*t45*u1*9.0-l2*t16*t19*t45*u2*9.0-l3*t14*t19*t59*u1*1.8*10.0+l3*t14*t27*t52*u1-l1*t16*t19*t67*u2*9.0+l1*t16*t19*t67*u3*9.0+l3*t14*t52*t59*u2-l1*l2*l3*t3*t19*u3*3.6*10.0+l1*l2*l3*t3*t52*u2-l1*l2*l3*t19*t46*u2*1.8*10.0+l1*l2*l3*t19*t46*u3*3.6*10.0-l2*m2*m3*t2*t16*u1*1.2*10.0+l2*m2*m3*t2*t16*u2*1.2*10.0+l1*m2*m3*t14*t23*u3*3.6*10.0+l3*m2*m3*t14*t27*u1*3.0-l3*m2*m3*t14*t27*u2*3.0-l3*m2*m3*t14*t59*u1*9.0+l3*m2*m3*t14*t59*u2*9.0+l1*l2*t6*t9*t17*t20*(3.0/2.0)+l1*l2*t6*t10*t17*t20*(3.0/2.0)+l1*l2*t6*t11*t17*t20*(3.0/2.0)-l1*l2*t9*t17*t20*t50*(3.0/2.0)-l1*l2*t10*t17*t20*t50*(3.0/2.0)-l1*l2*t11*t17*t20*t50*(3.0/2.0)-l2*t5*t9*t12*t16*t20*6.0-l1*t9*t14*t16*t20*t25*3.0-l1*t10*t14*t16*t20*t25*3.0-g*l1*l2*t16*t20*t28*3.0+g*l1*l2*t16*t20*t60*3.0+dth1*dth2*l1*l2*t6*t17*t20*3.0+dth1*dth3*l1*l2*t6*t17*t20*3.0+dth2*dth3*l1*l2*t6*t17*t20*3.0-dth1*dth2*l1*l2*t17*t20*t50*3.0-dth1*dth3*l1*l2*t17*t20*t50*3.0-dth2*dth3*l1*l2*t17*t20*t50*3.0-dth1*dth2*l1*t14*t16*t20*t25*6.0+l1*l2*l3*m1*m3*t3*u2*1.2*10.0-l1*l2*l3*m1*m3*t3*u3*2.4*10.0+l1*l2*l3*m2*m3*t3*u2*2.7*10.0-l1*l2*l3*m2*m3*t3*u3*5.4*10.0-l1*l2*l3*m2*m3*t46*u2*9.0+l1*l2*l3*m2*m3*t46*u3*1.8*10.0-g*l1*l2*m1*t16*t19*t28*(5.0/4.0)-g*l1*l2*m2*t16*t19*t28*(4.5*10.0/4.0)-g*l1*l2*m3*t16*t18*t28*6.0+g*l1*l3*m1*t14*t19*t38*(3.0/2.0)+g*l1*l3*m2*t14*t19*t38*(3.0/2.0)-g*l1*l3*m3*t14*t18*t38*(3.0/4.0)+g*l1*l2*m1*t16*t19*t60*(1.5*10.0/4.0)+g*l1*l2*m2*t16*t19*t60*(4.5*10.0/4.0)+g*l1*l2*m3*t16*t18*t60*6.0+g*l1*l2*m1*t16*t19*t66*(3.0/4.0)+g*l1*l2*m2*t16*t19*t66*(9.0/4.0)-g*l1*l3*m1*t14*t19*t70*(3.0/2.0)+g*l1*l3*m1*t14*t19*t71*(9.0/2.0)-g*l1*l3*m2*t14*t19*t70*(9.0/2.0)-g*l1*l3*m3*t14*t18*t70*(9.0/4.0)+g*l1*l3*m1*t14*t19*t72*(9.0/2.0)+g*l1*l3*m2*t14*t19*t71*(9.0/2.0)+g*l1*l3*m3*t14*t18*t71*(9.0/4.0)+g*l1*l3*m2*t14*t19*t72*(3.0/2.0)-g*l1*l3*m3*t14*t18*t72*(3.0/4.0)+g*l1*l2*m1*t16*t19*t83*(9.0/4.0)+g*l1*l2*m2*t16*t19*t83*(9.0/4.0)+l1*l3*m1*t6*t9*t15*t19*1.2*10.0+l1*l2*m1*t6*t9*t17*t19*4.0+l1*l3*m1*t6*t10*t15*t19*1.2*10.0+l1*l3*m2*t6*t9*t15*t19*1.5*10.0+l1*l3*m3*t6*t9*t15*t18*(9.0/2.0)+l1*l2*m1*t6*t10*t17*t19*4.0+l1*l2*m2*t6*t9*t17*t19*9.0+l1*l3*m2*t6*t10*t15*t19*1.5*10.0+l1*l3*m3*t6*t10*t15*t18*(9.0/2.0)+l1*l2*m1*t6*t11*t17*t19*4.0+l1*l2*m2*t6*t10*t17*t19*9.0+l1*l2*m2*t6*t11*t17*t19*9.0-l1*l3*m2*t9*t15*t19*t50*3.0-l1*l3*m3*t9*t15*t18*t50*(3.0/2.0)-l1*l2*m2*t9*t17*t19*t50*3.0-l1*l3*m2*t10*t15*t19*t50*3.0-l1*l3*m3*t10*t15*t18*t50*(3.0/2.0)-l1*l2*m2*t10*t17*t19*t50*3.0-l1*l2*m2*t11*t17*t19*t50*3.0-l2*m1*t5*t9*t12*t16*t19*5.0-l2*m2*t5*t9*t12*t16*t19*(4.5*10.0/2.0)-l2*m3*t5*t9*t12*t16*t18*1.2*10.0+l1*m1*t9*t14*t16*t19*t26*6.0-l1*m2*t9*t14*t16*t19*t25*(1.5*10.0/2.0)-l1*m3*t9*t14*t16*t18*t25*3.0+l1*m1*t10*t14*t16*t19*t26*6.0+l1*m2*t9*t14*t16*t19*t26*9.0-l1*m2*t10*t14*t16*t19*t25*(1.5*10.0/2.0)-l1*m3*t10*t14*t16*t18*t25*3.0+l3*m1*t9*t12*t14*t19*t29*6.0+l1*m1*t11*t14*t16*t19*t26*3.0+l1*m2*t10*t14*t16*t19*t26*9.0+l3*m2*t9*t12*t14*t19*t29*3.0-l3*m3*t9*t12*t14*t18*t29*(3.0/2.0)+l1*m2*t11*t14*t16*t19*t26*(9.0/2.0)+l2*m1*t9*t12*t16*t19*t49*3.0+l2*m2*t9*t12*t16*t19*t49*(9.0/2.0)-l3*m1*t9*t12*t14*t19*t62*6.0-l3*m2*t9*t12*t14*t19*t62*9.0-l3*m3*t9*t12*t14*t18*t62*(9.0/2.0)-l1*m2*t9*t14*t16*t19*t68*(3.0/2.0)-l1*m2*t10*t14*t16*t19*t68*(3.0/2.0)-l1*m2*t11*t14*t16*t19*t68*(3.0/2.0)+dth1*dth2*l1*l3*m1*t6*t15*t19*2.4*10.0+dth1*dth2*l1*l2*m1*t6*t17*t19*8.0+dth1*dth2*l1*l3*m2*t6*t15*t19*3.0*10.0+dth1*dth2*l1*l3*m3*t6*t15*t18*9.0+dth1*dth3*l1*l2*m1*t6*t17*t19*8.0+dth2*dth3*l1*l2*m1*t6*t17*t19*8.0+dth1*dth2*l1*l2*m2*t6*t17*t52+dth1*dth3*l1*l2*m2*t6*t17*t52+dth2*dth3*l1*l2*m2*t6*t17*t52-dth1*dth2*l1*l3*m2*t15*t19*t50*6.0-dth1*dth2*l1*l3*m3*t15*t18*t50*3.0-dth1*dth2*l1*l2*m2*t17*t19*t50*6.0-dth1*dth3*l1*l2*m2*t17*t19*t50*6.0-dth2*dth3*l1*l2*m2*t17*t19*t50*6.0-g*l1*l2*m1*m2*m3*t16*t28+(g*l1*l3*m1*m2*m3*t14*t38)/4.0+g*l1*l2*m1*m2*m3*t16*t60*3.0-g*l1*l3*m1*m2*m3*t14*t70*(3.0/4.0)+g*l1*l3*m1*m2*m3*t14*t71*(9.0/4.0)+g*l1*l3*m1*m2*m3*t14*t72*(3.0/4.0)+dth1*dth2*l1*m1*t14*t16*t19*t26*1.2*10.0-dth1*dth2*l1*m2*t14*t16*t19*t25*1.5*10.0-dth1*dth2*l1*m3*t14*t16*t18*t25*6.0+dth1*dth3*l1*m1*t14*t16*t19*t26*6.0+dth1*dth3*l1*m2*t14*t16*t19*t26*9.0+dth2*dth3*l1*m1*t14*t16*t19*t26*6.0+dth2*dth3*l1*m2*t14*t16*t19*t26*9.0+dth1*dth2*l1*m2*t14*t16*t26*t52-dth1*dth2*l1*m2*t14*t16*t19*t68*3.0-dth1*dth3*l1*m2*t14*t16*t19*t68*3.0-dth2*dth3*l1*m2*t14*t16*t19*t68*3.0+l1*l3*m1*m2*m3*t6*t9*t15*4.0+l1*l3*m1*m2*m3*t6*t10*t15*4.0-l2*m1*m2*m3*t5*t9*t12*t16*4.0+l3*m1*m2*m3*t9*t12*t14*t29-l3*m1*m2*m3*t9*t12*t14*t62*3.0+dth1*dth2*l1*l3*m1*m2*m3*t6*t15*8.0)*(-6.0))/(l1*m3*t16);

        index = index + num_threads;

        index1 = (grid_size[0] > 1) ? index : 0;
        index2 = (grid_size[1] > 1) ? index : 0;
        index3 = (grid_size[2] > 1) ? index : 0;
        index4 = (grid_size[3] > 1) ? index : 0;
        index5 = (grid_size[4] > 1) ? index : 0;
        index6 = (grid_size[5] > 1) ? index : 0;

        uindex1 = (active_actions[0] == 1) ? index : 0;
        uindex2 = (active_actions[1] == 1) ? index : 0;
        uindex3 = (active_actions[2] == 1) ? index : 0;
    }
}

__global__ void step(double* const x1n, // outputs
                     double* const x1, // current states
                     double* const d_x1, // current derivatives
                     const double delta, int32_t const* const grid_size)
{
    int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x * blockDim.y
        + (threadIdx.x + threadIdx.y * blockDim.x);
    int num_threads = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
    int num_elements = grid_size[0] * grid_size[1] * grid_size[2] * grid_size[3]
                        * grid_size[4] * grid_size[5];
    
    while (index < num_elements)
    {
        x1n[index] = x1[index] + delta * d_x1[index];
        index = index + num_threads;
    }
}

__global__ void rk4_wbounds(double* const x1n, // outputs
                    double* const x1, // current states
                    double* const k1_x1, // K1
                    double* const k2_x1, // K2
                    double* const k3_x1, // K3
                    double* const k4_x1, // K4
                    double dt, double lower_limit, double upper_limit, int32_t const* const grid_size)
{
    int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x * blockDim.y
        + (threadIdx.x + threadIdx.y * blockDim.x);
    int num_threads = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
    int num_elements = grid_size[0] * grid_size[1] * grid_size[2] * grid_size[3]
                        * grid_size[4] * grid_size[5];
    while (index < num_elements)
    {
        x1n[index] = x1[index] + dt / 6.0 * (k1_x1[index] + 2.0 * k2_x1[index] + 2.0 * k3_x1[index] + k4_x1[index]);        
        x1n[index] = max(lower_limit, min(upper_limit, x1n[index]));
        index = index + num_threads;
    }
}
/*
 * Host code
 */
void mexFunction(int nlhs, mxArray* plhs[],
    int nrhs, mxArray const* prhs[])
{
    /* Declare all variables*/
    // Inputs
    // GPU Arrays
    mxGPUArray const* X1;
    mxGPUArray const* X2;
    mxGPUArray const* X3;
    mxGPUArray const* dX1;
    mxGPUArray const* dX2;
    mxGPUArray const* dX3;
    mxGPUArray const* in1;
    mxGPUArray const* in2;
    mxGPUArray const* in3;
    mxGPUArray const* grid_size;
    mxGPUArray const* active_actions;
    // Pointers for GPU Arrays
    double* p_X1; 
    double* p_X2;
    double* p_X3; 
    double* p_dX1;
    double* p_dX2;
    double* p_dX3;
    double const* p_in1; 
    double const* p_in2;
    double const* p_in3; 
    int32_t const* p_grid_size;
    int32_t const* p_active_actions;
    // Pointers for normal inputs
    double* p_m;
    double* p_l;
    double* p_g;
    double* p_dt;
    double* p_limits;
    double* p_x_dims_free;
    
    // Intermediate variables
    mxGPUArray* k1_dX1;
    mxGPUArray* k1_dX2;
    mxGPUArray* k1_dX3;
    mxGPUArray* k2_X1;
    mxGPUArray* k2_X2;
    mxGPUArray* k2_X3;
    mxGPUArray* k2_dX1;
    mxGPUArray* k2_dX2;
    mxGPUArray* k2_dX3;
    mxGPUArray* k3_X1;
    mxGPUArray* k3_X2;
    mxGPUArray* k3_X3;
    mxGPUArray* k3_dX1;
    mxGPUArray* k3_dX2;
    mxGPUArray* k3_dX3;
    mxGPUArray* k4_X1;
    mxGPUArray* k4_X2;
    mxGPUArray* k4_X3;
    mxGPUArray* k4_dX1;
    mxGPUArray* k4_dX2;
    mxGPUArray* k4_dX3;
    // Pointers for intermediate variables
    double* p_k1_X1;
    double* p_k1_X2;
    double* p_k1_X3;
    double* p_k1_dX1;
    double* p_k1_dX2;
    double* p_k1_dX3;
    double* p_k2_X1;
    double* p_k2_X2;
    double* p_k2_X3;
    double* p_k2_dX1;
    double* p_k2_dX2;
    double* p_k2_dX3;
    double* p_k3_X1;
    double* p_k3_X2;
    double* p_k3_X3;
    double* p_k3_dX1;
    double* p_k3_dX2;
    double* p_k3_dX3;
    double* p_k4_X1;
    double* p_k4_X2;
    double* p_k4_X3;
    double* p_k4_dX1;
    double* p_k4_dX2;
    double* p_k4_dX3;

    // Outputs
    mxGPUArray* X1n;
    mxGPUArray* X2n;
    mxGPUArray* X3n;
    mxGPUArray* dX1n;
    mxGPUArray* dX2n;
    mxGPUArray* dX3n;
    // Pointers for outputs
    double* p_X1n;
    double* p_X2n;
    double* p_X3n;
    double* p_dX1n;
    double* p_dX2n;
    double* p_dX3n;

    char const* const errId = "parallel:gpu:mexGPUExample:InvalidInput";
    char const* const errMsg = "Invalid input to MEX file.";

    /* Choose a reasonably sized number of threads for the block. */
    int const threadsPerBlock = 256;
    int const blocksPerGrid = 1024;

    /* Initialize the MathWorks GPU API. */
    mxInitGPU();

    /* Throw an error if the input is not a GPU array. */
    if ((nrhs != 21) || !(mxIsGPUArray(prhs[0])) || !(mxIsGPUArray(prhs[1])) || !(mxIsGPUArray(prhs[2])) 
                     || !(mxIsGPUArray(prhs[3])) || !(mxIsGPUArray(prhs[4])) || !(mxIsGPUArray(prhs[5])) 
                     || !(mxIsGPUArray(prhs[6])) || !(mxIsGPUArray(prhs[7])) || !(mxIsGPUArray(prhs[8]))) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }

    X1 = mxGPUCreateFromMxArray(prhs[0]);
    X2 = mxGPUCreateFromMxArray(prhs[1]);
    X3 = mxGPUCreateFromMxArray(prhs[2]);
    dX1 = mxGPUCreateFromMxArray(prhs[3]);
    dX2 = mxGPUCreateFromMxArray(prhs[4]);
    dX3 = mxGPUCreateFromMxArray(prhs[5]);
    in1 = mxGPUCreateFromMxArray(prhs[6]);
    in2 = mxGPUCreateFromMxArray(prhs[7]);
    in3 = mxGPUCreateFromMxArray(prhs[8]);
    grid_size = mxGPUCreateFromMxArray(prhs[17]);
    active_actions = mxGPUCreateFromMxArray(prhs[18]);
    
    p_m = mxGetDoubles(prhs[9]); 
    double const m1 = p_m[0];
    p_m = mxGetDoubles(prhs[10]); 
    double const m2 = p_m[0];
    p_m = mxGetDoubles(prhs[11]); 
    double const m3 = p_m[0];

    p_l = mxGetDoubles(prhs[12]); 
    double const l1 = p_l[0];
    p_l = mxGetDoubles(prhs[13]); 
    double const l2 = p_l[0];
    p_l = mxGetDoubles(prhs[14]); 
    double const l3 = p_l[0];

    p_g = mxGetDoubles(prhs[15]);
    double const g = p_g[0];
    
    p_dt = mxGetDoubles(prhs[16]);
    double const dt = p_dt[0];
    
    p_limits = mxGetDoubles(prhs[19]);
    
    p_x_dims_free = mxGetDoubles(prhs[20]);
    mwSize const* num_x_dims = mxGetDimensions(prhs[20]);
    if ((mxGetNumberOfDimensions(prhs[20]) != 2) || (num_x_dims[1] > 1)) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }
    
    /*
     * Verify that inputs are of appropriate type before extracting the pointer.
     */
    if ((mxGPUGetClassID(X1) != mxDOUBLE_CLASS) || (mxGPUGetClassID(X2) != mxDOUBLE_CLASS) || (mxGPUGetClassID(X3) != mxDOUBLE_CLASS)
        || (mxGPUGetClassID(dX1) != mxDOUBLE_CLASS) || (mxGPUGetClassID(dX2) != mxDOUBLE_CLASS) || (mxGPUGetClassID(dX3) != mxDOUBLE_CLASS)     
        || (mxGPUGetClassID(in1) != mxDOUBLE_CLASS) || (mxGPUGetClassID(in2) != mxDOUBLE_CLASS) || (mxGPUGetClassID(in3) != mxDOUBLE_CLASS)
        || (mxGPUGetClassID(grid_size) != mxINT32_CLASS) || (mxGPUGetClassID(active_actions) != mxINT32_CLASS)) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }

    /*
     * Now that we have verified the data type, extract a pointer to the input
     * data on the device.
     */
    p_X1 = (double*)(mxGPUGetDataReadOnly(X1));
    p_X2 = (double*)(mxGPUGetDataReadOnly(X2));
    p_X3 = (double*)(mxGPUGetDataReadOnly(X3));
    p_dX1 = (double*)(mxGPUGetDataReadOnly(dX1));
    p_dX2 = (double*)(mxGPUGetDataReadOnly(dX2));
    p_dX3 = (double*)(mxGPUGetDataReadOnly(dX3));
    p_in1 = (double const*)(mxGPUGetDataReadOnly(in1));
    p_in2 = (double const*)(mxGPUGetDataReadOnly(in2));
    p_in3 = (double const*)(mxGPUGetDataReadOnly(in3));
    p_grid_size = (int32_t const*)(mxGPUGetDataReadOnly(grid_size));
    p_active_actions = (int32_t const*)(mxGPUGetDataReadOnly(active_actions));
    
    /* Create output arrays*/
    X1n = mxGPUCopyGPUArray(X1);
    p_X1n = (double*)(mxGPUGetData(X1n));
    
    X2n = mxGPUCopyGPUArray(X2);
    p_X2n = (double*)(mxGPUGetData(X2n));
    
    X3n = mxGPUCopyGPUArray(X3);
    p_X3n = (double*)(mxGPUGetData(X3n));
    
    dX1n = mxGPUCopyGPUArray(dX1);
    p_dX1n = (double*)(mxGPUGetData(dX1n));
    
    dX2n = mxGPUCopyGPUArray(dX2);
    p_dX2n = (double*)(mxGPUGetData(dX2n));
    
    dX3n = mxGPUCopyGPUArray(dX3);
    p_dX3n = (double*)(mxGPUGetData(dX3n));
    
    // RK4 - Step 1
    int32_t curr_free_dim;
    for (int32_t ii=0; ii < num_x_dims[0]; ii++) {
        curr_free_dim = int32_t(p_x_dims_free[ii]);
        if (curr_free_dim == 1) {
            p_k1_X1 = p_dX1;
            step << <blocksPerGrid, threadsPerBlock >> > (p_X1n, p_X1, p_k1_X1, 0.5 * dt, p_grid_size);
    
        } else if (curr_free_dim == 2) {
            p_k1_X2 = p_dX2;
            step << <blocksPerGrid, threadsPerBlock >> > (p_X2n, p_X2, p_k1_X2, 0.5 * dt, p_grid_size);
            
        } else if (curr_free_dim == 3) {
            p_k1_X3 = p_dX3;
            step << <blocksPerGrid, threadsPerBlock >> > (p_X3n, p_X3, p_k1_X3, 0.5 * dt, p_grid_size);
            
        } else if (curr_free_dim == 4) {
            k1_dX1 = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(dX1), mxGPUGetDimensions(dX1), mxGPUGetClassID(dX1), mxGPUGetComplexity(dX1), MX_GPU_DO_NOT_INITIALIZE);
            p_k1_dX1 = (double*)(mxGPUGetData(k1_dX1));
            dyn4_mex_continuous << <blocksPerGrid, threadsPerBlock >> > (p_k1_dX1, p_X1, p_X2, p_X3, p_dX1, p_dX2, p_dX3, 
                                                                         p_in1, p_in2, p_in3, m1, m2, m3, l1, l2, l3, 
                                                                         g, p_grid_size, p_active_actions);
            step << <blocksPerGrid, threadsPerBlock >> > (p_dX1n, p_dX1, p_k1_dX1, 0.5 * dt, p_grid_size);
            
        } else if (curr_free_dim == 5) {
            k1_dX2 = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(dX2), mxGPUGetDimensions(dX2), mxGPUGetClassID(dX2), mxGPUGetComplexity(dX2), MX_GPU_DO_NOT_INITIALIZE);
            p_k1_dX2 = (double*)(mxGPUGetData(k1_dX2));
            dyn5_mex_continuous << <blocksPerGrid, threadsPerBlock >> > (p_k1_dX2, p_X1, p_X2, p_X3, p_dX1, p_dX2, p_dX3, 
                                                                         p_in1, p_in2, p_in3, m1, m2, m3, l1, l2, l3, 
                                                                         g, p_grid_size, p_active_actions);
            step << <blocksPerGrid, threadsPerBlock >> > (p_dX2n, p_dX2, p_k1_dX2, 0.5 * dt, p_grid_size);
            
        } else if (curr_free_dim == 6) {
            k1_dX3 = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(dX3), mxGPUGetDimensions(dX3), mxGPUGetClassID(dX3), mxGPUGetComplexity(dX3), MX_GPU_DO_NOT_INITIALIZE);
            p_k1_dX3 = (double*)(mxGPUGetData(k1_dX3));
            dyn6_mex_continuous << <blocksPerGrid, threadsPerBlock >> > (p_k1_dX3, p_X1, p_X2, p_X3, p_dX1, p_dX2, p_dX3, 
                                                                         p_in1, p_in2, p_in3, m1, m2, m3, l1, l2, l3, 
                                                                         g, p_grid_size, p_active_actions);
            step << <blocksPerGrid, threadsPerBlock >> > (p_dX3n, p_dX3, p_k1_dX3, 0.5 * dt, p_grid_size);
            
        } 
    }
    
    // RK4 - Step 2
    // Continuous Dynamics
    for (int32_t ii=0; ii < num_x_dims[0]; ii++) {
        curr_free_dim = int32_t(p_x_dims_free[ii]);
        if (curr_free_dim == 1) {
            k2_X1 = mxGPUCopyGPUArray(dX1n);
            p_k2_X1 = (double*)(mxGPUGetData(k2_X1));
    
        } else if (curr_free_dim == 2) {
            k2_X2 = mxGPUCopyGPUArray(dX2n);
            p_k2_X2 = (double*)(mxGPUGetData(k2_X2));
            
        } else if (curr_free_dim == 3) {
            k2_X3 = mxGPUCopyGPUArray(dX3n);
            p_k2_X3 = (double*)(mxGPUGetData(k2_X3));
            
        } else if (curr_free_dim == 4) {
            k2_dX1 = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(dX1), mxGPUGetDimensions(dX1), mxGPUGetClassID(dX1), mxGPUGetComplexity(dX1), MX_GPU_DO_NOT_INITIALIZE);
            p_k2_dX1 = (double*)(mxGPUGetData(k2_dX1));
            dyn4_mex_continuous << <blocksPerGrid, threadsPerBlock >> > (p_k2_dX1, p_X1n, p_X2n, p_X3n, p_dX1n, p_dX2n, p_dX3n, 
                                                                         p_in1, p_in2, p_in3, m1, m2, m3, l1, l2, l3, 
                                                                         g, p_grid_size, p_active_actions);
        } else if (curr_free_dim == 5) {
            k2_dX2 = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(dX2), mxGPUGetDimensions(dX2), mxGPUGetClassID(dX2), mxGPUGetComplexity(dX2), MX_GPU_DO_NOT_INITIALIZE);
            p_k2_dX2 = (double*)(mxGPUGetData(k2_dX2));
            dyn5_mex_continuous << <blocksPerGrid, threadsPerBlock >> > (p_k2_dX2, p_X1n, p_X2n, p_X3n, p_dX1n, p_dX2n, p_dX3n, 
                                                                         p_in1, p_in2, p_in3, m1, m2, m3, l1, l2, l3, 
                                                                         g, p_grid_size, p_active_actions);            
        } else if (curr_free_dim == 6) {
            k2_dX3 = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(dX3), mxGPUGetDimensions(dX3), mxGPUGetClassID(dX3), mxGPUGetComplexity(dX3), MX_GPU_DO_NOT_INITIALIZE);
            p_k2_dX3 = (double*)(mxGPUGetData(k2_dX3));
            dyn6_mex_continuous << <blocksPerGrid, threadsPerBlock >> > (p_k2_dX3, p_X1n, p_X2n, p_X3n, p_dX1n, p_dX2n, p_dX3n, 
                                                                         p_in1, p_in2, p_in3, m1, m2, m3, l1, l2, l3, 
                                                                         g, p_grid_size, p_active_actions); 
        }
    }
    // Discrete step
    for (int32_t ii=0; ii < num_x_dims[0]; ii++) {
        curr_free_dim = int32_t(p_x_dims_free[ii]);
        if (curr_free_dim == 1) {
            step << <blocksPerGrid, threadsPerBlock >> > (p_X1n, p_X1, p_k2_X1, 0.5 * dt, p_grid_size);
    
        } else if (curr_free_dim == 2) {
            step << <blocksPerGrid, threadsPerBlock >> > (p_X2n, p_X2, p_k2_X2, 0.5 * dt, p_grid_size);
            
        } else if (curr_free_dim == 3) {
            step << <blocksPerGrid, threadsPerBlock >> > (p_X3n, p_X3, p_k2_X3, 0.5 * dt, p_grid_size);
            
        } else if (curr_free_dim == 4) {
            step << <blocksPerGrid, threadsPerBlock >> > (p_dX1n, p_dX1, p_k2_dX1, 0.5 * dt, p_grid_size);
            
        } else if (curr_free_dim == 5) {
            step << <blocksPerGrid, threadsPerBlock >> > (p_dX2n, p_dX2, p_k2_dX2, 0.5 * dt, p_grid_size);
            
        } else if (curr_free_dim == 6) {
            step << <blocksPerGrid, threadsPerBlock >> > (p_dX3n, p_dX3, p_k2_dX3, 0.5 * dt, p_grid_size);
            
        } 
    }
    
    // RK4 - Step 3
    // Continuous Dynamics
    for (int32_t ii=0; ii < num_x_dims[0]; ii++) {
        curr_free_dim = int32_t(p_x_dims_free[ii]);
        if (curr_free_dim == 1) {
            k3_X1 = mxGPUCopyGPUArray(dX1n);
            p_k3_X1 = (double*)(mxGPUGetData(k3_X1));
    
        } else if (curr_free_dim == 2) {
            k3_X2 = mxGPUCopyGPUArray(dX2n);
            p_k3_X2 = (double*)(mxGPUGetData(k3_X2));
            
        } else if (curr_free_dim == 3) {
            k3_X3 = mxGPUCopyGPUArray(dX3n);
            p_k3_X3 = (double*)(mxGPUGetData(k3_X3));
            
        } else if (curr_free_dim == 4) {
            k3_dX1 = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(dX1), mxGPUGetDimensions(dX1), mxGPUGetClassID(dX1), mxGPUGetComplexity(dX1), MX_GPU_DO_NOT_INITIALIZE);
            p_k3_dX1 = (double*)(mxGPUGetData(k3_dX1));
            dyn4_mex_continuous << <blocksPerGrid, threadsPerBlock >> > (p_k3_dX1, p_X1n, p_X2n, p_X3n, p_dX1n, p_dX2n, p_dX3n, 
                                                                         p_in1, p_in2, p_in3, m1, m2, m3, l1, l2, l3,
                                                                         g, p_grid_size, p_active_actions);
        } else if (curr_free_dim == 5) {
            k3_dX2 = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(dX2), mxGPUGetDimensions(dX2), mxGPUGetClassID(dX2), mxGPUGetComplexity(dX2), MX_GPU_DO_NOT_INITIALIZE);
            p_k3_dX2 = (double*)(mxGPUGetData(k3_dX2));
            dyn5_mex_continuous << <blocksPerGrid, threadsPerBlock >> > (p_k3_dX2, p_X1n, p_X2n, p_X3n, p_dX1n, p_dX2n, p_dX3n, 
                                                                         p_in1, p_in2, p_in3, m1, m2, m3, l1, l2, l3, 
                                                                         g, p_grid_size, p_active_actions);            
        } else if (curr_free_dim == 6) {
            k3_dX3 = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(dX3), mxGPUGetDimensions(dX3), mxGPUGetClassID(dX3), mxGPUGetComplexity(dX3), MX_GPU_DO_NOT_INITIALIZE);
            p_k3_dX3 = (double*)(mxGPUGetData(k3_dX3));
            dyn6_mex_continuous << <blocksPerGrid, threadsPerBlock >> > (p_k3_dX3, p_X1n, p_X2n, p_X3n, p_dX1n, p_dX2n, p_dX3n, 
                                                                         p_in1, p_in2, p_in3, m1, m2, m3, l1, l2, l3, 
                                                                         g, p_grid_size, p_active_actions); 
        }
    }
    // Discrete step
    for (int32_t ii=0; ii < num_x_dims[0]; ii++) {
        curr_free_dim = int32_t(p_x_dims_free[ii]);
        if (curr_free_dim == 1) {
            step << <blocksPerGrid, threadsPerBlock >> > (p_X1n, p_X1, p_k3_X1, dt, p_grid_size);
    
        } else if (curr_free_dim == 2) {
            step << <blocksPerGrid, threadsPerBlock >> > (p_X2n, p_X2, p_k3_X2, dt, p_grid_size);
            
        } else if (curr_free_dim == 3) {
            step << <blocksPerGrid, threadsPerBlock >> > (p_X3n, p_X3, p_k3_X3, dt, p_grid_size);
            
        } else if (curr_free_dim == 4) {
            step << <blocksPerGrid, threadsPerBlock >> > (p_dX1n, p_dX1, p_k3_dX1, dt, p_grid_size);
            
        } else if (curr_free_dim == 5) {
            step << <blocksPerGrid, threadsPerBlock >> > (p_dX2n, p_dX2, p_k3_dX2, dt, p_grid_size);
            
        } else if (curr_free_dim == 6) {
            step << <blocksPerGrid, threadsPerBlock >> > (p_dX3n, p_dX3, p_k3_dX3, dt, p_grid_size);
            
        }
    }
    
    // RK4 - Step 4
    // Continuous Dynamics
    for (int32_t ii=0; ii < num_x_dims[0]; ii++) {
        curr_free_dim = int32_t(p_x_dims_free[ii]);
        if (curr_free_dim == 1) {
            k4_X1 = mxGPUCopyGPUArray(dX1n);
            p_k4_X1 = (double*)(mxGPUGetData(k4_X1));
    
        } else if (curr_free_dim == 2) {
            k4_X2 = mxGPUCopyGPUArray(dX2n);
            p_k4_X2 = (double*)(mxGPUGetData(k4_X2));
            
        } else if (curr_free_dim == 3) {
            k4_X3 = mxGPUCopyGPUArray(dX3n);
            p_k4_X3 = (double*)(mxGPUGetData(k4_X3));
            
        } else if (curr_free_dim == 4) {
            k4_dX1 = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(dX1), mxGPUGetDimensions(dX1), mxGPUGetClassID(dX1), mxGPUGetComplexity(dX1), MX_GPU_DO_NOT_INITIALIZE);
            p_k4_dX1 = (double*)(mxGPUGetData(k4_dX1));
            dyn4_mex_continuous << <blocksPerGrid, threadsPerBlock >> > (p_k4_dX1, p_X1n, p_X2n, p_X3n, p_dX1n, p_dX2n, p_dX3n, 
                                                                         p_in1, p_in2, p_in3, m1, m2, m3, l1, l2, l3, 
                                                                         g, p_grid_size, p_active_actions);
        } else if (curr_free_dim == 5) {
            k4_dX2 = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(dX2), mxGPUGetDimensions(dX2), mxGPUGetClassID(dX2), mxGPUGetComplexity(dX2), MX_GPU_DO_NOT_INITIALIZE);
            p_k4_dX2 = (double*)(mxGPUGetData(k4_dX2));
            dyn5_mex_continuous << <blocksPerGrid, threadsPerBlock >> > (p_k4_dX2, p_X1n, p_X2n, p_X3n, p_dX1n, p_dX2n, p_dX3n, 
                                                                         p_in1, p_in2, p_in3, m1, m2, m3, l1, l2, l3, 
                                                                         g, p_grid_size, p_active_actions);            
        } else if (curr_free_dim == 6) {
            k4_dX3 = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(dX3), mxGPUGetDimensions(dX3), mxGPUGetClassID(dX3), mxGPUGetComplexity(dX3), MX_GPU_DO_NOT_INITIALIZE);
            p_k4_dX3 = (double*)(mxGPUGetData(k4_dX3));
            dyn6_mex_continuous << <blocksPerGrid, threadsPerBlock >> > (p_k4_dX3, p_X1n, p_X2n, p_X3n, p_dX1n, p_dX2n, p_dX3n, 
                                                                         p_in1, p_in2, p_in3, m1, m2, m3, l1, l2, l3, 
                                                                         g, p_grid_size, p_active_actions); 
        }
    }

    // RK4 Ouput
    for (int32_t ii=0; ii < num_x_dims[0]; ii++) {
        curr_free_dim = int32_t(p_x_dims_free[ii]);
        if (curr_free_dim == 1) {
            rk4_wbounds << <blocksPerGrid, threadsPerBlock >> > (p_X1n, p_X1, p_k1_X1, p_k2_X1, p_k3_X1, p_k4_X1, dt, p_limits[0], p_limits[6], p_grid_size);
            mxGPUDestroyGPUArray(k2_X1);
            mxGPUDestroyGPUArray(k3_X1);
            mxGPUDestroyGPUArray(k4_X1);
        } else if (curr_free_dim == 2) {
            rk4_wbounds << <blocksPerGrid, threadsPerBlock >> > (p_X2n, p_X2, p_k1_X2, p_k2_X2, p_k3_X2, p_k4_X2, dt, p_limits[1], p_limits[7], p_grid_size);
            mxGPUDestroyGPUArray(k2_X2);
            mxGPUDestroyGPUArray(k3_X2);
            mxGPUDestroyGPUArray(k4_X2);
        } else if (curr_free_dim == 3) {
            rk4_wbounds << <blocksPerGrid, threadsPerBlock >> > (p_X3n, p_X3, p_k1_X3, p_k2_X3, p_k3_X3, p_k4_X3, dt, p_limits[2], p_limits[8], p_grid_size);
            mxGPUDestroyGPUArray(k2_X3);
            mxGPUDestroyGPUArray(k3_X3);
            mxGPUDestroyGPUArray(k4_X3);
        } else if (curr_free_dim == 4) {
            rk4_wbounds << <blocksPerGrid, threadsPerBlock >> > (p_dX1n, p_dX1, p_k1_dX1, p_k2_dX1, p_k3_dX1, p_k4_dX1, dt, p_limits[3], p_limits[9], p_grid_size);
            mxGPUDestroyGPUArray(k1_dX1);
            mxGPUDestroyGPUArray(k2_dX1);
            mxGPUDestroyGPUArray(k3_dX1);
            mxGPUDestroyGPUArray(k4_dX1);
        } else if (curr_free_dim == 5) {
            rk4_wbounds << <blocksPerGrid, threadsPerBlock >> > (p_dX2n, p_dX2, p_k1_dX2, p_k2_dX2, p_k3_dX2, p_k4_dX2, dt, p_limits[4], p_limits[10], p_grid_size);
            mxGPUDestroyGPUArray(k1_dX2);
            mxGPUDestroyGPUArray(k2_dX2);
            mxGPUDestroyGPUArray(k3_dX2);
            mxGPUDestroyGPUArray(k4_dX2);
        } else if (curr_free_dim == 6) {
            rk4_wbounds << <blocksPerGrid, threadsPerBlock >> > (p_dX3n, p_dX3, p_k1_dX3, p_k2_dX3, p_k3_dX3, p_k4_dX3, dt, p_limits[5], p_limits[11], p_grid_size);
            mxGPUDestroyGPUArray(k1_dX3);
            mxGPUDestroyGPUArray(k2_dX3);
            mxGPUDestroyGPUArray(k3_dX3);
            mxGPUDestroyGPUArray(k4_dX3);
        }
    }
 
    /* Wrap the result up as a MATLAB gpuArray for return. */
    plhs[0] = mxGPUCreateMxArrayOnGPU(X1n);
    plhs[1] = mxGPUCreateMxArrayOnGPU(X2n);
    plhs[2] = mxGPUCreateMxArrayOnGPU(X3n);
    plhs[3] = mxGPUCreateMxArrayOnGPU(dX1n);
    plhs[4] = mxGPUCreateMxArrayOnGPU(dX2n);
    plhs[5] = mxGPUCreateMxArrayOnGPU(dX3n);

    /*
     * The mxGPUArray pointers are host-side structures that refer to device
     * data. These must be destroyed before leaving the MEX function.
     */
    mxGPUDestroyGPUArray(X1);
    mxGPUDestroyGPUArray(X2);
    mxGPUDestroyGPUArray(X3);
    mxGPUDestroyGPUArray(dX1);
    mxGPUDestroyGPUArray(dX2);
    mxGPUDestroyGPUArray(dX3);
    mxGPUDestroyGPUArray(in1);
    mxGPUDestroyGPUArray(in2);
    mxGPUDestroyGPUArray(in3);
    mxGPUDestroyGPUArray(grid_size);
    mxGPUDestroyGPUArray(active_actions);
    mxGPUDestroyGPUArray(X1n);
    mxGPUDestroyGPUArray(X2n);
    mxGPUDestroyGPUArray(X3n);
    mxGPUDestroyGPUArray(dX1n);
    mxGPUDestroyGPUArray(dX2n);
    mxGPUDestroyGPUArray(dX3n);
}
