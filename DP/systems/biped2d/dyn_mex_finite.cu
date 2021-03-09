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

__global__ void dyn1_mex_continuous(double* const d_x1, // outputs
    double* const x1_, double* const x2_, double* const x3_,
    double* const x4_, double* const x5_, double* const x6_, // input states
    double const* const in1, double const* const in2, double const* const in3, double const* const in4, // input actions
    const double m, const double I, const double d,
    const double df, const double l0, const double g, 
    int32_t const* const grid_size, int32_t const* const active_actions)
{
    int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x * blockDim.y
        + (threadIdx.x + threadIdx.y * blockDim.x);
    int num_threads = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
    int num_elements = grid_size[0] * grid_size[1] * grid_size[2] * grid_size[3]
                        * grid_size[4] * grid_size[5];
    double x1, x2, x3, x4, x5, x6, u1, u2, u3, u4;

    int index1 = (grid_size[0] > 1) ? index : 0;
    int index2 = (grid_size[1] > 1) ? index : 0;
    int index3 = (grid_size[2] > 1) ? index : 0;
    int index4 = (grid_size[3] > 1) ? index : 0;
    int index5 = (grid_size[4] > 1) ? index : 0;
    int index6 = (grid_size[5] > 1) ? index : 0;

    int uindex1 = (active_actions[0] == 1) ? index : 0;
    int uindex2 = (active_actions[1] == 1) ? index : 0;
    int uindex3 = (active_actions[2] == 1) ? index : 0;
    int uindex4 = (active_actions[3] == 1) ? index : 0;

    while (index < num_elements)
    {
        x1 = x1_[index1];
        x2 = x2_[index2];
        x3 = x3_[index3];
        x4 = x4_[index4];
        x5 = x5_[index5];
        x6 = x6_[index6];

        u1 = in1[uindex1];
        u2 = in2[uindex2];
        u3 = in3[uindex3];
        u4 = in4[uindex4];
        
        double t2 = cos(x2);
        double t3 = sin(x2);
        double l2 = sqrt((x1 * t2 + df) * (x1 * t2 + df) 
                         + (x1 * t3) * (x1 * t3));
        double contact1 = (x1 <= l0) ? 1.0 : 0.0;
        double contact2 = (l2 <= l0) ? 1.0 : 0.0;
        u1 = u1 * contact1;
        u2 = u2 * contact2;
        u3 = u3 * contact1;
        u4 = u4 * contact2;

        double t4 = df * df;
        double t5 = x1 * x1;
//         double t8 = 1.0/m;
        double t9 = -x5;
//         double t10 = 1.0/x1;
        double t6 = t2 * t2;
        double t7 = t2 * x1;
        double t13 = t9+x2;
        double t11 = df * t7 * 2.0;
        double t12 = df+t7;
        double t14 = cos(t13);
        double t15 = sin(t13);
        double t16 = t6-1.0;
        double t17 = t4+t5+t11;
        double t18 = 1.0/t17;
        double t19 = 1.0/sqrt(t17);
        double t20 = t12 * t19;
        double t22 = t5 * t16 * t18;
        double t21 = acos(t20);
        double t24 = -t22;
//         double t23 = -t21;
        double t26 = sqrt(t24);
//         double t25 = t23+x5;
        
        d_x1[index] = (t2 * x3) + (t3 * x4) + (d * t14 * x6);

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
        uindex4 = (active_actions[3] == 1) ? index : 0;
    }
}

__global__ void dyn2_mex_continuous(double* const d_x2, // outputs
    double* const x1_, double* const x2_, double* const x3_,
    double* const x4_, double* const x5_, double* const x6_, // input states
    double const* const in1, double const* const in2, double const* const in3, double const* const in4, // input actions
    const double m, const double I, const double d,
    const double df, const double l0, const double g, 
    int32_t const* const grid_size, int32_t const* const active_actions)
{
    int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x * blockDim.y
        + (threadIdx.x + threadIdx.y * blockDim.x);
    int num_threads = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
    int num_elements = grid_size[0] * grid_size[1] * grid_size[2] * grid_size[3]
                        * grid_size[4] * grid_size[5];
    double x1, x2, x3, x4, x5, x6, u1, u2, u3, u4;

    int index1 = (grid_size[0] > 1) ? index : 0;
    int index2 = (grid_size[1] > 1) ? index : 0;
    int index3 = (grid_size[2] > 1) ? index : 0;
    int index4 = (grid_size[3] > 1) ? index : 0;
    int index5 = (grid_size[4] > 1) ? index : 0;
    int index6 = (grid_size[5] > 1) ? index : 0;

    int uindex1 = (active_actions[0] == 1) ? index : 0;
    int uindex2 = (active_actions[1] == 1) ? index : 0;
    int uindex3 = (active_actions[2] == 1) ? index : 0;
    int uindex4 = (active_actions[3] == 1) ? index : 0;

    while (index < num_elements)
    {
        x1 = x1_[index1];
        x2 = x2_[index2];
        x3 = x3_[index3];
        x4 = x4_[index4];
        x5 = x5_[index5];
        x6 = x6_[index6];

        u1 = in1[uindex1];
        u2 = in2[uindex2];
        u3 = in3[uindex3];
        u4 = in4[uindex4];
        
        double t2 = cos(x2);
        double t3 = sin(x2);
        double l2 = sqrt((x1 * t2 + df) * (x1 * t2 + df) 
                         + (x1 * t3) * (x1 * t3));
        double contact1 = (x1 <= l0) ? 1.0 : 0.0;
        double contact2 = (l2 <= l0) ? 1.0 : 0.0;
        u1 = u1 * contact1;
        u2 = u2 * contact2;
        u3 = u3 * contact1;
        u4 = u4 * contact2;

        double t4 = df * df;
        double t5 = x1 * x1;
//         double t8 = 1.0/m;
        double t9 = -x5;
        double t10 = 1.0/x1;
        double t6 = t2 * t2;
        double t7 = t2 * x1;
        double t13 = t9+x2;
        double t11 = df * t7 * 2.0;
        double t12 = df+t7;
        double t14 = cos(t13);
        double t15 = sin(t13);
        double t16 = t6-1.0;
        double t17 = t4+t5+t11;
        double t18 = 1.0/t17;
        double t19 = 1.0/sqrt(t17);
        double t20 = t12 * t19;
        double t22 = t5 * t16 * t18;
        double t21 = acos(t20);
        double t24 = -t22;
//         double t23 = -t21;
        double t26 = sqrt(t24);
//         double t25 = t23+x5;
        
        d_x2[index] = -t10 * ((t3 * x3) + (d * t15 * x6) - (t2 * x4));

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
        uindex4 = (active_actions[3] == 1) ? index : 0;
    }
}

__global__ void dyn3_mex_continuous(double* const d_x3, // outputs
    double* const x1_, double* const x2_, double* const x3_,
    double* const x4_, double* const x5_, double* const x6_, // input states
    double const* const in1, double const* const in2, double const* const in3, double const* const in4, // input actions
    const double m, const double I, const double d,
    const double df, const double l0, const double g, 
    int32_t const* const grid_size, int32_t const* const active_actions)
{
    int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x * blockDim.y
        + (threadIdx.x + threadIdx.y * blockDim.x);
    int num_threads = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
    int num_elements = grid_size[0] * grid_size[1] * grid_size[2] * grid_size[3]
                        * grid_size[4] * grid_size[5];
    double x1, x2, x5, u1, u2, u3, u4;
//     double x3, x4, x6;
    int index1 = (grid_size[0] > 1) ? index : 0;
    int index2 = (grid_size[1] > 1) ? index : 0;
//     int index3 = (grid_size[2] > 1) ? index : 0;
//     int index4 = (grid_size[3] > 1) ? index : 0;
    int index5 = (grid_size[4] > 1) ? index : 0;
//     int index6 = (grid_size[5] > 1) ? index : 0;

    int uindex1 = (active_actions[0] == 1) ? index : 0;
    int uindex2 = (active_actions[1] == 1) ? index : 0;
    int uindex3 = (active_actions[2] == 1) ? index : 0;
    int uindex4 = (active_actions[3] == 1) ? index : 0;

    while (index < num_elements)
    {
        x1 = x1_[index1];
        x2 = x2_[index2];
//         x3 = x3_[index3];
//         x4 = x4_[index4];
        x5 = x5_[index5];
//         x6 = x6_[index6];

        u1 = in1[uindex1];
        u2 = in2[uindex2];
        u3 = in3[uindex3];
        u4 = in4[uindex4];
        
        double t2 = cos(x2);
        double t3 = sin(x2);
        double l2 = sqrt((x1 * t2 + df) * (x1 * t2 + df) 
                         + (x1 * t3) * (x1 * t3));
        double contact1 = (x1 <= l0) ? 1.0 : 0.0;
        double contact2 = (l2 <= l0) ? 1.0 : 0.0;
        u1 = u1 * contact1;
        u2 = u2 * contact2;
        u3 = u3 * contact1;
        u4 = u4 * contact2;

        double t4 = df * df;
        double t5 = x1 * x1;
        double t8 = 1.0/m;
        double t9 = -x5;
        double t10 = 1.0/x1;
        double t6 = t2 * t2;
        double t7 = t2 * x1;
        double t13 = t9+x2;
        double t11 = df * t7 * 2.0;
        double t12 = df+t7;
        double t14 = cos(t13);
        double t15 = sin(t13);
        double t16 = t6-1.0;
        double t17 = t4+t5+t11;
        double t18 = 1.0/t17;
        double t19 = 1.0/sqrt(t17);
        double t20 = t12 * t19;
        double t22 = t5 * t16 * t18;
        double t21 = acos(t20);
        double t24 = -t22;
//         double t23 = -t21;
        double t26 = sqrt(t24);
//         double t25 = t23+x5;
        
        d_x3[index] = t8 * ((t2 * u1) + (t20 * u2) + (t3 * t10 * u3) + (t19 * t26 * u4));

        index = index + num_threads;

        index1 = (grid_size[0] > 1) ? index : 0;
        index2 = (grid_size[1] > 1) ? index : 0;
//         index3 = (grid_size[2] > 1) ? index : 0;
//         index4 = (grid_size[3] > 1) ? index : 0;
        index5 = (grid_size[4] > 1) ? index : 0;
//         index6 = (grid_size[5] > 1) ? index : 0;

        uindex1 = (active_actions[0] == 1) ? index : 0;
        uindex2 = (active_actions[1] == 1) ? index : 0;
        uindex3 = (active_actions[2] == 1) ? index : 0;
        uindex4 = (active_actions[3] == 1) ? index : 0;
    }
}

__global__ void dyn4_mex_continuous(double* const d_x4, // outputs
    double* const x1_, double* const x2_, double* const x3_,
    double* const x4_, double* const x5_, double* const x6_, // input states
    double const* const in1, double const* const in2, double const* const in3, double const* const in4, // input actions
    const double m, const double I, const double d,
    const double df, const double l0, const double g, 
    int32_t const* const grid_size, int32_t const* const active_actions)
{
    int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x * blockDim.y
        + (threadIdx.x + threadIdx.y * blockDim.x);
    int num_threads = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
    int num_elements = grid_size[0] * grid_size[1] * grid_size[2] * grid_size[3]
                        * grid_size[4] * grid_size[5];
    double x1, x2, x5, u1, u2, u3, u4;
//     double x3, x4, x6;

    int index1 = (grid_size[0] > 1) ? index : 0;
    int index2 = (grid_size[1] > 1) ? index : 0;
//     int index3 = (grid_size[2] > 1) ? index : 0;
//     int index4 = (grid_size[3] > 1) ? index : 0;
    int index5 = (grid_size[4] > 1) ? index : 0;
//     int index6 = (grid_size[5] > 1) ? index : 0;

    int uindex1 = (active_actions[0] == 1) ? index : 0;
    int uindex2 = (active_actions[1] == 1) ? index : 0;
    int uindex3 = (active_actions[2] == 1) ? index : 0;
    int uindex4 = (active_actions[3] == 1) ? index : 0;

    while (index < num_elements)
    {
        x1 = x1_[index1];
        x2 = x2_[index2];
//         x3 = x3_[index3];
//         x4 = x4_[index4];
        x5 = x5_[index5];
//         x6 = x6_[index6];

        u1 = in1[uindex1];
        u2 = in2[uindex2];
        u3 = in3[uindex3];
        u4 = in4[uindex4];
        
        double t2 = cos(x2);
        double t3 = sin(x2);
        double l2 = sqrt((x1 * t2 + df) * (x1 * t2 + df) 
                         + (x1 * t3) * (x1 * t3));
        double contact1 = (x1 <= l0) ? 1.0 : 0.0;
        double contact2 = (l2 <= l0) ? 1.0 : 0.0;
        u1 = u1 * contact1;
        u2 = u2 * contact2;
        u3 = u3 * contact1;
        u4 = u4 * contact2;

        double t4 = df * df;
        double t5 = x1 * x1;
        double t8 = 1.0/m;
        double t9 = -x5;
        double t10 = 1.0/x1;
        double t6 = t2 * t2;
        double t7 = t2 * x1;
        double t13 = t9+x2;
        double t11 = df * t7 * 2.0;
        double t12 = df+t7;
        double t14 = cos(t13);
        double t15 = sin(t13);
        double t16 = t6-1.0;
        double t17 = t4+t5+t11;
        double t18 = 1.0/t17;
        double t19 = 1.0/sqrt(t17);
        double t20 = t12 * t19;
        double t22 = t5 * t16 * t18;
        double t21 = acos(t20);
        double t24 = -t22;
//         double t23 = -t21;
        double t26 = sqrt(t24);
//         double t25 = t23+x5;
        
        d_x4[index] = -g + t8 * ((t3 * u1) + (t26 * u2) - (t2 * t10 * u3) - (t12 * t18 * u4));

        index = index + num_threads;

        index1 = (grid_size[0] > 1) ? index : 0;
        index2 = (grid_size[1] > 1) ? index : 0;
//         index3 = (grid_size[2] > 1) ? index : 0;
//         index4 = (grid_size[3] > 1) ? index : 0;
        index5 = (grid_size[4] > 1) ? index : 0;
//         index6 = (grid_size[5] > 1) ? index : 0;

        uindex1 = (active_actions[0] == 1) ? index : 0;
        uindex2 = (active_actions[1] == 1) ? index : 0;
        uindex3 = (active_actions[2] == 1) ? index : 0;
        uindex4 = (active_actions[3] == 1) ? index : 0;
    }
}

__global__ void dyn6_mex_continuous(double* const d_x6, // outputs
    double* const x1_, double* const x2_, double* const x3_,
    double* const x4_, double* const x5_, double* const x6_, // input states
    double const* const in1, double const* const in2, double const* const in3, double const* const in4, // input actions
    const double m, const double I, const double d,
    const double df, const double l0, const double g, 
    int32_t const* const grid_size, int32_t const* const active_actions)
{
    int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x * blockDim.y
        + (threadIdx.x + threadIdx.y * blockDim.x);
    int num_threads = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
    int num_elements = grid_size[0] * grid_size[1] * grid_size[2] * grid_size[3]
                        * grid_size[4] * grid_size[5];
    double x1, x2, x5, u1, u2, u3, u4;
//     double x3, x4, x6;

    int index1 = (grid_size[0] > 1) ? index : 0;
    int index2 = (grid_size[1] > 1) ? index : 0;
//     int index3 = (grid_size[2] > 1) ? index : 0;
//     int index4 = (grid_size[3] > 1) ? index : 0;
    int index5 = (grid_size[4] > 1) ? index : 0;
//     int index6 = (grid_size[5] > 1) ? index : 0;

    int uindex1 = (active_actions[0] == 1) ? index : 0;
    int uindex2 = (active_actions[1] == 1) ? index : 0;
    int uindex3 = (active_actions[2] == 1) ? index : 0;
    int uindex4 = (active_actions[3] == 1) ? index : 0;

    while (index < num_elements)
    {
        x1 = x1_[index1];
        x2 = x2_[index2];
//         x3 = x3_[index3];
//         x4 = x4_[index4];
        x5 = x5_[index5];
//         x6 = x6_[index6];

        u1 = in1[uindex1];
        u2 = in2[uindex2];
        u3 = in3[uindex3];
        u4 = in4[uindex4];
        
        double t2 = cos(x2);
        double t3 = sin(x2);
        double l2 = sqrt((x1 * t2 + df) * (x1 * t2 + df) 
                         + (x1 * t3) * (x1 * t3));
        double contact1 = (x1 <= l0) ? 1.0 : 0.0;
        double contact2 = (l2 <= l0) ? 1.0 : 0.0;
        u1 = u1 * contact1;
        u2 = u2 * contact2;
        u3 = u3 * contact1;
        u4 = u4 * contact2;

        double t4 = df * df;
        double t5 = x1 * x1;
//         double t8 = 1.0/m;
        double t9 = -x5;
        double t10 = 1.0/x1;
        double t6 = t2 * t2;
        double t7 = t2 * x1;
        double t13 = t9+x2;
        double t11 = df * t7 * 2.0;
        double t12 = df+t7;
        double t14 = cos(t13);
        double t15 = sin(t13);
        double t16 = t6-1.0;
        double t17 = t4+t5+t11;
        double t18 = 1.0/t17;
        double t19 = 1.0/sqrt(t17);
        double t20 = t12 * t19;
        double t22 = t5 * t16 * t18;
        double t21 = acos(t20);
        double t24 = -t22;
        double t23 = -t21;
        double t26 = sqrt(t24);
        double t25 = t23+x5;
        
        d_x6[index] = (u3 + u4 + (d * u2 * cos(t25)) + (d * t14 * u1) 
                       + (d * t10 * t15 * u3) - (d * t19 * u4 * sin(t25))) / I;

        index = index + num_threads;

        index1 = (grid_size[0] > 1) ? index : 0;
        index2 = (grid_size[1] > 1) ? index : 0;
//         index3 = (grid_size[2] > 1) ? index : 0;
//         index4 = (grid_size[3] > 1) ? index : 0;
        index5 = (grid_size[4] > 1) ? index : 0;
//         index6 = (grid_size[5] > 1) ? index : 0;

        uindex1 = (active_actions[0] == 1) ? index : 0;
        uindex2 = (active_actions[1] == 1) ? index : 0;
        uindex3 = (active_actions[2] == 1) ? index : 0;
        uindex4 = (active_actions[3] == 1) ? index : 0;
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
    mxGPUArray const* X4;
    mxGPUArray const* X5;
    mxGPUArray const* X6;
    mxGPUArray const* in1;
    mxGPUArray const* in2;
    mxGPUArray const* in3;
    mxGPUArray const* in4;
    mxGPUArray const* grid_size;
    mxGPUArray const* active_actions;
    // Pointers for GPU Arrays
    double* p_X1; 
    double* p_X2;
    double* p_X3; 
    double* p_X4;
    double* p_X5;
    double* p_X6;
    double const* p_in1; 
    double const* p_in2;
    double const* p_in3; 
    double const* p_in4; 
    int32_t const* p_grid_size;
    int32_t const* p_active_actions;
    // Pointers for normal inputs
    double* p_m;
    double* p_I;
    double* p_d;
    double* p_df;
    double* p_l0;
    double* p_g;
    double* p_dt;
    double* p_limits;
    double* p_x_dims_free;
    
    // Intermediate variables
    mxGPUArray* k1_X1;
    mxGPUArray* k1_X2;
    mxGPUArray* k1_X3;
    mxGPUArray* k1_X4;
    mxGPUArray* k1_X6;
    
    mxGPUArray* k2_X1;
    mxGPUArray* k2_X2;
    mxGPUArray* k2_X3;
    mxGPUArray* k2_X4;
    mxGPUArray* k2_X5;
    mxGPUArray* k2_X6;
    
    mxGPUArray* k3_X1;
    mxGPUArray* k3_X2;
    mxGPUArray* k3_X3;
    mxGPUArray* k3_X4;
    mxGPUArray* k3_X5;
    mxGPUArray* k3_X6;
    
    mxGPUArray* k4_X1;
    mxGPUArray* k4_X2;
    mxGPUArray* k4_X3;
    mxGPUArray* k4_X4;
    mxGPUArray* k4_X5;
    mxGPUArray* k4_X6;
    // Pointers for intermediate variables
    double* p_k1_X1;
    double* p_k1_X2;
    double* p_k1_X3;
    double* p_k1_X4;
    double* p_k1_X5;
    double* p_k1_X6;
    
    double* p_k2_X1;
    double* p_k2_X2;
    double* p_k2_X3;
    double* p_k2_X4;
    double* p_k2_X5;
    double* p_k2_X6;
    
    double* p_k3_X1;
    double* p_k3_X2;
    double* p_k3_X3;
    double* p_k3_X4;
    double* p_k3_X5;
    double* p_k3_X6;
    
    double* p_k4_X1;
    double* p_k4_X2;
    double* p_k4_X3;
    double* p_k4_X4;
    double* p_k4_X5;
    double* p_k4_X6;

    // Outputs
    mxGPUArray* X1n;
    mxGPUArray* X2n;
    mxGPUArray* X3n;
    mxGPUArray* X4n;
    mxGPUArray* X5n;
    mxGPUArray* X6n;
    // Pointers for outputs
    double* p_X1n;
    double* p_X2n;
    double* p_X3n;
    double* p_X4n;
    double* p_X5n;
    double* p_X6n;

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
                     || !(mxIsGPUArray(prhs[6])) || !(mxIsGPUArray(prhs[7])) || !(mxIsGPUArray(prhs[8]))
                     || !(mxIsGPUArray(prhs[9]))) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }

    X1 = mxGPUCreateFromMxArray(prhs[0]);
    X2 = mxGPUCreateFromMxArray(prhs[1]);
    X3 = mxGPUCreateFromMxArray(prhs[2]);
    X4 = mxGPUCreateFromMxArray(prhs[3]);
    X5 = mxGPUCreateFromMxArray(prhs[4]);
    X6 = mxGPUCreateFromMxArray(prhs[5]);
    in1 = mxGPUCreateFromMxArray(prhs[6]);
    in2 = mxGPUCreateFromMxArray(prhs[7]);
    in3 = mxGPUCreateFromMxArray(prhs[8]);
    in4 = mxGPUCreateFromMxArray(prhs[9]);
    grid_size = mxGPUCreateFromMxArray(prhs[17]);
    active_actions = mxGPUCreateFromMxArray(prhs[18]);
    
    p_m = mxGetDoubles(prhs[10]); 
    double const m = p_m[0];
    p_I = mxGetDoubles(prhs[11]); 
    double const I = p_I[0];
    p_d = mxGetDoubles(prhs[12]); 
    double const d = p_d[0];
    p_df = mxGetDoubles(prhs[13]); 
    double const df = p_df[0];
    p_l0 = mxGetDoubles(prhs[14]); 
    double const l0 = p_l0[0];
    
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
        || (mxGPUGetClassID(X4) != mxDOUBLE_CLASS) || (mxGPUGetClassID(X5) != mxDOUBLE_CLASS) || (mxGPUGetClassID(X6) != mxDOUBLE_CLASS)     
        || (mxGPUGetClassID(in1) != mxDOUBLE_CLASS) || (mxGPUGetClassID(in2) != mxDOUBLE_CLASS) || (mxGPUGetClassID(in3) != mxDOUBLE_CLASS)
        || (mxGPUGetClassID(in4) != mxDOUBLE_CLASS)
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
    p_X4 = (double*)(mxGPUGetDataReadOnly(X4));
    p_X5 = (double*)(mxGPUGetDataReadOnly(X5));
    p_X6 = (double*)(mxGPUGetDataReadOnly(X6));
    p_in1 = (double const*)(mxGPUGetDataReadOnly(in1));
    p_in2 = (double const*)(mxGPUGetDataReadOnly(in2));
    p_in3 = (double const*)(mxGPUGetDataReadOnly(in3));
    p_in4 = (double const*)(mxGPUGetDataReadOnly(in4));
    p_grid_size = (int32_t const*)(mxGPUGetDataReadOnly(grid_size));
    p_active_actions = (int32_t const*)(mxGPUGetDataReadOnly(active_actions));
    
    /* Create output arrays*/
    X1n = mxGPUCopyGPUArray(X1);
    p_X1n = (double*)(mxGPUGetData(X1n));
    
    X2n = mxGPUCopyGPUArray(X2);
    p_X2n = (double*)(mxGPUGetData(X2n));
    
    X3n = mxGPUCopyGPUArray(X3);
    p_X3n = (double*)(mxGPUGetData(X3n));
    
    X4n = mxGPUCopyGPUArray(X4);
    p_X4n = (double*)(mxGPUGetData(X4n));
    
    X5n = mxGPUCopyGPUArray(X5);
    p_X5n = (double*)(mxGPUGetData(X5n));
    
    X6n = mxGPUCopyGPUArray(X6);
    p_X6n = (double*)(mxGPUGetData(X6n));
    
    // RK4 - Step 1
    int32_t curr_free_dim;
    for (int32_t ii=0; ii < num_x_dims[0]; ii++) {
        curr_free_dim = int32_t(p_x_dims_free[ii]);
        if (curr_free_dim == 1) {
            k1_X1 = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(X1), mxGPUGetDimensions(X1), mxGPUGetClassID(X1), mxGPUGetComplexity(X1), MX_GPU_DO_NOT_INITIALIZE);
            p_k1_X1 = (double*)(mxGPUGetData(k1_X1));
            dyn1_mex_continuous << <blocksPerGrid, threadsPerBlock >> > (p_k1_X1, p_X1, p_X2, p_X3, p_X4, p_X5, p_X6, 
                                                                         p_in1, p_in2, p_in3, p_in4, m, I, d, df, l0, 
                                                                         g, p_grid_size, p_active_actions);
            step << <blocksPerGrid, threadsPerBlock >> > (p_X1n, p_X1, p_k1_X1, 0.5 * dt, p_grid_size);
            
        } else if (curr_free_dim == 2) {
            k1_X2 = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(X2), mxGPUGetDimensions(X2), mxGPUGetClassID(X2), mxGPUGetComplexity(X2), MX_GPU_DO_NOT_INITIALIZE);
            p_k1_X2 = (double*)(mxGPUGetData(k1_X2));
            dyn2_mex_continuous << <blocksPerGrid, threadsPerBlock >> > (p_k1_X2, p_X1, p_X2, p_X3, p_X4, p_X5, p_X6, 
                                                                         p_in1, p_in2, p_in3, p_in4, m, I, d, df, l0, 
                                                                         g, p_grid_size, p_active_actions);
            step << <blocksPerGrid, threadsPerBlock >> > (p_X2n, p_X2, p_k1_X2, 0.5 * dt, p_grid_size);
            
        } else if (curr_free_dim == 3) {
            k1_X3 = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(X3), mxGPUGetDimensions(X3), mxGPUGetClassID(X3), mxGPUGetComplexity(X3), MX_GPU_DO_NOT_INITIALIZE);
            p_k1_X3 = (double*)(mxGPUGetData(k1_X3));
            dyn3_mex_continuous << <blocksPerGrid, threadsPerBlock >> > (p_k1_X3, p_X1, p_X2, p_X3, p_X4, p_X5, p_X6, 
                                                                         p_in1, p_in2, p_in3, p_in4, m, I, d, df, l0, 
                                                                         g, p_grid_size, p_active_actions);
            step << <blocksPerGrid, threadsPerBlock >> > (p_X3n, p_X3, p_k1_X3, 0.5 * dt, p_grid_size);
            
        } else if (curr_free_dim == 4) {
            k1_X4 = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(X4), mxGPUGetDimensions(X4), mxGPUGetClassID(X4), mxGPUGetComplexity(X4), MX_GPU_DO_NOT_INITIALIZE);
            p_k1_X4 = (double*)(mxGPUGetData(k1_X4));
            dyn4_mex_continuous << <blocksPerGrid, threadsPerBlock >> > (p_k1_X4, p_X1, p_X2, p_X3, p_X4, p_X5, p_X6, 
                                                                         p_in1, p_in2, p_in3, p_in4, m, I, d, df, l0, 
                                                                         g, p_grid_size, p_active_actions);
            step << <blocksPerGrid, threadsPerBlock >> > (p_X4n, p_X4, p_k1_X4, 0.5 * dt, p_grid_size);
            
        } else if (curr_free_dim == 5) {
            p_k1_X5 = p_X6;
            step << <blocksPerGrid, threadsPerBlock >> > (p_X5n, p_X5, p_k1_X5, 0.5 * dt, p_grid_size);
            
        } else if (curr_free_dim == 6) {
            k1_X6 = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(X6), mxGPUGetDimensions(X6), mxGPUGetClassID(X6), mxGPUGetComplexity(X6), MX_GPU_DO_NOT_INITIALIZE);
            p_k1_X6 = (double*)(mxGPUGetData(k1_X6));
            dyn6_mex_continuous << <blocksPerGrid, threadsPerBlock >> > (p_k1_X6, p_X1, p_X2, p_X3, p_X4, p_X5, p_X6, 
                                                                         p_in1, p_in2, p_in3, p_in4, m, I, d, df, l0, 
                                                                         g, p_grid_size, p_active_actions);
            step << <blocksPerGrid, threadsPerBlock >> > (p_X6n, p_X6, p_k1_X6, 0.5 * dt, p_grid_size);
            
        } 
    }
    
    // RK4 - Step 2
    // Continuous Dynamics
    for (int32_t ii=0; ii < num_x_dims[0]; ii++) {
        curr_free_dim = int32_t(p_x_dims_free[ii]);
        if (curr_free_dim == 1) {
            k2_X1 = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(X1), mxGPUGetDimensions(X1), mxGPUGetClassID(X1), mxGPUGetComplexity(X1), MX_GPU_DO_NOT_INITIALIZE);
            p_k2_X1 = (double*)(mxGPUGetData(k2_X1));
            dyn1_mex_continuous << <blocksPerGrid, threadsPerBlock >> > (p_k2_X1, p_X1n, p_X2n, p_X3n, p_X4n, p_X5n, p_X6n, 
                                                                         p_in1, p_in2, p_in3, p_in4, m, I, d, df, l0, 
                                                                         g, p_grid_size, p_active_actions);
        } else if (curr_free_dim == 2) {
            k2_X2 = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(X2), mxGPUGetDimensions(X2), mxGPUGetClassID(X2), mxGPUGetComplexity(X2), MX_GPU_DO_NOT_INITIALIZE);
            p_k2_X2 = (double*)(mxGPUGetData(k2_X2));
            dyn2_mex_continuous << <blocksPerGrid, threadsPerBlock >> > (p_k2_X2, p_X1n, p_X2n, p_X3n, p_X4n, p_X5n, p_X6n, 
                                                                         p_in1, p_in2, p_in3, p_in4, m, I, d, df, l0, 
                                                                         g, p_grid_size, p_active_actions);
        } else if (curr_free_dim == 3) {
            k2_X3 = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(X3), mxGPUGetDimensions(X3), mxGPUGetClassID(X3), mxGPUGetComplexity(X3), MX_GPU_DO_NOT_INITIALIZE);
            p_k2_X3 = (double*)(mxGPUGetData(k2_X3));
            dyn3_mex_continuous << <blocksPerGrid, threadsPerBlock >> > (p_k2_X3, p_X1n, p_X2n, p_X3n, p_X4n, p_X5n, p_X6n, 
                                                                         p_in1, p_in2, p_in3, p_in4, m, I, d, df, l0, 
                                                                         g, p_grid_size, p_active_actions);
        } else if (curr_free_dim == 4) {
            k2_X4 = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(X4), mxGPUGetDimensions(X4), mxGPUGetClassID(X4), mxGPUGetComplexity(X4), MX_GPU_DO_NOT_INITIALIZE);
            p_k2_X4 = (double*)(mxGPUGetData(k2_X4));
            dyn4_mex_continuous << <blocksPerGrid, threadsPerBlock >> > (p_k2_X4, p_X1n, p_X2n, p_X3n, p_X4n, p_X5n, p_X6n, 
                                                                         p_in1, p_in2, p_in3, p_in4, m, I, d, df, l0, 
                                                                         g, p_grid_size, p_active_actions);
        } else if (curr_free_dim == 5) {
            k2_X5 = mxGPUCopyGPUArray(X6n);
            p_k2_X5 = (double*)(mxGPUGetData(k2_X5));
            
        } else if (curr_free_dim == 6) {
            k2_X6 = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(X6), mxGPUGetDimensions(X6), mxGPUGetClassID(X6), mxGPUGetComplexity(X6), MX_GPU_DO_NOT_INITIALIZE);
            p_k2_X6 = (double*)(mxGPUGetData(k2_X6));
            dyn6_mex_continuous << <blocksPerGrid, threadsPerBlock >> > (p_k2_X6, p_X1n, p_X2n, p_X3n, p_X4n, p_X5n, p_X6n, 
                                                                         p_in1, p_in2, p_in3, p_in4, m, I, d, df, l0, 
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
            step << <blocksPerGrid, threadsPerBlock >> > (p_X4n, p_X4, p_k2_X4, 0.5 * dt, p_grid_size);
            
        } else if (curr_free_dim == 5) {
            step << <blocksPerGrid, threadsPerBlock >> > (p_X5n, p_X5, p_k2_X5, 0.5 * dt, p_grid_size);
            
        } else if (curr_free_dim == 6) {
            step << <blocksPerGrid, threadsPerBlock >> > (p_X6n, p_X6, p_k2_X6, 0.5 * dt, p_grid_size);
            
        } 
    }
    
    // RK4 - Step 3
    // Continuous Dynamics
    for (int32_t ii=0; ii < num_x_dims[0]; ii++) {
        curr_free_dim = int32_t(p_x_dims_free[ii]);
        if (curr_free_dim == 1) {
            k3_X1 = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(X1), mxGPUGetDimensions(X1), mxGPUGetClassID(X1), mxGPUGetComplexity(X1), MX_GPU_DO_NOT_INITIALIZE);
            p_k3_X1 = (double*)(mxGPUGetData(k3_X1));
            dyn1_mex_continuous << <blocksPerGrid, threadsPerBlock >> > (p_k3_X1, p_X1n, p_X2n, p_X3n, p_X4n, p_X5n, p_X6n, 
                                                                         p_in1, p_in2, p_in3, p_in4, m, I, d, df, l0, 
                                                                         g, p_grid_size, p_active_actions);
        } else if (curr_free_dim == 2) {
            k3_X2 = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(X2), mxGPUGetDimensions(X2), mxGPUGetClassID(X2), mxGPUGetComplexity(X2), MX_GPU_DO_NOT_INITIALIZE);
            p_k3_X2 = (double*)(mxGPUGetData(k3_X2));
            dyn2_mex_continuous << <blocksPerGrid, threadsPerBlock >> > (p_k3_X2, p_X1n, p_X2n, p_X3n, p_X4n, p_X5n, p_X6n, 
                                                                         p_in1, p_in2, p_in3, p_in4, m, I, d, df, l0, 
                                                                         g, p_grid_size, p_active_actions);
        } else if (curr_free_dim == 3) {
            k3_X3 = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(X3), mxGPUGetDimensions(X3), mxGPUGetClassID(X3), mxGPUGetComplexity(X3), MX_GPU_DO_NOT_INITIALIZE);
            p_k3_X3 = (double*)(mxGPUGetData(k3_X3));
            dyn3_mex_continuous << <blocksPerGrid, threadsPerBlock >> > (p_k3_X3, p_X1n, p_X2n, p_X3n, p_X4n, p_X5n, p_X6n, 
                                                                         p_in1, p_in2, p_in3, p_in4, m, I, d, df, l0, 
                                                                         g, p_grid_size, p_active_actions);
        } else if (curr_free_dim == 4) {
            k3_X4 = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(X4), mxGPUGetDimensions(X4), mxGPUGetClassID(X4), mxGPUGetComplexity(X4), MX_GPU_DO_NOT_INITIALIZE);
            p_k3_X4 = (double*)(mxGPUGetData(k3_X4));
            dyn4_mex_continuous << <blocksPerGrid, threadsPerBlock >> > (p_k3_X4, p_X1n, p_X2n, p_X3n, p_X4n, p_X5n, p_X6n, 
                                                                         p_in1, p_in2, p_in3, p_in4, m, I, d, df, l0, 
                                                                         g, p_grid_size, p_active_actions);
        } else if (curr_free_dim == 5) {
            k3_X5 = mxGPUCopyGPUArray(X6n);
            p_k3_X5 = (double*)(mxGPUGetData(k3_X5));
            
        } else if (curr_free_dim == 6) {
            k3_X6 = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(X6), mxGPUGetDimensions(X6), mxGPUGetClassID(X6), mxGPUGetComplexity(X6), MX_GPU_DO_NOT_INITIALIZE);
            p_k3_X6 = (double*)(mxGPUGetData(k3_X6));
            dyn6_mex_continuous << <blocksPerGrid, threadsPerBlock >> > (p_k3_X6, p_X1n, p_X2n, p_X3n, p_X4n, p_X5n, p_X6n, 
                                                                         p_in1, p_in2, p_in3, p_in4, m, I, d, df, l0, 
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
            step << <blocksPerGrid, threadsPerBlock >> > (p_X4n, p_X4, p_k3_X4, dt, p_grid_size);
            
        } else if (curr_free_dim == 5) {
            step << <blocksPerGrid, threadsPerBlock >> > (p_X5n, p_X5, p_k3_X5, dt, p_grid_size);
            
        } else if (curr_free_dim == 6) {
            step << <blocksPerGrid, threadsPerBlock >> > (p_X6n, p_X6, p_k3_X6, dt, p_grid_size);
            
        }
    }
    
    // RK4 - Step 4
    // Continuous Dynamics
    for (int32_t ii=0; ii < num_x_dims[0]; ii++) {
        curr_free_dim = int32_t(p_x_dims_free[ii]);
        if (curr_free_dim == 1) {
            k4_X1 = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(X1), mxGPUGetDimensions(X1), mxGPUGetClassID(X1), mxGPUGetComplexity(X1), MX_GPU_DO_NOT_INITIALIZE);
            p_k4_X1 = (double*)(mxGPUGetData(k4_X1));
            dyn1_mex_continuous << <blocksPerGrid, threadsPerBlock >> > (p_k4_X1, p_X1n, p_X2n, p_X3n, p_X4n, p_X5n, p_X6n, 
                                                                         p_in1, p_in2, p_in3, p_in4, m, I, d, df, l0, 
                                                                         g, p_grid_size, p_active_actions);
        } else if (curr_free_dim == 2) {
            k4_X2 = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(X2), mxGPUGetDimensions(X2), mxGPUGetClassID(X2), mxGPUGetComplexity(X2), MX_GPU_DO_NOT_INITIALIZE);
            p_k4_X2 = (double*)(mxGPUGetData(k4_X2));
            dyn2_mex_continuous << <blocksPerGrid, threadsPerBlock >> > (p_k4_X2, p_X1n, p_X2n, p_X3n, p_X4n, p_X5n, p_X6n, 
                                                                         p_in1, p_in2, p_in3, p_in4, m, I, d, df, l0, 
                                                                         g, p_grid_size, p_active_actions);
        } else if (curr_free_dim == 3) {
            k4_X3 = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(X3), mxGPUGetDimensions(X3), mxGPUGetClassID(X3), mxGPUGetComplexity(X3), MX_GPU_DO_NOT_INITIALIZE);
            p_k4_X3 = (double*)(mxGPUGetData(k4_X3));
            dyn3_mex_continuous << <blocksPerGrid, threadsPerBlock >> > (p_k4_X3, p_X1n, p_X2n, p_X3n, p_X4n, p_X5n, p_X6n, 
                                                                         p_in1, p_in2, p_in3, p_in4, m, I, d, df, l0, 
                                                                         g, p_grid_size, p_active_actions);
        } else if (curr_free_dim == 4) {
            k4_X4 = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(X4), mxGPUGetDimensions(X4), mxGPUGetClassID(X4), mxGPUGetComplexity(X4), MX_GPU_DO_NOT_INITIALIZE);
            p_k4_X4 = (double*)(mxGPUGetData(k4_X4));
            dyn4_mex_continuous << <blocksPerGrid, threadsPerBlock >> > (p_k4_X4, p_X1n, p_X2n, p_X3n, p_X4n, p_X5n, p_X6n, 
                                                                         p_in1, p_in2, p_in3, p_in4, m, I, d, df, l0, 
                                                                         g, p_grid_size, p_active_actions);
        } else if (curr_free_dim == 5) {
            k4_X5 = mxGPUCopyGPUArray(X6n);
            p_k4_X5 = (double*)(mxGPUGetData(k4_X5));
            
        } else if (curr_free_dim == 6) {
            k4_X6 = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(X6), mxGPUGetDimensions(X6), mxGPUGetClassID(X6), mxGPUGetComplexity(X6), MX_GPU_DO_NOT_INITIALIZE);
            p_k4_X6 = (double*)(mxGPUGetData(k4_X6));
            dyn6_mex_continuous << <blocksPerGrid, threadsPerBlock >> > (p_k4_X6, p_X1n, p_X2n, p_X3n, p_X4n, p_X5n, p_X6n, 
                                                                         p_in1, p_in2, p_in3, p_in4, m, I, d, df, l0, 
                                                                         g, p_grid_size, p_active_actions);
        }
    }

    // RK4 Ouput
    for (int32_t ii=0; ii < num_x_dims[0]; ii++) {
        curr_free_dim = int32_t(p_x_dims_free[ii]);
        if (curr_free_dim == 1) {
            rk4_wbounds << <blocksPerGrid, threadsPerBlock >> > (p_X1n, p_X1, p_k1_X1, p_k2_X1, p_k3_X1, p_k4_X1, dt, p_limits[0], p_limits[6], p_grid_size);
            mxGPUDestroyGPUArray(k1_X1);
            mxGPUDestroyGPUArray(k2_X1);
            mxGPUDestroyGPUArray(k3_X1);
            mxGPUDestroyGPUArray(k4_X1);
        } else if (curr_free_dim == 2) {
            rk4_wbounds << <blocksPerGrid, threadsPerBlock >> > (p_X2n, p_X2, p_k1_X2, p_k2_X2, p_k3_X2, p_k4_X2, dt, p_limits[1], p_limits[7], p_grid_size);
            mxGPUDestroyGPUArray(k1_X2);
            mxGPUDestroyGPUArray(k2_X2);
            mxGPUDestroyGPUArray(k3_X2);
            mxGPUDestroyGPUArray(k4_X2);
        } else if (curr_free_dim == 3) {
            rk4_wbounds << <blocksPerGrid, threadsPerBlock >> > (p_X3n, p_X3, p_k1_X3, p_k2_X3, p_k3_X3, p_k4_X3, dt, p_limits[2], p_limits[8], p_grid_size);
            mxGPUDestroyGPUArray(k1_X3);
            mxGPUDestroyGPUArray(k2_X3);
            mxGPUDestroyGPUArray(k3_X3);
            mxGPUDestroyGPUArray(k4_X3);
        } else if (curr_free_dim == 4) {
            rk4_wbounds << <blocksPerGrid, threadsPerBlock >> > (p_X4n, p_X4, p_k1_X4, p_k2_X4, p_k3_X4, p_k4_X4, dt, p_limits[3], p_limits[9], p_grid_size);
            mxGPUDestroyGPUArray(k1_X4);
            mxGPUDestroyGPUArray(k2_X4);
            mxGPUDestroyGPUArray(k3_X4);
            mxGPUDestroyGPUArray(k4_X4);
        } else if (curr_free_dim == 5) {
            rk4_wbounds << <blocksPerGrid, threadsPerBlock >> > (p_X5n, p_X5, p_k1_X5, p_k2_X5, p_k3_X5, p_k4_X5, dt, p_limits[4], p_limits[10], p_grid_size);
            mxGPUDestroyGPUArray(k2_X5);
            mxGPUDestroyGPUArray(k3_X5);
            mxGPUDestroyGPUArray(k4_X5);
        } else if (curr_free_dim == 6) {
            rk4_wbounds << <blocksPerGrid, threadsPerBlock >> > (p_X6n, p_X6, p_k1_X6, p_k2_X6, p_k3_X6, p_k4_X6, dt, p_limits[5], p_limits[11], p_grid_size);
            mxGPUDestroyGPUArray(k1_X6);
            mxGPUDestroyGPUArray(k2_X6);
            mxGPUDestroyGPUArray(k3_X6);
            mxGPUDestroyGPUArray(k4_X6);
        }
    }
 
    /* Wrap the result up as a MATLAB gpuArray for return. */
    plhs[0] = mxGPUCreateMxArrayOnGPU(X1n);
    plhs[1] = mxGPUCreateMxArrayOnGPU(X2n);
    plhs[2] = mxGPUCreateMxArrayOnGPU(X3n);
    plhs[3] = mxGPUCreateMxArrayOnGPU(X4n);
    plhs[4] = mxGPUCreateMxArrayOnGPU(X5n);
    plhs[5] = mxGPUCreateMxArrayOnGPU(X6n);

    /*
     * The mxGPUArray pointers are host-side structures that refer to device
     * data. These must be destroyed before leaving the MEX function.
     */
    mxGPUDestroyGPUArray(X1);
    mxGPUDestroyGPUArray(X2);
    mxGPUDestroyGPUArray(X3);
    mxGPUDestroyGPUArray(X4);
    mxGPUDestroyGPUArray(X5);
    mxGPUDestroyGPUArray(X6);
    mxGPUDestroyGPUArray(in1);
    mxGPUDestroyGPUArray(in2);
    mxGPUDestroyGPUArray(in3);
    mxGPUDestroyGPUArray(in4);
    mxGPUDestroyGPUArray(grid_size);
    mxGPUDestroyGPUArray(active_actions);
    mxGPUDestroyGPUArray(X1n);
    mxGPUDestroyGPUArray(X2n);
    mxGPUDestroyGPUArray(X3n);
    mxGPUDestroyGPUArray(X4n);
    mxGPUDestroyGPUArray(X5n);
    mxGPUDestroyGPUArray(X6n);
}
