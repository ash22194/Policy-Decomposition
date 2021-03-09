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
    double* const x1, double* const x2, double* const x3, double* const x4,
    double* const dx1, double* const dx2, double* const dx3, double* const dx4, // input states
    double const* const in1, double const* const in2, double const* const in3, double const* const in4, // input actions
    const double m1, const double m2, const double m3, const double m4,
    const double l1, const double l2, const double l3, const double l4, const double g, 
    int32_t const* const grid_size, int32_t const* const active_actions)
{
    int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x * blockDim.y
        + (threadIdx.x + threadIdx.y * blockDim.x);
    int num_threads = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
    int num_elements = grid_size[0] * grid_size[1] * grid_size[2] * grid_size[3]
                        * grid_size[4] * grid_size[5] * grid_size[6] * grid_size[7];
    double dth1;
    int index5 = (grid_size[4] > 1) ? index : 0;

    while (index < num_elements)
    {
        dth1 = dx1[index5];
        d_x1[index] = dth1;
        
        index = index + num_threads;
        index5 = (grid_size[4] > 1) ? index : 0;
    }
}

__global__ void dyn2_mex_continuous(double* const d_x2, // outputs
    double* const x1, double* const x2, double* const x3, double* const x4,
    double* const dx1, double* const dx2, double* const dx3, double* const dx4, // input states
    double const* const in1, double const* const in2, double const* const in3, double const* const in4, // input actions
    const double m1, const double m2, const double m3, const double m4,
    const double l1, const double l2, const double l3, const double l4, const double g, 
    int32_t const* const grid_size, int32_t const* const active_actions)
{
    int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x * blockDim.y
        + (threadIdx.x + threadIdx.y * blockDim.x);
    int num_threads = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
    int num_elements = grid_size[0] * grid_size[1] * grid_size[2] * grid_size[3]
                        * grid_size[4] * grid_size[5] * grid_size[6] * grid_size[7];
    double dth2;
    int index6 = (grid_size[5] > 1) ? index : 0;

    while (index < num_elements)
    {
        dth2 = dx2[index6];
        d_x2[index] = dth2;
        
        index = index + num_threads;
        index6 = (grid_size[5] > 1) ? index : 0;
    }
}

__global__ void dyn3_mex_continuous(double* const d_x3, // outputs
    double* const x1, double* const x2, double* const x3, double* const x4,
    double* const dx1, double* const dx2, double* const dx3, double* const dx4, // input states
    double const* const in1, double const* const in2, double const* const in3, double const* const in4, // input actions
    const double m1, const double m2, const double m3, const double m4,
    const double l1, const double l2, const double l3, const double l4, const double g, 
    int32_t const* const grid_size, int32_t const* const active_actions)
{
    int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x * blockDim.y
        + (threadIdx.x + threadIdx.y * blockDim.x);
    int num_threads = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
    int num_elements = grid_size[0] * grid_size[1] * grid_size[2] * grid_size[3]
                        * grid_size[4] * grid_size[5] * grid_size[6] * grid_size[7];
    double dth3;
    int index7 = (grid_size[6] > 1) ? index : 0;

    while (index < num_elements)
    {
        dth3 = dx3[index7];
        d_x3[index] = dth3;
        
        index = index + num_threads;
        index7 = (grid_size[6] > 1) ? index : 0;
    }
}

__global__ void dyn4_mex(double* const d_x4, // outputs
    double* const x1, double* const x2, double* const x3, double* const x4,
    double* const dx1, double* const dx2, double* const dx3, double* const dx4, // input states
    double const* const in1, double const* const in2, double const* const in3, double const* const in4, // input actions
    const double m1, const double m2, const double m3, const double m4,
    const double l1, const double l2, const double l3, const double l4, const double g, 
    int32_t const* const grid_size, int32_t const* const active_actions)
{
    int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x * blockDim.y
        + (threadIdx.x + threadIdx.y * blockDim.x);
    int num_threads = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
    int num_elements = grid_size[0] * grid_size[1] * grid_size[2] * grid_size[3]
                        * grid_size[4] * grid_size[5] * grid_size[6] * grid_size[7];
    double dth4;
    int index8 = (grid_size[7] > 1) ? index : 0;

    while (index < num_elements)
    {
        dth4 = dx4[index8];
        d_x4[index] = dth4;
        
        index = index + num_threads;
        index8 = (grid_size[7] > 1) ? index : 0;
    }
}

__global__ void dyn5_mex_continuous(double* const d_dx1, // outputs
    double* const x1, double* const x2, double* const x3, double* const x4,
    double* const dx1, double* const dx2, double* const dx3, double* const dx4, // input states
    double const* const in1, double const* const in2, double const* const in3, double const* const in4, // input actions
    const double m1, const double m2, const double m3, const double m4,
    const double l1, const double l2, const double l3, const double l4, const double g, 
    int32_t const* const grid_size, int32_t const* const active_actions)
{
    int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x * blockDim.y
        + (threadIdx.x + threadIdx.y * blockDim.x);
    int num_threads = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
    int num_elements = grid_size[0] * grid_size[1] * grid_size[2] * grid_size[3]
                        * grid_size[4] * grid_size[5] * grid_size[6] * grid_size[7];
    double th1, th2, th3, th4, dth1, dth2, dth3, dth4, u1, u2, u3, u4;

    int index1 = (grid_size[0] > 1) ? index : 0;
    int index2 = (grid_size[1] > 1) ? index : 0;
    int index3 = (grid_size[2] > 1) ? index : 0;
    int index4 = (grid_size[3] > 1) ? index : 0;
    int index5 = (grid_size[4] > 1) ? index : 0;
    int index6 = (grid_size[5] > 1) ? index : 0;
    int index7 = (grid_size[6] > 1) ? index : 0;
    int index8 = (grid_size[7] > 1) ? index : 0;

    int uindex1 = (active_actions[0] == 1) ? index : 0;
    int uindex2 = (active_actions[1] == 1) ? index : 0;
    int uindex3 = (active_actions[2] == 1) ? index : 0;
    int uindex4 = (active_actions[3] == 1) ? index : 0;

    while (index < num_elements)
    {
        th1 = x1[index1];
        th2 = x2[index2];
        th3 = x3[index3];
        th4 = x4[index4];

        dth1 = dx1[index5];
        dth2 = dx2[index6];
        dth3 = dx3[index7];
        dth4 = dx4[index8];

        u1 = in1[uindex1];
        u2 = in2[uindex2];
        u3 = in3[uindex3];
        u4 = in4[uindex4];

        double t2 = cos(th1);
        double t3 = cos(th2);
        double t4 = cos(th3);
        double t5 = cos(th4);
        double t6 = sin(th1);
        double t7 = sin(th2);
        double t8 = sin(th3);
        double t9 = sin(th4);
        double t10 = th1 + th2;
        double t11 = th2 + th3;
        double t12 = th3 + th4;
        double t13 = dth1 * dth1;
        double t14 = dth2 * dth2;
        double t15 = dth3 * dth3;
        double t16 = dth4 * dth4;
        double t17 = l1 * l1;
//         double t18 = t17 * l1;
        double t19 = l2 * l2;
//         double t20 = t19 * l2;
        double t21 = l3 * l3;
//         double t22 = t21 * l3;
        double t23 = l4 * l4;
//         double t24 = t23 * l4;
        double t25 = m2 * m2;
        double t26 = m3 * m3;
        double t27 = t26 * m3;
        double t28 = m4 * m4;
//         double t29 = t28 * m4;
        double t30 = th2 * 2.0;
        double t31 = th3 * 2.0;
        double t32 = th4 * 2.0;
//         double t46 = 1.0 / l1;
        double t48 = 1.0 / l2;
        double t50 = 1.0 / l3;
        double t52 = 1.0 / l4;
        double t53 = -th1;
        double t54 = -th2;
        double t55 = -th3;
        double t57 = -th4;
        double t125 = m1 * m2 * m3 * 128.0;
        double t126 = m1 * m2 * m4 * 240.0;
        double t163 = m1 * m3 * m4 * 600.0;
        double t164 = m2 * m3 * m4 * 1500.0;
        double t33 = cos(t30);
        double t34 = cos(t31);
        double t35 = cos(t32);
        double t36 = sin(t30);
        double t37 = sin(t31);
        double t38 = sin(t32);
        double t39 = cos(t11);
        double t40 = cos(t12);
        double t41 = sin(t10);
        double t42 = sin(t11);
        double t43 = sin(t12);
        double t44 = t10 + th3;
        double t45 = t11 + th4;
        double t47 = 1.0 / t17;
//         double t49 = 1.0 / t19;
//         double t51 = 1.0 / t21;
        double t56 = -t31;
        double t58 = -t32;
        double t62 = t10 + t12;
        double t63 = t10 + th2;
        double t64 = t31 + th1;
        double t65 = t32 + th1;
        double t66 = t11 + th3;
        double t67 = t11 + th2;
        double t68 = t32 + th2;
        double t69 = t30 + th4;
        double t70 = t12 + th4;
        double t71 = t12 + th3;
        double t87 = t10 + t31;
        double t88 = t10 + t32;
        double t89 = t11 + t32;
        double t92 = t54 + th1;
        double t94 = t55 + th2;
        double t97 = t57 + th3;
        double t98 = t11 * 2.0;
        double t99 = t30 + t32;
        double t100 = t12 * 2.0;
        double t110 = t10 + t55;
        double t112 = t11 + t53;
        double t114 = t11 + t57;
        double t116 = t12 + t54;
        double t117 = t27 * 144.0;
        double t128 = t30 + t57;
        double t156 = m1 * t28 * 144.0;
        double t157 = m3 * t28 * 144.0;
        double t158 = m1 * t26 * 240.0;
        double t159 = m3 * t25 * 240.0;
        double t160 = m2 * t28 * 360.0;
        double t161 = m4 * t25 * 450.0;
        double t162 = m4 * t26 * 450.0;
        double t179 = m2 * t26 * 600.0;
        double t211 = t10 - t12;
        double t213 = -t11 + th1 + th4;
        double t59 = cos(t45);
        double t60 = sin(t44);
        double t61 = sin(t45);
        double t72 = cos(t66);
        double t73 = cos(t67);
        double t74 = cos(t68);
        double t75 = cos(t69);
        double t76 = cos(t70);
        double t77 = cos(t71);
        double t78 = sin(t63);
        double t79 = sin(t64);
        double t80 = sin(t65);
        double t81 = sin(t66);
        double t82 = sin(t67);
        double t83 = sin(t68);
        double t84 = sin(t69);
        double t85 = sin(t70);
        double t86 = sin(t71);
        double t90 = t45 + th2;
        double t91 = sin(t62);
        double t93 = t56 + th1;
        double t95 = t58 + th1;
        double t96 = t58 + th2;
        double t101 = cos(t94);
        double t103 = cos(t97);
        double t104 = sin(t92);
        double t106 = sin(t94);
        double t109 = sin(t97);
        double t111 = t92 + th3;
        double t113 = t10 + t58;
        double t115 = t94 + th4;
        double t118 = cos(t89);
        double t120 = sin(t87);
        double t121 = sin(t88);
        double t122 = sin(t89);
        double t124 = t62 + th4;
        double t129 = t30 + t58;
        double t130 = cos(t98);
        double t131 = cos(t99);
        double t132 = cos(t100);
        double t133 = sin(t98);
        double t134 = sin(t99);
        double t135 = sin(t100);
        double t136 = t11 + t44;
        double t137 = t32 + t63;
        double t138 = t100 + th1;
        double t139 = t12 + t45;
        double t140 = t32 + t67;
        double t141 = t11 + t45;
        double t142 = cos(t114);
        double t144 = cos(t116);
        double t145 = sin(t110);
        double t147 = sin(t112);
        double t149 = sin(t114);
        double t151 = sin(t116);
        double t152 = t44 + t57;
        double t153 = t110 + th4;
        double t154 = t12 + t92;
        double t155 = t45 + t53;
        double t169 = cos(t128);
        double t171 = sin(t128);
        double t173 = t53 + t66;
        double t174 = t54 + t65;
        double t175 = t53 + t68;
        double t176 = t58 + t63;
        double t177 = t54 + t70;
        double t178 = t57 + t67;
        double t189 = t12 + t62;
        double t191 = t11 + t89;
        double t200 = t70 + t92;
        double t201 = t53 + t89;
        double t207 = t53 + t100;
        double t208 = m1 * m2 * m4 * t35 * 144.0;
        double t209 = m1 * m3 * m4 * t35 * 216.0;
        double t210 = m1 * m3 * m4 * t34 * 360.0;
        double t212 = t92 + t97;
        double t214 = t57 + t211;
        double t217 = -t27 * t33 * 144.0;
        double t218 = sin(t211);
        double t220 = sin(t213);
        double t222 = m1 * t26 * t34 * 144.0;
        double t223 = m3 * t25 * t33 * 144.0;
        double t226 = m4 * t26 * t35 * 162.0;
        double t227 = m2 * t26 * t34 * 216.0;
        double t228 = m2 * t28 * t33 * 216.0;
        double t229 = m2 * t28 * t34 * 216.0;
        double t230 = m4 * t25 * t33 * 270.0;
        double t231 = m4 * t25 * t35 * 270.0;
        double t232 = m2 * t26 * t33 * 360.0;
        double t237 = m2 * m3 * m4 * t34 * 540.0;
        double t238 = m2 * m3 * m4 * t35 * 540.0;
        double t239 = m2 * m3 * m4 * t33 * 900.0;
        double t243 = - m1 * t28 * t34 * 144.0;
        double t244 = - m3 * t28 * t33 * 144.0;
        double t252 = - m4 * t26 * t33 * 450.0;
        double t102 = cos(t96);
        double t105 = sin(t93);
        double t107 = sin(t95);
        double t108 = sin(t96);
        double t119 = cos(t90);
        double t123 = sin(t90);
        double t127 = sin(t124);
        double t143 = cos(t115);
        double t146 = sin(t111);
        double t148 = sin(t113);
        double t150 = sin(t115);
        double t165 = sin(t152);
        double t166 = sin(t153);
        double t167 = sin(t154);
        double t168 = sin(t155);
        double t170 = cos(t129);
        double t172 = sin(t129);
        double t180 = cos(t139);
        double t181 = cos(t140);
        double t182 = cos(t141);
        double t183 = sin(t136);
        double t184 = sin(t137);
        double t185 = sin(t138);
        double t186 = sin(t139);
        double t187 = sin(t140);
        double t188 = sin(t141);
        double t190 = sin(t189);
        double t192 = cos(t177);
        double t193 = cos(t178);
        double t194 = sin(t173);
        double t195 = sin(t174);
        double t196 = sin(t175);
        double t197 = sin(t176);
        double t198 = sin(t177);
        double t199 = sin(t178);
        double t202 = cos(t191);
        double t203 = sin(t191);
        double t205 = sin(t200);
        double t206 = sin(t201);
        double t215 = sin(t207);
        double t216 = t53 + t139;
        double t219 = sin(t212);
        double t221 = sin(t214);
        double t234 = -t208;
        double t235 = -t209;
        double t236 = -t210;
        double t241 = -t222;
        double t242 = -t223;
        double t245 = -t226;
        double t246 = -t227;
        double t247 = -t228;
        double t248 = -t229;
        double t249 = -t230;
        double t250 = -t231;
        double t251 = -t232;
        double t253 = -t237;
        double t254 = -t238;
        double t255 = -t239;
        double t256 = m1 * m3 * m4 * t132 * 72.0;
        double t257 = m2 * m3 * m4 * t132 * 108.0;
        double t258 = m2 * m3 * m4 * t131 * 162.0;
        double t259 = m2 * m3 * m4 * t130 * 180.0;
        double t260 = m2 * t26 * t130 * 72.0;
        double t261 = m2 * t28 * t130 * 72.0;
        double t262 = m4 * t25 * t131 * 81.0;
        double t263 = m4 * t26 * t131 * 81.0;
        double t240 = sin(t216);
        double t264 = m2 * m3 * m4 * t170 * 162.0;
        double t265 = m4 * t25 * t170 * 81.0;
        double t266 = m4 * t26 * t170 * 81.0;
        double t267 = m2 * m3 * m4 * t202 * 36.0;
        double t268 = -t267;
        double t269 = t117 + t125 + t126 + t156 + t157 + t158 + t159 + t160 + t161 + t162 + t163 + t164 + t179 + t217 + t234 + t235 + t236 + t241 + t242 + t243 + t244 + t245 + t246 + t247 + t248 + t249 + t250 + t251 + t252 + t253 + t254 + t255 + t256 + t257 + t258 + t259 + t260 + t261 + t262 + t263 + t264 + t265 + t266 + t268;
        double t270 = 1.0 / t269;

        d_dx1[index] = t47*t48*t50*t52*t270*(l2*l3*l4*t26*u1*1.2*100.0 - l2*l3*l4*t26*u2*1.2*100.0 + l2*l3*l4*t28*u1*7.2*10.0 - l2*l3*l4*t28*u2*7.2*10.0 - l1*l3*l4*t3*t26*u2*1.2*100.0 + l1*l3*l4*t3*t26*u3*1.2*100.0 - l1*l3*l4*t3*t28*u2*7.2*10.0 + l1*l3*l4*t3*t28*u3*7.2*10.0 - l2*l3*l4*t26*t34*u1*7.2*10.0 + l2*l3*l4*t26*t34*u2*7.2*10.0 - l2*l3*l4*t28*t34*u1*7.2*10.0 + l2*l3*l4*t28*t34*u2*7.2*10.0 + l2*l3*l4*m2*m3*u1*6.4*10.0 - l2*l3*l4*m2*m3*u2*6.4*10.0 + l2*l3*l4*m2*m4*u1*1.2*100.0 - l2*l3*l4*m2*m4*u2*1.2*100.0 + l2*l3*l4*m3*m4*u1*3.0*100.0 - l2*l3*l4*m3*m4*u2*3.0*100.0 - g*l1*l2*l3*l4*t6*t27*2.4*10.0 - l1*l3*l4*m2*m3*t3*u2*9.6*10.0 + l1*l3*l4*m2*m3*t3*u3*9.6*10.0 - l1*l3*l4*m2*m4*t3*u2*1.8*100.0 + l1*l3*l4*m2*m4*t3*u3*1.8*100.0 - l1*l3*l4*m3*m4*t3*u2*3.0*100.0 + l1*l3*l4*m3*m4*t3*u3*3.0*100.0 - l2*l3*l4*m2*m4*t35*u1*7.2*10.0 - l2*l3*l4*m3*m4*t34*u1*1.8*100.0 + l2*l3*l4*m2*m4*t35*u2*7.2*10.0 + l2*l3*l4*m3*m4*t34*u2*1.8*100.0 - l2*l3*l4*m3*m4*t35*u1*1.08*100.0 + l2*l3*l4*m3*m4*t35*u2*1.08*100.0 + l1*l3*l4*t7*t13*t19*t27*4.8*10.0 + l1*l3*l4*t7*t14*t19*t27*4.8*10.0 + l2*l3*l4*t13*t17*t27*t36*2.4*10.0 + l1*l2*l4*t7*t8*t26*u3*2.88*100.0 - l1*l2*l4*t7*t8*t26*u4*2.88*100.0 + l1*l2*l4*t7*t8*t28*u3*1.44*100.0 - l1*l2*l4*t7*t8*t28*u4*1.44*100.0 + l1*l3*l4*t3*t26*t34*u2*7.2*10.0 - l1*l3*l4*t3*t26*t34*u3*7.2*10.0 + l1*l3*l4*t3*t28*t34*u2*7.2*10.0 - l1*l3*l4*t3*t28*t34*u3*7.2*10.0 - l1*l3*l4*t7*t26*t37*u2*7.2*10.0 + l1*l3*l4*t7*t26*t37*u3*7.2*10.0 - l1*l3*l4*t7*t28*t37*u2*7.2*10.0 + l1*l3*l4*t7*t28*t37*u3*7.2*10.0 + dth1*dth2*l1*l3*l4*t7*t19*t27*9.6*10.0 - g*l1*l2*l3*l4*m1*t6*t26*6.0*10.0 - g*l1*l2*l3*l4*m2*t6*t26*1.0*100.0 - g*l1*l2*l3*l4*m3*t6*t25*4.0*10.0 - g*l1*l2*l3*l4*m1*t6*t28*3.6*10.0 - g*l1*l2*l3*l4*m4*t6*t25*7.5*10.0 - g*l1*l2*l3*l4*m2*t6*t28*6.0*10.0 - g*l1*l2*l3*l4*m4*t6*t26*7.5*10.0 - g*l1*l2*l3*l4*m3*t6*t28*2.4*10.0 + g*l1*l2*l3*l4*t2*t27*t36*2.4*10.0 + g*l1*l2*l3*l4*t6*t27*t33*2.4*10.0 + l1*l2*l4*m2*m3*t3*t4*u3*4.8*10.0 - l1*l2*l4*m2*m3*t3*t4*u4*4.8*10.0 + l1*l2*l4*m2*m4*t3*t4*u3*6.0*10.0 - l1*l2*l4*m2*m4*t3*t4*u4*6.0*10.0 + l1*l2*l4*m2*m3*t7*t8*u3*9.6*10.0 - l1*l2*l4*m2*m3*t7*t8*u4*9.6*10.0 + l1*l2*l4*m2*m4*t7*t8*u3*1.2*100.0 - l1*l2*l4*m2*m4*t7*t8*u4*1.2*100.0 + l1*l2*l4*m3*m4*t7*t8*u3*5.4*100.0 - l1*l2*l4*m3*m4*t7*t8*u4*5.4*100.0 + l1*l3*l4*m2*m4*t3*t35*u2*1.08*100.0 + l1*l3*l4*m3*m4*t3*t34*u2*1.8*100.0 - l1*l3*l4*m2*m4*t3*t35*u3*1.08*100.0 - l1*l3*l4*m3*m4*t3*t34*u3*1.8*100.0 + l1*l3*l4*m3*m4*t3*t35*u2*1.08*100.0 - l1*l3*l4*m3*m4*t3*t35*u3*1.08*100.0 - l1*l3*l4*m3*m4*t7*t37*u2*1.8*100.0 + l1*l3*l4*m3*m4*t7*t37*u3*1.8*100.0 + l2*l3*l4*m3*m4*t34*t35*u1*3.6*10.0 - l2*l3*l4*m3*m4*t34*t35*u2*3.6*10.0 - l2*l3*l4*m3*m4*t37*t38*u1*3.6*10.0 + l2*l3*l4*m3*m4*t37*t38*u2*3.6*10.0 + l1*l3*l4*m2*t7*t13*t19*t26*1.0*100.0 + l1*l3*l4*m3*t7*t13*t19*t25*3.2*10.0 + l1*l3*l4*m2*t7*t14*t19*t26*1.0*100.0 + l1*l3*l4*m3*t7*t14*t19*t25*3.2*10.0 + l1*l3*l4*m4*t7*t13*t19*t25*6.0*10.0 + l1*l3*l4*m2*t7*t13*t19*t28*6.0*10.0 + l1*l3*l4*m4*t7*t13*t19*t26*1.5*100.0 + l1*l3*l4*m4*t7*t14*t19*t25*6.0*10.0 + l1*l3*l4*m2*t7*t14*t19*t28*6.0*10.0 + l1*l3*l4*m3*t7*t13*t19*t28*4.8*10.0 + l1*l3*l4*m4*t7*t14*t19*t26*1.5*100.0 + l1*l3*l4*m3*t7*t14*t19*t28*4.8*10.0 + l2*l3*l4*m2*t13*t17*t26*t36*6.0*10.0 + l2*l3*l4*m3*t13*t17*t25*t36*2.4*10.0 + l2*l3*l4*m4*t13*t17*t25*t36*4.5*10.0 + l2*l3*l4*m2*t13*t17*t28*t36*3.6*10.0 + l2*l3*l4*m4*t13*t17*t26*t36*7.5*10.0 + l2*l3*l4*m3*t13*t17*t28*t36*2.4*10.0 + l1*l2*l4*t4*t7*t13*t21*t27*2.4*10.0 + l1*l2*l4*t4*t7*t14*t21*t27*2.4*10.0 + l1*l2*l4*t4*t7*t15*t21*t27*2.4*10.0 + l1*l2*l3*t4*t7*t9*t26*u4*7.2*10.0 - l1*l2*l3*t5*t7*t8*t26*u4*1.44*100.0 - g*l1*l2*l3*l4*m1*m2*m3*t6*3.2*10.0 - g*l1*l2*l3*l4*m1*m2*m4*t6*6.0*10.0 - g*l1*l2*l3*l4*m1*m3*m4*t6*1.5*100.0 - g*l1*l2*l3*l4*m2*m3*m4*t6*2.5*100.0 + dth1*dth2*l1*l3*l4*m2*t7*t19*t26*2.0*100.0 + dth1*dth2*l1*l3*l4*m3*t7*t19*t25*6.4*10.0 + dth1*dth2*l1*l3*l4*m4*t7*t19*t25*1.2*100.0 + dth1*dth2*l1*l3*l4*m2*t7*t19*t28*1.2*100.0 + dth1*dth2*l1*l3*l4*m4*t7*t19*t26*3.0*100.0 + dth1*dth2*l1*l3*l4*m3*t7*t19*t28*9.6*10.0 + dth1*dth2*l1*l2*l4*t4*t7*t21*t27*4.8*10.0 + dth1*dth3*l1*l2*l4*t4*t7*t21*t27*4.8*10.0 + dth2*dth3*l1*l2*l4*t4*t7*t21*t27*4.8*10.0 + g*l1*l2*l3*l4*m2*t2*t26*t36*6.0*10.0 + g*l1*l2*l3*l4*m3*t2*t25*t36*2.4*10.0 + g*l1*l2*l3*l4*m1*t6*t26*t34*3.6*10.0 + g*l1*l2*l3*l4*m2*t6*t26*t33*6.0*10.0 + g*l1*l2*l3*l4*m3*t6*t25*t33*2.4*10.0 + g*l1*l2*l3*l4*m4*t2*t25*t36*4.5*10.0 + g*l1*l2*l3*l4*m2*t2*t28*t36*3.6*10.0 + g*l1*l2*l3*l4*m2*t6*t26*t34*3.6*10.0 + g*l1*l2*l3*l4*m4*t2*t26*t36*7.5*10.0 + g*l1*l2*l3*l4*m4*t6*t25*t33*4.5*10.0 + g*l1*l2*l3*l4*m1*t6*t28*t34*3.6*10.0 + g*l1*l2*l3*l4*m2*t6*t28*t33*3.6*10.0 + g*l1*l2*l3*l4*m3*t2*t28*t36*2.4*10.0 + g*l1*l2*l3*l4*m4*t6*t26*t33*7.5*10.0 + g*l1*l2*l3*l4*m2*t6*t28*t34*3.6*10.0 + g*l1*l2*l3*l4*m3*t6*t28*t33*2.4*10.0 + g*l1*l2*l3*l4*m4*t6*t25*t35*4.5*10.0 + g*l1*l2*l3*l4*m4*t6*t26*t35*2.7*10.0 + l1*l3*l4*m2*m3*m4*t7*t13*t19*2.5*100.0 + l1*l3*l4*m2*m3*m4*t7*t14*t19*2.5*100.0 + l2*l3*l4*m2*m3*m4*t13*t17*t36*1.5*100.0 - l1*l2*l3*m2*m3*t3*t4*t5*u4*2.4*10.0 - l1*l2*l3*m2*m3*t3*t8*t9*u4*4.8*10.0 + l1*l2*l3*m2*m3*t4*t7*t9*u4*9.6*10.0 - l1*l2*l3*m2*m3*t5*t7*t8*u4*4.8*10.0 - l1*l2*l3*m2*m4*t3*t8*t9*u4*1.44*100.0 + l1*l2*l3*m2*m4*t4*t7*t9*u4*2.88*100.0 + l1*l2*l3*m3*m4*t4*t7*t9*u4*2.88*100.0 - l1*l2*l3*m3*m4*t5*t7*t8*u4*1.44*100.0 - l1*l2*l4*m2*m4*t3*t4*t35*u3*3.6*10.0 + l1*l2*l4*m2*m4*t3*t4*t35*u4*3.6*10.0 + l1*l2*l4*m2*m4*t3*t8*t38*u3*3.6*10.0 - l1*l2*l4*m2*m4*t4*t7*t38*u3*7.2*10.0 - l1*l2*l4*m2*m4*t3*t8*t38*u4*3.6*10.0 + l1*l2*l4*m2*m4*t4*t7*t38*u4*7.2*10.0 - l1*l2*l4*m2*m4*t7*t8*t35*u3*7.2*10.0 - l1*l2*l4*m3*m4*t4*t7*t38*u3*1.08*100.0 + l1*l2*l4*m2*m4*t7*t8*t35*u4*7.2*10.0 + l1*l2*l4*m3*m4*t4*t7*t38*u4*1.08*100.0 - l1*l2*l4*m3*m4*t7*t8*t35*u3*1.08*100.0 + l1*l2*l4*m3*m4*t7*t8*t35*u4*1.08*100.0 - l1*l3*l4*m3*m4*t3*t34*t35*u2*3.6*10.0 + l1*l3*l4*m3*m4*t3*t34*t35*u3*3.6*10.0 + l1*l3*l4*m3*m4*t3*t37*t38*u2*3.6*10.0 - l1*l3*l4*m3*m4*t3*t37*t38*u3*3.6*10.0 + l1*l3*l4*m3*m4*t7*t34*t38*u2*3.6*10.0 + l1*l3*l4*m3*m4*t7*t35*t37*u2*3.6*10.0 - l1*l3*l4*m3*m4*t7*t34*t38*u3*3.6*10.0 - l1*l3*l4*m3*m4*t7*t35*t37*u3*3.6*10.0 - l1*l2*l4*m2*t3*t8*t13*t21*t26*1.6*10.0 + l1*l2*l4*m2*t4*t7*t13*t21*t26*3.2*10.0 - l1*l2*l4*m2*t3*t8*t14*t21*t26*1.6*10.0 + l1*l2*l4*m2*t4*t7*t14*t21*t26*3.2*10.0 - l1*l2*l4*m2*t3*t8*t13*t21*t28*2.4*10.0 - l1*l2*l4*m2*t3*t8*t15*t21*t26*1.6*10.0 + l1*l2*l4*m2*t4*t7*t13*t21*t28*4.8*10.0 + l1*l2*l4*m2*t4*t7*t15*t21*t26*3.2*10.0 + l1*l2*l4*m4*t4*t7*t13*t21*t26*9.0*10.0 - l1*l2*l4*m2*t3*t8*t14*t21*t28*2.4*10.0 + l1*l2*l4*m2*t4*t7*t14*t21*t28*4.8*10.0 + l1*l2*l4*m3*t4*t7*t13*t21*t28*4.8*10.0 + l1*l2*l4*m4*t4*t7*t14*t21*t26*9.0*10.0 - l1*l2*l4*m2*t3*t8*t15*t21*t28*2.4*10.0 + l1*l2*l4*m2*t4*t7*t15*t21*t28*4.8*10.0 + l1*l2*l4*m3*t4*t7*t14*t21*t28*4.8*10.0 + l1*l2*l4*m4*t4*t7*t15*t21*t26*9.0*10.0 + l1*l2*l4*m3*t4*t7*t15*t21*t28*4.8*10.0 - l1*l3*l4*m2*t3*t13*t19*t26*t37*1.2*10.0 - l1*l3*l4*m2*t3*t14*t19*t26*t37*1.2*10.0 - l1*l3*l4*m2*t7*t13*t19*t26*t34*1.2*10.0 - l1*l3*l4*m2*t3*t13*t19*t28*t37*1.2*10.0 - l1*l3*l4*m2*t7*t14*t19*t26*t34*1.2*10.0 - l1*l3*l4*m2*t3*t14*t19*t28*t37*1.2*10.0 - l1*l3*l4*m2*t7*t13*t19*t28*t34*1.2*10.0 - l1*l3*l4*m4*t7*t13*t19*t25*t35*3.6*10.0 - l1*l3*l4*m2*t7*t14*t19*t28*t34*1.2*10.0 - l1*l3*l4*m4*t7*t13*t19*t26*t35*5.4*10.0 - l1*l3*l4*m4*t7*t14*t19*t25*t35*3.6*10.0 - l1*l3*l4*m4*t7*t14*t19*t26*t35*5.4*10.0 - l2*l3*l4*m2*t13*t17*t26*t33*t37*1.2*10.0 - l2*l3*l4*m2*t13*t17*t26*t34*t36*1.2*10.0 - l2*l3*l4*m2*t13*t17*t28*t33*t37*1.2*10.0 - l2*l3*l4*m2*t13*t17*t28*t34*t36*1.2*10.0 - l2*l3*l4*m4*t13*t17*t25*t35*t36*2.7*10.0 - l2*l3*l4*m4*t13*t17*t26*t35*t36*2.7*10.0 + dth1*dth2*l1*l3*l4*m2*m3*m4*t7*t19*5.0*100.0 + g*l1*l2*l3*l4*m2*m3*m4*t2*t36*1.5*100.0 + g*l1*l2*l3*l4*m1*m2*m4*t6*t35*3.6*10.0 + g*l1*l2*l3*l4*m1*m3*m4*t6*t34*9.0*10.0 + g*l1*l2*l3*l4*m2*m3*m4*t6*t33*1.5*100.0 + g*l1*l2*l3*l4*m1*m3*m4*t6*t35*5.4*10.0 + g*l1*l2*l3*l4*m2*m3*m4*t6*t34*9.0*10.0 + g*l1*l2*l3*l4*m2*m3*m4*t6*t35*9.0*10.0 - dth1*dth2*l1*l2*l4*m2*t3*t8*t21*t26*3.2*10.0 + dth1*dth2*l1*l2*l4*m2*t4*t7*t21*t26*6.4*10.0 - dth1*dth3*l1*l2*l4*m2*t3*t8*t21*t26*3.2*10.0 + dth1*dth3*l1*l2*l4*m2*t4*t7*t21*t26*6.4*10.0 - dth1*dth2*l1*l2*l4*m2*t3*t8*t21*t28*4.8*10.0 + dth1*dth2*l1*l2*l4*m2*t4*t7*t21*t28*9.6*10.0 + dth1*dth2*l1*l2*l4*m4*t4*t7*t21*t26*1.8*100.0 - dth2*dth3*l1*l2*l4*m2*t3*t8*t21*t26*3.2*10.0 + dth2*dth3*l1*l2*l4*m2*t4*t7*t21*t26*6.4*10.0 + dth1*dth2*l1*l2*l4*m3*t4*t7*t21*t28*9.6*10.0 - dth1*dth3*l1*l2*l4*m2*t3*t8*t21*t28*4.8*10.0 + dth1*dth3*l1*l2*l4*m2*t4*t7*t21*t28*9.6*10.0 + dth1*dth3*l1*l2*l4*m4*t4*t7*t21*t26*1.8*100.0 + dth1*dth3*l1*l2*l4*m3*t4*t7*t21*t28*9.6*10.0 - dth2*dth3*l1*l2*l4*m2*t3*t8*t21*t28*4.8*10.0 + dth2*dth3*l1*l2*l4*m2*t4*t7*t21*t28*9.6*10.0 + dth2*dth3*l1*l2*l4*m4*t4*t7*t21*t26*1.8*100.0 + dth2*dth3*l1*l2*l4*m3*t4*t7*t21*t28*9.6*10.0 - dth1*dth2*l1*l3*l4*m2*t3*t19*t26*t37*2.4*10.0 - dth1*dth2*l1*l3*l4*m2*t7*t19*t26*t34*2.4*10.0 - dth1*dth2*l1*l3*l4*m2*t3*t19*t28*t37*2.4*10.0 - dth1*dth2*l1*l3*l4*m2*t7*t19*t28*t34*2.4*10.0 - dth1*dth2*l1*l3*l4*m4*t7*t19*t25*t35*7.2*10.0 - dth1*dth2*l1*l3*l4*m4*t7*t19*t26*t35*1.08*100.0 - g*l1*l2*l3*l4*m2*t2*t26*t33*t37*1.2*10.0 - g*l1*l2*l3*l4*m2*t2*t26*t34*t36*1.2*10.0 - g*l1*l2*l3*l4*m2*t6*t26*t33*t34*1.2*10.0 - g*l1*l2*l3*l4*m2*t2*t28*t33*t37*1.2*10.0 - g*l1*l2*l3*l4*m2*t2*t28*t34*t36*1.2*10.0 - g*l1*l2*l3*l4*m4*t2*t25*t35*t36*2.7*10.0 - g*l1*l2*l3*l4*m2*t6*t28*t33*t34*1.2*10.0 - g*l1*l2*l3*l4*m4*t2*t26*t35*t36*2.7*10.0 - g*l1*l2*l3*l4*m4*t6*t25*t33*t35*2.7*10.0 - g*l1*l2*l3*l4*m4*t6*t26*t33*t35*2.7*10.0 + g*l1*l2*l3*l4*m2*t6*t26*t36*t37*1.2*10.0 + g*l1*l2*l3*l4*m2*t6*t28*t36*t37*1.2*10.0 - l1*l2*l4*m2*m3*m4*t3*t8*t13*t21*5.0*10.0 + l1*l2*l4*m2*m3*m4*t4*t7*t13*t21*1.0*100.0 - l1*l2*l4*m2*m3*m4*t3*t8*t14*t21*5.0*10.0 + l1*l2*l4*m2*m3*m4*t4*t7*t14*t21*1.0*100.0 - l1*l2*l4*m2*m3*m4*t3*t8*t15*t21*5.0*10.0 + l1*l2*l4*m2*m3*m4*t4*t7*t15*t21*1.0*100.0 - l1*l3*l4*m2*m3*m4*t3*t13*t19*t37*3.0*10.0 - l1*l3*l4*m2*m3*m4*t3*t14*t19*t37*3.0*10.0 - l1*l3*l4*m2*m3*m4*t7*t13*t19*t34*3.0*10.0 - l1*l3*l4*m2*m3*m4*t7*t13*t19*t35*9.0*10.0 - l1*l3*l4*m2*m3*m4*t7*t14*t19*t34*3.0*10.0 - l1*l3*l4*m2*m3*m4*t7*t14*t19*t35*9.0*10.0 - l2*l3*l4*m2*m3*m4*t13*t17*t33*t37*3.0*10.0 - l2*l3*l4*m2*m3*m4*t13*t17*t34*t36*3.0*10.0 - l2*l3*l4*m2*m3*m4*t13*t17*t35*t36*5.4*10.0 - l1*l2*l3*m2*t3*t5*t8*t13*t23*t28*1.2*10.0 + l1*l2*l3*m2*t4*t5*t7*t13*t23*t28*2.4*10.0 + l1*l2*l3*m4*t4*t5*t7*t13*t23*t26*2.4*10.0 - l1*l2*l3*m2*t3*t5*t8*t14*t23*t28*1.2*10.0 + l1*l2*l3*m2*t4*t5*t7*t14*t23*t28*2.4*10.0 + l1*l2*l3*m3*t4*t5*t7*t13*t23*t28*2.4*10.0 + l1*l2*l3*m4*t4*t5*t7*t14*t23*t26*2.4*10.0 - l1*l2*l3*m2*t3*t5*t8*t15*t23*t28*1.2*10.0 + l1*l2*l3*m2*t4*t5*t7*t15*t23*t28*2.4*10.0 + l1*l2*l3*m3*t4*t5*t7*t14*t23*t28*2.4*10.0 + l1*l2*l3*m4*t4*t5*t7*t15*t23*t26*2.4*10.0 - l1*l2*l3*m2*t3*t5*t8*t16*t23*t28*1.2*10.0 + l1*l2*l3*m2*t4*t5*t7*t16*t23*t28*2.4*10.0 + l1*l2*l3*m3*t4*t5*t7*t15*t23*t28*2.4*10.0 + l1*l2*l3*m4*t4*t5*t7*t16*t23*t26*2.4*10.0 + l1*l2*l3*m3*t4*t5*t7*t16*t23*t28*2.4*10.0 + l1*l2*l3*m4*t7*t8*t9*t13*t23*t26*4.8*10.0 + l1*l2*l3*m3*t7*t8*t9*t13*t23*t28*1.2*10.0 + l1*l2*l3*m4*t7*t8*t9*t14*t23*t26*4.8*10.0 + l1*l2*l3*m3*t7*t8*t9*t14*t23*t28*1.2*10.0 + l1*l2*l3*m4*t7*t8*t9*t15*t23*t26*4.8*10.0 + l1*l2*l3*m3*t7*t8*t9*t15*t23*t28*1.2*10.0 + l1*l2*l3*m4*t7*t8*t9*t16*t23*t26*4.8*10.0 + l1*l2*l3*m3*t7*t8*t9*t16*t23*t28*1.2*10.0 - l1*l2*l4*m4*t4*t7*t13*t21*t26*t35*1.8*10.0 - l1*l2*l4*m4*t4*t7*t14*t21*t26*t35*1.8*10.0 - l1*l2*l4*m4*t4*t7*t15*t21*t26*t35*1.8*10.0 + l1*l2*l4*m4*t7*t8*t13*t21*t26*t38*1.8*10.0 + l1*l2*l4*m4*t7*t8*t14*t21*t26*t38*1.8*10.0 + l1*l2*l4*m4*t7*t8*t15*t21*t26*t38*1.8*10.0 - dth1*dth2*l1*l2*l4*m2*m3*m4*t3*t8*t21*1.0*100.0 + dth1*dth2*l1*l2*l4*m2*m3*m4*t4*t7*t21*2.0*100.0 - dth1*dth3*l1*l2*l4*m2*m3*m4*t3*t8*t21*1.0*100.0 + dth1*dth3*l1*l2*l4*m2*m3*m4*t4*t7*t21*2.0*100.0 - dth2*dth3*l1*l2*l4*m2*m3*m4*t3*t8*t21*1.0*100.0 + dth2*dth3*l1*l2*l4*m2*m3*m4*t4*t7*t21*2.0*100.0 - dth1*dth2*l1*l3*l4*m2*m3*m4*t3*t19*t37*6.0*10.0 - dth1*dth2*l1*l3*l4*m2*m3*m4*t7*t19*t34*6.0*10.0 - dth1*dth2*l1*l3*l4*m2*m3*m4*t7*t19*t35*1.8*100.0 - g*l1*l2*l3*l4*m2*m3*m4*t2*t33*t37*3.0*10.0 - g*l1*l2*l3*l4*m2*m3*m4*t2*t34*t36*3.0*10.0 - g*l1*l2*l3*l4*m2*m3*m4*t2*t35*t36*5.4*10.0 - g*l1*l2*l3*l4*m2*m3*m4*t6*t33*t34*3.0*10.0 - g*l1*l2*l3*l4*m1*m3*m4*t6*t34*t35*1.8*10.0 - g*l1*l2*l3*l4*m2*m3*m4*t6*t33*t35*5.4*10.0 - g*l1*l2*l3*l4*m2*m3*m4*t6*t34*t35*1.8*10.0 + g*l1*l2*l3*l4*m2*m3*m4*t6*t36*t37*3.0*10.0 + g*l1*l2*l3*l4*m1*m3*m4*t6*t37*t38*1.8*10.0 + g*l1*l2*l3*l4*m2*m3*m4*t6*t37*t38*1.8*10.0 - dth1*dth2*l1*l2*l3*m2*t3*t5*t8*t23*t28*2.4*10.0 + dth1*dth2*l1*l2*l3*m2*t4*t5*t7*t23*t28*4.8*10.0 + dth1*dth2*l1*l2*l3*m4*t4*t5*t7*t23*t26*4.8*10.0 + dth1*dth2*l1*l2*l3*m3*t4*t5*t7*t23*t28*4.8*10.0 - dth1*dth3*l1*l2*l3*m2*t3*t5*t8*t23*t28*2.4*10.0 + dth1*dth3*l1*l2*l3*m2*t4*t5*t7*t23*t28*4.8*10.0 + dth1*dth3*l1*l2*l3*m4*t4*t5*t7*t23*t26*4.8*10.0 + dth1*dth3*l1*l2*l3*m3*t4*t5*t7*t23*t28*4.8*10.0 - dth1*dth4*l1*l2*l3*m2*t3*t5*t8*t23*t28*2.4*10.0 + dth1*dth4*l1*l2*l3*m2*t4*t5*t7*t23*t28*4.8*10.0 + dth1*dth4*l1*l2*l3*m4*t4*t5*t7*t23*t26*4.8*10.0 - dth2*dth3*l1*l2*l3*m2*t3*t5*t8*t23*t28*2.4*10.0 + dth2*dth3*l1*l2*l3*m2*t4*t5*t7*t23*t28*4.8*10.0 + dth2*dth3*l1*l2*l3*m4*t4*t5*t7*t23*t26*4.8*10.0 + dth1*dth4*l1*l2*l3*m3*t4*t5*t7*t23*t28*4.8*10.0 + dth2*dth3*l1*l2*l3*m3*t4*t5*t7*t23*t28*4.8*10.0 - dth2*dth4*l1*l2*l3*m2*t3*t5*t8*t23*t28*2.4*10.0 + dth2*dth4*l1*l2*l3*m2*t4*t5*t7*t23*t28*4.8*10.0 + dth2*dth4*l1*l2*l3*m4*t4*t5*t7*t23*t26*4.8*10.0 + dth2*dth4*l1*l2*l3*m3*t4*t5*t7*t23*t28*4.8*10.0 - dth3*dth4*l1*l2*l3*m2*t3*t5*t8*t23*t28*2.4*10.0 + dth3*dth4*l1*l2*l3*m2*t4*t5*t7*t23*t28*4.8*10.0 + dth3*dth4*l1*l2*l3*m4*t4*t5*t7*t23*t26*4.8*10.0 + dth3*dth4*l1*l2*l3*m3*t4*t5*t7*t23*t28*4.8*10.0 + dth1*dth2*l1*l2*l3*m4*t7*t8*t9*t23*t26*9.6*10.0 + dth1*dth2*l1*l2*l3*m3*t7*t8*t9*t23*t28*2.4*10.0 + dth1*dth3*l1*l2*l3*m4*t7*t8*t9*t23*t26*9.6*10.0 + dth1*dth3*l1*l2*l3*m3*t7*t8*t9*t23*t28*2.4*10.0 + dth1*dth4*l1*l2*l3*m4*t7*t8*t9*t23*t26*9.6*10.0 + dth2*dth3*l1*l2*l3*m4*t7*t8*t9*t23*t26*9.6*10.0 + dth1*dth4*l1*l2*l3*m3*t7*t8*t9*t23*t28*2.4*10.0 + dth2*dth3*l1*l2*l3*m3*t7*t8*t9*t23*t28*2.4*10.0 + dth2*dth4*l1*l2*l3*m4*t7*t8*t9*t23*t26*9.6*10.0 + dth2*dth4*l1*l2*l3*m3*t7*t8*t9*t23*t28*2.4*10.0 + dth3*dth4*l1*l2*l3*m4*t7*t8*t9*t23*t26*9.6*10.0 + dth3*dth4*l1*l2*l3*m3*t7*t8*t9*t23*t28*2.4*10.0 - dth1*dth2*l1*l2*l4*m4*t4*t7*t21*t26*t35*3.6*10.0 - dth1*dth3*l1*l2*l4*m4*t4*t7*t21*t26*t35*3.6*10.0 - dth2*dth3*l1*l2*l4*m4*t4*t7*t21*t26*t35*3.6*10.0 + dth1*dth2*l1*l2*l4*m4*t7*t8*t21*t26*t38*3.6*10.0 + dth1*dth3*l1*l2*l4*m4*t7*t8*t21*t26*t38*3.6*10.0 + dth2*dth3*l1*l2*l4*m4*t7*t8*t21*t26*t38*3.6*10.0 + l1*l2*l3*m2*m3*m4*t3*t4*t9*t13*t23*8.0 - l1*l2*l3*m2*m3*m4*t3*t5*t8*t13*t23*1.6*10.0 + l1*l2*l3*m2*m3*m4*t4*t5*t7*t13*t23*3.2*10.0 + l1*l2*l3*m2*m3*m4*t3*t4*t9*t14*t23*8.0 - l1*l2*l3*m2*m3*m4*t3*t5*t8*t14*t23*1.6*10.0 + l1*l2*l3*m2*m3*m4*t4*t5*t7*t14*t23*3.2*10.0 + l1*l2*l3*m2*m3*m4*t3*t4*t9*t15*t23*8.0 - l1*l2*l3*m2*m3*m4*t3*t5*t8*t15*t23*1.6*10.0 + l1*l2*l3*m2*m3*m4*t4*t5*t7*t15*t23*3.2*10.0 + l1*l2*l3*m2*m3*m4*t3*t4*t9*t16*t23*8.0 - l1*l2*l3*m2*m3*m4*t3*t5*t8*t16*t23*1.6*10.0 + l1*l2*l3*m2*m3*m4*t4*t5*t7*t16*t23*3.2*10.0 + l1*l2*l3*m2*m3*m4*t7*t8*t9*t13*t23*1.6*10.0 + l1*l2*l3*m2*m3*m4*t7*t8*t9*t14*t23*1.6*10.0 + l1*l2*l3*m2*m3*m4*t7*t8*t9*t15*t23*1.6*10.0 + l1*l2*l3*m2*m3*m4*t7*t8*t9*t16*t23*1.6*10.0 + l1*l2*l4*m2*m3*m4*t3*t4*t13*t21*t38*6.0 + l1*l2*l4*m2*m3*m4*t3*t4*t14*t21*t38*6.0 + l1*l2*l4*m2*m3*m4*t3*t8*t13*t21*t35*6.0 - l1*l2*l4*m2*m3*m4*t4*t7*t13*t21*t35*1.2*10.0 + l1*l2*l4*m2*m3*m4*t3*t4*t15*t21*t38*6.0 + l1*l2*l4*m2*m3*m4*t3*t8*t14*t21*t35*6.0 - l1*l2*l4*m2*m3*m4*t4*t7*t14*t21*t35*1.2*10.0 + l1*l2*l4*m2*m3*m4*t3*t8*t15*t21*t35*6.0 - l1*l2*l4*m2*m3*m4*t4*t7*t15*t21*t35*1.2*10.0 + l1*l2*l4*m2*m3*m4*t7*t8*t13*t21*t38*1.2*10.0 + l1*l2*l4*m2*m3*m4*t7*t8*t14*t21*t38*1.2*10.0 + l1*l2*l4*m2*m3*m4*t7*t8*t15*t21*t38*1.2*10.0 + l1*l3*l4*m2*m3*m4*t3*t13*t19*t34*t38*6.0 + l1*l3*l4*m2*m3*m4*t3*t13*t19*t35*t37*6.0 + l1*l3*l4*m2*m3*m4*t3*t14*t19*t34*t38*6.0 + l1*l3*l4*m2*m3*m4*t3*t14*t19*t35*t37*6.0 + l1*l3*l4*m2*m3*m4*t7*t13*t19*t34*t35*6.0 + l1*l3*l4*m2*m3*m4*t7*t14*t19*t34*t35*6.0 - l1*l3*l4*m2*m3*m4*t7*t13*t19*t37*t38*6.0 - l1*l3*l4*m2*m3*m4*t7*t14*t19*t37*t38*6.0 + l2*l3*l4*m2*m3*m4*t13*t17*t33*t34*t38*6.0 + l2*l3*l4*m2*m3*m4*t13*t17*t33*t35*t37*6.0 + l2*l3*l4*m2*m3*m4*t13*t17*t34*t35*t36*6.0 - l2*l3*l4*m2*m3*m4*t13*t17*t36*t37*t38*6.0 + dth1*dth2*l1*l2*l3*m2*m3*m4*t3*t4*t9*t23*1.6*10.0 - dth1*dth2*l1*l2*l3*m2*m3*m4*t3*t5*t8*t23*3.2*10.0 + dth1*dth2*l1*l2*l3*m2*m3*m4*t4*t5*t7*t23*6.4*10.0 + dth1*dth3*l1*l2*l3*m2*m3*m4*t3*t4*t9*t23*1.6*10.0 - dth1*dth3*l1*l2*l3*m2*m3*m4*t3*t5*t8*t23*3.2*10.0 + dth1*dth3*l1*l2*l3*m2*m3*m4*t4*t5*t7*t23*6.4*10.0 + dth1*dth4*l1*l2*l3*m2*m3*m4*t3*t4*t9*t23*1.6*10.0 - dth1*dth4*l1*l2*l3*m2*m3*m4*t3*t5*t8*t23*3.2*10.0 + dth1*dth4*l1*l2*l3*m2*m3*m4*t4*t5*t7*t23*6.4*10.0 + dth2*dth3*l1*l2*l3*m2*m3*m4*t3*t4*t9*t23*1.6*10.0 - dth2*dth3*l1*l2*l3*m2*m3*m4*t3*t5*t8*t23*3.2*10.0 + dth2*dth3*l1*l2*l3*m2*m3*m4*t4*t5*t7*t23*6.4*10.0 + dth2*dth4*l1*l2*l3*m2*m3*m4*t3*t4*t9*t23*1.6*10.0 - dth2*dth4*l1*l2*l3*m2*m3*m4*t3*t5*t8*t23*3.2*10.0 + dth2*dth4*l1*l2*l3*m2*m3*m4*t4*t5*t7*t23*6.4*10.0 + dth3*dth4*l1*l2*l3*m2*m3*m4*t3*t4*t9*t23*1.6*10.0 - dth3*dth4*l1*l2*l3*m2*m3*m4*t3*t5*t8*t23*3.2*10.0 + dth3*dth4*l1*l2*l3*m2*m3*m4*t4*t5*t7*t23*6.4*10.0 + dth1*dth2*l1*l2*l3*m2*m3*m4*t7*t8*t9*t23*3.2*10.0 + dth1*dth3*l1*l2*l3*m2*m3*m4*t7*t8*t9*t23*3.2*10.0 + dth1*dth4*l1*l2*l3*m2*m3*m4*t7*t8*t9*t23*3.2*10.0 + dth2*dth3*l1*l2*l3*m2*m3*m4*t7*t8*t9*t23*3.2*10.0 + dth2*dth4*l1*l2*l3*m2*m3*m4*t7*t8*t9*t23*3.2*10.0 + dth3*dth4*l1*l2*l3*m2*m3*m4*t7*t8*t9*t23*3.2*10.0 + dth1*dth2*l1*l2*l4*m2*m3*m4*t3*t4*t21*t38*1.2*10.0 + dth1*dth2*l1*l2*l4*m2*m3*m4*t3*t8*t21*t35*1.2*10.0 - dth1*dth2*l1*l2*l4*m2*m3*m4*t4*t7*t21*t35*2.4*10.0 + dth1*dth3*l1*l2*l4*m2*m3*m4*t3*t4*t21*t38*1.2*10.0 + dth1*dth3*l1*l2*l4*m2*m3*m4*t3*t8*t21*t35*1.2*10.0 - dth1*dth3*l1*l2*l4*m2*m3*m4*t4*t7*t21*t35*2.4*10.0 + dth2*dth3*l1*l2*l4*m2*m3*m4*t3*t4*t21*t38*1.2*10.0 + dth2*dth3*l1*l2*l4*m2*m3*m4*t3*t8*t21*t35*1.2*10.0 - dth2*dth3*l1*l2*l4*m2*m3*m4*t4*t7*t21*t35*2.4*10.0 + dth1*dth2*l1*l2*l4*m2*m3*m4*t7*t8*t21*t38*2.4*10.0 + dth1*dth3*l1*l2*l4*m2*m3*m4*t7*t8*t21*t38*2.4*10.0 + dth2*dth3*l1*l2*l4*m2*m3*m4*t7*t8*t21*t38*2.4*10.0 + dth1*dth2*l1*l3*l4*m2*m3*m4*t3*t19*t34*t38*1.2*10.0 + dth1*dth2*l1*l3*l4*m2*m3*m4*t3*t19*t35*t37*1.2*10.0 + dth1*dth2*l1*l3*l4*m2*m3*m4*t7*t19*t34*t35*1.2*10.0 - dth1*dth2*l1*l3*l4*m2*m3*m4*t7*t19*t37*t38*1.2*10.0 + g*l1*l2*l3*l4*m2*m3*m4*t2*t33*t34*t38*6.0 + g*l1*l2*l3*l4*m2*m3*m4*t2*t33*t35*t37*6.0 + g*l1*l2*l3*l4*m2*m3*m4*t2*t34*t35*t36*6.0 + g*l1*l2*l3*l4*m2*m3*m4*t6*t33*t34*t35*6.0 - g*l1*l2*l3*l4*m2*m3*m4*t2*t36*t37*t38*6.0 - g*l1*l2*l3*l4*m2*m3*m4*t6*t33*t37*t38*6.0 - g*l1*l2*l3*l4*m2*m3*m4*t6*t34*t36*t38*6.0 - g*l1*l2*l3*l4*m2*m3*m4*t6*t35*t36*t37*6.0)*6.0;

        index = index + num_threads;

        index1 = (grid_size[0] > 1) ? index : 0;
        index2 = (grid_size[1] > 1) ? index : 0;
        index3 = (grid_size[2] > 1) ? index : 0;
        index4 = (grid_size[3] > 1) ? index : 0;
        index5 = (grid_size[4] > 1) ? index : 0;
        index6 = (grid_size[5] > 1) ? index : 0;
        index7 = (grid_size[6] > 1) ? index : 0;
        index8 = (grid_size[7] > 1) ? index : 0;

        uindex1 = (active_actions[0] == 1) ? index : 0;
        uindex2 = (active_actions[1] == 1) ? index : 0;
        uindex3 = (active_actions[2] == 1) ? index : 0;
        uindex4 = (active_actions[3] == 1) ? index : 0;
    }
}

__global__ void dyn6_mex_continuous(double* const d_dx2, // outputs
    double* const x1, double* const x2, double* const x3, double* const x4,
    double* const dx1, double* const dx2, double* const dx3, double* const dx4, // input states
    double const* const in1, double const* const in2, double const* const in3, double const* const in4, // input actions
    const double m1, const double m2, const double m3, const double m4,
    const double l1, const double l2, const double l3, const double l4, const double g, 
    int32_t const* const grid_size, int32_t const* const active_actions)
{
    int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x * blockDim.y
        + (threadIdx.x + threadIdx.y * blockDim.x);
    int num_threads = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
    int num_elements = grid_size[0] * grid_size[1] * grid_size[2] * grid_size[3]
                        * grid_size[4] * grid_size[5] * grid_size[6] * grid_size[7];
    double th1, th2, th3, th4, dth1, dth2, dth3, dth4, u1, u2, u3, u4;

    int index1 = (grid_size[0] > 1) ? index : 0;
    int index2 = (grid_size[1] > 1) ? index : 0;
    int index3 = (grid_size[2] > 1) ? index : 0;
    int index4 = (grid_size[3] > 1) ? index : 0;
    int index5 = (grid_size[4] > 1) ? index : 0;
    int index6 = (grid_size[5] > 1) ? index : 0;
    int index7 = (grid_size[6] > 1) ? index : 0;
    int index8 = (grid_size[7] > 1) ? index : 0;

    int uindex1 = (active_actions[0] == 1) ? index : 0;
    int uindex2 = (active_actions[1] == 1) ? index : 0;
    int uindex3 = (active_actions[2] == 1) ? index : 0;
    int uindex4 = (active_actions[3] == 1) ? index : 0;

    while (index < num_elements)
    {
        th1 = x1[index1];
        th2 = x2[index2];
        th3 = x3[index3];
        th4 = x4[index4];

        dth1 = dx1[index5];
        dth2 = dx2[index6];
        dth3 = dx3[index7];
        dth4 = dx4[index8];

        u1 = in1[uindex1];
        u2 = in2[uindex2];
        u3 = in3[uindex3];
        u4 = in4[uindex4];

        double t2 = cos(th1);
        double t3 = cos(th2);
        double t4 = cos(th3);
        double t5 = cos(th4);
        double t6 = sin(th1);
        double t7 = sin(th2);
        double t8 = sin(th3);
        double t9 = sin(th4);
        double t10 = th1 + th2;
        double t11 = th2 + th3;
        double t12 = th3 + th4;
        double t13 = dth1 * dth1;
        double t14 = dth2 * dth2;
        double t15 = dth3 * dth3;
        double t16 = dth4 * dth4;
        double t17 = l1 * l1;
        double t18 = t17 * l1;
        double t19 = l2 * l2;
        double t20 = t19 * l2;
        double t21 = l3 * l3;
//         double t22 = t21 * l3;
        double t23 = l4 * l4;
//         double t24 = t23 * l4;
        double t25 = m2 * m2;
        double t26 = m3 * m3;
        double t27 = t26 * m3;
        double t28 = m4 * m4;
//         double t29 = t28 * m4;
        double t30 = th2 * 2.0;
        double t31 = th3 * 2.0;
        double t32 = th4 * 2.0;
//         double t46 = 1.0 / l1;
//         double t48 = 1.0 / l2;
        double t50 = 1.0 / l3;
        double t52 = 1.0 / l4;
        double t53 = -th1;
        double t54 = -th2;
        double t55 = -th3;
        double t57 = -th4;
        double t125 = m1 * m2 * m3 * 128.0;
        double t126 = m1 * m2 * m4 * 240.0;
        double t163 = m1 * m3 * m4 * 600.0;
        double t164 = m2 * m3 * m4 * 1500.0;
        double t33 = cos(t30);
        double t34 = cos(t31);
        double t35 = cos(t32);
        double t36 = sin(t30);
        double t37 = sin(t31);
        double t38 = sin(t32);
        double t39 = cos(t11);
        double t40 = cos(t12);
        double t41 = sin(t10);
        double t42 = sin(t11);
        double t43 = sin(t12);
        double t44 = t10 + th3;
        double t45 = t11 + th4;
        double t47 = 1.0 / t17;
        double t49 = 1.0 / t19;
//         double t51 = 1.0 / t21;
        double t56 = -t31;
        double t58 = -t32;
        double t62 = t10 + t12;
        double t63 = t10 + th2;
        double t64 = t31 + th1;
        double t65 = t32 + th1;
        double t66 = t11 + th3;
        double t67 = t11 + th2;
        double t68 = t32 + th2;
        double t69 = t30 + th4;
        double t70 = t12 + th4;
        double t71 = t12 + th3;
        double t87 = t10 + t31;
        double t88 = t10 + t32;
        double t89 = t11 + t32;
        double t92 = t54 + th1;
        double t94 = t55 + th2;
        double t97 = t57 + th3;
        double t98 = t11 * 2.0;
        double t99 = t30 + t32;
        double t100 = t12 * 2.0;
        double t110 = t10 + t55;
        double t112 = t11 + t53;
        double t114 = t11 + t57;
        double t116 = t12 + t54;
        double t117 = t27 * 144.0;
        double t128 = t30 + t57;
        double t156 = m1 * t28 * 144.0;
        double t157 = m3 * t28 * 144.0;
        double t158 = m1 * t26 * 240.0;
        double t159 = m3 * t25 * 240.0;
        double t160 = m2 * t28 * 360.0;
        double t161 = m4 * t25 * 450.0;
        double t162 = m4 * t26 * 450.0;
        double t179 = m2 * t26 * 600.0;
        double t211 = t10 - t12;
        double t213 = -t11 + th1 + th4;
        double t59 = cos(t45);
        double t60 = sin(t44);
        double t61 = sin(t45);
        double t72 = cos(t66);
        double t73 = cos(t67);
        double t74 = cos(t68);
        double t75 = cos(t69);
        double t76 = cos(t70);
        double t77 = cos(t71);
        double t78 = sin(t63);
        double t79 = sin(t64);
        double t80 = sin(t65);
        double t81 = sin(t66);
        double t82 = sin(t67);
        double t83 = sin(t68);
        double t84 = sin(t69);
        double t85 = sin(t70);
        double t86 = sin(t71);
        double t90 = t45 + th2;
        double t91 = sin(t62);
        double t93 = t56 + th1;
        double t95 = t58 + th1;
        double t96 = t58 + th2;
        double t101 = cos(t94);
        double t103 = cos(t97);
        double t104 = sin(t92);
        double t106 = sin(t94);
        double t109 = sin(t97);
        double t111 = t92 + th3;
        double t113 = t10 + t58;
        double t115 = t94 + th4;
        double t118 = cos(t89);
        double t120 = sin(t87);
        double t121 = sin(t88);
        double t122 = sin(t89);
        double t124 = t62 + th4;
        double t129 = t30 + t58;
        double t130 = cos(t98);
        double t131 = cos(t99);
        double t132 = cos(t100);
        double t133 = sin(t98);
        double t134 = sin(t99);
        double t135 = sin(t100);
        double t136 = t11 + t44;
        double t137 = t32 + t63;
        double t138 = t100 + th1;
        double t139 = t12 + t45;
        double t140 = t32 + t67;
        double t141 = t11 + t45;
        double t142 = cos(t114);
        double t144 = cos(t116);
        double t145 = sin(t110);
        double t147 = sin(t112);
        double t149 = sin(t114);
        double t151 = sin(t116);
        double t152 = t44 + t57;
        double t153 = t110 + th4;
        double t154 = t12 + t92;
        double t155 = t45 + t53;
        double t169 = cos(t128);
        double t171 = sin(t128);
        double t173 = t53 + t66;
        double t174 = t54 + t65;
        double t175 = t53 + t68;
        double t176 = t58 + t63;
        double t177 = t54 + t70;
        double t178 = t57 + t67;
        double t189 = t12 + t62;
        double t191 = t11 + t89;
        double t200 = t70 + t92;
        double t201 = t53 + t89;
        double t207 = t53 + t100;
        double t208 = m1 * m2 * m4 * t35 * 144.0;
        double t209 = m1 * m3 * m4 * t35 * 216.0;
        double t210 = m1 * m3 * m4 * t34 * 360.0;
        double t212 = t92 + t97;
        double t214 = t57 + t211;
        double t217 = -t27 * t33 * 144.0;
        double t218 = sin(t211);
        double t220 = sin(t213);
        double t222 = m1 * t26 * t34 * 144.0;
        double t223 = m3 * t25 * t33 * 144.0;
        double t226 = m4 * t26 * t35 * 162.0;
        double t227 = m2 * t26 * t34 * 216.0;
        double t228 = m2 * t28 * t33 * 216.0;
        double t229 = m2 * t28 * t34 * 216.0;
        double t230 = m4 * t25 * t33 * 270.0;
        double t231 = m4 * t25 * t35 * 270.0;
        double t232 = m2 * t26 * t33 * 360.0;
        double t237 = m2 * m3 * m4 * t34 * 540.0;
        double t238 = m2 * m3 * m4 * t35 * 540.0;
        double t239 = m2 * m3 * m4 * t33 * 900.0;
        double t243 = - m1 * t28 * t34 * 144.0;
        double t244 = - m3 * t28 * t33 * 144.0;
        double t252 = - m4 * t26 * t33 * 450.0;
        double t102 = cos(t96);
        double t105 = sin(t93);
        double t107 = sin(t95);
        double t108 = sin(t96);
        double t119 = cos(t90);
        double t123 = sin(t90);
        double t127 = sin(t124);
        double t143 = cos(t115);
        double t146 = sin(t111);
        double t148 = sin(t113);
        double t150 = sin(t115);
        double t165 = sin(t152);
        double t166 = sin(t153);
        double t167 = sin(t154);
        double t168 = sin(t155);
        double t170 = cos(t129);
        double t172 = sin(t129);
        double t180 = cos(t139);
        double t181 = cos(t140);
        double t182 = cos(t141);
        double t183 = sin(t136);
        double t184 = sin(t137);
        double t185 = sin(t138);
        double t186 = sin(t139);
        double t187 = sin(t140);
        double t188 = sin(t141);
        double t190 = sin(t189);
        double t192 = cos(t177);
        double t193 = cos(t178);
        double t194 = sin(t173);
        double t195 = sin(t174);
        double t196 = sin(t175);
        double t197 = sin(t176);
        double t198 = sin(t177);
        double t199 = sin(t178);
        double t202 = cos(t191);
        double t203 = sin(t191);
        double t205 = sin(t200);
        double t206 = sin(t201);
        double t215 = sin(t207);
        double t216 = t53 + t139;
        double t219 = sin(t212);
        double t221 = sin(t214);
        double t234 = -t208;
        double t235 = -t209;
        double t236 = -t210;
        double t241 = -t222;
        double t242 = -t223;
        double t245 = -t226;
        double t246 = -t227;
        double t247 = -t228;
        double t248 = -t229;
        double t249 = -t230;
        double t250 = -t231;
        double t251 = -t232;
        double t253 = -t237;
        double t254 = -t238;
        double t255 = -t239;
        double t256 = m1 * m3 * m4 * t132 * 72.0;
        double t257 = m2 * m3 * m4 * t132 * 108.0;
        double t258 = m2 * m3 * m4 * t131 * 162.0;
        double t259 = m2 * m3 * m4 * t130 * 180.0;
        double t260 = m2 * t26 * t130 * 72.0;
        double t261 = m2 * t28 * t130 * 72.0;
        double t262 = m4 * t25 * t131 * 81.0;
        double t263 = m4 * t26 * t131 * 81.0;
        double t240 = sin(t216);
        double t264 = m2 * m3 * m4 * t170 * 162.0;
        double t265 = m4 * t25 * t170 * 81.0;
        double t266 = m4 * t26 * t170 * 81.0;
        double t267 = m2 * m3 * m4 * t202 * 36.0;
        double t268 = -t267;
        double t269 = t117 + t125 + t126 + t156 + t157 + t158 + t159 + t160 + t161 + t162 + t163 + t164 + t179 + t217 + t234 + t235 + t236 + t241 + t242 + t243 + t244 + t245 + t246 + t247 + t248 + t249 + t250 + t251 + t252 + t253 + t254 + t255 + t256 + t257 + t258 + t259 + t260 + t261 + t262 + t263 + t264 + t265 + t266 + t268;
        double t270 = 1.0 / t269;

        d_dx2[index] = t47*t49*t50*t52*t270*(l3*l4*t17*t26*u2* - 3.0*10.0 + l3*l4*t17*t26*u3*3.0*10.0 + l3*l4*t19*t26*u1*3.0*10.0 - l3*l4*t17*t28*u2*1.8*10.0 - l3*l4*t19*t26*u2*3.0*10.0 + l3*l4*t17*t28*u3*1.8*10.0 + l3*l4*t19*t28*u1*1.8*10.0 - l3*l4*t19*t28*u2*1.8*10.0 + l2*l4*t4*t17*t26*u3*3.6*10.0 - l2*l4*t4*t17*t26*u4*3.6*10.0 + l2*l4*t4*t17*t28*u3*1.8*10.0 - l2*l4*t4*t17*t28*u4*1.8*10.0 - l3*l4*t19*t26*t34*u1*1.8*10.0 + l3*l4*t19*t26*t34*u2*1.8*10.0 - l3*l4*t19*t28*t34*u1*1.8*10.0 + l3*l4*t19*t28*t34*u2*1.8*10.0 - l1*l4*t19*t26*t39*u3*3.6*10.0 - l2*l3*t17*t26*t40*u4*(9.0 / 2.0) + l1*l4*t19*t26*t39*u4*3.6*10.0 - l1*l4*t19*t28*t39*u3*1.8*10.0 + l1*l4*t19*t28*t39*u4*1.8*10.0 + l1*l3*t19*t26*t59*u4*(9.0 / 2.0) - l2*l4*t17*t26*t73*u3*3.6*10.0 + l2*l4*t17*t26*t73*u4*3.6*10.0 - l2*l4*t17*t28*t73*u3*1.8*10.0 + l2*l4*t17*t28*t73*u4*1.8*10.0 + l1*l4*t19*t26*t101*u3*3.6*10.0 - l1*l4*t19*t26*t101*u4*3.6*10.0 - l2*l3*t17*t26*t103*u4*(2.7*10.0 / 2.0) + l1*l4*t19*t28*t101*u3*1.8*10.0 - l1*l4*t19*t28*t101*u4*1.8*10.0 + l2*l3*t17*t26*t119*u4*(9.0 / 2.0) + l3*l4*t17*t26*t130*u2*1.8*10.0 - l3*l4*t17*t26*t130*u3*1.8*10.0 + l3*l4*t17*t28*t130*u2*1.8*10.0 - l3*l4*t17*t28*t130*u3*1.8*10.0 + l1*l3*t19*t26*t142*u4*(2.7*10.0 / 2.0) - l1*l3*t19*t26*t143*u4*(2.7*10.0 / 2.0) - l1*l3*t19*t26*t144*u4*(9.0 / 2.0) + l2*l3*t17*t26*t193*u4*(2.7*10.0 / 2.0) - l3*l4*m1*m3*t17*u2*1.6*10.0 + l3*l4*m1*m3*t17*u3*1.6*10.0 - l3*l4*m1*m4*t17*u2*3.0*10.0 - l3*l4*m2*m3*t17*u2*4.8*10.0 + l3*l4*m1*m4*t17*u3*3.0*10.0 + l3*l4*m2*m3*t17*u3*4.8*10.0 + l3*l4*m2*m3*t19*u1*1.6*10.0 - l3*l4*m2*m4*t17*u2*9.0*10.0 - l3*l4*m2*m3*t19*u2*1.6*10.0 + l3*l4*m2*m4*t17*u3*9.0*10.0 + l3*l4*m2*m4*t19*u1*3.0*10.0 - l3*l4*m3*m4*t17*u2*7.5*10.0 - l3*l4*m2*m4*t19*u2*3.0*10.0 + l3*l4*m3*m4*t17*u3*7.5*10.0 + l3*l4*m3*m4*t19*u1*7.5*10.0 - l3*l4*m3*m4*t19*u2*7.5*10.0 - g*l1*l3*l4*t6*t19*t27*6.0 + g*l2*l3*l4*t17*t27*t41*6.0 + g*l1*l3*l4*t19*t27*t78*6.0 - g*l2*l3*l4*t17*t27*t104*6.0 + l1*l2*l3*l4*t3*t26*u1*3.0*10.0 - l1*l2*l3*l4*t3*t26*u2*6.0*10.0 + l1*l2*l3*l4*t3*t26*u3*3.0*10.0 + l1*l2*l3*l4*t3*t28*u1*1.8*10.0 - l1*l2*l3*l4*t3*t28*u2*3.6*10.0 + l1*l2*l3*l4*t3*t28*u3*1.8*10.0 - l1*l2*l3*l4*t26*t72*u1*1.8*10.0 + l1*l2*l3*l4*t26*t72*u2*3.6*10.0 - l1*l2*l3*l4*t26*t72*u3*1.8*10.0 - l1*l2*l3*l4*t28*t72*u1*1.8*10.0 + l1*l2*l3*l4*t28*t72*u2*3.6*10.0 - l1*l2*l3*l4*t28*t72*u3*1.8*10.0 + l2*l4*m1*m3*t4*t17*u3*2.4*10.0 - l2*l4*m1*m3*t4*t17*u4*2.4*10.0 + l2*l4*m1*m4*t4*t17*u3*3.0*10.0 + l2*l4*m2*m3*t4*t17*u3*5.4*10.0 - l2*l4*m1*m4*t4*t17*u4*3.0*10.0 - l2*l4*m2*m3*t4*t17*u4*5.4*10.0 + l2*l4*m2*m4*t4*t17*u3*(1.35*100.0 / 2.0) - l2*l4*m2*m4*t4*t17*u4*(1.35*100.0 / 2.0) + l2*l4*m3*m4*t4*t17*u3*(1.35*100.0 / 2.0) - l2*l4*m3*m4*t4*t17*u4*(1.35*100.0 / 2.0) + l3*l4*m1*m4*t17*t35*u2*1.8*10.0 - l3*l4*m1*m4*t17*t35*u3*1.8*10.0 + l3*l4*m2*m4*t17*t35*u2*5.4*10.0 - l3*l4*m2*m4*t17*t35*u3*5.4*10.0 - l3*l4*m2*m4*t19*t35*u1*1.8*10.0 + l3*l4*m3*m4*t17*t35*u2*2.7*10.0 - l3*l4*m3*m4*t19*t34*u1*4.5*10.0 + l3*l4*m2*m4*t19*t35*u2*1.8*10.0 - l3*l4*m3*m4*t17*t35*u3*2.7*10.0 + l3*l4*m3*m4*t19*t34*u2*4.5*10.0 - l3*l4*m3*m4*t19*t35*u1*2.7*10.0 + l2*l3*m1*m3*t17*t40*u4*6.0 + l3*l4*m3*m4*t19*t35*u2*2.7*10.0 - l1*l4*m2*m3*t19*t39*u3*6.0 + l2*l3*m1*m4*t17*t40*u4*3.6*10.0 + l2*l3*m2*m3*t17*t40*u4*(2.7*10.0 / 2.0) + l1*l4*m2*m3*t19*t39*u4*6.0 - l1*l4*m2*m4*t19*t39*u3*(1.5*10.0 / 2.0) + l2*l3*m2*m4*t17*t40*u4*8.1*10.0 + l1*l4*m2*m4*t19*t39*u4*(1.5*10.0 / 2.0) - l1*l4*m3*m4*t19*t39*u3*(1.35*100.0 / 2.0) + l2*l3*m3*m4*t17*t40*u4*9.0 + l1*l4*m3*m4*t19*t39*u4*(1.35*100.0 / 2.0) - l1*l3*m2*m3*t19*t59*u4*(3.0 / 2.0) - l1*l3*m2*m4*t19*t59*u4*9.0 - l1*l3*m3*m4*t19*t59*u4*9.0 - l2*l4*m2*m3*t17*t73*u3*1.8*10.0 + l2*l4*m2*m3*t17*t73*u4*1.8*10.0 - l2*l4*m2*m4*t17*t73*u3*(4.5*10.0 / 2.0) + l2*l4*m2*m4*t17*t73*u4*(4.5*10.0 / 2.0) - l2*l4*m3*m4*t17*t73*u3*(1.35*100.0 / 2.0) - l2*l4*m1*m4*t17*t76*u3*1.8*10.0 + l2*l4*m3*m4*t17*t73*u4*(1.35*100.0 / 2.0) + l2*l4*m1*m4*t17*t76*u4*1.8*10.0 - l2*l4*m2*m4*t17*t76*u3*(8.1*10.0 / 2.0) + l2*l4*m2*m4*t17*t76*u4*(8.1*10.0 / 2.0) - l2*l4*m3*m4*t17*t76*u3*(2.7*10.0 / 2.0) + l2*l4*m3*m4*t17*t76*u4*(2.7*10.0 / 2.0) + l1*l4*m2*m3*t19*t101*u3*1.8*10.0 - l2*l3*m1*m3*t17*t103*u4*1.8*10.0 - l1*l4*m2*m3*t19*t101*u4*1.8*10.0 + l1*l4*m2*m4*t19*t101*u3*(4.5*10.0 / 2.0) - l2*l3*m1*m4*t17*t103*u4*3.6*10.0 - l2*l3*m2*m3*t17*t103*u4*(8.1*10.0 / 2.0) - l1*l4*m2*m4*t19*t101*u4*(4.5*10.0 / 2.0) + l1*l4*m3*m4*t19*t101*u3*(1.35*100.0 / 2.0) - l2*l3*m2*m4*t17*t103*u4*8.1*10.0 - l1*l4*m3*m4*t19*t101*u4*(1.35*100.0 / 2.0) - l2*l3*m3*m4*t17*t103*u4*2.7*10.0 - l2*l3*m2*m3*t17*t119*u4*(9.0 / 2.0) + l1*l4*m2*m4*t19*t118*u3*(9.0 / 2.0) - l2*l3*m2*m4*t17*t119*u4*2.7*10.0 - l1*l4*m2*m4*t19*t118*u4*(9.0 / 2.0) + l1*l4*m3*m4*t19*t118*u3*(2.7*10.0 / 2.0) - l2*l3*m3*m4*t17*t119*u4*9.0 - l1*l4*m3*m4*t19*t118*u4*(2.7*10.0 / 2.0) + l3*l4*m3*m4*t17*t130*u2*4.5*10.0 - l3*l4*m3*m4*t17*t130*u3*4.5*10.0 + l3*l4*m3*m4*t19*t132*u1*9.0 - l3*l4*m3*m4*t19*t132*u2*9.0 + l1*l3*m2*m3*t19*t142*u4*(9.0 / 2.0) - l1*l3*m2*m3*t19*t143*u4*(2.7*10.0 / 2.0) + l1*l3*m2*m4*t19*t142*u4*9.0 + l1*l3*m2*m3*t19*t144*u4*(9.0 / 2.0) - l1*l3*m2*m4*t19*t143*u4*2.7*10.0 + l1*l3*m3*m4*t19*t142*u4*2.7*10.0 + l1*l3*m2*m4*t19*t144*u4*2.7*10.0 - l1*l3*m3*m4*t19*t143*u4*2.7*10.0 + l1*l3*m3*m4*t19*t144*u4*9.0 + l2*l4*m2*m4*t17*t181*u3*(2.7*10.0 / 2.0) - l2*l4*m2*m4*t17*t181*u4*(2.7*10.0 / 2.0) + l2*l4*m3*m4*t17*t181*u3*(2.7*10.0 / 2.0) - l2*l4*m3*m4*t17*t181*u4*(2.7*10.0 / 2.0) + l2*l3*m2*m3*t17*t193*u4*(2.7*10.0 / 2.0) - l1*l4*m2*m4*t19*t192*u3*(2.7*10.0 / 2.0) + l2*l3*m2*m4*t17*t193*u4*2.7*10.0 + l1*l4*m2*m4*t19*t192*u4*(2.7*10.0 / 2.0) - l1*l4*m3*m4*t19*t192*u3*(2.7*10.0 / 2.0) + l2*l3*m3*m4*t17*t193*u4*2.7*10.0 + l1*l4*m3*m4*t19*t192*u4*(2.7*10.0 / 2.0) - l3*l4*m3*m4*t17*t202*u2*9.0 + l3*l4*m3*m4*t17*t202*u3*9.0 + l2*l3*l4*t7*t13*t18*t27*1.2*10.0 + l1*l3*l4*t7*t13*t20*t27*1.2*10.0 + l1*l3*l4*t7*t14*t20*t27*1.2*10.0 - l2*l4*t8*t13*t17*t21*t27*3.0 - l2*l4*t8*t14*t17*t21*t27*3.0 - l2*l4*t8*t15*t17*t21*t27*3.0 + l3*l4*t13*t17*t19*t27*t36*1.2*10.0 + l3*l4*t14*t17*t19*t27*t36*6.0 + l1*l4*t13*t19*t21*t27*t42*3.0 + l1*l4*t14*t19*t21*t27*t42*3.0 + l1*l4*t15*t19*t21*t27*t42*3.0 + l2*l4*t13*t17*t21*t27*t82*3.0 + l2*l4*t14*t17*t21*t27*t82*3.0 + l2*l4*t15*t17*t21*t27*t82*3.0 + l1*l4*t13*t19*t21*t27*t106*3.0 + l1*l4*t14*t19*t21*t27*t106*3.0 + l1*l4*t15*t19*t21*t27*t106*3.0 + dth1*dth2*l1*l3*l4*t7*t20*t27*2.4*10.0 - dth1*dth2*l2*l4*t8*t17*t21*t27*6.0 - dth1*dth3*l2*l4*t8*t17*t21*t27*6.0 - dth2*dth3*l2*l4*t8*t17*t21*t27*6.0 + dth1*dth2*l3*l4*t17*t19*t27*t36*1.2*10.0 + dth1*dth2*l1*l4*t19*t21*t27*t42*6.0 + dth1*dth3*l1*l4*t19*t21*t27*t42*6.0 + dth2*dth3*l1*l4*t19*t21*t27*t42*6.0 + dth1*dth2*l2*l4*t17*t21*t27*t82*6.0 + dth1*dth3*l2*l4*t17*t21*t27*t82*6.0 + dth2*dth3*l2*l4*t17*t21*t27*t82*6.0 + dth1*dth2*l1*l4*t19*t21*t27*t106*6.0 + dth1*dth3*l1*l4*t19*t21*t27*t106*6.0 + dth2*dth3*l1*l4*t19*t21*t27*t106*6.0 + l1*l2*l3*l4*m2*m3*t3*u1*2.4*10.0 - l1*l2*l3*l4*m2*m3*t3*u2*4.8*10.0 + l1*l2*l3*l4*m2*m4*t3*u1*4.5*10.0 + l1*l2*l3*l4*m2*m3*t3*u3*2.4*10.0 - l1*l2*l3*l4*m2*m4*t3*u2*9.0*10.0 + l1*l2*l3*l4*m3*m4*t3*u1*7.5*10.0 + l1*l2*l3*l4*m2*m4*t3*u3*4.5*10.0 - l1*l2*l3*l4*m3*m4*t3*u2*1.5*100.0 + l1*l2*l3*l4*m3*m4*t3*u3*7.5*10.0 - l1*l2*l3*l4*m3*m4*t72*u1*4.5*10.0 - l1*l2*l3*l4*m2*m4*t74*u1*(2.7*10.0 / 2.0) + l1*l2*l3*l4*m3*m4*t72*u2*9.0*10.0 + l1*l2*l3*l4*m2*m4*t74*u2*2.7*10.0 - l1*l2*l3*l4*m3*m4*t72*u3*4.5*10.0 - l1*l2*l3*l4*m3*m4*t74*u1*(2.7*10.0 / 2.0) - l1*l2*l3*l4*m2*m4*t74*u3*(2.7*10.0 / 2.0) + l1*l2*l3*l4*m3*m4*t74*u2*2.7*10.0 - l1*l2*l3*l4*m3*m4*t74*u3*(2.7*10.0 / 2.0) - l1*l2*l3*l4*m2*m4*t102*u1*(2.7*10.0 / 2.0) + l1*l2*l3*l4*m2*m4*t102*u2*2.7*10.0 - l1*l2*l3*l4*m3*m4*t102*u1*(2.7*10.0 / 2.0) - l1*l2*l3*l4*m2*m4*t102*u3*(2.7*10.0 / 2.0) + l1*l2*l3*l4*m3*m4*t102*u2*2.7*10.0 - l1*l2*l3*l4*m3*m4*t102*u3*(2.7*10.0 / 2.0) + l1*l2*l3*l4*m3*m4*t180*u1*9.0 - l1*l2*l3*l4*m3*m4*t180*u2*1.8*10.0 + l1*l2*l3*l4*m3*m4*t180*u3*9.0 - g*l1*l3*l4*m1*t6*t19*t26*1.5*10.0 - g*l1*l3*l4*m2*t6*t19*t26*2.5*10.0 - g*l1*l3*l4*m3*t6*t19*t25*1.0*10.0 - g*l1*l3*l4*m1*t6*t19*t28*9.0 - g*l1*l3*l4*m4*t6*t19*t25*(7.5*10.0 / 4.0) - g*l1*l3*l4*m2*t6*t19*t28*1.5*10.0 - g*l1*l3*l4*m4*t6*t19*t26*(7.5*10.0 / 4.0) - g*l1*l3*l4*m3*t6*t19*t28*6.0 + g*l2*l3*l4*m1*t17*t26*t41*(5.0 / 2.0) + g*l2*l3*l4*m2*t17*t26*t41*(4.5*10.0 / 2.0) + g*l2*l3*l4*m3*t17*t25*t41*1.2*10.0 + g*l2*l3*l4*m1*t17*t28*t41*(3.0 / 2.0) + g*l2*l3*l4*m4*t17*t25*t41*(4.5*10.0 / 2.0) + g*l2*l3*l4*m2*t17*t28*t41*(2.7*10.0 / 2.0) + g*l2*l3*l4*m4*t17*t26*t41*(7.5*10.0 / 4.0) + g*l2*l3*l4*m3*t17*t28*t41*6.0 + g*l1*l3*l4*m1*t19*t26*t79*(9.0 / 2.0) + g*l1*l3*l4*m2*t19*t26*t78*1.5*10.0 + g*l1*l3*l4*m3*t19*t25*t78*6.0 + g*l1*l3*l4*m2*t19*t26*t79*(9.0 / 2.0) + g*l1*l3*l4*m4*t19*t25*t78*(4.5*10.0 / 4.0) + g*l1*l3*l4*m1*t19*t28*t79*(9.0 / 2.0) + g*l1*l3*l4*m2*t19*t28*t78*9.0 + g*l1*l3*l4*m4*t19*t26*t78*(7.5*10.0 / 4.0) + g*l1*l3*l4*m2*t19*t28*t79*(9.0 / 2.0) + g*l1*l3*l4*m3*t19*t28*t78*6.0 + g*l1*l3*l4*m4*t19*t25*t80*(4.5*10.0 / 8.0) + g*l1*l3*l4*m4*t19*t26*t80*(2.7*10.0 / 8.0) - g*l2*l3*l4*m1*t17*t26*t104*(1.5*10.0 / 2.0) - g*l2*l3*l4*m2*t17*t26*t104*(4.5*10.0 / 2.0) - g*l2*l3*l4*m3*t17*t25*t104*1.2*10.0 + g*l1*l3*l4*m1*t19*t26*t105*(9.0 / 2.0) - g*l2*l3*l4*m1*t17*t28*t104*(9.0 / 2.0) - g*l2*l3*l4*m4*t17*t25*t104*(4.5*10.0 / 2.0) + g*l1*l3*l4*m2*t19*t26*t105*(9.0 / 2.0) - g*l2*l3*l4*m2*t17*t28*t104*(2.7*10.0 / 2.0) - g*l2*l3*l4*m4*t17*t26*t104*(7.5*10.0 / 4.0) + g*l1*l3*l4*m1*t19*t28*t105*(9.0 / 2.0) - g*l2*l3*l4*m3*t17*t28*t104*6.0 + g*l1*l3*l4*m2*t19*t28*t105*(9.0 / 2.0) + g*l1*l3*l4*m4*t19*t25*t107*(4.5*10.0 / 8.0) + g*l1*l3*l4*m4*t19*t26*t107*(2.7*10.0 / 8.0) - g*l2*l3*l4*m1*t17*t26*t120*(3.0 / 2.0) - g*l2*l3*l4*m2*t17*t26*t120*(9.0 / 2.0) - g*l2*l3*l4*m1*t17*t28*t120*(3.0 / 2.0) - g*l2*l3*l4*m2*t17*t28*t120*(9.0 / 2.0) - g*l2*l3*l4*m4*t17*t25*t121*(2.7*10.0 / 4.0) - g*l2*l3*l4*m4*t17*t26*t121*(2.7*10.0 / 8.0) - g*l2*l3*l4*m4*t17*t25*t148*(2.7*10.0 / 4.0) - g*l2*l3*l4*m4*t17*t26*t148*(2.7*10.0 / 8.0) - g*l1*l3*l4*m2*t19*t26*t183*3.0 - g*l1*l3*l4*m2*t19*t28*t183*3.0 - g*l1*l3*l4*m4*t19*t25*t184*(2.7*10.0 / 8.0) - g*l1*l3*l4*m4*t19*t26*t184*(2.7*10.0 / 8.0) - g*l2*l3*l4*m1*t17*t26*t194*(9.0 / 2.0) - g*l2*l3*l4*m2*t17*t26*t194*(9.0 / 2.0) - g*l2*l3*l4*m1*t17*t28*t194*(9.0 / 2.0) - g*l2*l3*l4*m2*t17*t28*t194*(9.0 / 2.0) + g*l2*l3*l4*m4*t17*t25*t195*(2.7*10.0 / 4.0) - g*l2*l3*l4*m4*t17*t25*t196*(2.7*10.0 / 4.0) + g*l2*l3*l4*m4*t17*t26*t195*(2.7*10.0 / 8.0) - g*l2*l3*l4*m4*t17*t26*t196*(2.7*10.0 / 8.0) - g*l1*l3*l4*m4*t19*t25*t197*(2.7*10.0 / 8.0) - g*l1*l3*l4*m4*t19*t26*t197*(2.7*10.0 / 8.0) + l2*l3*l4*m1*t7*t13*t18*t26*1.0*10.0 + l2*l3*l4*m2*t7*t13*t18*t26*4.5*10.0 + l2*l3*l4*m3*t7*t13*t18*t25*2.4*10.0 + l1*l3*l4*m2*t7*t13*t20*t26*2.5*10.0 + l1*l3*l4*m3*t7*t13*t20*t25*8.0 + l2*l3*l4*m1*t7*t13*t18*t28*6.0 + l2*l3*l4*m4*t7*t13*t18*t25*4.5*10.0 + l1*l3*l4*m2*t7*t14*t20*t26*2.5*10.0 + l1*l3*l4*m3*t7*t14*t20*t25*8.0 + l1*l3*l4*m4*t7*t13*t20*t25*1.5*10.0 + l2*l3*l4*m2*t7*t13*t18*t28*2.7*10.0 + l2*l3*l4*m4*t7*t13*t18*t26*(7.5*10.0 / 2.0) + l1*l3*l4*m2*t7*t13*t20*t28*1.5*10.0 + l1*l3*l4*m4*t7*t13*t20*t26*(7.5*10.0 / 2.0) + l1*l3*l4*m4*t7*t14*t20*t25*1.5*10.0 + l2*l3*l4*m3*t7*t13*t18*t28*1.2*10.0 + l1*l3*l4*m2*t7*t14*t20*t28*1.5*10.0 + l1*l3*l4*m3*t7*t13*t20*t28*1.2*10.0 + l1*l3*l4*m4*t7*t14*t20*t26*(7.5*10.0 / 2.0) + l1*l3*l4*m3*t7*t14*t20*t28*1.2*10.0 - l2*l3*l4*m1*t13*t18*t26*t81*6.0 - l2*l3*l4*m2*t13*t18*t26*t81*9.0 - l1*l3*l4*m2*t13*t20*t26*t81*3.0 - l2*l3*l4*m1*t13*t18*t28*t81*6.0 - l1*l3*l4*m2*t14*t20*t26*t81*3.0 - l2*l3*l4*m2*t13*t18*t28*t81*9.0 - l1*l3*l4*m2*t13*t20*t28*t81*3.0 - l2*l3*l4*m4*t13*t18*t25*t83*(2.7*10.0 / 2.0) - l1*l3*l4*m2*t14*t20*t28*t81*3.0 - l1*l3*l4*m4*t13*t20*t25*t83*(9.0 / 2.0) - l2*l3*l4*m4*t13*t18*t26*t83*(2.7*10.0 / 4.0) - l1*l3*l4*m4*t13*t20*t26*t83*(2.7*10.0 / 4.0) - l1*l3*l4*m4*t14*t20*t25*t83*(9.0 / 2.0) - l1*l3*l4*m4*t14*t20*t26*t83*(2.7*10.0 / 4.0) - l2*l3*l4*m4*t13*t18*t25*t108*(2.7*10.0 / 2.0) - l1*l3*l4*m4*t13*t20*t25*t108*(9.0 / 2.0) - l2*l3*l4*m4*t13*t18*t26*t108*(2.7*10.0 / 4.0) - l1*l3*l4*m4*t13*t20*t26*t108*(2.7*10.0 / 4.0) - l1*l3*l4*m4*t14*t20*t25*t108*(9.0 / 2.0) - l1*l3*l4*m4*t14*t20*t26*t108*(2.7*10.0 / 4.0) - l2*l4*m1*t8*t13*t17*t21*t26*8.0 - l2*l4*m1*t8*t14*t17*t21*t26*8.0 - l2*l4*m2*t8*t13*t17*t21*t26*1.8*10.0 - l2*l4*m1*t8*t13*t17*t21*t28*1.2*10.0 - l2*l4*m1*t8*t15*t17*t21*t26*8.0 - l2*l4*m2*t8*t14*t17*t21*t26*1.8*10.0 - l2*l4*m1*t8*t14*t17*t21*t28*1.2*10.0 - l2*l4*m2*t8*t13*t17*t21*t28*2.7*10.0 - l2*l4*m2*t8*t15*t17*t21*t26*1.8*10.0 - l2*l4*m4*t8*t13*t17*t21*t26*(4.5*10.0 / 4.0) - l2*l4*m1*t8*t15*t17*t21*t28*1.2*10.0 - l2*l4*m2*t8*t14*t17*t21*t28*2.7*10.0 - l2*l4*m3*t8*t13*t17*t21*t28*6.0 - l2*l4*m4*t8*t14*t17*t21*t26*(4.5*10.0 / 4.0) - l2*l4*m2*t8*t15*t17*t21*t28*2.7*10.0 - l2*l4*m3*t8*t14*t17*t21*t28*6.0 - l2*l4*m4*t8*t15*t17*t21*t26*(4.5*10.0 / 4.0) - l2*l4*m3*t8*t15*t17*t21*t28*6.0 - l3*l4*m1*t13*t17*t19*t26*t37*6.0 + l3*l4*m2*t13*t17*t19*t26*t36*3.0*10.0 + l3*l4*m3*t13*t17*t19*t25*t36*1.2*10.0 - l3*l4*m1*t14*t17*t19*t26*t37*6.0 - l3*l4*m2*t13*t17*t19*t26*t37*9.0 + l3*l4*m2*t14*t17*t19*t26*t36*1.5*10.0 + l3*l4*m3*t14*t17*t19*t25*t36*6.0 + l3*l4*m4*t13*t17*t19*t25*t36*(4.5*10.0 / 2.0) - l3*l4*m1*t13*t17*t19*t28*t37*6.0 + l3*l4*m2*t13*t17*t19*t28*t36*1.8*10.0 - l3*l4*m2*t14*t17*t19*t26*t37*9.0 + l3*l4*m4*t13*t17*t19*t26*t36*(7.5*10.0 / 2.0) + l3*l4*m4*t14*t17*t19*t25*t36*(4.5*10.0 / 4.0) - l3*l4*m1*t14*t17*t19*t28*t37*6.0 - l3*l4*m2*t13*t17*t19*t28*t37*9.0 + l3*l4*m2*t14*t17*t19*t28*t36*9.0 + l3*l4*m3*t13*t17*t19*t28*t36*1.2*10.0 + l3*l4*m4*t14*t17*t19*t26*t36*(7.5*10.0 / 4.0) - l3*l4*m2*t14*t17*t19*t28*t37*9.0 + l3*l4*m3*t14*t17*t19*t28*t36*6.0 + l1*l4*m2*t13*t19*t21*t26*t42*2.0 + l1*l4*m2*t14*t19*t21*t26*t42*2.0 + l1*l4*m2*t13*t19*t21*t28*t42*3.0 + l1*l4*m2*t15*t19*t21*t26*t42*2.0 + l1*l4*m4*t13*t19*t21*t26*t42*(4.5*10.0 / 4.0) - l2*l3*m1*t13*t17*t23*t28*t43*3.0 + l1*l4*m2*t14*t19*t21*t28*t42*3.0 + l1*l4*m3*t13*t19*t21*t28*t42*6.0 + l1*l4*m4*t14*t19*t21*t26*t42*(4.5*10.0 / 4.0) - l2*l3*m1*t14*t17*t23*t28*t43*3.0 - l2*l3*m2*t13*t17*t23*t28*t43*(2.7*10.0 / 4.0) + l2*l3*m4*t13*t17*t23*t26*t43*(3.0 / 2.0) + l1*l4*m2*t15*t19*t21*t28*t42*3.0 + l1*l4*m3*t14*t19*t21*t28*t42*6.0 + l1*l4*m4*t15*t19*t21*t26*t42*(4.5*10.0 / 4.0) - l2*l3*m1*t15*t17*t23*t28*t43*3.0 - l2*l3*m2*t14*t17*t23*t28*t43*(2.7*10.0 / 4.0) - l2*l3*m3*t13*t17*t23*t28*t43*(3.0 / 4.0) + l2*l3*m4*t14*t17*t23*t26*t43*(3.0 / 2.0) + l1*l4*m3*t15*t19*t21*t28*t42*6.0 - l2*l3*m1*t16*t17*t23*t28*t43*3.0 - l2*l3*m2*t15*t17*t23*t28*t43*(2.7*10.0 / 4.0) - l2*l3*m3*t14*t17*t23*t28*t43*(3.0 / 4.0) + l2*l3*m4*t15*t17*t23*t26*t43*(3.0 / 2.0) - l2*l3*m2*t16*t17*t23*t28*t43*(2.7*10.0 / 4.0) - l2*l3*m3*t15*t17*t23*t28*t43*(3.0 / 4.0) + l2*l3*m4*t16*t17*t23*t26*t43*(3.0 / 2.0) - l2*l3*m3*t16*t17*t23*t28*t43*(3.0 / 4.0) + l1*l3*m2*t13*t19*t23*t28*t61*(3.0 / 4.0) - l1*l3*m4*t13*t19*t23*t26*t61*(3.0 / 2.0) + l1*l3*m2*t14*t19*t23*t28*t61*(3.0 / 4.0) + l1*l3*m3*t13*t19*t23*t28*t61*(3.0 / 4.0) - l1*l3*m4*t14*t19*t23*t26*t61*(3.0 / 2.0) + l1*l3*m2*t15*t19*t23*t28*t61*(3.0 / 4.0) + l1*l3*m3*t14*t19*t23*t28*t61*(3.0 / 4.0) - l1*l3*m4*t15*t19*t23*t26*t61*(3.0 / 2.0) + l1*l3*m2*t16*t19*t23*t28*t61*(3.0 / 4.0) + l1*l3*m3*t15*t19*t23*t28*t61*(3.0 / 4.0) - l1*l3*m4*t16*t19*t23*t26*t61*(3.0 / 2.0) + l1*l3*m3*t16*t19*t23*t28*t61*(3.0 / 4.0) + l2*l4*m2*t13*t17*t21*t26*t82*6.0 + l2*l4*m2*t14*t17*t21*t26*t82*6.0 + l2*l4*m2*t13*t17*t21*t28*t82*9.0 + l2*l4*m2*t15*t17*t21*t26*t82*6.0 + l2*l4*m4*t13*t17*t21*t26*t82*(4.5*10.0 / 4.0) + l2*l4*m2*t14*t17*t21*t28*t82*9.0 + l2*l4*m3*t13*t17*t21*t28*t82*6.0 + l2*l4*m4*t14*t17*t21*t26*t82*(4.5*10.0 / 4.0) + l2*l4*m2*t15*t17*t21*t28*t82*9.0 + l2*l4*m3*t14*t17*t21*t28*t82*6.0 + l2*l4*m4*t15*t17*t21*t26*t82*(4.5*10.0 / 4.0) + l2*l4*m3*t15*t17*t21*t28*t82*6.0 + l2*l4*m4*t13*t17*t21*t26*t85*(9.0 / 4.0) + l2*l4*m4*t14*t17*t21*t26*t85*(9.0 / 4.0) + l2*l4*m4*t15*t17*t21*t26*t85*(9.0 / 4.0) + l1*l4*m2*t13*t19*t21*t26*t106*6.0 + l1*l4*m2*t14*t19*t21*t26*t106*6.0 + l1*l4*m2*t13*t19*t21*t28*t106*9.0 + l1*l4*m2*t15*t19*t21*t26*t106*6.0 + l1*l4*m4*t13*t19*t21*t26*t106*(4.5*10.0 / 4.0) + l1*l4*m2*t14*t19*t21*t28*t106*9.0 + l1*l4*m3*t13*t19*t21*t28*t106*6.0 + l1*l4*m4*t14*t19*t21*t26*t106*(4.5*10.0 / 4.0) + l1*l4*m2*t15*t19*t21*t28*t106*9.0 + l1*l4*m3*t14*t19*t21*t28*t106*6.0 + l1*l4*m4*t15*t19*t21*t26*t106*(4.5*10.0 / 4.0) - l2*l3*m1*t13*t17*t23*t28*t109*3.0 + l1*l4*m3*t15*t19*t21*t28*t106*6.0 - l2*l3*m1*t14*t17*t23*t28*t109*3.0 - l2*l3*m2*t13*t17*t23*t28*t109*(2.7*10.0 / 4.0) - l2*l3*m4*t13*t17*t23*t26*t109*(9.0 / 2.0) - l2*l3*m1*t15*t17*t23*t28*t109*3.0 - l2*l3*m2*t14*t17*t23*t28*t109*(2.7*10.0 / 4.0) - l2*l3*m3*t13*t17*t23*t28*t109*(9.0 / 4.0) - l2*l3*m4*t14*t17*t23*t26*t109*(9.0 / 2.0) - l2*l3*m1*t16*t17*t23*t28*t109*3.0 - l2*l3*m2*t15*t17*t23*t28*t109*(2.7*10.0 / 4.0) - l2*l3*m3*t14*t17*t23*t28*t109*(9.0 / 4.0) - l2*l3*m4*t15*t17*t23*t26*t109*(9.0 / 2.0) - l2*l3*m2*t16*t17*t23*t28*t109*(2.7*10.0 / 4.0) - l2*l3*m3*t15*t17*t23*t28*t109*(9.0 / 4.0) - l2*l3*m4*t16*t17*t23*t26*t109*(9.0 / 2.0) - l2*l3*m3*t16*t17*t23*t28*t109*(9.0 / 4.0) - l1*l4*m4*t13*t19*t21*t26*t122*(9.0 / 4.0) - l1*l4*m4*t14*t19*t21*t26*t122*(9.0 / 4.0) + l2*l3*m2*t13*t17*t23*t28*t123*(9.0 / 4.0) - l2*l3*m4*t13*t17*t23*t26*t123*(3.0 / 2.0) - l1*l4*m4*t15*t19*t21*t26*t122*(9.0 / 4.0) + l2*l3*m2*t14*t17*t23*t28*t123*(9.0 / 4.0) + l2*l3*m3*t13*t17*t23*t28*t123*(3.0 / 4.0) - l2*l3*m4*t14*t17*t23*t26*t123*(3.0 / 2.0) + l2*l3*m2*t15*t17*t23*t28*t123*(9.0 / 4.0) + l2*l3*m3*t14*t17*t23*t28*t123*(3.0 / 4.0) - l2*l3*m4*t15*t17*t23*t26*t123*(3.0 / 2.0) + l2*l3*m2*t16*t17*t23*t28*t123*(9.0 / 4.0) + l2*l3*m3*t15*t17*t23*t28*t123*(3.0 / 4.0) - l2*l3*m4*t16*t17*t23*t26*t123*(3.0 / 2.0) + l2*l3*m3*t16*t17*t23*t28*t123*(3.0 / 4.0) - l3*l4*m2*t13*t17*t19*t26*t133*3.0 - l3*l4*m2*t13*t17*t19*t28*t133*3.0 - l3*l4*m4*t13*t17*t19*t25*t134*(2.7*10.0 / 4.0) - l3*l4*m4*t13*t17*t19*t26*t134*(2.7*10.0 / 4.0) - l3*l4*m4*t14*t17*t19*t25*t134*(2.7*10.0 / 8.0) - l3*l4*m4*t14*t17*t19*t26*t134*(2.7*10.0 / 8.0) + l1*l3*m2*t13*t19*t23*t28*t149*(3.0 / 4.0) + l1*l3*m4*t13*t19*t23*t26*t149*(9.0 / 2.0) + l1*l3*m2*t13*t19*t23*t28*t150*(9.0 / 4.0) + l1*l3*m2*t14*t19*t23*t28*t149*(3.0 / 4.0) + l1*l3*m3*t13*t19*t23*t28*t149*(9.0 / 4.0) + l1*l3*m4*t13*t19*t23*t26*t150*(9.0 / 2.0) + l1*l3*m4*t14*t19*t23*t26*t149*(9.0 / 2.0) - l1*l3*m2*t13*t19*t23*t28*t151*(9.0 / 4.0) + l1*l3*m2*t14*t19*t23*t28*t150*(9.0 / 4.0) + l1*l3*m2*t15*t19*t23*t28*t149*(3.0 / 4.0) + l1*l3*m3*t13*t19*t23*t28*t150*(9.0 / 4.0) + l1*l3*m3*t14*t19*t23*t28*t149*(9.0 / 4.0) + l1*l3*m4*t13*t19*t23*t26*t151*(3.0 / 2.0) + l1*l3*m4*t14*t19*t23*t26*t150*(9.0 / 2.0) + l1*l3*m4*t15*t19*t23*t26*t149*(9.0 / 2.0) - l1*l3*m2*t14*t19*t23*t28*t151*(9.0 / 4.0) + l1*l3*m2*t15*t19*t23*t28*t150*(9.0 / 4.0) + l1*l3*m2*t16*t19*t23*t28*t149*(3.0 / 4.0) - l1*l3*m3*t13*t19*t23*t28*t151*(3.0 / 4.0) + l1*l3*m3*t14*t19*t23*t28*t150*(9.0 / 4.0) + l1*l3*m3*t15*t19*t23*t28*t149*(9.0 / 4.0) + l1*l3*m4*t14*t19*t23*t26*t151*(3.0 / 2.0) + l1*l3*m4*t15*t19*t23*t26*t150*(9.0 / 2.0) + l1*l3*m4*t16*t19*t23*t26*t149*(9.0 / 2.0) - l1*l3*m2*t15*t19*t23*t28*t151*(9.0 / 4.0) + l1*l3*m2*t16*t19*t23*t28*t150*(9.0 / 4.0) - l1*l3*m3*t14*t19*t23*t28*t151*(3.0 / 4.0) + l1*l3*m3*t15*t19*t23*t28*t150*(9.0 / 4.0) + l1*l3*m3*t16*t19*t23*t28*t149*(9.0 / 4.0) + l1*l3*m4*t15*t19*t23*t26*t151*(3.0 / 2.0) + l1*l3*m4*t16*t19*t23*t26*t150*(9.0 / 2.0) - l1*l3*m2*t16*t19*t23*t28*t151*(9.0 / 4.0) - l1*l3*m3*t15*t19*t23*t28*t151*(3.0 / 4.0) + l1*l3*m3*t16*t19*t23*t28*t150*(9.0 / 4.0) + l1*l3*m4*t16*t19*t23*t26*t151*(3.0 / 2.0) - l1*l3*m3*t16*t19*t23*t28*t151*(3.0 / 4.0) - l3*l4*m4*t13*t17*t19*t25*t172*(2.7*10.0 / 4.0) - l3*l4*m4*t13*t17*t19*t26*t172*(2.7*10.0 / 4.0) - l3*l4*m4*t14*t17*t19*t25*t172*(2.7*10.0 / 8.0) - l3*l4*m4*t14*t17*t19*t26*t172*(2.7*10.0 / 8.0) - l2*l4*m4*t13*t17*t21*t26*t187*(9.0 / 4.0) - l2*l4*m4*t14*t17*t21*t26*t187*(9.0 / 4.0) - l2*l4*m4*t15*t17*t21*t26*t187*(9.0 / 4.0) + l1*l4*m4*t13*t19*t21*t26*t198*(9.0 / 4.0) + l1*l4*m4*t14*t19*t21*t26*t198*(9.0 / 4.0) + l2*l3*m2*t13*t17*t23*t28*t199*(9.0 / 4.0) + l2*l3*m4*t13*t17*t23*t26*t199*(9.0 / 2.0) + l1*l4*m4*t15*t19*t21*t26*t198*(9.0 / 4.0) + l2*l3*m2*t14*t17*t23*t28*t199*(9.0 / 4.0) + l2*l3*m3*t13*t17*t23*t28*t199*(9.0 / 4.0) + l2*l3*m4*t14*t17*t23*t26*t199*(9.0 / 2.0) + l2*l3*m2*t15*t17*t23*t28*t199*(9.0 / 4.0) + l2*l3*m3*t14*t17*t23*t28*t199*(9.0 / 4.0) + l2*l3*m4*t15*t17*t23*t26*t199*(9.0 / 2.0) + l2*l3*m2*t16*t17*t23*t28*t199*(9.0 / 4.0) + l2*l3*m3*t15*t17*t23*t28*t199*(9.0 / 4.0) + l2*l3*m4*t16*t17*t23*t26*t199*(9.0 / 2.0) + l2*l3*m3*t16*t17*t23*t28*t199*(9.0 / 4.0) + dth1*dth2*l1*l3*l4*m2*t7*t20*t26*5.0*10.0 + dth1*dth2*l1*l3*l4*m3*t7*t20*t25*1.6*10.0 + dth1*dth2*l1*l3*l4*m4*t7*t20*t25*3.0*10.0 + dth1*dth2*l1*l3*l4*m2*t7*t20*t28*3.0*10.0 + dth1*dth2*l1*l3*l4*m4*t7*t20*t26*7.5*10.0 + dth1*dth2*l1*l3*l4*m3*t7*t20*t28*2.4*10.0 - dth1*dth2*l1*l3*l4*m2*t20*t26*t81*6.0 - dth1*dth2*l1*l3*l4*m2*t20*t28*t81*6.0 - dth1*dth2*l1*l3*l4*m4*t20*t25*t83*9.0 - dth1*dth2*l1*l3*l4*m4*t20*t26*t83*(2.7*10.0 / 2.0) - dth1*dth2*l1*l3*l4*m4*t20*t25*t108*9.0 - dth1*dth2*l1*l3*l4*m4*t20*t26*t108*(2.7*10.0 / 2.0) - g*l1*l3*l4*m1*m2*m3*t6*t19*8.0 - g*l1*l3*l4*m1*m2*m4*t6*t19*1.5*10.0 - g*l1*l3*l4*m1*m3*m4*t6*t19*(7.5*10.0 / 2.0) - g*l1*l3*l4*m2*m3*m4*t6*t19*(1.25*100.0 / 2.0) + g*l2*l3*l4*m1*m2*m3*t17*t41*2.0 + g*l2*l3*l4*m1*m2*m4*t17*t41*(1.5*10.0 / 4.0) + g*l2*l3*l4*m1*m3*m4*t17*t41*(2.5*10.0 / 4.0) + g*l2*l3*l4*m2*m3*m4*t17*t41*(2.25*100.0 / 4.0) + g*l1*l3*l4*m1*m2*m4*t19*t80*(9.0 / 2.0) + g*l1*l3*l4*m1*m3*m4*t19*t79*(4.5*10.0 / 4.0) + g*l1*l3*l4*m2*m3*m4*t19*t78*(7.5*10.0 / 2.0) + g*l1*l3*l4*m1*m3*m4*t19*t80*(2.7*10.0 / 4.0) + g*l1*l3*l4*m2*m3*m4*t19*t79*(4.5*10.0 / 4.0) + g*l1*l3*l4*m2*m3*m4*t19*t80*(4.5*10.0 / 4.0) - g*l2*l3*l4*m1*m2*m3*t17*t104*6.0 - g*l2*l3*l4*m1*m2*m4*t17*t104*(4.5*10.0 / 4.0) - g*l2*l3*l4*m1*m3*m4*t17*t104*(7.5*10.0 / 4.0) - g*l2*l3*l4*m2*m3*m4*t17*t104*(2.25*100.0 / 4.0) + g*l1*l3*l4*m1*m3*m4*t19*t105*(4.5*10.0 / 4.0) + g*l1*l3*l4*m1*m2*m4*t19*t107*(9.0 / 2.0) + g*l1*l3*l4*m2*m3*m4*t19*t105*(4.5*10.0 / 4.0) + g*l1*l3*l4*m1*m3*m4*t19*t107*(2.7*10.0 / 4.0) + g*l1*l3*l4*m2*m3*m4*t19*t107*(4.5*10.0 / 4.0) - g*l2*l3*l4*m1*m2*m4*t17*t121*(9.0 / 8.0) - g*l2*l3*l4*m1*m3*m4*t17*t120*(1.5*10.0 / 4.0) - g*l2*l3*l4*m1*m3*m4*t17*t121*(9.0 / 8.0) - g*l2*l3*l4*m2*m3*m4*t17*t120*(4.5*10.0 / 4.0) - g*l2*l3*l4*m2*m3*m4*t17*t121*(8.1*10.0 / 8.0) - g*l2*l3*l4*m1*m2*m4*t17*t148*(9.0 / 8.0) - g*l2*l3*l4*m1*m3*m4*t17*t148*(9.0 / 8.0) - g*l2*l3*l4*m2*m3*m4*t17*t148*(8.1*10.0 / 8.0) - g*l1*l3*l4*m2*m3*m4*t19*t183*(1.5*10.0 / 2.0) - g*l1*l3*l4*m1*m3*m4*t19*t185*(9.0 / 4.0) - g*l1*l3*l4*m2*m3*m4*t19*t184*(2.7*10.0 / 4.0) - g*l1*l3*l4*m2*m3*m4*t19*t185*(9.0 / 4.0) + g*l2*l3*l4*m1*m3*m4*t17*t190*(3.0 / 4.0) + g*l2*l3*l4*m2*m3*m4*t17*t190*(9.0 / 4.0) + g*l2*l3*l4*m1*m2*m4*t17*t195*(2.7*10.0 / 8.0) - g*l2*l3*l4*m1*m3*m4*t17*t194*(4.5*10.0 / 4.0) - g*l2*l3*l4*m1*m2*m4*t17*t196*(2.7*10.0 / 8.0) + g*l2*l3*l4*m1*m3*m4*t17*t195*(2.7*10.0 / 8.0) - g*l2*l3*l4*m2*m3*m4*t17*t194*(4.5*10.0 / 4.0) - g*l2*l3*l4*m1*m3*m4*t17*t196*(2.7*10.0 / 8.0) + g*l2*l3*l4*m2*m3*m4*t17*t195*(8.1*10.0 / 8.0) - g*l2*l3*l4*m2*m3*m4*t17*t196*(8.1*10.0 / 8.0) - g*l1*l3*l4*m2*m3*m4*t19*t197*(2.7*10.0 / 4.0) + g*l1*l3*l4*m1*m3*m4*t19*t215*(9.0 / 4.0) + g*l1*l3*l4*m2*m3*m4*t19*t215*(9.0 / 4.0) + g*l2*l3*l4*m1*m3*m4*t17*t240*(9.0 / 4.0) + g*l2*l3*l4*m2*m3*m4*t17*t240*(9.0 / 4.0) - dth1*dth2*l2*l4*m1*t8*t17*t21*t26*1.6*10.0 - dth1*dth2*l2*l4*m2*t8*t17*t21*t26*3.6*10.0 - dth1*dth3*l2*l4*m1*t8*t17*t21*t26*1.6*10.0 - dth1*dth2*l2*l4*m1*t8*t17*t21*t28*2.4*10.0 - dth1*dth3*l2*l4*m2*t8*t17*t21*t26*3.6*10.0 - dth2*dth3*l2*l4*m1*t8*t17*t21*t26*1.6*10.0 - dth1*dth2*l2*l4*m2*t8*t17*t21*t28*5.4*10.0 - dth1*dth2*l2*l4*m4*t8*t17*t21*t26*(4.5*10.0 / 2.0) - dth1*dth3*l2*l4*m1*t8*t17*t21*t28*2.4*10.0 - dth2*dth3*l2*l4*m2*t8*t17*t21*t26*3.6*10.0 - dth1*dth2*l2*l4*m3*t8*t17*t21*t28*1.2*10.0 - dth1*dth3*l2*l4*m2*t8*t17*t21*t28*5.4*10.0 - dth1*dth3*l2*l4*m4*t8*t17*t21*t26*(4.5*10.0 / 2.0) - dth2*dth3*l2*l4*m1*t8*t17*t21*t28*2.4*10.0 - dth1*dth3*l2*l4*m3*t8*t17*t21*t28*1.2*10.0 - dth2*dth3*l2*l4*m2*t8*t17*t21*t28*5.4*10.0 - dth2*dth3*l2*l4*m4*t8*t17*t21*t26*(4.5*10.0 / 2.0) - dth2*dth3*l2*l4*m3*t8*t17*t21*t28*1.2*10.0 - dth1*dth2*l3*l4*m1*t17*t19*t26*t37*1.2*10.0 + dth1*dth2*l3*l4*m2*t17*t19*t26*t36*3.0*10.0 + dth1*dth2*l3*l4*m3*t17*t19*t25*t36*1.2*10.0 - dth1*dth2*l3*l4*m2*t17*t19*t26*t37*1.8*10.0 + dth1*dth2*l3*l4*m4*t17*t19*t25*t36*(4.5*10.0 / 2.0) - dth1*dth2*l3*l4*m1*t17*t19*t28*t37*1.2*10.0 + dth1*dth2*l3*l4*m2*t17*t19*t28*t36*1.8*10.0 + dth1*dth2*l3*l4*m4*t17*t19*t26*t36*(7.5*10.0 / 2.0) - dth1*dth2*l3*l4*m2*t17*t19*t28*t37*1.8*10.0 + dth1*dth2*l3*l4*m3*t17*t19*t28*t36*1.2*10.0 + dth1*dth2*l1*l4*m2*t19*t21*t26*t42*4.0 + dth1*dth3*l1*l4*m2*t19*t21*t26*t42*4.0 + dth1*dth2*l1*l4*m2*t19*t21*t28*t42*6.0 + dth1*dth2*l1*l4*m4*t19*t21*t26*t42*(4.5*10.0 / 2.0) - dth1*dth2*l2*l3*m1*t17*t23*t28*t43*6.0 + dth2*dth3*l1*l4*m2*t19*t21*t26*t42*4.0 + dth1*dth2*l1*l4*m3*t19*t21*t28*t42*1.2*10.0 - dth1*dth2*l2*l3*m2*t17*t23*t28*t43*(2.7*10.0 / 2.0) + dth1*dth2*l2*l3*m4*t17*t23*t26*t43*3.0 + dth1*dth3*l1*l4*m2*t19*t21*t28*t42*6.0 + dth1*dth3*l1*l4*m4*t19*t21*t26*t42*(4.5*10.0 / 2.0) - dth1*dth3*l2*l3*m1*t17*t23*t28*t43*6.0 - dth1*dth2*l2*l3*m3*t17*t23*t28*t43*(3.0 / 2.0) + dth1*dth3*l1*l4*m3*t19*t21*t28*t42*1.2*10.0 - dth1*dth3*l2*l3*m2*t17*t23*t28*t43*(2.7*10.0 / 2.0) + dth1*dth3*l2*l3*m4*t17*t23*t26*t43*3.0 - dth1*dth4*l2*l3*m1*t17*t23*t28*t43*6.0 + dth2*dth3*l1*l4*m2*t19*t21*t28*t42*6.0 + dth2*dth3*l1*l4*m4*t19*t21*t26*t42*(4.5*10.0 / 2.0) - dth2*dth3*l2*l3*m1*t17*t23*t28*t43*6.0 - dth1*dth3*l2*l3*m3*t17*t23*t28*t43*(3.0 / 2.0) - dth1*dth4*l2*l3*m2*t17*t23*t28*t43*(2.7*10.0 / 2.0) + dth1*dth4*l2*l3*m4*t17*t23*t26*t43*3.0 + dth2*dth3*l1*l4*m3*t19*t21*t28*t42*1.2*10.0 - dth2*dth3*l2*l3*m2*t17*t23*t28*t43*(2.7*10.0 / 2.0) + dth2*dth3*l2*l3*m4*t17*t23*t26*t43*3.0 - dth2*dth4*l2*l3*m1*t17*t23*t28*t43*6.0 - dth1*dth4*l2*l3*m3*t17*t23*t28*t43*(3.0 / 2.0) - dth2*dth3*l2*l3*m3*t17*t23*t28*t43*(3.0 / 2.0) - dth2*dth4*l2*l3*m2*t17*t23*t28*t43*(2.7*10.0 / 2.0) + dth2*dth4*l2*l3*m4*t17*t23*t26*t43*3.0 - dth3*dth4*l2*l3*m1*t17*t23*t28*t43*6.0 - dth2*dth4*l2*l3*m3*t17*t23*t28*t43*(3.0 / 2.0) - dth3*dth4*l2*l3*m2*t17*t23*t28*t43*(2.7*10.0 / 2.0) + dth3*dth4*l2*l3*m4*t17*t23*t26*t43*3.0 - dth3*dth4*l2*l3*m3*t17*t23*t28*t43*(3.0 / 2.0) + dth1*dth2*l1*l3*m2*t19*t23*t28*t61*(3.0 / 2.0) - dth1*dth2*l1*l3*m4*t19*t23*t26*t61*3.0 + dth1*dth2*l1*l3*m3*t19*t23*t28*t61*(3.0 / 2.0) + dth1*dth3*l1*l3*m2*t19*t23*t28*t61*(3.0 / 2.0) - dth1*dth3*l1*l3*m4*t19*t23*t26*t61*3.0 + dth1*dth3*l1*l3*m3*t19*t23*t28*t61*(3.0 / 2.0) + dth1*dth4*l1*l3*m2*t19*t23*t28*t61*(3.0 / 2.0) - dth1*dth4*l1*l3*m4*t19*t23*t26*t61*3.0 + dth2*dth3*l1*l3*m2*t19*t23*t28*t61*(3.0 / 2.0) - dth2*dth3*l1*l3*m4*t19*t23*t26*t61*3.0 + dth1*dth4*l1*l3*m3*t19*t23*t28*t61*(3.0 / 2.0) + dth2*dth3*l1*l3*m3*t19*t23*t28*t61*(3.0 / 2.0) + dth2*dth4*l1*l3*m2*t19*t23*t28*t61*(3.0 / 2.0) - dth2*dth4*l1*l3*m4*t19*t23*t26*t61*3.0 + dth2*dth4*l1*l3*m3*t19*t23*t28*t61*(3.0 / 2.0) + dth3*dth4*l1*l3*m2*t19*t23*t28*t61*(3.0 / 2.0) - dth3*dth4*l1*l3*m4*t19*t23*t26*t61*3.0 + dth3*dth4*l1*l3*m3*t19*t23*t28*t61*(3.0 / 2.0) + dth1*dth2*l2*l4*m2*t17*t21*t26*t82*1.2*10.0 + dth1*dth3*l2*l4*m2*t17*t21*t26*t82*1.2*10.0 + dth1*dth2*l2*l4*m2*t17*t21*t28*t82*1.8*10.0 + dth1*dth2*l2*l4*m4*t17*t21*t26*t82*(4.5*10.0 / 2.0) + dth2*dth3*l2*l4*m2*t17*t21*t26*t82*1.2*10.0 + dth1*dth2*l2*l4*m3*t17*t21*t28*t82*1.2*10.0 + dth1*dth3*l2*l4*m2*t17*t21*t28*t82*1.8*10.0 + dth1*dth3*l2*l4*m4*t17*t21*t26*t82*(4.5*10.0 / 2.0) + dth1*dth3*l2*l4*m3*t17*t21*t28*t82*1.2*10.0 + dth2*dth3*l2*l4*m2*t17*t21*t28*t82*1.8*10.0 + dth2*dth3*l2*l4*m4*t17*t21*t26*t82*(4.5*10.0 / 2.0) + dth1*dth2*l2*l4*m4*t17*t21*t26*t85*(9.0 / 2.0) + dth2*dth3*l2*l4*m3*t17*t21*t28*t82*1.2*10.0 + dth1*dth3*l2*l4*m4*t17*t21*t26*t85*(9.0 / 2.0) + dth2*dth3*l2*l4*m4*t17*t21*t26*t85*(9.0 / 2.0) + dth1*dth2*l1*l4*m2*t19*t21*t26*t106*1.2*10.0 + dth1*dth3*l1*l4*m2*t19*t21*t26*t106*1.2*10.0 + dth1*dth2*l1*l4*m2*t19*t21*t28*t106*1.8*10.0 + dth1*dth2*l1*l4*m4*t19*t21*t26*t106*(4.5*10.0 / 2.0) + dth2*dth3*l1*l4*m2*t19*t21*t26*t106*1.2*10.0 + dth1*dth2*l1*l4*m3*t19*t21*t28*t106*1.2*10.0 + dth1*dth3*l1*l4*m2*t19*t21*t28*t106*1.8*10.0 + dth1*dth3*l1*l4*m4*t19*t21*t26*t106*(4.5*10.0 / 2.0) - dth1*dth2*l2*l3*m1*t17*t23*t28*t109*6.0 + dth1*dth3*l1*l4*m3*t19*t21*t28*t106*1.2*10.0 + dth2*dth3*l1*l4*m2*t19*t21*t28*t106*1.8*10.0 + dth2*dth3*l1*l4*m4*t19*t21*t26*t106*(4.5*10.0 / 2.0) - dth1*dth2*l2*l3*m2*t17*t23*t28*t109*(2.7*10.0 / 2.0) - dth1*dth2*l2*l3*m4*t17*t23*t26*t109*9.0 - dth1*dth3*l2*l3*m1*t17*t23*t28*t109*6.0 + dth2*dth3*l1*l4*m3*t19*t21*t28*t106*1.2*10.0 - dth1*dth2*l2*l3*m3*t17*t23*t28*t109*(9.0 / 2.0) - dth1*dth3*l2*l3*m2*t17*t23*t28*t109*(2.7*10.0 / 2.0) - dth1*dth3*l2*l3*m4*t17*t23*t26*t109*9.0 - dth1*dth4*l2*l3*m1*t17*t23*t28*t109*6.0 - dth2*dth3*l2*l3*m1*t17*t23*t28*t109*6.0 - dth1*dth3*l2*l3*m3*t17*t23*t28*t109*(9.0 / 2.0) - dth1*dth4*l2*l3*m2*t17*t23*t28*t109*(2.7*10.0 / 2.0) - dth1*dth4*l2*l3*m4*t17*t23*t26*t109*9.0 - dth2*dth3*l2*l3*m2*t17*t23*t28*t109*(2.7*10.0 / 2.0) - dth2*dth3*l2*l3*m4*t17*t23*t26*t109*9.0 - dth2*dth4*l2*l3*m1*t17*t23*t28*t109*6.0 - dth1*dth4*l2*l3*m3*t17*t23*t28*t109*(9.0 / 2.0) - dth2*dth3*l2*l3*m3*t17*t23*t28*t109*(9.0 / 2.0) - dth2*dth4*l2*l3*m2*t17*t23*t28*t109*(2.7*10.0 / 2.0) - dth2*dth4*l2*l3*m4*t17*t23*t26*t109*9.0 - dth3*dth4*l2*l3*m1*t17*t23*t28*t109*6.0 - dth2*dth4*l2*l3*m3*t17*t23*t28*t109*(9.0 / 2.0) - dth3*dth4*l2*l3*m2*t17*t23*t28*t109*(2.7*10.0 / 2.0) - dth3*dth4*l2*l3*m4*t17*t23*t26*t109*9.0 - dth3*dth4*l2*l3*m3*t17*t23*t28*t109*(9.0 / 2.0) - dth1*dth2*l1*l4*m4*t19*t21*t26*t122*(9.0 / 2.0) + dth1*dth2*l2*l3*m2*t17*t23*t28*t123*(9.0 / 2.0) - dth1*dth2*l2*l3*m4*t17*t23*t26*t123*3.0 - dth1*dth3*l1*l4*m4*t19*t21*t26*t122*(9.0 / 2.0) + dth1*dth2*l2*l3*m3*t17*t23*t28*t123*(3.0 / 2.0) + dth1*dth3*l2*l3*m2*t17*t23*t28*t123*(9.0 / 2.0) - dth1*dth3*l2*l3*m4*t17*t23*t26*t123*3.0 - dth2*dth3*l1*l4*m4*t19*t21*t26*t122*(9.0 / 2.0) + dth1*dth3*l2*l3*m3*t17*t23*t28*t123*(3.0 / 2.0) + dth1*dth4*l2*l3*m2*t17*t23*t28*t123*(9.0 / 2.0) - dth1*dth4*l2*l3*m4*t17*t23*t26*t123*3.0 + dth2*dth3*l2*l3*m2*t17*t23*t28*t123*(9.0 / 2.0) - dth2*dth3*l2*l3*m4*t17*t23*t26*t123*3.0 + dth1*dth4*l2*l3*m3*t17*t23*t28*t123*(3.0 / 2.0) + dth2*dth3*l2*l3*m3*t17*t23*t28*t123*(3.0 / 2.0) + dth2*dth4*l2*l3*m2*t17*t23*t28*t123*(9.0 / 2.0) - dth2*dth4*l2*l3*m4*t17*t23*t26*t123*3.0 + dth2*dth4*l2*l3*m3*t17*t23*t28*t123*(3.0 / 2.0) + dth3*dth4*l2*l3*m2*t17*t23*t28*t123*(9.0 / 2.0) - dth3*dth4*l2*l3*m4*t17*t23*t26*t123*3.0 + dth3*dth4*l2*l3*m3*t17*t23*t28*t123*(3.0 / 2.0) - dth1*dth2*l3*l4*m4*t17*t19*t25*t134*(2.7*10.0 / 4.0) - dth1*dth2*l3*l4*m4*t17*t19*t26*t134*(2.7*10.0 / 4.0) + dth1*dth2*l1*l3*m2*t19*t23*t28*t149*(3.0 / 2.0) + dth1*dth2*l1*l3*m4*t19*t23*t26*t149*9.0 + dth1*dth2*l1*l3*m2*t19*t23*t28*t150*(9.0 / 2.0) + dth1*dth2*l1*l3*m3*t19*t23*t28*t149*(9.0 / 2.0) + dth1*dth2*l1*l3*m4*t19*t23*t26*t150*9.0 + dth1*dth3*l1*l3*m2*t19*t23*t28*t149*(3.0 / 2.0) + dth1*dth3*l1*l3*m4*t19*t23*t26*t149*9.0 - dth1*dth2*l1*l3*m2*t19*t23*t28*t151*(9.0 / 2.0) + dth1*dth2*l1*l3*m3*t19*t23*t28*t150*(9.0 / 2.0) + dth1*dth2*l1*l3*m4*t19*t23*t26*t151*3.0 + dth1*dth3*l1*l3*m2*t19*t23*t28*t150*(9.0 / 2.0) + dth1*dth3*l1*l3*m3*t19*t23*t28*t149*(9.0 / 2.0) + dth1*dth3*l1*l3*m4*t19*t23*t26*t150*9.0 + dth1*dth4*l1*l3*m2*t19*t23*t28*t149*(3.0 / 2.0) + dth1*dth4*l1*l3*m4*t19*t23*t26*t149*9.0 + dth2*dth3*l1*l3*m2*t19*t23*t28*t149*(3.0 / 2.0) + dth2*dth3*l1*l3*m4*t19*t23*t26*t149*9.0 - dth1*dth2*l1*l3*m3*t19*t23*t28*t151*(3.0 / 2.0) - dth1*dth3*l1*l3*m2*t19*t23*t28*t151*(9.0 / 2.0) + dth1*dth3*l1*l3*m3*t19*t23*t28*t150*(9.0 / 2.0) + dth1*dth3*l1*l3*m4*t19*t23*t26*t151*3.0 + dth1*dth4*l1*l3*m2*t19*t23*t28*t150*(9.0 / 2.0) + dth1*dth4*l1*l3*m3*t19*t23*t28*t149*(9.0 / 2.0) + dth1*dth4*l1*l3*m4*t19*t23*t26*t150*9.0 + dth2*dth3*l1*l3*m2*t19*t23*t28*t150*(9.0 / 2.0) + dth2*dth3*l1*l3*m3*t19*t23*t28*t149*(9.0 / 2.0) + dth2*dth3*l1*l3*m4*t19*t23*t26*t150*9.0 + dth2*dth4*l1*l3*m2*t19*t23*t28*t149*(3.0 / 2.0) + dth2*dth4*l1*l3*m4*t19*t23*t26*t149*9.0 - dth1*dth3*l1*l3*m3*t19*t23*t28*t151*(3.0 / 2.0) - dth1*dth4*l1*l3*m2*t19*t23*t28*t151*(9.0 / 2.0) + dth1*dth4*l1*l3*m3*t19*t23*t28*t150*(9.0 / 2.0) + dth1*dth4*l1*l3*m4*t19*t23*t26*t151*3.0 - dth2*dth3*l1*l3*m2*t19*t23*t28*t151*(9.0 / 2.0) + dth2*dth3*l1*l3*m3*t19*t23*t28*t150*(9.0 / 2.0) + dth2*dth3*l1*l3*m4*t19*t23*t26*t151*3.0 + dth2*dth4*l1*l3*m2*t19*t23*t28*t150*(9.0 / 2.0) + dth2*dth4*l1*l3*m3*t19*t23*t28*t149*(9.0 / 2.0) + dth2*dth4*l1*l3*m4*t19*t23*t26*t150*9.0 + dth3*dth4*l1*l3*m2*t19*t23*t28*t149*(3.0 / 2.0) + dth3*dth4*l1*l3*m4*t19*t23*t26*t149*9.0 - dth1*dth4*l1*l3*m3*t19*t23*t28*t151*(3.0 / 2.0) - dth2*dth3*l1*l3*m3*t19*t23*t28*t151*(3.0 / 2.0) - dth2*dth4*l1*l3*m2*t19*t23*t28*t151*(9.0 / 2.0) + dth2*dth4*l1*l3*m3*t19*t23*t28*t150*(9.0 / 2.0) + dth2*dth4*l1*l3*m4*t19*t23*t26*t151*3.0 + dth3*dth4*l1*l3*m2*t19*t23*t28*t150*(9.0 / 2.0) + dth3*dth4*l1*l3*m3*t19*t23*t28*t149*(9.0 / 2.0) + dth3*dth4*l1*l3*m4*t19*t23*t26*t150*9.0 - dth2*dth4*l1*l3*m3*t19*t23*t28*t151*(3.0 / 2.0) - dth3*dth4*l1*l3*m2*t19*t23*t28*t151*(9.0 / 2.0) + dth3*dth4*l1*l3*m3*t19*t23*t28*t150*(9.0 / 2.0) + dth3*dth4*l1*l3*m4*t19*t23*t26*t151*3.0 - dth3*dth4*l1*l3*m3*t19*t23*t28*t151*(3.0 / 2.0) - dth1*dth2*l3*l4*m4*t17*t19*t25*t172*(2.7*10.0 / 4.0) - dth1*dth2*l3*l4*m4*t17*t19*t26*t172*(2.7*10.0 / 4.0) - dth1*dth2*l2*l4*m4*t17*t21*t26*t187*(9.0 / 2.0) - dth1*dth3*l2*l4*m4*t17*t21*t26*t187*(9.0 / 2.0) - dth2*dth3*l2*l4*m4*t17*t21*t26*t187*(9.0 / 2.0) + dth1*dth2*l1*l4*m4*t19*t21*t26*t198*(9.0 / 2.0) + dth1*dth2*l2*l3*m2*t17*t23*t28*t199*(9.0 / 2.0) + dth1*dth2*l2*l3*m4*t17*t23*t26*t199*9.0 + dth1*dth3*l1*l4*m4*t19*t21*t26*t198*(9.0 / 2.0) + dth1*dth2*l2*l3*m3*t17*t23*t28*t199*(9.0 / 2.0) + dth1*dth3*l2*l3*m2*t17*t23*t28*t199*(9.0 / 2.0) + dth1*dth3*l2*l3*m4*t17*t23*t26*t199*9.0 + dth2*dth3*l1*l4*m4*t19*t21*t26*t198*(9.0 / 2.0) + dth1*dth3*l2*l3*m3*t17*t23*t28*t199*(9.0 / 2.0) + dth1*dth4*l2*l3*m2*t17*t23*t28*t199*(9.0 / 2.0) + dth1*dth4*l2*l3*m4*t17*t23*t26*t199*9.0 + dth2*dth3*l2*l3*m2*t17*t23*t28*t199*(9.0 / 2.0) + dth2*dth3*l2*l3*m4*t17*t23*t26*t199*9.0 + dth1*dth4*l2*l3*m3*t17*t23*t28*t199*(9.0 / 2.0) + dth2*dth3*l2*l3*m3*t17*t23*t28*t199*(9.0 / 2.0) + dth2*dth4*l2*l3*m2*t17*t23*t28*t199*(9.0 / 2.0) + dth2*dth4*l2*l3*m4*t17*t23*t26*t199*9.0 + dth2*dth4*l2*l3*m3*t17*t23*t28*t199*(9.0 / 2.0) + dth3*dth4*l2*l3*m2*t17*t23*t28*t199*(9.0 / 2.0) + dth3*dth4*l2*l3*m4*t17*t23*t26*t199*9.0 + dth3*dth4*l2*l3*m3*t17*t23*t28*t199*(9.0 / 2.0) + l2*l3*l4*m1*m2*m3*t7*t13*t18*8.0 + l2*l3*l4*m1*m2*m4*t7*t13*t18*1.5*10.0 + l2*l3*l4*m1*m3*m4*t7*t13*t18*2.5*10.0 + l2*l3*l4*m2*m3*m4*t7*t13*t18*(2.25*100.0 / 2.0) + l1*l3*l4*m2*m3*m4*t7*t13*t20*(1.25*100.0 / 2.0) + l1*l3*l4*m2*m3*m4*t7*t14*t20*(1.25*100.0 / 2.0) - l2*l3*l4*m1*m3*m4*t13*t18*t81*1.5*10.0 - l2*l3*l4*m1*m2*m4*t13*t18*t83*(9.0 / 2.0) - l2*l3*l4*m2*m3*m4*t13*t18*t81*(4.5*10.0 / 2.0) - l1*l3*l4*m2*m3*m4*t13*t20*t81*(1.5*10.0 / 2.0) - l2*l3*l4*m1*m3*m4*t13*t18*t83*(9.0 / 2.0) - l1*l3*l4*m2*m3*m4*t14*t20*t81*(1.5*10.0 / 2.0) - l2*l3*l4*m2*m3*m4*t13*t18*t83*(8.1*10.0 / 4.0) - l1*l3*l4*m2*m3*m4*t13*t20*t83*(4.5*10.0 / 4.0) - l1*l3*l4*m2*m3*m4*t14*t20*t83*(4.5*10.0 / 4.0) - l2*l3*l4*m1*m2*m4*t13*t18*t108*(9.0 / 2.0) - l2*l3*l4*m1*m3*m4*t13*t18*t108*(9.0 / 2.0) - l2*l3*l4*m2*m3*m4*t13*t18*t108*(8.1*10.0 / 4.0) - l1*l3*l4*m2*m3*m4*t13*t20*t108*(4.5*10.0 / 4.0) - l1*l3*l4*m2*m3*m4*t14*t20*t108*(4.5*10.0 / 4.0) + l2*l3*l4*m1*m3*m4*t13*t18*t186*3.0 + l2*l3*l4*m2*m3*m4*t13*t18*t186*(9.0 / 2.0) + l1*l3*l4*m2*m3*m4*t13*t20*t186*(3.0 / 2.0) + l1*l3*l4*m2*m3*m4*t14*t20*t186*(3.0 / 2.0) - l2*l4*m1*m3*m4*t8*t13*t17*t21*2.5*10.0 - l2*l4*m1*m3*m4*t8*t14*t17*t21*2.5*10.0 - l2*l4*m2*m3*m4*t8*t13*t17*t21*(2.25*100.0 / 4.0) - l2*l4*m1*m3*m4*t8*t15*t17*t21*2.5*10.0 - l2*l4*m2*m3*m4*t8*t14*t17*t21*(2.25*100.0 / 4.0) - l2*l4*m2*m3*m4*t8*t15*t17*t21*(2.25*100.0 / 4.0) - l3*l4*m1*m3*m4*t13*t17*t19*t37*1.5*10.0 + l3*l4*m2*m3*m4*t13*t17*t19*t36*7.5*10.0 - l3*l4*m1*m3*m4*t14*t17*t19*t37*1.5*10.0 - l3*l4*m2*m3*m4*t13*t17*t19*t37*(4.5*10.0 / 2.0) + l3*l4*m2*m3*m4*t14*t17*t19*t36*(7.5*10.0 / 2.0) - l3*l4*m2*m3*m4*t14*t17*t19*t37*(4.5*10.0 / 2.0) + l1*l4*m2*m3*m4*t13*t19*t21*t42*(2.5*10.0 / 4.0) - l2*l3*m1*m3*m4*t13*t17*t23*t43*2.0 + l1*l4*m2*m3*m4*t14*t19*t21*t42*(2.5*10.0 / 4.0) - l2*l3*m1*m3*m4*t14*t17*t23*t43*2.0 - l2*l3*m2*m3*m4*t13*t17*t23*t43*(9.0 / 2.0) + l1*l4*m2*m3*m4*t15*t19*t21*t42*(2.5*10.0 / 4.0) - l2*l3*m1*m3*m4*t15*t17*t23*t43*2.0 - l2*l3*m2*m3*m4*t14*t17*t23*t43*(9.0 / 2.0) - l2*l3*m1*m3*m4*t16*t17*t23*t43*2.0 - l2*l3*m2*m3*m4*t15*t17*t23*t43*(9.0 / 2.0) - l2*l3*m2*m3*m4*t16*t17*t23*t43*(9.0 / 2.0) + (l1*l3*m2*m3*m4*t13*t19*t23*t61) / 2.0 + (l1*l3*m2*m3*m4*t14*t19*t23*t61) / 2.0 + (l1*l3*m2*m3*m4*t15*t19*t23*t61) / 2.0 + (l1*l3*m2*m3*m4*t16*t19*t23*t61) / 2.0 + l2*l4*m2*m3*m4*t13*t17*t21*t82*(7.5*10.0 / 4.0) + l2*l4*m2*m3*m4*t14*t17*t21*t82*(7.5*10.0 / 4.0) + l2*l4*m1*m3*m4*t13*t17*t21*t85*3.0 + l2*l4*m2*m3*m4*t15*t17*t21*t82*(7.5*10.0 / 4.0) + l2*l4*m1*m3*m4*t14*t17*t21*t85*3.0 + l2*l4*m2*m3*m4*t13*t17*t21*t85*(2.7*10.0 / 4.0) + l2*l4*m1*m3*m4*t15*t17*t21*t85*3.0 + l2*l4*m2*m3*m4*t14*t17*t21*t85*(2.7*10.0 / 4.0) + l2*l4*m2*m3*m4*t15*t17*t21*t85*(2.7*10.0 / 4.0) + l1*l4*m2*m3*m4*t13*t19*t21*t106*(7.5*10.0 / 4.0) + l1*l4*m2*m3*m4*t14*t19*t21*t106*(7.5*10.0 / 4.0) + l1*l4*m2*m3*m4*t15*t19*t21*t106*(7.5*10.0 / 4.0) - l2*l3*m1*m3*m4*t13*t17*t23*t109*6.0 - l2*l3*m1*m3*m4*t14*t17*t23*t109*6.0 - l2*l3*m2*m3*m4*t13*t17*t23*t109*(2.7*10.0 / 2.0) - l2*l3*m1*m3*m4*t15*t17*t23*t109*6.0 - l2*l3*m2*m3*m4*t14*t17*t23*t109*(2.7*10.0 / 2.0) - l2*l3*m1*m3*m4*t16*t17*t23*t109*6.0 - l2*l3*m2*m3*m4*t15*t17*t23*t109*(2.7*10.0 / 2.0) - l2*l3*m2*m3*m4*t16*t17*t23*t109*(2.7*10.0 / 2.0) - l1*l4*m2*m3*m4*t13*t19*t21*t122*(3.0 / 4.0) - l1*l4*m2*m3*m4*t14*t19*t21*t122*(3.0 / 4.0) + l2*l3*m2*m3*m4*t13*t17*t23*t123*(3.0 / 2.0) - l1*l4*m2*m3*m4*t15*t19*t21*t122*(3.0 / 4.0) + l2*l3*m2*m3*m4*t14*t17*t23*t123*(3.0 / 2.0) + l2*l3*m2*m3*m4*t15*t17*t23*t123*(3.0 / 2.0) + l2*l3*m2*m3*m4*t16*t17*t23*t123*(3.0 / 2.0) - l3*l4*m2*m3*m4*t13*t17*t19*t133*(1.5*10.0 / 2.0) + l3*l4*m1*m3*m4*t13*t17*t19*t135*3.0 - l3*l4*m2*m3*m4*t13*t17*t19*t134*(2.7*10.0 / 2.0) + l3*l4*m1*m3*m4*t14*t17*t19*t135*3.0 + l3*l4*m2*m3*m4*t13*t17*t19*t135*(9.0 / 2.0) - l3*l4*m2*m3*m4*t14*t17*t19*t134*(2.7*10.0 / 4.0) + l3*l4*m2*m3*m4*t14*t17*t19*t135*(9.0 / 2.0) + l1*l3*m2*m3*m4*t13*t19*t23*t149*(3.0 / 2.0) + l1*l3*m2*m3*m4*t13*t19*t23*t150*(9.0 / 2.0) + l1*l3*m2*m3*m4*t14*t19*t23*t149*(3.0 / 2.0) - l1*l3*m2*m3*m4*t13*t19*t23*t151*(3.0 / 2.0) + l1*l3*m2*m3*m4*t14*t19*t23*t150*(9.0 / 2.0) + l1*l3*m2*m3*m4*t15*t19*t23*t149*(3.0 / 2.0) - l1*l3*m2*m3*m4*t14*t19*t23*t151*(3.0 / 2.0) + l1*l3*m2*m3*m4*t15*t19*t23*t150*(9.0 / 2.0) + l1*l3*m2*m3*m4*t16*t19*t23*t149*(3.0 / 2.0) - l1*l3*m2*m3*m4*t15*t19*t23*t151*(3.0 / 2.0) + l1*l3*m2*m3*m4*t16*t19*t23*t150*(9.0 / 2.0) - l1*l3*m2*m3*m4*t16*t19*t23*t151*(3.0 / 2.0) - l3*l4*m2*m3*m4*t13*t17*t19*t172*(2.7*10.0 / 2.0) - l3*l4*m2*m3*m4*t14*t17*t19*t172*(2.7*10.0 / 4.0) - l2*l4*m2*m3*m4*t13*t17*t21*t187*(9.0 / 4.0) - l2*l4*m2*m3*m4*t14*t17*t21*t187*(9.0 / 4.0) - l2*l4*m2*m3*m4*t15*t17*t21*t187*(9.0 / 4.0) + l1*l4*m2*m3*m4*t13*t19*t21*t198*(9.0 / 4.0) + l1*l4*m2*m3*m4*t14*t19*t21*t198*(9.0 / 4.0) + l2*l3*m2*m3*m4*t13*t17*t23*t199*(9.0 / 2.0) + l1*l4*m2*m3*m4*t15*t19*t21*t198*(9.0 / 4.0) + l2*l3*m2*m3*m4*t14*t17*t23*t199*(9.0 / 2.0) + l2*l3*m2*m3*m4*t15*t17*t23*t199*(9.0 / 2.0) + l3*l4*m2*m3*m4*t13*t17*t19*t203*(3.0 / 2.0) + l2*l3*m2*m3*m4*t16*t17*t23*t199*(9.0 / 2.0) + g*l1*l3*l4*m2*m3*m4*t19*sin(t45 + t62)*(3.0 / 2.0) + dth1*dth2*l1*l3*l4*m2*m3*m4*t7*t20*1.25*100.0 - dth1*dth2*l1*l3*l4*m2*m3*m4*t20*t81*1.5*10.0 - dth1*dth2*l1*l3*l4*m2*m3*m4*t20*t83*(4.5*10.0 / 2.0) - dth1*dth2*l1*l3*l4*m2*m3*m4*t20*t108*(4.5*10.0 / 2.0) + dth1*dth2*l1*l3*l4*m2*m3*m4*t20*t186*3.0 - dth1*dth2*l2*l4*m1*m3*m4*t8*t17*t21*5.0*10.0 - dth1*dth2*l2*l4*m2*m3*m4*t8*t17*t21*(2.25*100.0 / 2.0) - dth1*dth3*l2*l4*m1*m3*m4*t8*t17*t21*5.0*10.0 - dth1*dth3*l2*l4*m2*m3*m4*t8*t17*t21*(2.25*100.0 / 2.0) - dth2*dth3*l2*l4*m1*m3*m4*t8*t17*t21*5.0*10.0 - dth2*dth3*l2*l4*m2*m3*m4*t8*t17*t21*(2.25*100.0 / 2.0) - dth1*dth2*l3*l4*m1*m3*m4*t17*t19*t37*3.0*10.0 + dth1*dth2*l3*l4*m2*m3*m4*t17*t19*t36*7.5*10.0 - dth1*dth2*l3*l4*m2*m3*m4*t17*t19*t37*4.5*10.0 + dth1*dth2*l1*l4*m2*m3*m4*t19*t21*t42*(2.5*10.0 / 2.0) - dth1*dth2*l2*l3*m1*m3*m4*t17*t23*t43*4.0 - dth1*dth2*l2*l3*m2*m3*m4*t17*t23*t43*9.0 + dth1*dth3*l1*l4*m2*m3*m4*t19*t21*t42*(2.5*10.0 / 2.0) - dth1*dth3*l2*l3*m1*m3*m4*t17*t23*t43*4.0 - dth1*dth3*l2*l3*m2*m3*m4*t17*t23*t43*9.0 - dth1*dth4*l2*l3*m1*m3*m4*t17*t23*t43*4.0 + dth2*dth3*l1*l4*m2*m3*m4*t19*t21*t42*(2.5*10.0 / 2.0) - dth2*dth3*l2*l3*m1*m3*m4*t17*t23*t43*4.0 - dth1*dth4*l2*l3*m2*m3*m4*t17*t23*t43*9.0 - dth2*dth3*l2*l3*m2*m3*m4*t17*t23*t43*9.0 - dth2*dth4*l2*l3*m1*m3*m4*t17*t23*t43*4.0 - dth2*dth4*l2*l3*m2*m3*m4*t17*t23*t43*9.0 - dth3*dth4*l2*l3*m1*m3*m4*t17*t23*t43*4.0 - dth3*dth4*l2*l3*m2*m3*m4*t17*t23*t43*9.0 + dth1*dth2*l1*l3*m2*m3*m4*t19*t23*t61 + dth1*dth3*l1*l3*m2*m3*m4*t19*t23*t61 + dth1*dth4*l1*l3*m2*m3*m4*t19*t23*t61 + dth2*dth3*l1*l3*m2*m3*m4*t19*t23*t61 + dth2*dth4*l1*l3*m2*m3*m4*t19*t23*t61 + dth3*dth4*l1*l3*m2*m3*m4*t19*t23*t61 + dth1*dth2*l2*l4*m2*m3*m4*t17*t21*t82*(7.5*10.0 / 2.0) + dth1*dth3*l2*l4*m2*m3*m4*t17*t21*t82*(7.5*10.0 / 2.0) + dth1*dth2*l2*l4*m1*m3*m4*t17*t21*t85*6.0 + dth2*dth3*l2*l4*m2*m3*m4*t17*t21*t82*(7.5*10.0 / 2.0) + dth1*dth2*l2*l4*m2*m3*m4*t17*t21*t85*(2.7*10.0 / 2.0) + dth1*dth3*l2*l4*m1*m3*m4*t17*t21*t85*6.0 + dth1*dth3*l2*l4*m2*m3*m4*t17*t21*t85*(2.7*10.0 / 2.0) + dth2*dth3*l2*l4*m1*m3*m4*t17*t21*t85*6.0 + dth2*dth3*l2*l4*m2*m3*m4*t17*t21*t85*(2.7*10.0 / 2.0) + dth1*dth2*l1*l4*m2*m3*m4*t19*t21*t106*(7.5*10.0 / 2.0) + dth1*dth3*l1*l4*m2*m3*m4*t19*t21*t106*(7.5*10.0 / 2.0) - dth1*dth2*l2*l3*m1*m3*m4*t17*t23*t109*1.2*10.0 + dth2*dth3*l1*l4*m2*m3*m4*t19*t21*t106*(7.5*10.0 / 2.0) - dth1*dth2*l2*l3*m2*m3*m4*t17*t23*t109*2.7*10.0 - dth1*dth3*l2*l3*m1*m3*m4*t17*t23*t109*1.2*10.0 - dth1*dth3*l2*l3*m2*m3*m4*t17*t23*t109*2.7*10.0 - dth1*dth4*l2*l3*m1*m3*m4*t17*t23*t109*1.2*10.0 - dth2*dth3*l2*l3*m1*m3*m4*t17*t23*t109*1.2*10.0 - dth1*dth4*l2*l3*m2*m3*m4*t17*t23*t109*2.7*10.0 - dth2*dth3*l2*l3*m2*m3*m4*t17*t23*t109*2.7*10.0 - dth2*dth4*l2*l3*m1*m3*m4*t17*t23*t109*1.2*10.0 - dth2*dth4*l2*l3*m2*m3*m4*t17*t23*t109*2.7*10.0 - dth3*dth4*l2*l3*m1*m3*m4*t17*t23*t109*1.2*10.0 - dth3*dth4*l2*l3*m2*m3*m4*t17*t23*t109*2.7*10.0 - dth1*dth2*l1*l4*m2*m3*m4*t19*t21*t122*(3.0 / 2.0) + dth1*dth2*l2*l3*m2*m3*m4*t17*t23*t123*3.0 - dth1*dth3*l1*l4*m2*m3*m4*t19*t21*t122*(3.0 / 2.0) + dth1*dth3*l2*l3*m2*m3*m4*t17*t23*t123*3.0 - dth2*dth3*l1*l4*m2*m3*m4*t19*t21*t122*(3.0 / 2.0) + dth1*dth4*l2*l3*m2*m3*m4*t17*t23*t123*3.0 + dth2*dth3*l2*l3*m2*m3*m4*t17*t23*t123*3.0 + dth2*dth4*l2*l3*m2*m3*m4*t17*t23*t123*3.0 + dth3*dth4*l2*l3*m2*m3*m4*t17*t23*t123*3.0 + dth1*dth2*l3*l4*m1*m3*m4*t17*t19*t135*6.0 - dth1*dth2*l3*l4*m2*m3*m4*t17*t19*t134*(2.7*10.0 / 2.0) + dth1*dth2*l3*l4*m2*m3*m4*t17*t19*t135*9.0 + dth1*dth2*l1*l3*m2*m3*m4*t19*t23*t149*3.0 + dth1*dth2*l1*l3*m2*m3*m4*t19*t23*t150*9.0 + dth1*dth3*l1*l3*m2*m3*m4*t19*t23*t149*3.0 - dth1*dth2*l1*l3*m2*m3*m4*t19*t23*t151*3.0 + dth1*dth3*l1*l3*m2*m3*m4*t19*t23*t150*9.0 + dth1*dth4*l1*l3*m2*m3*m4*t19*t23*t149*3.0 + dth2*dth3*l1*l3*m2*m3*m4*t19*t23*t149*3.0 - dth1*dth3*l1*l3*m2*m3*m4*t19*t23*t151*3.0 + dth1*dth4*l1*l3*m2*m3*m4*t19*t23*t150*9.0 + dth2*dth3*l1*l3*m2*m3*m4*t19*t23*t150*9.0 + dth2*dth4*l1*l3*m2*m3*m4*t19*t23*t149*3.0 - dth1*dth4*l1*l3*m2*m3*m4*t19*t23*t151*3.0 - dth2*dth3*l1*l3*m2*m3*m4*t19*t23*t151*3.0 + dth2*dth4*l1*l3*m2*m3*m4*t19*t23*t150*9.0 + dth3*dth4*l1*l3*m2*m3*m4*t19*t23*t149*3.0 - dth2*dth4*l1*l3*m2*m3*m4*t19*t23*t151*3.0 + dth3*dth4*l1*l3*m2*m3*m4*t19*t23*t150*9.0 - dth3*dth4*l1*l3*m2*m3*m4*t19*t23*t151*3.0 - dth1*dth2*l3*l4*m2*m3*m4*t17*t19*t172*(2.7*10.0 / 2.0) - dth1*dth2*l2*l4*m2*m3*m4*t17*t21*t187*(9.0 / 2.0) - dth1*dth3*l2*l4*m2*m3*m4*t17*t21*t187*(9.0 / 2.0) - dth2*dth3*l2*l4*m2*m3*m4*t17*t21*t187*(9.0 / 2.0) + dth1*dth2*l1*l4*m2*m3*m4*t19*t21*t198*(9.0 / 2.0) + dth1*dth2*l2*l3*m2*m3*m4*t17*t23*t199*9.0 + dth1*dth3*l1*l4*m2*m3*m4*t19*t21*t198*(9.0 / 2.0) + dth1*dth3*l2*l3*m2*m3*m4*t17*t23*t199*9.0 + dth2*dth3*l1*l4*m2*m3*m4*t19*t21*t198*(9.0 / 2.0) + dth1*dth4*l2*l3*m2*m3*m4*t17*t23*t199*9.0 + dth2*dth3*l2*l3*m2*m3*m4*t17*t23*t199*9.0 + dth2*dth4*l2*l3*m2*m3*m4*t17*t23*t199*9.0 + dth3*dth4*l2*l3*m2*m3*m4*t17*t23*t199*9.0)* - 2.4*10.0;

        index = index + num_threads;

        index1 = (grid_size[0] > 1) ? index : 0;
        index2 = (grid_size[1] > 1) ? index : 0;
        index3 = (grid_size[2] > 1) ? index : 0;
        index4 = (grid_size[3] > 1) ? index : 0;
        index5 = (grid_size[4] > 1) ? index : 0;
        index6 = (grid_size[5] > 1) ? index : 0;
        index7 = (grid_size[6] > 1) ? index : 0;
        index8 = (grid_size[7] > 1) ? index : 0;

        uindex1 = (active_actions[0] == 1) ? index : 0;
        uindex2 = (active_actions[1] == 1) ? index : 0;
        uindex3 = (active_actions[2] == 1) ? index : 0;
        uindex4 = (active_actions[3] == 1) ? index : 0;
    }
}

__global__ void dyn7_mex_continuous(double* const d_dx3, // outputs
    double* const x1, double* const x2, double* const x3, double* const x4,
    double* const dx1, double* const dx2, double* const dx3, double* const dx4, // input states
    double const* const in1, double const* const in2, double const* const in3, double const* const in4, // input actions
    const double m1, const double m2, const double m3, const double m4,
    const double l1, const double l2, const double l3, const double l4, const double g, 
    int32_t const* const grid_size, int32_t const* const active_actions)
{
    int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x * blockDim.y
        + (threadIdx.x + threadIdx.y * blockDim.x);
    int num_threads = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
    int num_elements = grid_size[0] * grid_size[1] * grid_size[2] * grid_size[3]
                        * grid_size[4] * grid_size[5] * grid_size[6] * grid_size[7];
    double th1, th2, th3, th4, dth1, dth2, dth3, dth4, u1, u2, u3, u4;

    int index1 = (grid_size[0] > 1) ? index : 0;
    int index2 = (grid_size[1] > 1) ? index : 0;
    int index3 = (grid_size[2] > 1) ? index : 0;
    int index4 = (grid_size[3] > 1) ? index : 0;
    int index5 = (grid_size[4] > 1) ? index : 0;
    int index6 = (grid_size[5] > 1) ? index : 0;
    int index7 = (grid_size[6] > 1) ? index : 0;
    int index8 = (grid_size[7] > 1) ? index : 0;

    int uindex1 = (active_actions[0] == 1) ? index : 0;
    int uindex2 = (active_actions[1] == 1) ? index : 0;
    int uindex3 = (active_actions[2] == 1) ? index : 0;
    int uindex4 = (active_actions[3] == 1) ? index : 0;

    while (index < num_elements)
    {
        th1 = x1[index1];
        th2 = x2[index2];
        th3 = x3[index3];
        th4 = x4[index4];

        dth1 = dx1[index5];
        dth2 = dx2[index6];
        dth3 = dx3[index7];
        dth4 = dx4[index8];

        u1 = in1[uindex1];
        u2 = in2[uindex2];
        u3 = in3[uindex3];
        u4 = in4[uindex4];

        double t2 = cos(th1);
        double t3 = cos(th2);
        double t4 = cos(th3);
        double t5 = cos(th4);
        double t6 = sin(th1);
        double t7 = sin(th2);
        double t8 = sin(th3);
        double t9 = sin(th4);
        double t10 = th1 + th2;
        double t11 = th2 + th3;
        double t12 = th3 + th4;
        double t13 = dth1 * dth1;
        double t14 = dth2 * dth2;
        double t15 = dth3 * dth3;
        double t16 = dth4 * dth4;
        double t17 = l1 * l1;
//         double t18 = t17 * l1;
        double t19 = l2 * l2;
        double t20 = t19 * l2;
        double t21 = l3 * l3;
        double t22 = t21 * l3;
        double t23 = l4 * l4;
//         double t24 = t23 * l4;
        double t25 = m2 * m2;
        double t26 = m3 * m3;
        double t27 = t26 * m3;
        double t28 = m4 * m4;
//         double t29 = t28 * m4;
        double t30 = th2 * 2.0;
        double t31 = th3 * 2.0;
        double t32 = th4 * 2.0;
        double t46 = 1.0 / l1;
//         double t48 = 1.0 / l2;
//         double t50 = 1.0 / l3;
        double t52 = 1.0 / l4;
        double t53 = -th1;
        double t54 = -th2;
        double t55 = -th3;
        double t57 = -th4;
        double t125 = m1 * m2 * m3 * 128.0;
        double t126 = m1 * m2 * m4 * 240.0;
        double t163 = m1 * m3 * m4 * 600.0;
        double t164 = m2 * m3 * m4 * 1500.0;
        double t33 = cos(t30);
        double t34 = cos(t31);
        double t35 = cos(t32);
        double t36 = sin(t30);
        double t37 = sin(t31);
        double t38 = sin(t32);
        double t39 = cos(t11);
        double t40 = cos(t12);
        double t41 = sin(t10);
        double t42 = sin(t11);
        double t43 = sin(t12);
        double t44 = t10 + th3;
        double t45 = t11 + th4;
//         double t47 = 1.0 / t17;
        double t49 = 1.0 / t19;
        double t51 = 1.0 / t21;
        double t56 = -t31;
        double t58 = -t32;
        double t62 = t10 + t12;
        double t63 = t10 + th2;
        double t64 = t31 + th1;
        double t65 = t32 + th1;
        double t66 = t11 + th3;
        double t67 = t11 + th2;
        double t68 = t32 + th2;
        double t69 = t30 + th4;
        double t70 = t12 + th4;
        double t71 = t12 + th3;
        double t87 = t10 + t31;
        double t88 = t10 + t32;
        double t89 = t11 + t32;
        double t92 = t54 + th1;
        double t94 = t55 + th2;
        double t97 = t57 + th3;
        double t98 = t11 * 2.0;
        double t99 = t30 + t32;
        double t100 = t12 * 2.0;
        double t110 = t10 + t55;
        double t112 = t11 + t53;
        double t114 = t11 + t57;
        double t116 = t12 + t54;
        double t117 = t27 * 144.0;
        double t128 = t30 + t57;
        double t156 = m1 * t28 * 144.0;
        double t157 = m3 * t28 * 144.0;
        double t158 = m1 * t26 * 240.0;
        double t159 = m3 * t25 * 240.0;
        double t160 = m2 * t28 * 360.0;
        double t161 = m4 * t25 * 450.0;
        double t162 = m4 * t26 * 450.0;
        double t179 = m2 * t26 * 600.0;
        double t211 = t10 - t12;
        double t213 = -t11 + th1 + th4;
        double t59 = cos(t45);
        double t60 = sin(t44);
        double t61 = sin(t45);
        double t72 = cos(t66);
        double t73 = cos(t67);
        double t74 = cos(t68);
        double t75 = cos(t69);
        double t76 = cos(t70);
        double t77 = cos(t71);
        double t78 = sin(t63);
        double t79 = sin(t64);
        double t80 = sin(t65);
        double t81 = sin(t66);
        double t82 = sin(t67);
        double t83 = sin(t68);
        double t84 = sin(t69);
        double t85 = sin(t70);
        double t86 = sin(t71);
        double t90 = t45 + th2;
        double t91 = sin(t62);
        double t93 = t56 + th1;
        double t95 = t58 + th1;
        double t96 = t58 + th2;
        double t101 = cos(t94);
        double t103 = cos(t97);
        double t104 = sin(t92);
        double t106 = sin(t94);
        double t109 = sin(t97);
        double t111 = t92 + th3;
        double t113 = t10 + t58;
        double t115 = t94 + th4;
        double t118 = cos(t89);
        double t120 = sin(t87);
        double t121 = sin(t88);
        double t122 = sin(t89);
        double t124 = t62 + th4;
        double t129 = t30 + t58;
        double t130 = cos(t98);
        double t131 = cos(t99);
        double t132 = cos(t100);
        double t133 = sin(t98);
        double t134 = sin(t99);
        double t135 = sin(t100);
        double t136 = t11 + t44;
        double t137 = t32 + t63;
        double t138 = t100 + th1;
        double t139 = t12 + t45;
        double t140 = t32 + t67;
        double t141 = t11 + t45;
        double t142 = cos(t114);
        double t144 = cos(t116);
        double t145 = sin(t110);
        double t147 = sin(t112);
        double t149 = sin(t114);
        double t151 = sin(t116);
        double t152 = t44 + t57;
        double t153 = t110 + th4;
        double t154 = t12 + t92;
        double t155 = t45 + t53;
        double t169 = cos(t128);
        double t171 = sin(t128);
        double t173 = t53 + t66;
        double t174 = t54 + t65;
        double t175 = t53 + t68;
        double t176 = t58 + t63;
        double t177 = t54 + t70;
        double t178 = t57 + t67;
        double t189 = t12 + t62;
        double t191 = t11 + t89;
        double t200 = t70 + t92;
        double t201 = t53 + t89;
        double t207 = t53 + t100;
        double t208 = m1 * m2 * m4 * t35 * 144.0;
        double t209 = m1 * m3 * m4 * t35 * 216.0;
        double t210 = m1 * m3 * m4 * t34 * 360.0;
        double t212 = t92 + t97;
        double t214 = t57 + t211;
        double t217 = -t27 * t33 * 144.0;
        double t218 = sin(t211);
        double t220 = sin(t213);
        double t222 = m1 * t26 * t34 * 144.0;
        double t223 = m3 * t25 * t33 * 144.0;
        double t226 = m4 * t26 * t35 * 162.0;
        double t227 = m2 * t26 * t34 * 216.0;
        double t228 = m2 * t28 * t33 * 216.0;
        double t229 = m2 * t28 * t34 * 216.0;
        double t230 = m4 * t25 * t33 * 270.0;
        double t231 = m4 * t25 * t35 * 270.0;
        double t232 = m2 * t26 * t33 * 360.0;
        double t237 = m2 * m3 * m4 * t34 * 540.0;
        double t238 = m2 * m3 * m4 * t35 * 540.0;
        double t239 = m2 * m3 * m4 * t33 * 900.0;
        double t243 = - m1 * t28 * t34 * 144.0;
        double t244 = - m3 * t28 * t33 * 144.0;
        double t252 = - m4 * t26 * t33 * 450.0;
        double t102 = cos(t96);
        double t105 = sin(t93);
        double t107 = sin(t95);
        double t108 = sin(t96);
        double t119 = cos(t90);
        double t123 = sin(t90);
        double t127 = sin(t124);
        double t143 = cos(t115);
        double t146 = sin(t111);
        double t148 = sin(t113);
        double t150 = sin(t115);
        double t165 = sin(t152);
        double t166 = sin(t153);
        double t167 = sin(t154);
        double t168 = sin(t155);
        double t170 = cos(t129);
        double t172 = sin(t129);
        double t180 = cos(t139);
        double t181 = cos(t140);
        double t182 = cos(t141);
        double t183 = sin(t136);
        double t184 = sin(t137);
        double t185 = sin(t138);
        double t186 = sin(t139);
        double t187 = sin(t140);
        double t188 = sin(t141);
        double t190 = sin(t189);
        double t192 = cos(t177);
        double t193 = cos(t178);
        double t194 = sin(t173);
        double t195 = sin(t174);
        double t196 = sin(t175);
        double t197 = sin(t176);
        double t198 = sin(t177);
        double t199 = sin(t178);
        double t202 = cos(t191);
        double t203 = sin(t191);
        double t205 = sin(t200);
        double t206 = sin(t201);
        double t215 = sin(t207);
        double t216 = t53 + t139;
        double t219 = sin(t212);
        double t221 = sin(t214);
        double t234 = -t208;
        double t235 = -t209;
        double t236 = -t210;
        double t241 = -t222;
        double t242 = -t223;
        double t245 = -t226;
        double t246 = -t227;
        double t247 = -t228;
        double t248 = -t229;
        double t249 = -t230;
        double t250 = -t231;
        double t251 = -t232;
        double t253 = -t237;
        double t254 = -t238;
        double t255 = -t239;
        double t256 = m1 * m3 * m4 * t132 * 72.0;
        double t257 = m2 * m3 * m4 * t132 * 108.0;
        double t258 = m2 * m3 * m4 * t131 * 162.0;
        double t259 = m2 * m3 * m4 * t130 * 180.0;
        double t260 = m2 * t26 * t130 * 72.0;
        double t261 = m2 * t28 * t130 * 72.0;
        double t262 = m4 * t25 * t131 * 81.0;
        double t263 = m4 * t26 * t131 * 81.0;
        double t240 = sin(t216);
        double t264 = m2 * m3 * m4 * t170 * 162.0;
        double t265 = m4 * t25 * t170 * 81.0;
        double t266 = m4 * t26 * t170 * 81.0;
        double t267 = m2 * m3 * m4 * t202 * 36.0;
        double t268 = -t267;
        double t269 = t117 + t125 + t126 + t156 + t157 + t158 + t159 + t160 + t161 + t162 + t163 + t164 + t179 + t217 + t234 + t235 + t236 + t241 + t242 + t243 + t244 + t245 + t246 + t247 + t248 + t249 + t250 + t251 + t252 + t253 + t254 + t255 + t256 + t257 + t258 + t259 + t260 + t261 + t262 + t263 + t264 + t265 + t266 + t268;
        double t270 = 1.0 / t269;

        d_dx3[index] = t46*t49*t51*t52*t270*(l1*l4*t19*t25*u3*3.0*10.0 - l1*l4*t19*t25*u4*3.0*10.0 + l1*l4*t19*t26*u3*7.2*10.0 - l1*l4*t19*t26*u4*7.2*10.0 - l1*l4*t21*t26*u2*3.0*10.0 + l1*l4*t19*t28*u3*1.8*10.0 + l1*l4*t21*t26*u3*3.0*10.0 - l1*l4*t19*t28*u4*1.8*10.0 - l1*l4*t21*t28*u2*1.8*10.0 + l1*l4*t21*t28*u3*1.8*10.0 - l1*l3*t5*t19*t25*u4*4.5*10.0 + l2*l4*t3*t21*t26*u1*3.0*10.0 - l1*l3*t5*t19*t26*u4*5.4*10.0 - l2*l4*t3*t21*t26*u2*3.0*10.0 + l2*l4*t3*t21*t28*u1*1.8*10.0 - l2*l4*t3*t21*t28*u2*1.8*10.0 - l1*l4*t19*t25*t33*u3*1.8*10.0 + l1*l4*t19*t25*t33*u4*1.8*10.0 - l1*l4*t19*t26*t33*u3*7.2*10.0 + l1*l4*t19*t26*t33*u4*7.2*10.0 - l1*l4*t19*t28*t33*u3*1.8*10.0 + l1*l4*t19*t28*t33*u4*1.8*10.0 - l3*l4*t19*t26*t39*u1*3.6*10.0 + l3*l4*t19*t26*t39*u2*3.6*10.0 - l1*l2*t21*t26*t40*u4*(9.0 / 2.0) - l3*l4*t19*t28*t39*u1*1.8*10.0 + l3*l4*t19*t28*t39*u2*1.8*10.0 - l2*l4*t21*t26*t72*u1*1.8*10.0 + l1*l3*t19*t25*t75*u4*(2.7*10.0 / 2.0) + l2*l4*t21*t26*t72*u2*1.8*10.0 + l1*l3*t19*t26*t75*u4*2.7*10.0 - l2*l4*t21*t28*t72*u1*1.8*10.0 + l2*l4*t21*t28*t72*u2*1.8*10.0 + l3*l4*t19*t26*t101*u1*3.6*10.0 - l3*l4*t19*t26*t101*u2*3.6*10.0 + l3*l4*t19*t28*t101*u1*1.8*10.0 - l1*l2*t21*t26*t103*u4*(2.7*10.0 / 2.0) - l3*l4*t19*t28*t101*u2*1.8*10.0 + l1*l2*t21*t26*t119*u4*(9.0 / 2.0) + l1*l4*t21*t26*t130*u2*1.8*10.0 - l1*l4*t21*t26*t130*u3*1.8*10.0 + l1*l4*t21*t28*t130*u2*1.8*10.0 - l1*l4*t21*t28*t130*u3*1.8*10.0 + l1*l3*t19*t25*t169*u4*(2.7*10.0 / 2.0) + l1*l3*t19*t26*t169*u4*2.7*10.0 + l1*l2*t21*t26*t193*u4*(2.7*10.0 / 2.0) + l1*l4*m1*m2*t19*u3*1.6*10.0 - l1*l4*m1*m2*t19*u4*1.6*10.0 + l1*l4*m1*m3*t19*u3*4.8*10.0 - l1*l4*m1*m3*t19*u4*4.8*10.0 - l1*l4*m1*m3*t21*u2*1.6*10.0 + l1*l4*m1*m4*t19*u3*3.0*10.0 + l1*l4*m2*m3*t19*u3*1.2*100.0 + l1*l4*m1*m3*t21*u3*1.6*10.0 - l1*l4*m1*m4*t19*u4*3.0*10.0 - l1*l4*m1*m4*t21*u2*3.0*10.0 - l1*l4*m2*m3*t19*u4*1.2*100.0 - l1*l4*m2*m3*t21*u2*4.8*10.0 + l1*l4*m2*m4*t19*u3*7.5*10.0 + l1*l4*m1*m4*t21*u3*3.0*10.0 + l1*l4*m2*m3*t21*u3*4.8*10.0 - l1*l4*m2*m4*t19*u4*7.5*10.0 - l1*l4*m2*m4*t21*u2*9.0*10.0 + l1*l4*m3*m4*t19*u3*9.0*10.0 + l1*l4*m2*m4*t21*u3*9.0*10.0 - l1*l4*m3*m4*t19*u4*9.0*10.0 - l1*l4*m3*m4*t21*u2*7.5*10.0 + l1*l4*m3*m4*t21*u3*7.5*10.0 + g*l1*l2*l4*t21*t27*t41*6.0 - g*l1*l2*l4*t21*t27*t104*6.0 - l1*l2*l3*l4*t4*t26*u2*3.6*10.0 + l1*l2*l3*l4*t4*t26*u3*7.2*10.0 - l1*l2*l3*l4*t4*t26*u4*3.6*10.0 - l1*l2*l3*l4*t4*t28*u2*1.8*10.0 + l1*l2*l3*l4*t4*t28*u3*3.6*10.0 - l1*l2*l3*l4*t4*t28*u4*1.8*10.0 + l1*l2*l3*l4*t26*t73*u2*3.6*10.0 - l1*l2*l3*l4*t26*t73*u3*7.2*10.0 + l1*l2*l3*l4*t26*t73*u4*3.6*10.0 + l1*l2*l3*l4*t28*t73*u2*1.8*10.0 - l1*l2*l3*l4*t28*t73*u3*3.6*10.0 + l1*l2*l3*l4*t28*t73*u4*1.8*10.0 - l1*l3*m1*m2*t5*t19*u4*2.4*10.0 - l1*l3*m1*m3*t5*t19*u4*5.4*10.0 + l2*l4*m2*m3*t3*t21*u1*2.4*10.0 - l1*l3*m1*m4*t5*t19*u4*3.6*10.0 - l1*l3*m2*m3*t5*t19*u4*1.35*100.0 - l2*l4*m2*m3*t3*t21*u2*2.4*10.0 + l2*l4*m2*m4*t3*t21*u1*4.5*10.0 - l1*l3*m2*m4*t5*t19*u4*9.0*10.0 - l2*l4*m2*m4*t3*t21*u2*4.5*10.0 + l2*l4*m3*m4*t3*t21*u1*7.5*10.0 - l1*l3*m3*m4*t5*t19*u4*5.4*10.0 - l2*l4*m3*m4*t3*t21*u2*7.5*10.0 - l1*l4*m2*m3*t19*t33*u3*7.2*10.0 + l1*l4*m2*m3*t19*t33*u4*7.2*10.0 - l1*l4*m2*m4*t19*t33*u3*4.5*10.0 + l1*l4*m2*m4*t19*t33*u4*4.5*10.0 - l1*l4*m3*m4*t19*t33*u3*9.0*10.0 + l1*l4*m1*m4*t21*t35*u2*1.8*10.0 + l1*l4*m3*m4*t19*t33*u4*9.0*10.0 - l1*l4*m1*m4*t21*t35*u3*1.8*10.0 + l1*l4*m2*m4*t21*t35*u2*5.4*10.0 - l1*l4*m2*m4*t21*t35*u3*5.4*10.0 + l1*l4*m3*m4*t21*t35*u2*2.7*10.0 - l1*l4*m3*m4*t21*t35*u3*2.7*10.0 - l3*l4*m2*m3*t19*t39*u1*6.0 + l1*l2*m1*m3*t21*t40*u4*6.0 + l3*l4*m2*m3*t19*t39*u2*6.0 - l3*l4*m2*m4*t19*t39*u1*(1.5*10.0 / 2.0) + l1*l2*m1*m4*t21*t40*u4*3.6*10.0 + l1*l2*m2*m3*t21*t40*u4*(2.7*10.0 / 2.0) + l3*l4*m2*m4*t19*t39*u2*(1.5*10.0 / 2.0) - l3*l4*m3*m4*t19*t39*u1*(1.35*100.0 / 2.0) + l1*l2*m2*m4*t21*t40*u4*8.1*10.0 + l3*l4*m3*m4*t19*t39*u2*(1.35*100.0 / 2.0) + l1*l2*m3*m4*t21*t40*u4*9.0 + l1*l3*m2*m3*t19*t75*u4*(8.1*10.0 / 2.0) - l2*l4*m3*m4*t21*t72*u1*4.5*10.0 + l1*l3*m1*m3*t19*t77*u4*1.8*10.0 + l1*l3*m2*m4*t19*t75*u4*2.7*10.0 - l2*l4*m2*m4*t21*t74*u1*(2.7*10.0 / 2.0) + l2*l4*m3*m4*t21*t72*u2*4.5*10.0 + l1*l3*m1*m4*t19*t77*u4*3.6*10.0 + l1*l3*m2*m3*t19*t77*u4*2.7*10.0 + l1*l3*m3*m4*t19*t75*u4*2.7*10.0 + l2*l4*m2*m4*t21*t74*u2*(2.7*10.0 / 2.0) - l2*l4*m3*m4*t21*t74*u1*(2.7*10.0 / 2.0) + l1*l3*m2*m4*t19*t77*u4*5.4*10.0 + l2*l4*m3*m4*t21*t74*u2*(2.7*10.0 / 2.0) + l3*l4*m2*m3*t19*t101*u1*1.8*10.0 - l3*l4*m2*m3*t19*t101*u2*1.8*10.0 + l3*l4*m2*m4*t19*t101*u1*(4.5*10.0 / 2.0) - l1*l2*m1*m3*t21*t103*u4*1.8*10.0 - l3*l4*m2*m4*t19*t101*u2*(4.5*10.0 / 2.0) + l3*l4*m3*m4*t19*t101*u1*(1.35*100.0 / 2.0) - l1*l2*m1*m4*t21*t103*u4*3.6*10.0 - l1*l2*m2*m3*t21*t103*u4*(8.1*10.0 / 2.0) - l2*l4*m2*m4*t21*t102*u1*(2.7*10.0 / 2.0) - l3*l4*m3*m4*t19*t101*u2*(1.35*100.0 / 2.0) - l1*l2*m2*m4*t21*t103*u4*8.1*10.0 + l2*l4*m2*m4*t21*t102*u2*(2.7*10.0 / 2.0) - l2*l4*m3*m4*t21*t102*u1*(2.7*10.0 / 2.0) - l1*l2*m3*m4*t21*t103*u4*2.7*10.0 + l2*l4*m3*m4*t21*t102*u2*(2.7*10.0 / 2.0) + l3*l4*m2*m4*t19*t118*u1*(9.0 / 2.0) - l1*l2*m2*m3*t21*t119*u4*(9.0 / 2.0) - l3*l4*m2*m4*t19*t118*u2*(9.0 / 2.0) + l3*l4*m3*m4*t19*t118*u1*(2.7*10.0 / 2.0) - l1*l2*m2*m4*t21*t119*u4*2.7*10.0 - l3*l4*m3*m4*t19*t118*u2*(2.7*10.0 / 2.0) - l1*l2*m3*m4*t21*t119*u4*9.0 - l1*l4*m1*m4*t19*t132*u3*1.8*10.0 + l1*l4*m1*m4*t19*t132*u4*1.8*10.0 - l1*l4*m2*m4*t19*t132*u3*2.7*10.0 + l1*l4*m3*m4*t21*t130*u2*4.5*10.0 + l1*l4*m2*m4*t19*t132*u4*2.7*10.0 - l1*l4*m3*m4*t21*t130*u3*4.5*10.0 + l1*l3*m2*m3*t19*t169*u4*(8.1*10.0 / 2.0) + l1*l3*m2*m4*t19*t169*u4*2.7*10.0 + l1*l3*m3*m4*t19*t169*u4*2.7*10.0 - l1*l3*m2*m3*t19*t182*u4*9.0 - l1*l3*m2*m4*t19*t182*u4*1.8*10.0 + l2*l4*m3*m4*t21*t180*u1*9.0 - l2*l4*m3*m4*t21*t180*u2*9.0 - l3*l4*m2*m4*t19*t192*u1*(2.7*10.0 / 2.0) + l1*l2*m2*m3*t21*t193*u4*(2.7*10.0 / 2.0) + l3*l4*m2*m4*t19*t192*u2*(2.7*10.0 / 2.0) - l3*l4*m3*m4*t19*t192*u1*(2.7*10.0 / 2.0) + l1*l2*m2*m4*t21*t193*u4*2.7*10.0 + l3*l4*m3*m4*t19*t192*u2*(2.7*10.0 / 2.0) + l1*l2*m3*m4*t21*t193*u4*2.7*10.0 + l1*l4*m2*m4*t19*t202*u3*9.0 - l1*l4*m2*m4*t19*t202*u4*9.0 - l1*l4*m3*m4*t21*t202*u2*9.0 + l1*l4*m3*m4*t21*t202*u3*9.0 - l1*l2*l4*t8*t13*t22*t27*3.0 - l1*l2*l4*t8*t14*t22*t27*3.0 - l1*l2*l4*t8*t15*t22*t27*3.0 + l1*l2*l4*t13*t22*t27*t82*3.0 + l1*l2*l4*t14*t22*t27*t82*3.0 + l1*l2*l4*t15*t22*t27*t82*3.0 + l2*l4*t7*t13*t17*t21*t27*1.2*10.0 + l1*l4*t13*t19*t21*t27*t36*6.0 + l1*l4*t14*t19*t21*t27*t36*6.0 - dth1*dth2*l1*l2*l4*t8*t22*t27*6.0 - dth1*dth3*l1*l2*l4*t8*t22*t27*6.0 - dth2*dth3*l1*l2*l4*t8*t22*t27*6.0 + dth1*dth2*l1*l2*l4*t22*t27*t82*6.0 + dth1*dth3*l1*l2*l4*t22*t27*t82*6.0 + dth2*dth3*l1*l2*l4*t22*t27*t82*6.0 + dth1*dth2*l1*l4*t19*t21*t27*t36*1.2*10.0 - l1*l2*l3*l4*m1*m3*t4*u2*2.4*10.0 + l1*l2*l3*l4*m1*m3*t4*u3*4.8*10.0 - l1*l2*l3*l4*m1*m4*t4*u2*3.0*10.0 - l1*l2*l3*l4*m2*m3*t4*u2*5.4*10.0 - l1*l2*l3*l4*m1*m3*t4*u4*2.4*10.0 + l1*l2*l3*l4*m1*m4*t4*u3*6.0*10.0 + l1*l2*l3*l4*m2*m3*t4*u3*1.08*100.0 - l1*l2*l3*l4*m2*m4*t4*u2*(1.35*100.0 / 2.0) - l1*l2*l3*l4*m1*m4*t4*u4*3.0*10.0 - l1*l2*l3*l4*m2*m3*t4*u4*5.4*10.0 + l1*l2*l3*l4*m2*m4*t4*u3*1.35*100.0 - l1*l2*l3*l4*m3*m4*t4*u2*(1.35*100.0 / 2.0) - l1*l2*l3*l4*m2*m4*t4*u4*(1.35*100.0 / 2.0) + l1*l2*l3*l4*m3*m4*t4*u3*1.35*100.0 - l1*l2*l3*l4*m3*m4*t4*u4*(1.35*100.0 / 2.0) + l1*l2*l3*l4*m2*m3*t73*u2*1.8*10.0 - l1*l2*l3*l4*m2*m3*t73*u3*3.6*10.0 + l1*l2*l3*l4*m2*m4*t73*u2*(4.5*10.0 / 2.0) + l1*l2*l3*l4*m2*m3*t73*u4*1.8*10.0 - l1*l2*l3*l4*m2*m4*t73*u3*4.5*10.0 + l1*l2*l3*l4*m3*m4*t73*u2*(1.35*100.0 / 2.0) + l1*l2*l3*l4*m1*m4*t76*u2*1.8*10.0 + l1*l2*l3*l4*m2*m4*t73*u4*(4.5*10.0 / 2.0) - l1*l2*l3*l4*m3*m4*t73*u3*1.35*100.0 - l1*l2*l3*l4*m1*m4*t76*u3*3.6*10.0 + l1*l2*l3*l4*m2*m4*t76*u2*(8.1*10.0 / 2.0) + l1*l2*l3*l4*m3*m4*t73*u4*(1.35*100.0 / 2.0) + l1*l2*l3*l4*m1*m4*t76*u4*1.8*10.0 - l1*l2*l3*l4*m2*m4*t76*u3*8.1*10.0 + l1*l2*l3*l4*m3*m4*t76*u2*(2.7*10.0 / 2.0) + l1*l2*l3*l4*m2*m4*t76*u4*(8.1*10.0 / 2.0) - l1*l2*l3*l4*m3*m4*t76*u3*2.7*10.0 + l1*l2*l3*l4*m3*m4*t76*u4*(2.7*10.0 / 2.0) - l1*l2*l3*l4*m2*m4*t181*u2*(2.7*10.0 / 2.0) + l1*l2*l3*l4*m2*m4*t181*u3*2.7*10.0 - l1*l2*l3*l4*m3*m4*t181*u2*(2.7*10.0 / 2.0) - l1*l2*l3*l4*m2*m4*t181*u4*(2.7*10.0 / 2.0) + l1*l2*l3*l4*m3*m4*t181*u3*2.7*10.0 - l1*l2*l3*l4*m3*m4*t181*u4*(2.7*10.0 / 2.0) + g*l1*l2*l4*m1*t21*t26*t41*(5.0 / 2.0) + g*l1*l2*l4*m2*t21*t26*t41*(4.5*10.0 / 2.0) + g*l1*l2*l4*m3*t21*t25*t41*1.2*10.0 + g*l1*l2*l4*m1*t21*t28*t41*(3.0 / 2.0) + g*l1*l2*l4*m4*t21*t25*t41*(4.5*10.0 / 2.0) + g*l1*l2*l4*m2*t21*t28*t41*(2.7*10.0 / 2.0) + g*l1*l2*l4*m4*t21*t26*t41*(7.5*10.0 / 4.0) + g*l1*l2*l4*m3*t21*t28*t41*6.0 - g*l1*l3*l4*m1*t19*t26*t60*3.0 - g*l1*l3*l4*m2*t19*t26*t60*3.0 + g*l1*l3*l4*m3*t19*t25*t60*(3.0 / 2.0) - g*l1*l3*l4*m1*t19*t28*t60*(3.0 / 2.0) + g*l1*l3*l4*m4*t19*t25*t60*(1.5*10.0 / 8.0) - g*l1*l3*l4*m2*t19*t28*t60*(3.0 / 2.0) - g*l1*l2*l4*m1*t21*t26*t104*(1.5*10.0 / 2.0) - g*l1*l2*l4*m2*t21*t26*t104*(4.5*10.0 / 2.0) - g*l1*l2*l4*m3*t21*t25*t104*1.2*10.0 - g*l1*l2*l4*m1*t21*t28*t104*(9.0 / 2.0) - g*l1*l2*l4*m4*t21*t25*t104*(4.5*10.0 / 2.0) - g*l1*l2*l4*m2*t21*t28*t104*(2.7*10.0 / 2.0) - g*l1*l2*l4*m4*t21*t26*t104*(7.5*10.0 / 4.0) - g*l1*l2*l4*m3*t21*t28*t104*6.0 - g*l1*l2*l4*m1*t21*t26*t120*(3.0 / 2.0) - g*l1*l2*l4*m2*t21*t26*t120*(9.0 / 2.0) - g*l1*l2*l4*m1*t21*t28*t120*(3.0 / 2.0) - g*l1*l2*l4*m2*t21*t28*t120*(9.0 / 2.0) - g*l1*l2*l4*m4*t21*t25*t121*(2.7*10.0 / 4.0) - g*l1*l2*l4*m4*t21*t26*t121*(2.7*10.0 / 8.0) - g*l1*l3*l4*m4*t19*t25*t127*(9.0 / 8.0) + g*l1*l3*l4*m1*t19*t26*t145*3.0 - g*l1*l3*l4*m1*t19*t26*t146*9.0 + g*l1*l3*l4*m2*t19*t26*t145*9.0 + g*l1*l3*l4*m3*t19*t25*t145*(9.0 / 2.0) - g*l1*l3*l4*m1*t19*t26*t147*9.0 + g*l1*l3*l4*m1*t19*t28*t145*(3.0 / 2.0) - g*l1*l3*l4*m2*t19*t26*t146*9.0 - g*l1*l3*l4*m3*t19*t25*t146*(9.0 / 2.0) + g*l1*l3*l4*m4*t19*t25*t145*(4.5*10.0 / 8.0) - g*l1*l3*l4*m1*t19*t28*t146*(9.0 / 2.0) - g*l1*l3*l4*m2*t19*t26*t147*3.0 + g*l1*l3*l4*m2*t19*t28*t145*(9.0 / 2.0) + g*l1*l3*l4*m3*t19*t25*t147*(3.0 / 2.0) - g*l1*l3*l4*m4*t19*t25*t146*(4.5*10.0 / 8.0) - g*l1*l3*l4*m1*t19*t28*t147*(9.0 / 2.0) - g*l1*l3*l4*m2*t19*t28*t146*(9.0 / 2.0) + g*l1*l3*l4*m4*t19*t25*t147*(1.5*10.0 / 8.0) - g*l1*l3*l4*m2*t19*t28*t147*(3.0 / 2.0) - g*l1*l2*l4*m4*t21*t25*t148*(2.7*10.0 / 4.0) - g*l1*l2*l4*m4*t21*t26*t148*(2.7*10.0 / 8.0) - g*l1*l2*l4*m1*t21*t26*t194*(9.0 / 2.0) - g*l1*l2*l4*m2*t21*t26*t194*(9.0 / 2.0) - g*l1*l2*l4*m1*t21*t28*t194*(9.0 / 2.0) - g*l1*l2*l4*m2*t21*t28*t194*(9.0 / 2.0) + g*l1*l2*l4*m4*t21*t25*t195*(2.7*10.0 / 4.0) - g*l1*l2*l4*m4*t21*t25*t196*(2.7*10.0 / 4.0) + g*l1*l2*l4*m4*t21*t26*t195*(2.7*10.0 / 8.0) - g*l1*l2*l4*m4*t21*t26*t196*(2.7*10.0 / 8.0) + g*l1*l3*l4*m4*t19*t25*t205*(2.7*10.0 / 8.0) - g*l1*l3*l4*m4*t19*t25*t206*(9.0 / 8.0) - g*l1*l3*l4*m4*t19*t25*t221*(2.7*10.0 / 8.0) - l1*l3*l4*m1*t8*t13*t20*t26*2.4*10.0 - l1*l2*l4*m1*t8*t13*t22*t26*8.0 - l1*l3*l4*m1*t8*t14*t20*t26*2.4*10.0 - l1*l3*l4*m2*t8*t13*t20*t26*3.0*10.0 - l1*l3*l4*m3*t8*t13*t20*t25*9.0 - l1*l2*l4*m1*t8*t14*t22*t26*8.0 - l1*l2*l4*m2*t8*t13*t22*t26*1.8*10.0 - l1*l3*l4*m1*t8*t13*t20*t28*1.2*10.0 - l1*l3*l4*m2*t8*t14*t20*t26*3.0*10.0 - l1*l3*l4*m3*t8*t14*t20*t25*9.0 - l1*l3*l4*m4*t8*t13*t20*t25*(4.5*10.0 / 4.0) - l1*l2*l4*m1*t8*t13*t22*t28*1.2*10.0 - l1*l2*l4*m1*t8*t15*t22*t26*8.0 - l1*l2*l4*m2*t8*t14*t22*t26*1.8*10.0 - l1*l3*l4*m1*t8*t14*t20*t28*1.2*10.0 - l1*l3*l4*m2*t8*t13*t20*t28*1.5*10.0 - l1*l3*l4*m4*t8*t14*t20*t25*(4.5*10.0 / 4.0) - l1*l2*l4*m1*t8*t14*t22*t28*1.2*10.0 - l1*l2*l4*m2*t8*t13*t22*t28*2.7*10.0 - l1*l2*l4*m2*t8*t15*t22*t26*1.8*10.0 - l1*l2*l4*m4*t8*t13*t22*t26*(4.5*10.0 / 4.0) - l1*l3*l4*m2*t8*t14*t20*t28*1.5*10.0 - l1*l2*l4*m1*t8*t15*t22*t28*1.2*10.0 - l1*l2*l4*m2*t8*t14*t22*t28*2.7*10.0 - l1*l2*l4*m3*t8*t13*t22*t28*6.0 - l1*l2*l4*m4*t8*t14*t22*t26*(4.5*10.0 / 4.0) - l1*l2*l4*m2*t8*t15*t22*t28*2.7*10.0 - l1*l2*l4*m3*t8*t14*t22*t28*6.0 - l1*l2*l4*m4*t8*t15*t22*t26*(4.5*10.0 / 4.0) - l1*l2*l4*m3*t8*t15*t22*t28*6.0 + l1*l3*l4*m2*t13*t20*t26*t82*6.0 + l1*l3*l4*m3*t13*t20*t25*t82*3.0 + l1*l2*l4*m2*t13*t22*t26*t82*6.0 + l1*l3*l4*m2*t14*t20*t26*t82*6.0 + l1*l3*l4*m3*t14*t20*t25*t82*3.0 + l1*l3*l4*m4*t13*t20*t25*t82*(1.5*10.0 / 4.0) + l1*l2*l4*m2*t14*t22*t26*t82*6.0 + l1*l3*l4*m2*t13*t20*t28*t82*3.0 + l1*l3*l4*m4*t14*t20*t25*t82*(1.5*10.0 / 4.0) + l1*l2*l4*m2*t13*t22*t28*t82*9.0 + l1*l2*l4*m2*t15*t22*t26*t82*6.0 + l1*l2*l4*m4*t13*t22*t26*t82*(4.5*10.0 / 4.0) + l1*l3*l4*m2*t14*t20*t28*t82*3.0 + l1*l2*l4*m2*t14*t22*t28*t82*9.0 + l1*l2*l4*m3*t13*t22*t28*t82*6.0 + l1*l2*l4*m4*t14*t22*t26*t82*(4.5*10.0 / 4.0) + l1*l3*l4*m4*t13*t20*t25*t85*(2.7*10.0 / 4.0) + l1*l2*l4*m2*t15*t22*t28*t82*9.0 + l1*l2*l4*m3*t14*t22*t28*t82*6.0 + l1*l2*l4*m4*t15*t22*t26*t82*(4.5*10.0 / 4.0) + l1*l3*l4*m4*t14*t20*t25*t85*(2.7*10.0 / 4.0) + l1*l2*l4*m3*t15*t22*t28*t82*6.0 + l1*l2*l4*m4*t13*t22*t26*t85*(9.0 / 4.0) + l1*l2*l4*m4*t14*t22*t26*t85*(9.0 / 4.0) + l1*l2*l4*m4*t15*t22*t26*t85*(9.0 / 4.0) - l1*l3*l4*m4*t13*t20*t25*t187*(9.0 / 4.0) - l1*l3*l4*m4*t14*t20*t25*t187*(9.0 / 4.0) - l1*l2*l4*m4*t13*t22*t26*t187*(9.0 / 4.0) - l1*l2*l4*m4*t14*t22*t26*t187*(9.0 / 4.0) - l1*l2*l4*m4*t15*t22*t26*t187*(9.0 / 4.0) + l2*l4*m1*t7*t13*t17*t21*t26*1.0*10.0 + l2*l4*m2*t7*t13*t17*t21*t26*4.5*10.0 + l2*l4*m3*t7*t13*t17*t21*t25*2.4*10.0 + l2*l4*m1*t7*t13*t17*t21*t28*6.0 + l2*l4*m4*t7*t13*t17*t21*t25*4.5*10.0 + l2*l4*m2*t7*t13*t17*t21*t28*2.7*10.0 + l2*l4*m4*t7*t13*t17*t21*t26*(7.5*10.0 / 2.0) + l2*l4*m3*t7*t13*t17*t21*t28*1.2*10.0 + l1*l3*m1*t9*t13*t19*t23*t28*3.0 + l1*l3*m4*t9*t13*t19*t23*t25*1.5*10.0 + l1*l3*m1*t9*t14*t19*t23*t28*3.0 + l1*l3*m2*t9*t13*t19*t23*t28*(1.5*10.0 / 2.0) + l1*l3*m4*t9*t13*t19*t23*t26*1.8*10.0 + l1*l3*m4*t9*t14*t19*t23*t25*1.5*10.0 + l1*l3*m1*t9*t15*t19*t23*t28*3.0 + l1*l3*m2*t9*t14*t19*t23*t28*(1.5*10.0 / 2.0) + l1*l3*m3*t9*t13*t19*t23*t28*(9.0 / 2.0) + l1*l3*m4*t9*t14*t19*t23*t26*1.8*10.0 + l1*l3*m4*t9*t15*t19*t23*t25*1.5*10.0 + l1*l3*m1*t9*t16*t19*t23*t28*3.0 + l1*l3*m2*t9*t15*t19*t23*t28*(1.5*10.0 / 2.0) + l1*l3*m3*t9*t14*t19*t23*t28*(9.0 / 2.0) + l1*l3*m4*t9*t15*t19*t23*t26*1.8*10.0 + l1*l3*m4*t9*t16*t19*t23*t25*1.5*10.0 + l1*l3*m2*t9*t16*t19*t23*t28*(1.5*10.0 / 2.0) + l1*l3*m3*t9*t15*t19*t23*t28*(9.0 / 2.0) + l1*l3*m4*t9*t16*t19*t23*t26*1.8*10.0 + l1*l3*m3*t9*t16*t19*t23*t28*(9.0 / 2.0) - l1*l4*m1*t13*t19*t21*t26*t37*1.2*10.0 + l1*l4*m2*t13*t19*t21*t26*t36*1.5*10.0 + l1*l4*m3*t13*t19*t21*t25*t36*6.0 - l1*l4*m1*t14*t19*t21*t26*t37*1.2*10.0 - l1*l4*m2*t13*t19*t21*t26*t37*1.8*10.0 + l1*l4*m2*t14*t19*t21*t26*t36*1.5*10.0 + l1*l4*m3*t14*t19*t21*t25*t36*6.0 + l1*l4*m4*t13*t19*t21*t25*t36*(4.5*10.0 / 4.0) - l1*l4*m1*t13*t19*t21*t28*t37*1.2*10.0 - l1*l4*m1*t15*t19*t21*t26*t37*6.0 + l1*l4*m2*t13*t19*t21*t28*t36*9.0 - l1*l4*m2*t14*t19*t21*t26*t37*1.8*10.0 + l1*l4*m4*t13*t19*t21*t26*t36*(7.5*10.0 / 4.0) + l1*l4*m4*t14*t19*t21*t25*t36*(4.5*10.0 / 4.0) - l1*l4*m1*t14*t19*t21*t28*t37*1.2*10.0 - l1*l4*m2*t13*t19*t21*t28*t37*1.8*10.0 + l1*l4*m2*t14*t19*t21*t28*t36*9.0 - l1*l4*m2*t15*t19*t21*t26*t37*9.0 + l1*l4*m3*t13*t19*t21*t28*t36*6.0 + l1*l4*m4*t13*t19*t21*t25*t38*(4.5*10.0 / 4.0) + l1*l4*m4*t14*t19*t21*t26*t36*(7.5*10.0 / 4.0) - l3*l4*m1*t13*t17*t19*t26*t42*1.2*10.0 - l1*l4*m1*t15*t19*t21*t28*t37*6.0 - l1*l4*m2*t14*t19*t21*t28*t37*1.8*10.0 + l1*l4*m3*t14*t19*t21*t28*t36*6.0 + l1*l4*m4*t13*t19*t21*t26*t38*(2.7*10.0 / 4.0) + l1*l4*m4*t14*t19*t21*t25*t38*(4.5*10.0 / 4.0) - l3*l4*m2*t13*t17*t19*t26*t42*6.0 + l3*l4*m3*t13*t17*t19*t25*t42*3.0 - l1*l4*m2*t15*t19*t21*t28*t37*9.0 + l1*l4*m4*t14*t19*t21*t26*t38*(2.7*10.0 / 4.0) + l1*l4*m4*t15*t19*t21*t25*t38*(4.5*10.0 / 4.0) - l3*l4*m1*t13*t17*t19*t28*t42*6.0 + l3*l4*m4*t13*t17*t19*t25*t42*(1.5*10.0 / 4.0) + l1*l4*m4*t15*t19*t21*t26*t38*(2.7*10.0 / 4.0) - l3*l4*m2*t13*t17*t19*t28*t42*3.0 - l1*l2*m1*t13*t21*t23*t28*t43*3.0 - l1*l2*m1*t14*t21*t23*t28*t43*3.0 - l1*l2*m2*t13*t21*t23*t28*t43*(2.7*10.0 / 4.0) + l1*l2*m4*t13*t21*t23*t26*t43*(3.0 / 2.0) - l1*l2*m1*t15*t21*t23*t28*t43*3.0 - l1*l2*m2*t14*t21*t23*t28*t43*(2.7*10.0 / 4.0) - l1*l2*m3*t13*t21*t23*t28*t43*(3.0 / 4.0) + l1*l2*m4*t14*t21*t23*t26*t43*(3.0 / 2.0) - l1*l2*m1*t16*t21*t23*t28*t43*3.0 - l1*l2*m2*t15*t21*t23*t28*t43*(2.7*10.0 / 4.0) - l1*l2*m3*t14*t21*t23*t28*t43*(3.0 / 4.0) + l1*l2*m4*t15*t21*t23*t26*t43*(3.0 / 2.0) - l1*l2*m2*t16*t21*t23*t28*t43*(2.7*10.0 / 4.0) - l1*l2*m3*t15*t21*t23*t28*t43*(3.0 / 4.0) + l1*l2*m4*t16*t21*t23*t26*t43*(3.0 / 2.0) - l1*l2*m3*t16*t21*t23*t28*t43*(3.0 / 4.0) - l2*l4*m1*t13*t17*t21*t26*t81*6.0 - l2*l4*m2*t13*t17*t21*t26*t81*9.0 - l2*l4*m1*t13*t17*t21*t28*t81*6.0 - l2*l4*m2*t13*t17*t21*t28*t81*9.0 - l2*l4*m4*t13*t17*t21*t25*t83*(2.7*10.0 / 2.0) - l2*l4*m4*t13*t17*t21*t26*t83*(2.7*10.0 / 4.0) - l1*l3*m4*t13*t19*t23*t25*t84*(9.0 / 2.0) - l1*l3*m2*t13*t19*t23*t28*t84*(9.0 / 4.0) - l1*l3*m4*t13*t19*t23*t26*t84*9.0 - l1*l3*m4*t14*t19*t23*t25*t84*(9.0 / 2.0) - l1*l3*m1*t13*t19*t23*t28*t86*3.0 - l1*l3*m2*t14*t19*t23*t28*t84*(9.0 / 4.0) - l1*l3*m3*t13*t19*t23*t28*t84*(9.0 / 4.0) - l1*l3*m4*t14*t19*t23*t26*t84*9.0 - l1*l3*m4*t15*t19*t23*t25*t84*(9.0 / 2.0) - l1*l3*m1*t14*t19*t23*t28*t86*3.0 - l1*l3*m2*t13*t19*t23*t28*t86*(9.0 / 2.0) - l1*l3*m2*t15*t19*t23*t28*t84*(9.0 / 4.0) - l1*l3*m3*t14*t19*t23*t28*t84*(9.0 / 4.0) - l1*l3*m4*t15*t19*t23*t26*t84*9.0 - l1*l3*m4*t16*t19*t23*t25*t84*(9.0 / 2.0) - l1*l3*m1*t15*t19*t23*t28*t86*3.0 - l1*l3*m2*t14*t19*t23*t28*t86*(9.0 / 2.0) - l1*l3*m2*t16*t19*t23*t28*t84*(9.0 / 4.0) - l1*l3*m3*t15*t19*t23*t28*t84*(9.0 / 4.0) - l1*l3*m4*t16*t19*t23*t26*t84*9.0 - l1*l3*m1*t16*t19*t23*t28*t86*3.0 - l1*l3*m2*t15*t19*t23*t28*t86*(9.0 / 2.0) - l1*l3*m3*t16*t19*t23*t28*t84*(9.0 / 4.0) - l1*l3*m2*t16*t19*t23*t28*t86*(9.0 / 2.0) + l3*l4*m1*t13*t17*t19*t26*t106*1.2*10.0 + l3*l4*m2*t13*t17*t19*t26*t106*1.8*10.0 + l3*l4*m3*t13*t17*t19*t25*t106*9.0 + l3*l4*m1*t13*t17*t19*t28*t106*6.0 + l3*l4*m4*t13*t17*t19*t25*t106*(4.5*10.0 / 4.0) + l3*l4*m2*t13*t17*t19*t28*t106*9.0 - l2*l4*m4*t13*t17*t21*t25*t108*(2.7*10.0 / 2.0) - l2*l4*m4*t13*t17*t21*t26*t108*(2.7*10.0 / 4.0) - l1*l2*m1*t13*t21*t23*t28*t109*3.0 - l1*l2*m1*t14*t21*t23*t28*t109*3.0 - l1*l2*m2*t13*t21*t23*t28*t109*(2.7*10.0 / 4.0) - l1*l2*m4*t13*t21*t23*t26*t109*(9.0 / 2.0) - l1*l2*m1*t15*t21*t23*t28*t109*3.0 - l1*l2*m2*t14*t21*t23*t28*t109*(2.7*10.0 / 4.0) - l1*l2*m3*t13*t21*t23*t28*t109*(9.0 / 4.0) - l1*l2*m4*t14*t21*t23*t26*t109*(9.0 / 2.0) - l1*l2*m1*t16*t21*t23*t28*t109*3.0 - l1*l2*m2*t15*t21*t23*t28*t109*(2.7*10.0 / 4.0) - l1*l2*m3*t14*t21*t23*t28*t109*(9.0 / 4.0) - l1*l2*m4*t15*t21*t23*t26*t109*(9.0 / 2.0) - l1*l2*m2*t16*t21*t23*t28*t109*(2.7*10.0 / 4.0) - l1*l2*m3*t15*t21*t23*t28*t109*(9.0 / 4.0) - l1*l2*m4*t16*t21*t23*t26*t109*(9.0 / 2.0) - l1*l2*m3*t16*t21*t23*t28*t109*(9.0 / 4.0) - l3*l4*m4*t13*t17*t19*t25*t122*(9.0 / 4.0) + l1*l2*m2*t13*t21*t23*t28*t123*(9.0 / 4.0) - l1*l2*m4*t13*t21*t23*t26*t123*(3.0 / 2.0) + l1*l2*m2*t14*t21*t23*t28*t123*(9.0 / 4.0) + l1*l2*m3*t13*t21*t23*t28*t123*(3.0 / 4.0) - l1*l2*m4*t14*t21*t23*t26*t123*(3.0 / 2.0) + l1*l2*m2*t15*t21*t23*t28*t123*(9.0 / 4.0) + l1*l2*m3*t14*t21*t23*t28*t123*(3.0 / 4.0) - l1*l2*m4*t15*t21*t23*t26*t123*(3.0 / 2.0) + l1*l2*m2*t16*t21*t23*t28*t123*(9.0 / 4.0) + l1*l2*m3*t15*t21*t23*t28*t123*(3.0 / 4.0) - l1*l2*m4*t16*t21*t23*t26*t123*(3.0 / 2.0) + l1*l2*m3*t16*t21*t23*t28*t123*(3.0 / 4.0) + l1*l4*m2*t13*t19*t21*t26*t133*3.0 + l1*l4*m2*t14*t19*t21*t26*t133*3.0 + l1*l4*m2*t13*t19*t21*t28*t133*3.0 + l1*l4*m2*t15*t19*t21*t26*t133*3.0 - l1*l4*m4*t13*t19*t21*t25*t134*(2.7*10.0 / 4.0) + l1*l4*m2*t14*t19*t21*t28*t133*3.0 - l1*l4*m4*t13*t19*t21*t26*t134*(2.7*10.0 / 4.0) - l1*l4*m4*t14*t19*t21*t25*t134*(2.7*10.0 / 4.0) + l1*l4*m2*t15*t19*t21*t28*t133*3.0 - l1*l4*m4*t14*t19*t21*t26*t134*(2.7*10.0 / 4.0) - l1*l4*m4*t15*t19*t21*t25*t134*(2.7*10.0 / 8.0) - l1*l4*m4*t15*t19*t21*t26*t134*(2.7*10.0 / 8.0) + l1*l3*m4*t13*t19*t23*t25*t171*(9.0 / 2.0) + l1*l3*m2*t13*t19*t23*t28*t171*(9.0 / 4.0) + l1*l3*m4*t13*t19*t23*t26*t171*9.0 + l1*l3*m4*t14*t19*t23*t25*t171*(9.0 / 2.0) + l1*l3*m2*t14*t19*t23*t28*t171*(9.0 / 4.0) + l1*l3*m3*t13*t19*t23*t28*t171*(9.0 / 4.0) + l1*l3*m4*t14*t19*t23*t26*t171*9.0 + l1*l3*m4*t15*t19*t23*t25*t171*(9.0 / 2.0) + l1*l4*m4*t15*t19*t21*t25*t172*(2.7*10.0 / 8.0) + l1*l3*m2*t15*t19*t23*t28*t171*(9.0 / 4.0) + l1*l3*m3*t14*t19*t23*t28*t171*(9.0 / 4.0) + l1*l3*m4*t15*t19*t23*t26*t171*9.0 + l1*l3*m4*t16*t19*t23*t25*t171*(9.0 / 2.0) + l1*l4*m4*t15*t19*t21*t26*t172*(2.7*10.0 / 8.0) + l1*l3*m2*t16*t19*t23*t28*t171*(9.0 / 4.0) + l1*l3*m3*t15*t19*t23*t28*t171*(9.0 / 4.0) + l1*l3*m4*t16*t19*t23*t26*t171*9.0 + l1*l3*m3*t16*t19*t23*t28*t171*(9.0 / 4.0) + l1*l3*m2*t13*t19*t23*t28*t188*(3.0 / 2.0) + l1*l3*m2*t14*t19*t23*t28*t188*(3.0 / 2.0) + l1*l3*m2*t15*t19*t23*t28*t188*(3.0 / 2.0) + l1*l3*m2*t16*t19*t23*t28*t188*(3.0 / 2.0) + l3*l4*m4*t13*t17*t19*t25*t198*(2.7*10.0 / 4.0) + l1*l2*m2*t13*t21*t23*t28*t199*(9.0 / 4.0) + l1*l2*m4*t13*t21*t23*t26*t199*(9.0 / 2.0) + l1*l2*m2*t14*t21*t23*t28*t199*(9.0 / 4.0) + l1*l2*m3*t13*t21*t23*t28*t199*(9.0 / 4.0) + l1*l2*m4*t14*t21*t23*t26*t199*(9.0 / 2.0) + l1*l2*m2*t15*t21*t23*t28*t199*(9.0 / 4.0) + l1*l2*m3*t14*t21*t23*t28*t199*(9.0 / 4.0) + l1*l2*m4*t15*t21*t23*t26*t199*(9.0 / 2.0) + l1*l2*m2*t16*t21*t23*t28*t199*(9.0 / 4.0) + l1*l2*m3*t15*t21*t23*t28*t199*(9.0 / 4.0) + l1*l2*m4*t16*t21*t23*t26*t199*(9.0 / 2.0) + l1*l2*m3*t16*t21*t23*t28*t199*(9.0 / 4.0) - dth1*dth2*l1*l3*l4*m1*t8*t20*t26*4.8*10.0 - dth1*dth2*l1*l2*l4*m1*t8*t22*t26*1.6*10.0 - dth1*dth2*l1*l3*l4*m2*t8*t20*t26*6.0*10.0 - dth1*dth2*l1*l3*l4*m3*t8*t20*t25*1.8*10.0 - dth1*dth2*l1*l2*l4*m2*t8*t22*t26*3.6*10.0 - dth1*dth2*l1*l3*l4*m1*t8*t20*t28*2.4*10.0 - dth1*dth2*l1*l3*l4*m4*t8*t20*t25*(4.5*10.0 / 2.0) - dth1*dth3*l1*l2*l4*m1*t8*t22*t26*1.6*10.0 - dth1*dth2*l1*l2*l4*m1*t8*t22*t28*2.4*10.0 - dth1*dth2*l1*l3*l4*m2*t8*t20*t28*3.0*10.0 - dth1*dth3*l1*l2*l4*m2*t8*t22*t26*3.6*10.0 - dth2*dth3*l1*l2*l4*m1*t8*t22*t26*1.6*10.0 - dth1*dth2*l1*l2*l4*m2*t8*t22*t28*5.4*10.0 - dth1*dth2*l1*l2*l4*m4*t8*t22*t26*(4.5*10.0 / 2.0) - dth1*dth3*l1*l2*l4*m1*t8*t22*t28*2.4*10.0 - dth2*dth3*l1*l2*l4*m2*t8*t22*t26*3.6*10.0 - dth1*dth2*l1*l2*l4*m3*t8*t22*t28*1.2*10.0 - dth1*dth3*l1*l2*l4*m2*t8*t22*t28*5.4*10.0 - dth1*dth3*l1*l2*l4*m4*t8*t22*t26*(4.5*10.0 / 2.0) - dth2*dth3*l1*l2*l4*m1*t8*t22*t28*2.4*10.0 - dth1*dth3*l1*l2*l4*m3*t8*t22*t28*1.2*10.0 - dth2*dth3*l1*l2*l4*m2*t8*t22*t28*5.4*10.0 - dth2*dth3*l1*l2*l4*m4*t8*t22*t26*(4.5*10.0 / 2.0) - dth2*dth3*l1*l2*l4*m3*t8*t22*t28*1.2*10.0 + dth1*dth2*l1*l3*l4*m2*t20*t26*t82*1.2*10.0 + dth1*dth2*l1*l3*l4*m3*t20*t25*t82*6.0 + dth1*dth2*l1*l2*l4*m2*t22*t26*t82*1.2*10.0 + dth1*dth2*l1*l3*l4*m4*t20*t25*t82*(1.5*10.0 / 2.0) + dth1*dth2*l1*l3*l4*m2*t20*t28*t82*6.0 + dth1*dth3*l1*l2*l4*m2*t22*t26*t82*1.2*10.0 + dth1*dth2*l1*l2*l4*m2*t22*t28*t82*1.8*10.0 + dth1*dth2*l1*l2*l4*m4*t22*t26*t82*(4.5*10.0 / 2.0) + dth2*dth3*l1*l2*l4*m2*t22*t26*t82*1.2*10.0 + dth1*dth2*l1*l2*l4*m3*t22*t28*t82*1.2*10.0 + dth1*dth2*l1*l3*l4*m4*t20*t25*t85*(2.7*10.0 / 2.0) + dth1*dth3*l1*l2*l4*m2*t22*t28*t82*1.8*10.0 + dth1*dth3*l1*l2*l4*m4*t22*t26*t82*(4.5*10.0 / 2.0) + dth1*dth3*l1*l2*l4*m3*t22*t28*t82*1.2*10.0 + dth2*dth3*l1*l2*l4*m2*t22*t28*t82*1.8*10.0 + dth2*dth3*l1*l2*l4*m4*t22*t26*t82*(4.5*10.0 / 2.0) + dth1*dth2*l1*l2*l4*m4*t22*t26*t85*(9.0 / 2.0) + dth2*dth3*l1*l2*l4*m3*t22*t28*t82*1.2*10.0 + dth1*dth3*l1*l2*l4*m4*t22*t26*t85*(9.0 / 2.0) + dth2*dth3*l1*l2*l4*m4*t22*t26*t85*(9.0 / 2.0) - dth1*dth2*l1*l3*l4*m4*t20*t25*t187*(9.0 / 2.0) - dth1*dth2*l1*l2*l4*m4*t22*t26*t187*(9.0 / 2.0) - dth1*dth3*l1*l2*l4*m4*t22*t26*t187*(9.0 / 2.0) - dth2*dth3*l1*l2*l4*m4*t22*t26*t187*(9.0 / 2.0) + g*l1*l2*l4*m1*m2*m3*t21*t41*2.0 + g*l1*l2*l4*m1*m2*m4*t21*t41*(1.5*10.0 / 4.0) + g*l1*l2*l4*m1*m3*m4*t21*t41*(2.5*10.0 / 4.0) + g*l1*l2*l4*m2*m3*m4*t21*t41*(2.25*100.0 / 4.0) - (g*l1*l3*l4*m1*m2*m3*t19*t60) / 2.0 - g*l1*l3*l4*m1*m2*m4*t19*t60*(5.0 / 8.0) - g*l1*l3*l4*m1*m3*m4*t19*t60*(4.5*10.0 / 8.0) - g*l1*l3*l4*m2*m3*m4*t19*t60*(4.5*10.0 / 8.0) - g*l1*l2*l4*m1*m2*m3*t21*t104*6.0 - g*l1*l2*l4*m1*m2*m4*t21*t104*(4.5*10.0 / 4.0) - g*l1*l2*l4*m1*m3*m4*t21*t104*(7.5*10.0 / 4.0) - g*l1*l2*l4*m2*m3*m4*t21*t104*(2.25*100.0 / 4.0) - g*l1*l2*l4*m1*m2*m4*t21*t121*(9.0 / 8.0) - g*l1*l2*l4*m1*m3*m4*t21*t120*(1.5*10.0 / 4.0) - g*l1*l2*l4*m1*m3*m4*t21*t121*(9.0 / 8.0) - g*l1*l2*l4*m2*m3*m4*t21*t120*(4.5*10.0 / 4.0) - g*l1*l2*l4*m2*m3*m4*t21*t121*(8.1*10.0 / 8.0) + g*l1*l3*l4*m1*m2*m4*t19*t127*(3.0 / 8.0) + g*l1*l3*l4*m1*m3*m4*t19*t127*(9.0 / 8.0) + g*l1*l3*l4*m2*m3*m4*t19*t127*(9.0 / 8.0) + g*l1*l3*l4*m1*m2*m3*t19*t145*(3.0 / 2.0) - g*l1*l3*l4*m1*m2*m3*t19*t146*(9.0 / 2.0) + g*l1*l3*l4*m1*m2*m4*t19*t145*(1.5*10.0 / 8.0) - g*l1*l3*l4*m1*m2*m3*t19*t147*(3.0 / 2.0) - g*l1*l3*l4*m1*m2*m4*t19*t146*(4.5*10.0 / 8.0) + g*l1*l3*l4*m1*m3*m4*t19*t145*(4.5*10.0 / 8.0) - g*l1*l3*l4*m1*m2*m4*t19*t147*(1.5*10.0 / 8.0) - g*l1*l3*l4*m1*m3*m4*t19*t146*(1.35*100.0 / 8.0) + g*l1*l3*l4*m2*m3*m4*t19*t145*(1.35*100.0 / 8.0) - g*l1*l3*l4*m1*m3*m4*t19*t147*(1.35*100.0 / 8.0) - g*l1*l3*l4*m2*m3*m4*t19*t146*(1.35*100.0 / 8.0) - g*l1*l2*l4*m1*m2*m4*t21*t148*(9.0 / 8.0) - g*l1*l3*l4*m2*m3*m4*t19*t147*(4.5*10.0 / 8.0) - g*l1*l2*l4*m1*m3*m4*t21*t148*(9.0 / 8.0) - g*l1*l2*l4*m2*m3*m4*t21*t148*(8.1*10.0 / 8.0) + g*l1*l2*l4*m1*m3*m4*t21*t190*(3.0 / 4.0) + g*l1*l2*l4*m2*m3*m4*t21*t190*(9.0 / 4.0) + g*l1*l2*l4*m1*m2*m4*t21*t195*(2.7*10.0 / 8.0) - g*l1*l2*l4*m1*m3*m4*t21*t194*(4.5*10.0 / 4.0) - g*l1*l2*l4*m1*m2*m4*t21*t196*(2.7*10.0 / 8.0) + g*l1*l2*l4*m1*m3*m4*t21*t195*(2.7*10.0 / 8.0) - g*l1*l2*l4*m2*m3*m4*t21*t194*(4.5*10.0 / 4.0) - g*l1*l2*l4*m1*m3*m4*t21*t196*(2.7*10.0 / 8.0) + g*l1*l2*l4*m2*m3*m4*t21*t195*(8.1*10.0 / 8.0) - g*l1*l2*l4*m2*m3*m4*t21*t196*(8.1*10.0 / 8.0) + g*l1*l3*l4*m1*m2*m4*t19*t205*(2.7*10.0 / 8.0) + g*l1*l3*l4*m1*m2*m4*t19*t206*(9.0 / 8.0) + g*l1*l3*l4*m1*m3*m4*t19*t205*(2.7*10.0 / 8.0) + g*l1*l3*l4*m1*m3*m4*t19*t206*(2.7*10.0 / 8.0) + g*l1*l3*l4*m2*m3*m4*t19*t205*(2.7*10.0 / 8.0) + g*l1*l3*l4*m2*m3*m4*t19*t206*(9.0 / 8.0) - g*l1*l3*l4*m1*m2*m4*t19*t221*(9.0 / 8.0) - g*l1*l3*l4*m1*m3*m4*t19*t221*(9.0 / 8.0) - g*l1*l3*l4*m2*m3*m4*t19*t221*(2.7*10.0 / 8.0) + g*l1*l2*l4*m1*m3*m4*t21*t240*(9.0 / 4.0) + g*l1*l2*l4*m2*m3*m4*t21*t240*(9.0 / 4.0) + dth1*dth2*l1*l3*m1*t9*t19*t23*t28*6.0 + dth1*dth2*l1*l3*m4*t9*t19*t23*t25*3.0*10.0 + dth1*dth2*l1*l3*m2*t9*t19*t23*t28*1.5*10.0 + dth1*dth2*l1*l3*m4*t9*t19*t23*t26*3.6*10.0 + dth1*dth3*l1*l3*m1*t9*t19*t23*t28*6.0 + dth1*dth3*l1*l3*m4*t9*t19*t23*t25*3.0*10.0 + dth1*dth2*l1*l3*m3*t9*t19*t23*t28*9.0 + dth1*dth3*l1*l3*m2*t9*t19*t23*t28*1.5*10.0 + dth1*dth3*l1*l3*m4*t9*t19*t23*t26*3.6*10.0 + dth1*dth4*l1*l3*m1*t9*t19*t23*t28*6.0 + dth1*dth4*l1*l3*m4*t9*t19*t23*t25*3.0*10.0 + dth2*dth3*l1*l3*m1*t9*t19*t23*t28*6.0 + dth2*dth3*l1*l3*m4*t9*t19*t23*t25*3.0*10.0 + dth1*dth3*l1*l3*m3*t9*t19*t23*t28*9.0 + dth1*dth4*l1*l3*m2*t9*t19*t23*t28*1.5*10.0 + dth1*dth4*l1*l3*m4*t9*t19*t23*t26*3.6*10.0 + dth2*dth3*l1*l3*m2*t9*t19*t23*t28*1.5*10.0 + dth2*dth3*l1*l3*m4*t9*t19*t23*t26*3.6*10.0 + dth2*dth4*l1*l3*m1*t9*t19*t23*t28*6.0 + dth2*dth4*l1*l3*m4*t9*t19*t23*t25*3.0*10.0 + dth1*dth4*l1*l3*m3*t9*t19*t23*t28*9.0 + dth2*dth3*l1*l3*m3*t9*t19*t23*t28*9.0 + dth2*dth4*l1*l3*m2*t9*t19*t23*t28*1.5*10.0 + dth2*dth4*l1*l3*m4*t9*t19*t23*t26*3.6*10.0 + dth3*dth4*l1*l3*m1*t9*t19*t23*t28*6.0 + dth3*dth4*l1*l3*m4*t9*t19*t23*t25*3.0*10.0 + dth2*dth4*l1*l3*m3*t9*t19*t23*t28*9.0 + dth3*dth4*l1*l3*m2*t9*t19*t23*t28*1.5*10.0 + dth3*dth4*l1*l3*m4*t9*t19*t23*t26*3.6*10.0 + dth3*dth4*l1*l3*m3*t9*t19*t23*t28*9.0 - dth1*dth2*l1*l4*m1*t19*t21*t26*t37*2.4*10.0 + dth1*dth2*l1*l4*m2*t19*t21*t26*t36*3.0*10.0 + dth1*dth2*l1*l4*m3*t19*t21*t25*t36*1.2*10.0 - dth1*dth2*l1*l4*m2*t19*t21*t26*t37*3.6*10.0 + dth1*dth2*l1*l4*m4*t19*t21*t25*t36*(4.5*10.0 / 2.0) - dth1*dth3*l1*l4*m1*t19*t21*t26*t37*1.2*10.0 - dth1*dth2*l1*l4*m1*t19*t21*t28*t37*2.4*10.0 + dth1*dth2*l1*l4*m2*t19*t21*t28*t36*1.8*10.0 + dth1*dth2*l1*l4*m4*t19*t21*t26*t36*(7.5*10.0 / 2.0) - dth1*dth3*l1*l4*m2*t19*t21*t26*t37*1.8*10.0 - dth2*dth3*l1*l4*m1*t19*t21*t26*t37*1.2*10.0 - dth1*dth2*l1*l4*m2*t19*t21*t28*t37*3.6*10.0 + dth1*dth2*l1*l4*m3*t19*t21*t28*t36*1.2*10.0 + dth1*dth2*l1*l4*m4*t19*t21*t25*t38*(4.5*10.0 / 2.0) - dth1*dth3*l1*l4*m1*t19*t21*t28*t37*1.2*10.0 - dth2*dth3*l1*l4*m2*t19*t21*t26*t37*1.8*10.0 + dth1*dth2*l1*l4*m4*t19*t21*t26*t38*(2.7*10.0 / 2.0) - dth1*dth3*l1*l4*m2*t19*t21*t28*t37*1.8*10.0 + dth1*dth3*l1*l4*m4*t19*t21*t25*t38*(4.5*10.0 / 2.0) - dth2*dth3*l1*l4*m1*t19*t21*t28*t37*1.2*10.0 + dth1*dth3*l1*l4*m4*t19*t21*t26*t38*(2.7*10.0 / 2.0) - dth2*dth3*l1*l4*m2*t19*t21*t28*t37*1.8*10.0 + dth2*dth3*l1*l4*m4*t19*t21*t25*t38*(4.5*10.0 / 2.0) + dth2*dth3*l1*l4*m4*t19*t21*t26*t38*(2.7*10.0 / 2.0) - dth1*dth2*l1*l2*m1*t21*t23*t28*t43*6.0 - dth1*dth2*l1*l2*m2*t21*t23*t28*t43*(2.7*10.0 / 2.0) + dth1*dth2*l1*l2*m4*t21*t23*t26*t43*3.0 - dth1*dth3*l1*l2*m1*t21*t23*t28*t43*6.0 - dth1*dth2*l1*l2*m3*t21*t23*t28*t43*(3.0 / 2.0) - dth1*dth3*l1*l2*m2*t21*t23*t28*t43*(2.7*10.0 / 2.0) + dth1*dth3*l1*l2*m4*t21*t23*t26*t43*3.0 - dth1*dth4*l1*l2*m1*t21*t23*t28*t43*6.0 - dth2*dth3*l1*l2*m1*t21*t23*t28*t43*6.0 - dth1*dth3*l1*l2*m3*t21*t23*t28*t43*(3.0 / 2.0) - dth1*dth4*l1*l2*m2*t21*t23*t28*t43*(2.7*10.0 / 2.0) + dth1*dth4*l1*l2*m4*t21*t23*t26*t43*3.0 - dth2*dth3*l1*l2*m2*t21*t23*t28*t43*(2.7*10.0 / 2.0) + dth2*dth3*l1*l2*m4*t21*t23*t26*t43*3.0 - dth2*dth4*l1*l2*m1*t21*t23*t28*t43*6.0 - dth1*dth4*l1*l2*m3*t21*t23*t28*t43*(3.0 / 2.0) - dth2*dth3*l1*l2*m3*t21*t23*t28*t43*(3.0 / 2.0) - dth2*dth4*l1*l2*m2*t21*t23*t28*t43*(2.7*10.0 / 2.0) + dth2*dth4*l1*l2*m4*t21*t23*t26*t43*3.0 - dth3*dth4*l1*l2*m1*t21*t23*t28*t43*6.0 - dth2*dth4*l1*l2*m3*t21*t23*t28*t43*(3.0 / 2.0) - dth3*dth4*l1*l2*m2*t21*t23*t28*t43*(2.7*10.0 / 2.0) + dth3*dth4*l1*l2*m4*t21*t23*t26*t43*3.0 - dth3*dth4*l1*l2*m3*t21*t23*t28*t43*(3.0 / 2.0) - dth1*dth2*l1*l3*m4*t19*t23*t25*t84*9.0 - dth1*dth2*l1*l3*m2*t19*t23*t28*t84*(9.0 / 2.0) - dth1*dth2*l1*l3*m4*t19*t23*t26*t84*1.8*10.0 - dth1*dth3*l1*l3*m4*t19*t23*t25*t84*9.0 - dth1*dth2*l1*l3*m1*t19*t23*t28*t86*6.0 - dth1*dth2*l1*l3*m3*t19*t23*t28*t84*(9.0 / 2.0) - dth1*dth3*l1*l3*m2*t19*t23*t28*t84*(9.0 / 2.0) - dth1*dth3*l1*l3*m4*t19*t23*t26*t84*1.8*10.0 - dth1*dth4*l1*l3*m4*t19*t23*t25*t84*9.0 - dth2*dth3*l1*l3*m4*t19*t23*t25*t84*9.0 - dth1*dth2*l1*l3*m2*t19*t23*t28*t86*9.0 - dth1*dth3*l1*l3*m1*t19*t23*t28*t86*6.0 - dth1*dth3*l1*l3*m3*t19*t23*t28*t84*(9.0 / 2.0) - dth1*dth4*l1*l3*m2*t19*t23*t28*t84*(9.0 / 2.0) - dth1*dth4*l1*l3*m4*t19*t23*t26*t84*1.8*10.0 - dth2*dth3*l1*l3*m2*t19*t23*t28*t84*(9.0 / 2.0) - dth2*dth3*l1*l3*m4*t19*t23*t26*t84*1.8*10.0 - dth2*dth4*l1*l3*m4*t19*t23*t25*t84*9.0 - dth1*dth3*l1*l3*m2*t19*t23*t28*t86*9.0 - dth1*dth4*l1*l3*m1*t19*t23*t28*t86*6.0 - dth1*dth4*l1*l3*m3*t19*t23*t28*t84*(9.0 / 2.0) - dth2*dth3*l1*l3*m1*t19*t23*t28*t86*6.0 - dth2*dth3*l1*l3*m3*t19*t23*t28*t84*(9.0 / 2.0) - dth2*dth4*l1*l3*m2*t19*t23*t28*t84*(9.0 / 2.0) - dth2*dth4*l1*l3*m4*t19*t23*t26*t84*1.8*10.0 - dth3*dth4*l1*l3*m4*t19*t23*t25*t84*9.0 - dth1*dth4*l1*l3*m2*t19*t23*t28*t86*9.0 - dth2*dth3*l1*l3*m2*t19*t23*t28*t86*9.0 - dth2*dth4*l1*l3*m1*t19*t23*t28*t86*6.0 - dth2*dth4*l1*l3*m3*t19*t23*t28*t84*(9.0 / 2.0) - dth3*dth4*l1*l3*m2*t19*t23*t28*t84*(9.0 / 2.0) - dth3*dth4*l1*l3*m4*t19*t23*t26*t84*1.8*10.0 - dth2*dth4*l1*l3*m2*t19*t23*t28*t86*9.0 - dth3*dth4*l1*l3*m1*t19*t23*t28*t86*6.0 - dth3*dth4*l1*l3*m3*t19*t23*t28*t84*(9.0 / 2.0) - dth3*dth4*l1*l3*m2*t19*t23*t28*t86*9.0 - dth1*dth2*l1*l2*m1*t21*t23*t28*t109*6.0 - dth1*dth2*l1*l2*m2*t21*t23*t28*t109*(2.7*10.0 / 2.0) - dth1*dth2*l1*l2*m4*t21*t23*t26*t109*9.0 - dth1*dth3*l1*l2*m1*t21*t23*t28*t109*6.0 - dth1*dth2*l1*l2*m3*t21*t23*t28*t109*(9.0 / 2.0) - dth1*dth3*l1*l2*m2*t21*t23*t28*t109*(2.7*10.0 / 2.0) - dth1*dth3*l1*l2*m4*t21*t23*t26*t109*9.0 - dth1*dth4*l1*l2*m1*t21*t23*t28*t109*6.0 - dth2*dth3*l1*l2*m1*t21*t23*t28*t109*6.0 - dth1*dth3*l1*l2*m3*t21*t23*t28*t109*(9.0 / 2.0) - dth1*dth4*l1*l2*m2*t21*t23*t28*t109*(2.7*10.0 / 2.0) - dth1*dth4*l1*l2*m4*t21*t23*t26*t109*9.0 - dth2*dth3*l1*l2*m2*t21*t23*t28*t109*(2.7*10.0 / 2.0) - dth2*dth3*l1*l2*m4*t21*t23*t26*t109*9.0 - dth2*dth4*l1*l2*m1*t21*t23*t28*t109*6.0 - dth1*dth4*l1*l2*m3*t21*t23*t28*t109*(9.0 / 2.0) - dth2*dth3*l1*l2*m3*t21*t23*t28*t109*(9.0 / 2.0) - dth2*dth4*l1*l2*m2*t21*t23*t28*t109*(2.7*10.0 / 2.0) - dth2*dth4*l1*l2*m4*t21*t23*t26*t109*9.0 - dth3*dth4*l1*l2*m1*t21*t23*t28*t109*6.0 - dth2*dth4*l1*l2*m3*t21*t23*t28*t109*(9.0 / 2.0) - dth3*dth4*l1*l2*m2*t21*t23*t28*t109*(2.7*10.0 / 2.0) - dth3*dth4*l1*l2*m4*t21*t23*t26*t109*9.0 - dth3*dth4*l1*l2*m3*t21*t23*t28*t109*(9.0 / 2.0) + dth1*dth2*l1*l2*m2*t21*t23*t28*t123*(9.0 / 2.0) - dth1*dth2*l1*l2*m4*t21*t23*t26*t123*3.0 + dth1*dth2*l1*l2*m3*t21*t23*t28*t123*(3.0 / 2.0) + dth1*dth3*l1*l2*m2*t21*t23*t28*t123*(9.0 / 2.0) - dth1*dth3*l1*l2*m4*t21*t23*t26*t123*3.0 + dth1*dth3*l1*l2*m3*t21*t23*t28*t123*(3.0 / 2.0) + dth1*dth4*l1*l2*m2*t21*t23*t28*t123*(9.0 / 2.0) - dth1*dth4*l1*l2*m4*t21*t23*t26*t123*3.0 + dth2*dth3*l1*l2*m2*t21*t23*t28*t123*(9.0 / 2.0) - dth2*dth3*l1*l2*m4*t21*t23*t26*t123*3.0 + dth1*dth4*l1*l2*m3*t21*t23*t28*t123*(3.0 / 2.0) + dth2*dth3*l1*l2*m3*t21*t23*t28*t123*(3.0 / 2.0) + dth2*dth4*l1*l2*m2*t21*t23*t28*t123*(9.0 / 2.0) - dth2*dth4*l1*l2*m4*t21*t23*t26*t123*3.0 + dth2*dth4*l1*l2*m3*t21*t23*t28*t123*(3.0 / 2.0) + dth3*dth4*l1*l2*m2*t21*t23*t28*t123*(9.0 / 2.0) - dth3*dth4*l1*l2*m4*t21*t23*t26*t123*3.0 + dth3*dth4*l1*l2*m3*t21*t23*t28*t123*(3.0 / 2.0) + dth1*dth2*l1*l4*m2*t19*t21*t26*t133*6.0 + dth1*dth3*l1*l4*m2*t19*t21*t26*t133*6.0 + dth1*dth2*l1*l4*m2*t19*t21*t28*t133*6.0 - dth1*dth2*l1*l4*m4*t19*t21*t25*t134*(2.7*10.0 / 2.0) + dth2*dth3*l1*l4*m2*t19*t21*t26*t133*6.0 - dth1*dth2*l1*l4*m4*t19*t21*t26*t134*(2.7*10.0 / 2.0) + dth1*dth3*l1*l4*m2*t19*t21*t28*t133*6.0 - dth1*dth3*l1*l4*m4*t19*t21*t25*t134*(2.7*10.0 / 4.0) - dth1*dth3*l1*l4*m4*t19*t21*t26*t134*(2.7*10.0 / 4.0) + dth2*dth3*l1*l4*m2*t19*t21*t28*t133*6.0 - dth2*dth3*l1*l4*m4*t19*t21*t25*t134*(2.7*10.0 / 4.0) - dth2*dth3*l1*l4*m4*t19*t21*t26*t134*(2.7*10.0 / 4.0) + dth1*dth2*l1*l3*m4*t19*t23*t25*t171*9.0 + dth1*dth2*l1*l3*m2*t19*t23*t28*t171*(9.0 / 2.0) + dth1*dth2*l1*l3*m4*t19*t23*t26*t171*1.8*10.0 + dth1*dth3*l1*l3*m4*t19*t23*t25*t171*9.0 + dth1*dth3*l1*l4*m4*t19*t21*t25*t172*(2.7*10.0 / 4.0) + dth1*dth2*l1*l3*m3*t19*t23*t28*t171*(9.0 / 2.0) + dth1*dth3*l1*l3*m2*t19*t23*t28*t171*(9.0 / 2.0) + dth1*dth3*l1*l3*m4*t19*t23*t26*t171*1.8*10.0 + dth1*dth3*l1*l4*m4*t19*t21*t26*t172*(2.7*10.0 / 4.0) + dth1*dth4*l1*l3*m4*t19*t23*t25*t171*9.0 + dth2*dth3*l1*l3*m4*t19*t23*t25*t171*9.0 + dth2*dth3*l1*l4*m4*t19*t21*t25*t172*(2.7*10.0 / 4.0) + dth1*dth3*l1*l3*m3*t19*t23*t28*t171*(9.0 / 2.0) + dth1*dth4*l1*l3*m2*t19*t23*t28*t171*(9.0 / 2.0) + dth1*dth4*l1*l3*m4*t19*t23*t26*t171*1.8*10.0 + dth2*dth3*l1*l3*m2*t19*t23*t28*t171*(9.0 / 2.0) + dth2*dth3*l1*l3*m4*t19*t23*t26*t171*1.8*10.0 + dth2*dth3*l1*l4*m4*t19*t21*t26*t172*(2.7*10.0 / 4.0) + dth2*dth4*l1*l3*m4*t19*t23*t25*t171*9.0 + dth1*dth4*l1*l3*m3*t19*t23*t28*t171*(9.0 / 2.0) + dth2*dth3*l1*l3*m3*t19*t23*t28*t171*(9.0 / 2.0) + dth2*dth4*l1*l3*m2*t19*t23*t28*t171*(9.0 / 2.0) + dth2*dth4*l1*l3*m4*t19*t23*t26*t171*1.8*10.0 + dth3*dth4*l1*l3*m4*t19*t23*t25*t171*9.0 + dth2*dth4*l1*l3*m3*t19*t23*t28*t171*(9.0 / 2.0) + dth3*dth4*l1*l3*m2*t19*t23*t28*t171*(9.0 / 2.0) + dth3*dth4*l1*l3*m4*t19*t23*t26*t171*1.8*10.0 + dth3*dth4*l1*l3*m3*t19*t23*t28*t171*(9.0 / 2.0) + dth1*dth2*l1*l3*m2*t19*t23*t28*t188*3.0 + dth1*dth3*l1*l3*m2*t19*t23*t28*t188*3.0 + dth1*dth4*l1*l3*m2*t19*t23*t28*t188*3.0 + dth2*dth3*l1*l3*m2*t19*t23*t28*t188*3.0 + dth2*dth4*l1*l3*m2*t19*t23*t28*t188*3.0 + dth3*dth4*l1*l3*m2*t19*t23*t28*t188*3.0 + dth1*dth2*l1*l2*m2*t21*t23*t28*t199*(9.0 / 2.0) + dth1*dth2*l1*l2*m4*t21*t23*t26*t199*9.0 + dth1*dth2*l1*l2*m3*t21*t23*t28*t199*(9.0 / 2.0) + dth1*dth3*l1*l2*m2*t21*t23*t28*t199*(9.0 / 2.0) + dth1*dth3*l1*l2*m4*t21*t23*t26*t199*9.0 + dth1*dth3*l1*l2*m3*t21*t23*t28*t199*(9.0 / 2.0) + dth1*dth4*l1*l2*m2*t21*t23*t28*t199*(9.0 / 2.0) + dth1*dth4*l1*l2*m4*t21*t23*t26*t199*9.0 + dth2*dth3*l1*l2*m2*t21*t23*t28*t199*(9.0 / 2.0) + dth2*dth3*l1*l2*m4*t21*t23*t26*t199*9.0 + dth1*dth4*l1*l2*m3*t21*t23*t28*t199*(9.0 / 2.0) + dth2*dth3*l1*l2*m3*t21*t23*t28*t199*(9.0 / 2.0) + dth2*dth4*l1*l2*m2*t21*t23*t28*t199*(9.0 / 2.0) + dth2*dth4*l1*l2*m4*t21*t23*t26*t199*9.0 + dth2*dth4*l1*l2*m3*t21*t23*t28*t199*(9.0 / 2.0) + dth3*dth4*l1*l2*m2*t21*t23*t28*t199*(9.0 / 2.0) + dth3*dth4*l1*l2*m4*t21*t23*t26*t199*9.0 + dth3*dth4*l1*l2*m3*t21*t23*t28*t199*(9.0 / 2.0) - l1*l3*l4*m1*m2*m3*t8*t13*t20*8.0 - l1*l3*l4*m1*m2*m3*t8*t14*t20*8.0 - l1*l3*l4*m1*m2*m4*t8*t13*t20*1.0*10.0 - l1*l3*l4*m1*m2*m4*t8*t14*t20*1.0*10.0 - l1*l3*l4*m1*m3*m4*t8*t13*t20*4.5*10.0 - l1*l2*l4*m1*m3*m4*t8*t13*t22*2.5*10.0 - l1*l3*l4*m1*m3*m4*t8*t14*t20*4.5*10.0 - l1*l3*l4*m2*m3*m4*t8*t13*t20*(2.25*100.0 / 4.0) - l1*l2*l4*m1*m3*m4*t8*t14*t22*2.5*10.0 - l1*l2*l4*m2*m3*m4*t8*t13*t22*(2.25*100.0 / 4.0) - l1*l3*l4*m2*m3*m4*t8*t14*t20*(2.25*100.0 / 4.0) - l1*l2*l4*m1*m3*m4*t8*t15*t22*2.5*10.0 - l1*l2*l4*m2*m3*m4*t8*t14*t22*(2.25*100.0 / 4.0) - l1*l2*l4*m2*m3*m4*t8*t15*t22*(2.25*100.0 / 4.0) + l1*l3*l4*m2*m3*m4*t13*t20*t82*(4.5*10.0 / 4.0) + l1*l2*l4*m2*m3*m4*t13*t22*t82*(7.5*10.0 / 4.0) + l1*l3*l4*m1*m2*m4*t13*t20*t85*6.0 + l1*l3*l4*m2*m3*m4*t14*t20*t82*(4.5*10.0 / 4.0) + l1*l2*l4*m2*m3*m4*t14*t22*t82*(7.5*10.0 / 4.0) + l1*l3*l4*m1*m2*m4*t14*t20*t85*6.0 + l1*l3*l4*m1*m3*m4*t13*t20*t85*9.0 + l1*l2*l4*m1*m3*m4*t13*t22*t85*3.0 + l1*l2*l4*m2*m3*m4*t15*t22*t82*(7.5*10.0 / 4.0) + l1*l3*l4*m1*m3*m4*t14*t20*t85*9.0 + l1*l3*l4*m2*m3*m4*t13*t20*t85*(4.5*10.0 / 4.0) + l1*l2*l4*m1*m3*m4*t14*t22*t85*3.0 + l1*l2*l4*m2*m3*m4*t13*t22*t85*(2.7*10.0 / 4.0) + l1*l3*l4*m2*m3*m4*t14*t20*t85*(4.5*10.0 / 4.0) + l1*l2*l4*m1*m3*m4*t15*t22*t85*3.0 + l1*l2*l4*m2*m3*m4*t14*t22*t85*(2.7*10.0 / 4.0) + l1*l2*l4*m2*m3*m4*t15*t22*t85*(2.7*10.0 / 4.0) - l1*l3*l4*m2*m3*m4*t13*t20*t187*(9.0 / 4.0) - l1*l2*l4*m2*m3*m4*t13*t22*t187*(9.0 / 4.0) - l1*l3*l4*m2*m3*m4*t14*t20*t187*(9.0 / 4.0) - l1*l2*l4*m2*m3*m4*t14*t22*t187*(9.0 / 4.0) - l1*l2*l4*m2*m3*m4*t15*t22*t187*(9.0 / 4.0) + l2*l4*m1*m2*m3*t7*t13*t17*t21*8.0 + l2*l4*m1*m2*m4*t7*t13*t17*t21*1.5*10.0 + l2*l4*m1*m3*m4*t7*t13*t17*t21*2.5*10.0 + l2*l4*m2*m3*m4*t7*t13*t17*t21*(2.25*100.0 / 2.0) + l1*l3*m1*m2*m4*t9*t13*t19*t23*8.0 + l1*l3*m1*m2*m4*t9*t14*t19*t23*8.0 + l1*l3*m1*m3*m4*t9*t13*t19*t23*1.8*10.0 + l1*l3*m1*m2*m4*t9*t15*t19*t23*8.0 + l1*l3*m1*m3*m4*t9*t14*t19*t23*1.8*10.0 + l1*l3*m2*m3*m4*t9*t13*t19*t23*4.5*10.0 + l1*l3*m1*m2*m4*t9*t16*t19*t23*8.0 + l1*l3*m1*m3*m4*t9*t15*t19*t23*1.8*10.0 + l1*l3*m2*m3*m4*t9*t14*t19*t23*4.5*10.0 + l1*l3*m1*m3*m4*t9*t16*t19*t23*1.8*10.0 + l1*l3*m2*m3*m4*t9*t15*t19*t23*4.5*10.0 + l1*l3*m2*m3*m4*t9*t16*t19*t23*4.5*10.0 + l1*l4*m1*m2*m4*t13*t19*t21*t38*6.0 - l1*l4*m1*m3*m4*t13*t19*t21*t37*3.0*10.0 + l1*l4*m2*m3*m4*t13*t19*t21*t36*(7.5*10.0 / 2.0) + l1*l4*m1*m2*m4*t14*t19*t21*t38*6.0 + l1*l4*m1*m3*m4*t13*t19*t21*t38*9.0 - l1*l4*m1*m3*m4*t14*t19*t21*t37*3.0*10.0 - l1*l4*m2*m3*m4*t13*t19*t21*t37*4.5*10.0 + l1*l4*m2*m3*m4*t14*t19*t21*t36*(7.5*10.0 / 2.0) - l3*l4*m1*m2*m3*t13*t17*t19*t42*2.0 + l1*l4*m1*m2*m4*t15*t19*t21*t38*6.0 + l1*l4*m1*m3*m4*t14*t19*t21*t38*9.0 - l1*l4*m1*m3*m4*t15*t19*t21*t37*1.5*10.0 + l1*l4*m2*m3*m4*t13*t19*t21*t38*(4.5*10.0 / 2.0) - l1*l4*m2*m3*m4*t14*t19*t21*t37*4.5*10.0 - l3*l4*m1*m2*m4*t13*t17*t19*t42*(5.0 / 2.0) + l1*l4*m1*m3*m4*t15*t19*t21*t38*9.0 + l1*l4*m2*m3*m4*t14*t19*t21*t38*(4.5*10.0 / 2.0) - l1*l4*m2*m3*m4*t15*t19*t21*t37*(4.5*10.0 / 2.0) - l3*l4*m1*m3*m4*t13*t17*t19*t42*(4.5*10.0 / 2.0) + l1*l4*m2*m3*m4*t15*t19*t21*t38*(4.5*10.0 / 2.0) - l3*l4*m2*m3*m4*t13*t17*t19*t42*(4.5*10.0 / 4.0) - l1*l2*m1*m3*m4*t13*t21*t23*t43*2.0 - l1*l2*m1*m3*m4*t14*t21*t23*t43*2.0 - l1*l2*m2*m3*m4*t13*t21*t23*t43*(9.0 / 2.0) - l1*l2*m1*m3*m4*t15*t21*t23*t43*2.0 - l1*l2*m2*m3*m4*t14*t21*t23*t43*(9.0 / 2.0) - l1*l2*m1*m3*m4*t16*t21*t23*t43*2.0 - l1*l2*m2*m3*m4*t15*t21*t23*t43*(9.0 / 2.0) - l1*l2*m2*m3*m4*t16*t21*t23*t43*(9.0 / 2.0) - l2*l4*m1*m3*m4*t13*t17*t21*t81*1.5*10.0 - l2*l4*m1*m2*m4*t13*t17*t21*t83*(9.0 / 2.0) - l2*l4*m2*m3*m4*t13*t17*t21*t81*(4.5*10.0 / 2.0) - l2*l4*m1*m3*m4*t13*t17*t21*t83*(9.0 / 2.0) - l2*l4*m2*m3*m4*t13*t17*t21*t83*(8.1*10.0 / 4.0) - l1*l3*m2*m3*m4*t13*t19*t23*t84*(2.7*10.0 / 2.0) - l1*l3*m1*m3*m4*t13*t19*t23*t86*6.0 - l1*l3*m2*m3*m4*t14*t19*t23*t84*(2.7*10.0 / 2.0) - l1*l3*m1*m3*m4*t14*t19*t23*t86*6.0 - l1*l3*m2*m3*m4*t13*t19*t23*t86*9.0 - l1*l3*m2*m3*m4*t15*t19*t23*t84*(2.7*10.0 / 2.0) - l1*l3*m1*m3*m4*t15*t19*t23*t86*6.0 - l1*l3*m2*m3*m4*t14*t19*t23*t86*9.0 - l1*l3*m2*m3*m4*t16*t19*t23*t84*(2.7*10.0 / 2.0) - l1*l3*m1*m3*m4*t16*t19*t23*t86*6.0 - l1*l3*m2*m3*m4*t15*t19*t23*t86*9.0 - l1*l3*m2*m3*m4*t16*t19*t23*t86*9.0 + l3*l4*m1*m2*m3*t13*t17*t19*t106*6.0 + l3*l4*m1*m2*m4*t13*t17*t19*t106*(1.5*10.0 / 2.0) + l3*l4*m1*m3*m4*t13*t17*t19*t106*(4.5*10.0 / 2.0) + l3*l4*m2*m3*m4*t13*t17*t19*t106*(1.35*100.0 / 4.0) - l2*l4*m1*m2*m4*t13*t17*t21*t108*(9.0 / 2.0) - l2*l4*m1*m3*m4*t13*t17*t21*t108*(9.0 / 2.0) - l2*l4*m2*m3*m4*t13*t17*t21*t108*(8.1*10.0 / 4.0) - l1*l2*m1*m3*m4*t13*t21*t23*t109*6.0 - l1*l2*m1*m3*m4*t14*t21*t23*t109*6.0 - l1*l2*m2*m3*m4*t13*t21*t23*t109*(2.7*10.0 / 2.0) - l1*l2*m1*m3*m4*t15*t21*t23*t109*6.0 - l1*l2*m2*m3*m4*t14*t21*t23*t109*(2.7*10.0 / 2.0) - l1*l2*m1*m3*m4*t16*t21*t23*t109*6.0 - l1*l2*m2*m3*m4*t15*t21*t23*t109*(2.7*10.0 / 2.0) - l1*l2*m2*m3*m4*t16*t21*t23*t109*(2.7*10.0 / 2.0) + l3*l4*m1*m2*m4*t13*t17*t19*t122*(3.0 / 2.0) + l3*l4*m1*m3*m4*t13*t17*t19*t122*(9.0 / 2.0) + l3*l4*m2*m3*m4*t13*t17*t19*t122*(9.0 / 4.0) + l1*l2*m2*m3*m4*t13*t21*t23*t123*(3.0 / 2.0) + l1*l2*m2*m3*m4*t14*t21*t23*t123*(3.0 / 2.0) + l1*l2*m2*m3*m4*t15*t21*t23*t123*(3.0 / 2.0) + l1*l2*m2*m3*m4*t16*t21*t23*t123*(3.0 / 2.0) + l1*l4*m2*m3*m4*t13*t19*t21*t133*(1.5*10.0 / 2.0) + l1*l4*m1*m3*m4*t13*t19*t21*t135*3.0 - l1*l4*m2*m3*m4*t13*t19*t21*t134*(2.7*10.0 / 2.0) + l1*l4*m2*m3*m4*t14*t19*t21*t133*(1.5*10.0 / 2.0) + l1*l4*m1*m3*m4*t14*t19*t21*t135*3.0 + l1*l4*m2*m3*m4*t13*t19*t21*t135*(9.0 / 2.0) - l1*l4*m2*m3*m4*t14*t19*t21*t134*(2.7*10.0 / 2.0) + l1*l4*m2*m3*m4*t15*t19*t21*t133*(1.5*10.0 / 2.0) + l1*l4*m2*m3*m4*t14*t19*t21*t135*(9.0 / 2.0) - l1*l4*m2*m3*m4*t15*t19*t21*t134*(2.7*10.0 / 4.0) + l1*l3*m2*m3*m4*t13*t19*t23*t171*(2.7*10.0 / 2.0) + l1*l3*m2*m3*m4*t14*t19*t23*t171*(2.7*10.0 / 2.0) + l1*l3*m2*m3*m4*t15*t19*t23*t171*(2.7*10.0 / 2.0) + l1*l4*m2*m3*m4*t15*t19*t21*t172*(2.7*10.0 / 4.0) + l1*l3*m2*m3*m4*t16*t19*t23*t171*(2.7*10.0 / 2.0) + l2*l4*m1*m3*m4*t13*t17*t21*t186*3.0 + l2*l4*m2*m3*m4*t13*t17*t21*t186*(9.0 / 2.0) + l1*l3*m2*m3*m4*t13*t19*t23*t188*3.0 + l1*l3*m2*m3*m4*t14*t19*t23*t188*3.0 + l1*l3*m2*m3*m4*t15*t19*t23*t188*3.0 + l1*l3*m2*m3*m4*t16*t19*t23*t188*3.0 + l3*l4*m1*m2*m4*t13*t17*t19*t198*(9.0 / 2.0) + l3*l4*m1*m3*m4*t13*t17*t19*t198*(9.0 / 2.0) + l3*l4*m2*m3*m4*t13*t17*t19*t198*(2.7*10.0 / 4.0) + l1*l2*m2*m3*m4*t13*t21*t23*t199*(9.0 / 2.0) + l1*l2*m2*m3*m4*t14*t21*t23*t199*(9.0 / 2.0) + l1*l2*m2*m3*m4*t15*t21*t23*t199*(9.0 / 2.0) + l1*l2*m2*m3*m4*t16*t21*t23*t199*(9.0 / 2.0) - dth1*dth2*l1*l3*l4*m1*m2*m3*t8*t20*1.6*10.0 - dth1*dth2*l1*l3*l4*m1*m2*m4*t8*t20*2.0*10.0 - dth1*dth2*l1*l3*l4*m1*m3*m4*t8*t20*9.0*10.0 - dth1*dth2*l1*l2*l4*m1*m3*m4*t8*t22*5.0*10.0 - dth1*dth2*l1*l3*l4*m2*m3*m4*t8*t20*(2.25*100.0 / 2.0) - dth1*dth2*l1*l2*l4*m2*m3*m4*t8*t22*(2.25*100.0 / 2.0) - dth1*dth3*l1*l2*l4*m1*m3*m4*t8*t22*5.0*10.0 - dth1*dth3*l1*l2*l4*m2*m3*m4*t8*t22*(2.25*100.0 / 2.0) - dth2*dth3*l1*l2*l4*m1*m3*m4*t8*t22*5.0*10.0 - dth2*dth3*l1*l2*l4*m2*m3*m4*t8*t22*(2.25*100.0 / 2.0) + dth1*dth2*l1*l3*l4*m2*m3*m4*t20*t82*(4.5*10.0 / 2.0) + dth1*dth2*l1*l2*l4*m2*m3*m4*t22*t82*(7.5*10.0 / 2.0) + dth1*dth2*l1*l3*l4*m1*m2*m4*t20*t85*1.2*10.0 + dth1*dth2*l1*l3*l4*m1*m3*m4*t20*t85*1.8*10.0 + dth1*dth3*l1*l2*l4*m2*m3*m4*t22*t82*(7.5*10.0 / 2.0) + dth1*dth2*l1*l2*l4*m1*m3*m4*t22*t85*6.0 + dth1*dth2*l1*l3*l4*m2*m3*m4*t20*t85*(4.5*10.0 / 2.0) + dth2*dth3*l1*l2*l4*m2*m3*m4*t22*t82*(7.5*10.0 / 2.0) + dth1*dth2*l1*l2*l4*m2*m3*m4*t22*t85*(2.7*10.0 / 2.0) + dth1*dth3*l1*l2*l4*m1*m3*m4*t22*t85*6.0 + dth1*dth3*l1*l2*l4*m2*m3*m4*t22*t85*(2.7*10.0 / 2.0) + dth2*dth3*l1*l2*l4*m1*m3*m4*t22*t85*6.0 + dth2*dth3*l1*l2*l4*m2*m3*m4*t22*t85*(2.7*10.0 / 2.0) - dth1*dth2*l1*l3*l4*m2*m3*m4*t20*t187*(9.0 / 2.0) - dth1*dth2*l1*l2*l4*m2*m3*m4*t22*t187*(9.0 / 2.0) - dth1*dth3*l1*l2*l4*m2*m3*m4*t22*t187*(9.0 / 2.0) - dth2*dth3*l1*l2*l4*m2*m3*m4*t22*t187*(9.0 / 2.0) + dth1*dth2*l1*l3*m1*m2*m4*t9*t19*t23*1.6*10.0 + dth1*dth2*l1*l3*m1*m3*m4*t9*t19*t23*3.6*10.0 + dth1*dth3*l1*l3*m1*m2*m4*t9*t19*t23*1.6*10.0 + dth1*dth2*l1*l3*m2*m3*m4*t9*t19*t23*9.0*10.0 + dth1*dth3*l1*l3*m1*m3*m4*t9*t19*t23*3.6*10.0 + dth1*dth4*l1*l3*m1*m2*m4*t9*t19*t23*1.6*10.0 + dth2*dth3*l1*l3*m1*m2*m4*t9*t19*t23*1.6*10.0 + dth1*dth3*l1*l3*m2*m3*m4*t9*t19*t23*9.0*10.0 + dth1*dth4*l1*l3*m1*m3*m4*t9*t19*t23*3.6*10.0 + dth2*dth3*l1*l3*m1*m3*m4*t9*t19*t23*3.6*10.0 + dth2*dth4*l1*l3*m1*m2*m4*t9*t19*t23*1.6*10.0 + dth1*dth4*l1*l3*m2*m3*m4*t9*t19*t23*9.0*10.0 + dth2*dth3*l1*l3*m2*m3*m4*t9*t19*t23*9.0*10.0 + dth2*dth4*l1*l3*m1*m3*m4*t9*t19*t23*3.6*10.0 + dth3*dth4*l1*l3*m1*m2*m4*t9*t19*t23*1.6*10.0 + dth2*dth4*l1*l3*m2*m3*m4*t9*t19*t23*9.0*10.0 + dth3*dth4*l1*l3*m1*m3*m4*t9*t19*t23*3.6*10.0 + dth3*dth4*l1*l3*m2*m3*m4*t9*t19*t23*9.0*10.0 + dth1*dth2*l1*l4*m1*m2*m4*t19*t21*t38*1.2*10.0 - dth1*dth2*l1*l4*m1*m3*m4*t19*t21*t37*6.0*10.0 + dth1*dth2*l1*l4*m2*m3*m4*t19*t21*t36*7.5*10.0 + dth1*dth2*l1*l4*m1*m3*m4*t19*t21*t38*1.8*10.0 - dth1*dth2*l1*l4*m2*m3*m4*t19*t21*t37*9.0*10.0 + dth1*dth3*l1*l4*m1*m2*m4*t19*t21*t38*1.2*10.0 - dth1*dth3*l1*l4*m1*m3*m4*t19*t21*t37*3.0*10.0 + dth1*dth2*l1*l4*m2*m3*m4*t19*t21*t38*4.5*10.0 + dth1*dth3*l1*l4*m1*m3*m4*t19*t21*t38*1.8*10.0 - dth1*dth3*l1*l4*m2*m3*m4*t19*t21*t37*4.5*10.0 + dth2*dth3*l1*l4*m1*m2*m4*t19*t21*t38*1.2*10.0 - dth2*dth3*l1*l4*m1*m3*m4*t19*t21*t37*3.0*10.0 + dth1*dth3*l1*l4*m2*m3*m4*t19*t21*t38*4.5*10.0 + dth2*dth3*l1*l4*m1*m3*m4*t19*t21*t38*1.8*10.0 - dth2*dth3*l1*l4*m2*m3*m4*t19*t21*t37*4.5*10.0 + dth2*dth3*l1*l4*m2*m3*m4*t19*t21*t38*4.5*10.0 - dth1*dth2*l1*l2*m1*m3*m4*t21*t23*t43*4.0 - dth1*dth2*l1*l2*m2*m3*m4*t21*t23*t43*9.0 - dth1*dth3*l1*l2*m1*m3*m4*t21*t23*t43*4.0 - dth1*dth3*l1*l2*m2*m3*m4*t21*t23*t43*9.0 - dth1*dth4*l1*l2*m1*m3*m4*t21*t23*t43*4.0 - dth2*dth3*l1*l2*m1*m3*m4*t21*t23*t43*4.0 - dth1*dth4*l1*l2*m2*m3*m4*t21*t23*t43*9.0 - dth2*dth3*l1*l2*m2*m3*m4*t21*t23*t43*9.0 - dth2*dth4*l1*l2*m1*m3*m4*t21*t23*t43*4.0 - dth2*dth4*l1*l2*m2*m3*m4*t21*t23*t43*9.0 - dth3*dth4*l1*l2*m1*m3*m4*t21*t23*t43*4.0 - dth3*dth4*l1*l2*m2*m3*m4*t21*t23*t43*9.0 - dth1*dth2*l1*l3*m2*m3*m4*t19*t23*t84*2.7*10.0 - dth1*dth2*l1*l3*m1*m3*m4*t19*t23*t86*1.2*10.0 - dth1*dth3*l1*l3*m2*m3*m4*t19*t23*t84*2.7*10.0 - dth1*dth2*l1*l3*m2*m3*m4*t19*t23*t86*1.8*10.0 - dth1*dth3*l1*l3*m1*m3*m4*t19*t23*t86*1.2*10.0 - dth1*dth4*l1*l3*m2*m3*m4*t19*t23*t84*2.7*10.0 - dth2*dth3*l1*l3*m2*m3*m4*t19*t23*t84*2.7*10.0 - dth1*dth3*l1*l3*m2*m3*m4*t19*t23*t86*1.8*10.0 - dth1*dth4*l1*l3*m1*m3*m4*t19*t23*t86*1.2*10.0 - dth2*dth3*l1*l3*m1*m3*m4*t19*t23*t86*1.2*10.0 - dth2*dth4*l1*l3*m2*m3*m4*t19*t23*t84*2.7*10.0 - dth1*dth4*l1*l3*m2*m3*m4*t19*t23*t86*1.8*10.0 - dth2*dth3*l1*l3*m2*m3*m4*t19*t23*t86*1.8*10.0 - dth2*dth4*l1*l3*m1*m3*m4*t19*t23*t86*1.2*10.0 - dth3*dth4*l1*l3*m2*m3*m4*t19*t23*t84*2.7*10.0 - dth2*dth4*l1*l3*m2*m3*m4*t19*t23*t86*1.8*10.0 - dth3*dth4*l1*l3*m1*m3*m4*t19*t23*t86*1.2*10.0 - dth3*dth4*l1*l3*m2*m3*m4*t19*t23*t86*1.8*10.0 - dth1*dth2*l1*l2*m1*m3*m4*t21*t23*t109*1.2*10.0 - dth1*dth2*l1*l2*m2*m3*m4*t21*t23*t109*2.7*10.0 - dth1*dth3*l1*l2*m1*m3*m4*t21*t23*t109*1.2*10.0 - dth1*dth3*l1*l2*m2*m3*m4*t21*t23*t109*2.7*10.0 - dth1*dth4*l1*l2*m1*m3*m4*t21*t23*t109*1.2*10.0 - dth2*dth3*l1*l2*m1*m3*m4*t21*t23*t109*1.2*10.0 - dth1*dth4*l1*l2*m2*m3*m4*t21*t23*t109*2.7*10.0 - dth2*dth3*l1*l2*m2*m3*m4*t21*t23*t109*2.7*10.0 - dth2*dth4*l1*l2*m1*m3*m4*t21*t23*t109*1.2*10.0 - dth2*dth4*l1*l2*m2*m3*m4*t21*t23*t109*2.7*10.0 - dth3*dth4*l1*l2*m1*m3*m4*t21*t23*t109*1.2*10.0 - dth3*dth4*l1*l2*m2*m3*m4*t21*t23*t109*2.7*10.0 + dth1*dth2*l1*l2*m2*m3*m4*t21*t23*t123*3.0 + dth1*dth3*l1*l2*m2*m3*m4*t21*t23*t123*3.0 + dth1*dth4*l1*l2*m2*m3*m4*t21*t23*t123*3.0 + dth2*dth3*l1*l2*m2*m3*m4*t21*t23*t123*3.0 + dth2*dth4*l1*l2*m2*m3*m4*t21*t23*t123*3.0 + dth3*dth4*l1*l2*m2*m3*m4*t21*t23*t123*3.0 + dth1*dth2*l1*l4*m2*m3*m4*t19*t21*t133*1.5*10.0 + dth1*dth2*l1*l4*m1*m3*m4*t19*t21*t135*6.0 - dth1*dth2*l1*l4*m2*m3*m4*t19*t21*t134*2.7*10.0 + dth1*dth3*l1*l4*m2*m3*m4*t19*t21*t133*1.5*10.0 + dth1*dth2*l1*l4*m2*m3*m4*t19*t21*t135*9.0 - dth1*dth3*l1*l4*m2*m3*m4*t19*t21*t134*(2.7*10.0 / 2.0) + dth2*dth3*l1*l4*m2*m3*m4*t19*t21*t133*1.5*10.0 - dth2*dth3*l1*l4*m2*m3*m4*t19*t21*t134*(2.7*10.0 / 2.0) + dth1*dth2*l1*l3*m2*m3*m4*t19*t23*t171*2.7*10.0 + dth1*dth3*l1*l3*m2*m3*m4*t19*t23*t171*2.7*10.0 + dth1*dth3*l1*l4*m2*m3*m4*t19*t21*t172*(2.7*10.0 / 2.0) + dth1*dth4*l1*l3*m2*m3*m4*t19*t23*t171*2.7*10.0 + dth2*dth3*l1*l3*m2*m3*m4*t19*t23*t171*2.7*10.0 + dth2*dth3*l1*l4*m2*m3*m4*t19*t21*t172*(2.7*10.0 / 2.0) + dth2*dth4*l1*l3*m2*m3*m4*t19*t23*t171*2.7*10.0 + dth3*dth4*l1*l3*m2*m3*m4*t19*t23*t171*2.7*10.0 + dth1*dth2*l1*l3*m2*m3*m4*t19*t23*t188*6.0 + dth1*dth3*l1*l3*m2*m3*m4*t19*t23*t188*6.0 + dth1*dth4*l1*l3*m2*m3*m4*t19*t23*t188*6.0 + dth2*dth3*l1*l3*m2*m3*m4*t19*t23*t188*6.0 + dth2*dth4*l1*l3*m2*m3*m4*t19*t23*t188*6.0 + dth3*dth4*l1*l3*m2*m3*m4*t19*t23*t188*6.0 + dth1*dth2*l1*l2*m2*m3*m4*t21*t23*t199*9.0 + dth1*dth3*l1*l2*m2*m3*m4*t21*t23*t199*9.0 + dth1*dth4*l1*l2*m2*m3*m4*t21*t23*t199*9.0 + dth2*dth3*l1*l2*m2*m3*m4*t21*t23*t199*9.0 + dth2*dth4*l1*l2*m2*m3*m4*t21*t23*t199*9.0 + dth3*dth4*l1*l2*m2*m3*m4*t21*t23*t199*9.0)*2.4*10.0;

        index = index + num_threads;

        index1 = (grid_size[0] > 1) ? index : 0;
        index2 = (grid_size[1] > 1) ? index : 0;
        index3 = (grid_size[2] > 1) ? index : 0;
        index4 = (grid_size[3] > 1) ? index : 0;
        index5 = (grid_size[4] > 1) ? index : 0;
        index6 = (grid_size[5] > 1) ? index : 0;
        index7 = (grid_size[6] > 1) ? index : 0;
        index8 = (grid_size[7] > 1) ? index : 0;

        uindex1 = (active_actions[0] == 1) ? index : 0;
        uindex2 = (active_actions[1] == 1) ? index : 0;
        uindex3 = (active_actions[2] == 1) ? index : 0;
        uindex4 = (active_actions[3] == 1) ? index : 0;
    }
}

__global__ void dyn8_mex_continuous(double* const d_dx4, // outputs
    double* const x1, double* const x2, double* const x3, double* const x4,
    double* const dx1, double* const dx2, double* const dx3, double* const dx4, // input states
    double const* const in1, double const* const in2, double const* const in3, double const* const in4, // input actions
    const double m1, const double m2, const double m3, const double m4,
    const double l1, const double l2, const double l3, const double l4, const double g, 
    int32_t const* const grid_size, int32_t const* const active_actions)
{
    int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x * blockDim.y
        + (threadIdx.x + threadIdx.y * blockDim.x);
    int num_threads = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
    int num_elements = grid_size[0] * grid_size[1] * grid_size[2] * grid_size[3]
                        * grid_size[4] * grid_size[5] * grid_size[6] * grid_size[7];
    double th1, th2, th3, th4, dth1, dth2, dth3, dth4, u1, u2, u3, u4;

    int index1 = (grid_size[0] > 1) ? index : 0;
    int index2 = (grid_size[1] > 1) ? index : 0;
    int index3 = (grid_size[2] > 1) ? index : 0;
    int index4 = (grid_size[3] > 1) ? index : 0;
    int index5 = (grid_size[4] > 1) ? index : 0;
    int index6 = (grid_size[5] > 1) ? index : 0;
    int index7 = (grid_size[6] > 1) ? index : 0;
    int index8 = (grid_size[7] > 1) ? index : 0;

    int uindex1 = (active_actions[0] == 1) ? index : 0;
    int uindex2 = (active_actions[1] == 1) ? index : 0;
    int uindex3 = (active_actions[2] == 1) ? index : 0;
    int uindex4 = (active_actions[3] == 1) ? index : 0;

    while (index < num_elements)
    {
        th1 = x1[index1];
        th2 = x2[index2];
        th3 = x3[index3];
        th4 = x4[index4];

        dth1 = dx1[index5];
        dth2 = dx2[index6];
        dth3 = dx3[index7];
        dth4 = dx4[index8];

        u1 = in1[uindex1];
        u2 = in2[uindex2];
        u3 = in3[uindex3];
        u4 = in4[uindex4];

        double t2 = cos(th1);
        double t3 = cos(th2);
        double t4 = cos(th3);
        double t5 = cos(th4);
        double t6 = sin(th1);
        double t7 = sin(th2);
        double t8 = sin(th3);
        double t9 = sin(th4);
        double t10 = th1 + th2;
        double t11 = th2 + th3;
        double t12 = th3 + th4;
        double t13 = dth1 * dth1;
        double t14 = dth2 * dth2;
        double t15 = dth3 * dth3;
        double t16 = dth4 * dth4;
        double t17 = l1 * l1;
//         double t18 = t17 * l1;
        double t19 = l2 * l2;
//         double t20 = t19 * l2;
        double t21 = l3 * l3;
        double t22 = t21 * l3;
        double t23 = l4 * l4;
        double t24 = t23 * l4;
        double t25 = m2 * m2;
        double t26 = m3 * m3;
        double t27 = t26 * m3;
        double t28 = m4 * m4;
        double t29 = t28 * m4;
        double t30 = th2 * 2.0;
        double t31 = th3 * 2.0;
        double t32 = th4 * 2.0;
        double t46 = 1.0 / l1;
        double t48 = 1.0 / l2;
//         double t50 = 1.0 / l3;
//         double t52 = 1.0 / l4;
        double t53 = -th1;
        double t54 = -th2;
        double t55 = -th3;
        double t57 = -th4;
        double t125 = m1 * m2 * m3 * 128.0;
        double t126 = m1 * m2 * m4 * 240.0;
        double t163 = m1 * m3 * m4 * 600.0;
        double t164 = m2 * m3 * m4 * 1500.0;
        double t33 = cos(t30);
        double t34 = cos(t31);
        double t35 = cos(t32);
        double t36 = sin(t30);
        double t37 = sin(t31);
        double t38 = sin(t32);
        double t39 = cos(t11);
        double t40 = cos(t12);
        double t41 = sin(t10);
        double t42 = sin(t11);
        double t43 = sin(t12);
        double t44 = t10 + th3;
        double t45 = t11 + th4;
//         double t47 = 1.0 / t17;
//         double t49 = 1.0 / t19;
        double t51 = 1.0 / t21;
        double t56 = -t31;
        double t58 = -t32;
        double t62 = t10 + t12;
        double t63 = t10 + th2;
        double t64 = t31 + th1;
        double t65 = t32 + th1;
        double t66 = t11 + th3;
        double t67 = t11 + th2;
        double t68 = t32 + th2;
        double t69 = t30 + th4;
        double t70 = t12 + th4;
        double t71 = t12 + th3;
        double t87 = t10 + t31;
        double t88 = t10 + t32;
        double t89 = t11 + t32;
        double t92 = t54 + th1;
        double t94 = t55 + th2;
        double t97 = t57 + th3;
        double t98 = t11 * 2.0;
        double t99 = t30 + t32;
        double t100 = t12 * 2.0;
        double t110 = t10 + t55;
        double t112 = t11 + t53;
        double t114 = t11 + t57;
        double t116 = t12 + t54;
        double t117 = t27 * 144.0;
        double t128 = t30 + t57;
        double t156 = m1 * t28 * 144.0;
        double t157 = m3 * t28 * 144.0;
        double t158 = m1 * t26 * 240.0;
        double t159 = m3 * t25 * 240.0;
        double t160 = m2 * t28 * 360.0;
        double t161 = m4 * t25 * 450.0;
        double t162 = m4 * t26 * 450.0;
        double t179 = m2 * t26 * 600.0;
        double t211 = t10 - t12;
        double t213 = -t11 + th1 + th4;
        double t59 = cos(t45);
        double t60 = sin(t44);
        double t61 = sin(t45);
        double t72 = cos(t66);
        double t73 = cos(t67);
        double t74 = cos(t68);
        double t75 = cos(t69);
        double t76 = cos(t70);
        double t77 = cos(t71);
        double t78 = sin(t63);
        double t79 = sin(t64);
        double t80 = sin(t65);
        double t81 = sin(t66);
        double t82 = sin(t67);
        double t83 = sin(t68);
        double t84 = sin(t69);
        double t85 = sin(t70);
        double t86 = sin(t71);
        double t90 = t45 + th2;
        double t91 = sin(t62);
        double t93 = t56 + th1;
        double t95 = t58 + th1;
        double t96 = t58 + th2;
        double t101 = cos(t94);
        double t103 = cos(t97);
        double t104 = sin(t92);
        double t106 = sin(t94);
        double t109 = sin(t97);
        double t111 = t92 + th3;
        double t113 = t10 + t58;
        double t115 = t94 + th4;
        double t118 = cos(t89);
        double t120 = sin(t87);
        double t121 = sin(t88);
        double t122 = sin(t89);
        double t124 = t62 + th4;
        double t129 = t30 + t58;
        double t130 = cos(t98);
        double t131 = cos(t99);
        double t132 = cos(t100);
        double t133 = sin(t98);
        double t134 = sin(t99);
        double t135 = sin(t100);
        double t136 = t11 + t44;
        double t137 = t32 + t63;
        double t138 = t100 + th1;
        double t139 = t12 + t45;
        double t140 = t32 + t67;
        double t141 = t11 + t45;
        double t142 = cos(t114);
        double t144 = cos(t116);
        double t145 = sin(t110);
        double t147 = sin(t112);
        double t149 = sin(t114);
        double t151 = sin(t116);
        double t152 = t44 + t57;
        double t153 = t110 + th4;
        double t154 = t12 + t92;
        double t155 = t45 + t53;
        double t169 = cos(t128);
        double t171 = sin(t128);
        double t173 = t53 + t66;
        double t174 = t54 + t65;
        double t175 = t53 + t68;
        double t176 = t58 + t63;
        double t177 = t54 + t70;
        double t178 = t57 + t67;
        double t189 = t12 + t62;
        double t191 = t11 + t89;
        double t200 = t70 + t92;
        double t201 = t53 + t89;
        double t207 = t53 + t100;
        double t208 = m1 * m2 * m4 * t35 * 144.0;
        double t209 = m1 * m3 * m4 * t35 * 216.0;
        double t210 = m1 * m3 * m4 * t34 * 360.0;
        double t212 = t92 + t97;
        double t214 = t57 + t211;
        double t217 = -t27 * t33 * 144.0;
        double t218 = sin(t211);
        double t220 = sin(t213);
        double t222 = m1 * t26 * t34 * 144.0;
        double t223 = m3 * t25 * t33 * 144.0;
        double t226 = m4 * t26 * t35 * 162.0;
        double t227 = m2 * t26 * t34 * 216.0;
        double t228 = m2 * t28 * t33 * 216.0;
        double t229 = m2 * t28 * t34 * 216.0;
        double t230 = m4 * t25 * t33 * 270.0;
        double t231 = m4 * t25 * t35 * 270.0;
        double t232 = m2 * t26 * t33 * 360.0;
        double t237 = m2 * m3 * m4 * t34 * 540.0;
        double t238 = m2 * m3 * m4 * t35 * 540.0;
        double t239 = m2 * m3 * m4 * t33 * 900.0;
        double t243 = - m1 * t28 * t34 * 144.0;
        double t244 = - m3 * t28 * t33 * 144.0;
        double t252 = - m4 * t26 * t33 * 450.0;
        double t102 = cos(t96);
        double t105 = sin(t93);
        double t107 = sin(t95);
        double t108 = sin(t96);
        double t119 = cos(t90);
        double t123 = sin(t90);
        double t127 = sin(t124);
        double t143 = cos(t115);
        double t146 = sin(t111);
        double t148 = sin(t113);
        double t150 = sin(t115);
        double t165 = sin(t152);
        double t166 = sin(t153);
        double t167 = sin(t154);
        double t168 = sin(t155);
        double t170 = cos(t129);
        double t172 = sin(t129);
        double t180 = cos(t139);
        double t181 = cos(t140);
        double t182 = cos(t141);
        double t183 = sin(t136);
        double t184 = sin(t137);
        double t185 = sin(t138);
        double t186 = sin(t139);
        double t187 = sin(t140);
        double t188 = sin(t141);
        double t190 = sin(t189);
        double t192 = cos(t177);
        double t193 = cos(t178);
        double t194 = sin(t173);
        double t195 = sin(t174);
        double t196 = sin(t175);
        double t197 = sin(t176);
        double t198 = sin(t177);
        double t199 = sin(t178);
        double t202 = cos(t191);
        double t203 = sin(t191);
        double t205 = sin(t200);
        double t206 = sin(t201);
        double t215 = sin(t207);
        double t216 = t53 + t139;
        double t219 = sin(t212);
        double t221 = sin(t214);
        double t234 = -t208;
        double t235 = -t209;
        double t236 = -t210;
        double t241 = -t222;
        double t242 = -t223;
        double t245 = -t226;
        double t246 = -t227;
        double t247 = -t228;
        double t248 = -t229;
        double t249 = -t230;
        double t250 = -t231;
        double t251 = -t232;
        double t253 = -t237;
        double t254 = -t238;
        double t255 = -t239;
        double t256 = m1 * m3 * m4 * t132 * 72.0;
        double t257 = m2 * m3 * m4 * t132 * 108.0;
        double t258 = m2 * m3 * m4 * t131 * 162.0;
        double t259 = m2 * m3 * m4 * t130 * 180.0;
        double t260 = m2 * t26 * t130 * 72.0;
        double t261 = m2 * t28 * t130 * 72.0;
        double t262 = m4 * t25 * t131 * 81.0;
        double t263 = m4 * t26 * t131 * 81.0;
        double t240 = sin(t216);
        double t264 = m2 * m3 * m4 * t170 * 162.0;
        double t265 = m4 * t25 * t170 * 81.0;
        double t266 = m4 * t26 * t170 * 81.0;
        double t267 = m2 * m3 * m4 * t202 * 36.0;
        double t268 = -t267;
        double t269 = t117 + t125 + t126 + t156 + t157 + t158 + t159 + t160 + t161 + t162 + t163 + t164 + t179 + t217 + t234 + t235 + t236 + t241 + t242 + t243 + t244 + t245 + t246 + t247 + t248 + t249 + t250 + t251 + t252 + t253 + t254 + t255 + t256 + t257 + t258 + t259 + t260 + t261 + t262 + t263 + t264 + t265 + t266 + t268;
        double t270 = 1.0 / t269;

        d_dx4[index] = (t46*t48*t51*t270*(l1*l2*t21*t27*u4* - 1.8*10.0 + l1*l2*t23*t29*u3*1.8*10.0 - l1*l2*t23*t29*u4*1.8*10.0 - l1*l2*m1*t21*t26*u4*3.0*10.0 - l1*l2*m2*t21*t26*u4*7.5*10.0 - l1*l2*m3*t21*t25*u4*3.0*10.0 - l1*l2*m1*t21*t28*u4*7.2*10.0 - l1*l2*m4*t21*t25*u4*9.0*10.0 + l1*l2*m1*t23*t28*u3*3.0*10.0 - l1*l2*m2*t21*t28*u4*1.8*100.0 - l1*l2*m4*t21*t26*u4*9.0*10.0 + l1*l2*m4*t23*t25*u3*3.0*10.0 - l1*l2*m1*t23*t28*u4*3.0*10.0 + l1*l2*m2*t23*t28*u3*7.5*10.0 - l1*l2*m3*t21*t28*u4*7.2*10.0 - l1*l2*m4*t23*t25*u4*3.0*10.0 + l1*l2*m4*t23*t26*u3*7.2*10.0 - l1*l2*m2*t23*t28*u4*7.5*10.0 + l1*l2*m3*t23*t28*u3*9.0*10.0 - l1*l2*m4*t23*t26*u4*7.2*10.0 - l1*l2*m3*t23*t28*u4*9.0*10.0 - l1*l3*t4*t23*t29*u2*1.8*10.0 + l1*l3*t4*t23*t29*u3*1.8*10.0 + l1*l2*t21*t27*t33*u4*1.8*10.0 - l1*l2*t23*t29*t33*u3*1.8*10.0 + l1*l2*t23*t29*t33*u4*1.8*10.0 - l2*l3*t23*t29*t39*u1*1.8*10.0 + l2*l3*t23*t29*t39*u2*1.8*10.0 + l1*l3*t23*t29*t73*u2*1.8*10.0 - l1*l3*t23*t29*t73*u3*1.8*10.0 + l2*l3*t23*t29*t101*u1*1.8*10.0 - l2*l3*t23*t29*t101*u2*1.8*10.0 - l1*l2*m1*m2*m3*t21*u4*1.6*10.0 - l1*l2*m1*m2*m4*t21*u4*4.8*10.0 + l1*l2*m1*m2*m4*t23*u3*1.6*10.0 - l1*l2*m1*m3*m4*t21*u4*1.2*100.0 - l1*l2*m1*m2*m4*t23*u4*1.6*10.0 + l1*l2*m1*m3*m4*t23*u3*4.8*10.0 - l1*l2*m2*m3*m4*t21*u4*3.0*100.0 - l1*l2*m1*m3*m4*t23*u4*4.8*10.0 + l1*l2*m2*m3*m4*t23*u3*1.2*100.0 - l1*l2*m2*m3*m4*t23*u4*1.2*100.0 - l1*l3*m1*t4*t23*t28*u2*3.0*10.0 + l1*l3*m1*t4*t23*t28*u3*3.0*10.0 - l1*l3*m2*t4*t23*t28*u2*(1.35*100.0 / 2.0) - l1*l3*m4*t4*t23*t26*u2*3.6*10.0 + l1*l3*m2*t4*t23*t28*u3*(1.35*100.0 / 2.0) - l1*l3*m3*t4*t23*t28*u2*(1.35*100.0 / 2.0) + l1*l3*m4*t4*t23*t26*u3*3.6*10.0 + l1*l3*m3*t4*t23*t28*u3*(1.35*100.0 / 2.0) + l1*l2*m1*t21*t26*t34*u4*1.8*10.0 + l1*l2*m2*t21*t26*t33*u4*4.5*10.0 + l1*l2*m3*t21*t25*t33*u4*1.8*10.0 + l1*l2*m2*t21*t26*t34*u4*2.7*10.0 + l1*l2*m4*t21*t25*t33*u4*5.4*10.0 + l1*l2*m1*t21*t28*t34*u4*7.2*10.0 + l1*l2*m2*t21*t28*t33*u4*1.08*100.0 + l1*l2*m4*t21*t26*t33*u4*9.0*10.0 - l1*l2*m4*t23*t25*t33*u3*1.8*10.0 + l1*l2*m2*t21*t28*t34*u4*1.08*100.0 - l1*l2*m2*t23*t28*t33*u3*4.5*10.0 + l1*l2*m3*t21*t28*t33*u4*7.2*10.0 + l1*l2*m4*t23*t25*t33*u4*1.8*10.0 - l1*l2*m4*t23*t26*t33*u3*7.2*10.0 + l1*l2*m2*t23*t28*t33*u4*4.5*10.0 - l1*l2*m3*t23*t28*t33*u3*9.0*10.0 + l1*l2*m4*t23*t26*t33*u4*7.2*10.0 + l1*l2*m3*t23*t28*t33*u4*9.0*10.0 + l1*l4*m1*t21*t28*t40*u2*3.6*10.0 - l1*l4*m1*t21*t28*t40*u3*3.6*10.0 + l1*l4*m2*t21*t28*t40*u2*8.1*10.0 - l1*l4*m4*t21*t26*t40*u2*(9.0 / 2.0) - l2*l3*m2*t23*t28*t39*u1*(1.5*10.0 / 2.0) - l2*l3*m4*t23*t26*t39*u1*3.6*10.0 - l1*l4*m2*t21*t28*t40*u3*8.1*10.0 + l1*l4*m3*t21*t28*t40*u2*9.0 + l1*l4*m4*t21*t26*t40*u3*(9.0 / 2.0) + l2*l3*m2*t23*t28*t39*u2*(1.5*10.0 / 2.0) - l2*l3*m3*t23*t28*t39*u1*(1.35*100.0 / 2.0) + l2*l3*m4*t23*t26*t39*u2*3.6*10.0 - l1*l4*m3*t21*t28*t40*u3*9.0 + l2*l3*m3*t23*t28*t39*u2*(1.35*100.0 / 2.0) + l2*l4*m2*t21*t28*t59*u1*9.0 - l2*l4*m4*t21*t26*t59*u1*(9.0 / 2.0) - l2*l4*m2*t21*t28*t59*u2*9.0 + l2*l4*m3*t21*t28*t59*u1*9.0 + l2*l4*m4*t21*t26*t59*u2*(9.0 / 2.0) - l2*l4*m3*t21*t28*t59*u2*9.0 + l1*l3*m2*t23*t28*t73*u2*(4.5*10.0 / 2.0) + l1*l3*m4*t23*t26*t73*u2*3.6*10.0 - l1*l3*m2*t23*t28*t73*u3*(4.5*10.0 / 2.0) + l1*l3*m3*t23*t28*t73*u2*(1.35*100.0 / 2.0) - l1*l3*m4*t23*t26*t73*u3*3.6*10.0 + l1*l3*m1*t23*t28*t76*u2*1.8*10.0 - l1*l3*m3*t23*t28*t73*u3*(1.35*100.0 / 2.0) - l1*l3*m1*t23*t28*t76*u3*1.8*10.0 + l1*l3*m2*t23*t28*t76*u2*(8.1*10.0 / 2.0) - l1*l3*m2*t23*t28*t76*u3*(8.1*10.0 / 2.0) + l1*l3*m3*t23*t28*t76*u2*(2.7*10.0 / 2.0) - l1*l3*m3*t23*t28*t76*u3*(2.7*10.0 / 2.0) - l1*l4*m1*t21*t28*t103*u2*3.6*10.0 + l2*l3*m2*t23*t28*t101*u1*(4.5*10.0 / 2.0) + l2*l3*m4*t23*t26*t101*u1*3.6*10.0 + l1*l4*m1*t21*t28*t103*u3*3.6*10.0 - l1*l4*m2*t21*t28*t103*u2*8.1*10.0 - l1*l4*m4*t21*t26*t103*u2*(2.7*10.0 / 2.0) - l2*l3*m2*t23*t28*t101*u2*(4.5*10.0 / 2.0) + l2*l3*m3*t23*t28*t101*u1*(1.35*100.0 / 2.0) - l2*l3*m4*t23*t26*t101*u2*3.6*10.0 + l1*l4*m2*t21*t28*t103*u3*8.1*10.0 - l1*l4*m3*t21*t28*t103*u2*2.7*10.0 + l1*l4*m4*t21*t26*t103*u3*(2.7*10.0 / 2.0) - l2*l3*m3*t23*t28*t101*u2*(1.35*100.0 / 2.0) + l1*l4*m3*t21*t28*t103*u3*2.7*10.0 - l1*l4*m2*t21*t28*t119*u2*2.7*10.0 + l1*l4*m4*t21*t26*t119*u2*(9.0 / 2.0) + l2*l3*m2*t23*t28*t118*u1*(9.0 / 2.0) + l1*l4*m2*t21*t28*t119*u3*2.7*10.0 - l1*l4*m3*t21*t28*t119*u2*9.0 - l1*l4*m4*t21*t26*t119*u3*(9.0 / 2.0) - l2*l3*m2*t23*t28*t118*u2*(9.0 / 2.0) + l2*l3*m3*t23*t28*t118*u1*(2.7*10.0 / 2.0) + l1*l4*m3*t21*t28*t119*u3*9.0 - l2*l3*m3*t23*t28*t118*u2*(2.7*10.0 / 2.0) - l1*l2*m2*t21*t26*t130*u4*9.0 - l1*l2*m2*t21*t28*t130*u4*3.6*10.0 - l1*l2*m1*t23*t28*t132*u3*1.8*10.0 + l1*l2*m1*t23*t28*t132*u4*1.8*10.0 - l1*l2*m2*t23*t28*t132*u3*2.7*10.0 + l1*l2*m2*t23*t28*t132*u4*2.7*10.0 - l2*l4*m2*t21*t28*t142*u1*9.0 - l2*l4*m4*t21*t26*t142*u1*(2.7*10.0 / 2.0) + l2*l4*m2*t21*t28*t142*u2*9.0 + l2*l4*m2*t21*t28*t143*u1*2.7*10.0 - l2*l4*m3*t21*t28*t142*u1*2.7*10.0 + l2*l4*m4*t21*t26*t142*u2*(2.7*10.0 / 2.0) + l2*l4*m4*t21*t26*t143*u1*(2.7*10.0 / 2.0) - l2*l4*m2*t21*t28*t143*u2*2.7*10.0 - l2*l4*m2*t21*t28*t144*u1*2.7*10.0 + l2*l4*m3*t21*t28*t142*u2*2.7*10.0 + l2*l4*m3*t21*t28*t143*u1*2.7*10.0 - l2*l4*m4*t21*t26*t143*u2*(2.7*10.0 / 2.0) + l2*l4*m4*t21*t26*t144*u1*(9.0 / 2.0) + l2*l4*m2*t21*t28*t144*u2*2.7*10.0 - l2*l4*m3*t21*t28*t143*u2*2.7*10.0 - l2*l4*m3*t21*t28*t144*u1*9.0 - l2*l4*m4*t21*t26*t144*u2*(9.0 / 2.0) + l2*l4*m3*t21*t28*t144*u2*9.0 - l1*l3*m2*t23*t28*t181*u2*(2.7*10.0 / 2.0) + l1*l3*m2*t23*t28*t181*u3*(2.7*10.0 / 2.0) - l1*l3*m3*t23*t28*t181*u2*(2.7*10.0 / 2.0) + l1*l3*m3*t23*t28*t181*u3*(2.7*10.0 / 2.0) + l1*l4*m2*t21*t28*t193*u2*2.7*10.0 + l1*l4*m4*t21*t26*t193*u2*(2.7*10.0 / 2.0) - l2*l3*m2*t23*t28*t192*u1*(2.7*10.0 / 2.0) - l1*l4*m2*t21*t28*t193*u3*2.7*10.0 + l1*l4*m3*t21*t28*t193*u2*2.7*10.0 - l1*l4*m4*t21*t26*t193*u3*(2.7*10.0 / 2.0) + l2*l3*m2*t23*t28*t192*u2*(2.7*10.0 / 2.0) - l2*l3*m3*t23*t28*t192*u1*(2.7*10.0 / 2.0) - l1*l4*m3*t21*t28*t193*u3*2.7*10.0 + l2*l3*m3*t23*t28*t192*u2*(2.7*10.0 / 2.0) + l1*l2*m2*t23*t28*t202*u3*9.0 - l1*l2*m2*t23*t28*t202*u4*9.0 - g*l1*l2*l3*m1*t23*t29*t60*(3.0 / 2.0) - g*l1*l2*l3*m2*t23*t29*t60*(3.0 / 2.0) + g*l1*l2*l3*m1*t23*t29*t145*(3.0 / 2.0) - g*l1*l2*l3*m1*t23*t29*t146*(9.0 / 2.0) + g*l1*l2*l3*m2*t23*t29*t145*(9.0 / 2.0) - g*l1*l2*l3*m1*t23*t29*t147*(9.0 / 2.0) - g*l1*l2*l3*m2*t23*t29*t146*(9.0 / 2.0) - g*l1*l2*l3*m2*t23*t29*t147*(3.0 / 2.0) + l1*l2*l3*l4*m1*t5*t28*u3*3.6*10.0 + l1*l2*l3*l4*m4*t5*t25*u3*4.5*10.0 - l1*l2*l3*l4*m1*t5*t28*u4*7.2*10.0 + l1*l2*l3*l4*m2*t5*t28*u3*9.0*10.0 - l1*l2*l3*l4*m4*t5*t25*u4*9.0*10.0 + l1*l2*l3*l4*m4*t5*t26*u3*5.4*10.0 - l1*l2*l3*l4*m2*t5*t28*u4*1.8*100.0 + l1*l2*l3*l4*m3*t5*t28*u3*5.4*10.0 - l1*l2*l3*l4*m4*t5*t26*u4*1.08*100.0 - l1*l2*l3*l4*m3*t5*t28*u4*1.08*100.0 - l1*l2*l3*l4*m4*t25*t75*u3*(2.7*10.0 / 2.0) - l1*l2*l3*l4*m2*t28*t75*u3*2.7*10.0 + l1*l2*l3*l4*m4*t25*t75*u4*2.7*10.0 - l1*l2*l3*l4*m4*t26*t75*u3*2.7*10.0 - l1*l2*l3*l4*m1*t28*t77*u3*3.6*10.0 + l1*l2*l3*l4*m2*t28*t75*u4*5.4*10.0 - l1*l2*l3*l4*m3*t28*t75*u3*2.7*10.0 + l1*l2*l3*l4*m4*t26*t75*u4*5.4*10.0 + l1*l2*l3*l4*m1*t28*t77*u4*7.2*10.0 - l1*l2*l3*l4*m2*t28*t77*u3*5.4*10.0 + l1*l2*l3*l4*m3*t28*t75*u4*5.4*10.0 + l1*l2*l3*l4*m2*t28*t77*u4*1.08*100.0 - l1*l2*l3*l4*m4*t25*t169*u3*(2.7*10.0 / 2.0) - l1*l2*l3*l4*m2*t28*t169*u3*2.7*10.0 + l1*l2*l3*l4*m4*t25*t169*u4*2.7*10.0 - l1*l2*l3*l4*m4*t26*t169*u3*2.7*10.0 + l1*l2*l3*l4*m2*t28*t169*u4*5.4*10.0 - l1*l2*l3*l4*m3*t28*t169*u3*2.7*10.0 + l1*l2*l3*l4*m4*t26*t169*u4*5.4*10.0 + l1*l2*l3*l4*m3*t28*t169*u4*5.4*10.0 + l1*l2*l3*l4*m2*t28*t182*u3*1.8*10.0 - l1*l2*l3*l4*m2*t28*t182*u4*3.6*10.0 + g*l1*l2*l3*t23*t25*t28*t60*(1.5*10.0 / 8.0) - g*l1*l2*l4*t21*t25*t28*t91*(9.0 / 4.0) - g*l1*l2*l3*t23*t25*t28*t127*(9.0 / 8.0) + g*l1*l2*l3*t23*t25*t28*t145*(4.5*10.0 / 8.0) - g*l1*l2*l3*t23*t25*t28*t146*(4.5*10.0 / 8.0) + g*l1*l2*l3*t23*t25*t28*t147*(1.5*10.0 / 8.0) + g*l1*l2*l4*t21*t25*t28*t165*(9.0 / 4.0) + g*l1*l2*l4*t21*t25*t28*t166*(2.7*10.0 / 4.0) + g*l1*l2*l4*t21*t25*t28*t167*(2.7*10.0 / 4.0) - g*l1*l2*l4*t21*t25*t28*t168*(9.0 / 4.0) + g*l1*l2*l3*t23*t25*t28*t205*(2.7*10.0 / 8.0) - g*l1*l2*l3*t23*t25*t28*t206*(9.0 / 8.0) - g*l1*l2*l4*t21*t25*t28*t218*(2.7*10.0 / 4.0) - g*l1*l2*l4*t21*t25*t28*t219*(2.7*10.0 / 4.0) - g*l1*l2*l4*t21*t25*t28*t220*(9.0 / 4.0) - g*l1*l2*l3*t23*t25*t28*t221*(2.7*10.0 / 8.0) - l1*l3*m1*m3*m4*t4*t23*u2*2.4*10.0 + l1*l3*m1*m3*m4*t4*t23*u3*2.4*10.0 - l1*l3*m2*m3*m4*t4*t23*u2*5.4*10.0 + l1*l3*m2*m3*m4*t4*t23*u3*5.4*10.0 + l1*l2*m1*m3*m4*t21*t34*u4*7.2*10.0 + l1*l2*m2*m3*m4*t21*t33*u4*1.8*100.0 + l1*l2*m2*m3*m4*t21*t34*u4*1.08*100.0 - l1*l2*m2*m3*m4*t23*t33*u3*7.2*10.0 + l1*l2*m2*m3*m4*t23*t33*u4*7.2*10.0 + l1*l4*m1*m3*m4*t21*t40*u2*6.0 - l1*l4*m1*m3*m4*t21*t40*u3*6.0 + l1*l4*m2*m3*m4*t21*t40*u2*(2.7*10.0 / 2.0) - l2*l3*m2*m3*m4*t23*t39*u1*6.0 - l1*l4*m2*m3*m4*t21*t40*u3*(2.7*10.0 / 2.0) + l2*l3*m2*m3*m4*t23*t39*u2*6.0 + l2*l4*m2*m3*m4*t21*t59*u1*(3.0 / 2.0) - l2*l4*m2*m3*m4*t21*t59*u2*(3.0 / 2.0) + l1*l3*m2*m3*m4*t23*t73*u2*1.8*10.0 - l1*l3*m2*m3*m4*t23*t73*u3*1.8*10.0 - l1*l4*m1*m3*m4*t21*t103*u2*1.8*10.0 + l2*l3*m2*m3*m4*t23*t101*u1*1.8*10.0 + l1*l4*m1*m3*m4*t21*t103*u3*1.8*10.0 - l1*l4*m2*m3*m4*t21*t103*u2*(8.1*10.0 / 2.0) - l2*l3*m2*m3*m4*t23*t101*u2*1.8*10.0 + l1*l4*m2*m3*m4*t21*t103*u3*(8.1*10.0 / 2.0) - l1*l4*m2*m3*m4*t21*t119*u2*(9.0 / 2.0) + l1*l4*m2*m3*m4*t21*t119*u3*(9.0 / 2.0) - l1*l2*m2*m3*m4*t21*t130*u4*3.6*10.0 - l2*l4*m2*m3*m4*t21*t142*u1*(9.0 / 2.0) + l2*l4*m2*m3*m4*t21*t142*u2*(9.0 / 2.0) + l2*l4*m2*m3*m4*t21*t143*u1*(2.7*10.0 / 2.0) - l2*l4*m2*m3*m4*t21*t143*u2*(2.7*10.0 / 2.0) - l2*l4*m2*m3*m4*t21*t144*u1*(9.0 / 2.0) + l2*l4*m2*m3*m4*t21*t144*u2*(9.0 / 2.0) + l1*l4*m2*m3*m4*t21*t193*u2*(2.7*10.0 / 2.0) - l1*l4*m2*m3*m4*t21*t193*u3*(2.7*10.0 / 2.0) + l1*l2*l3*m1*t9*t13*t24*t29*3.0 + l1*l2*l4*m4*t9*t13*t22*t27*(9.0 / 2.0) + l1*l2*l3*m1*t9*t14*t24*t29*3.0 + l1*l2*l3*m2*t9*t13*t24*t29*(1.5*10.0 / 2.0) + l1*l2*l4*m4*t9*t14*t22*t27*(9.0 / 2.0) + l1*l2*l3*m1*t9*t15*t24*t29*3.0 + l1*l2*l3*m2*t9*t14*t24*t29*(1.5*10.0 / 2.0) + l1*l2*l3*m3*t9*t13*t24*t29*(9.0 / 2.0) + l1*l2*l4*m4*t9*t15*t22*t27*(9.0 / 2.0) + l1*l2*l3*m1*t9*t16*t24*t29*3.0 + l1*l2*l3*m2*t9*t15*t24*t29*(1.5*10.0 / 2.0) + l1*l2*l3*m3*t9*t14*t24*t29*(9.0 / 2.0) + l1*l2*l3*m2*t9*t16*t24*t29*(1.5*10.0 / 2.0) + l1*l2*l3*m3*t9*t15*t24*t29*(9.0 / 2.0) + l1*l2*l3*m3*t9*t16*t24*t29*(9.0 / 2.0) - l1*l2*l4*m4*t13*t22*t27*t84*(9.0 / 4.0) - l1*l2*l3*m2*t13*t24*t29*t84*(9.0 / 4.0) - l1*l2*l4*m4*t14*t22*t27*t84*(9.0 / 4.0) - l1*l2*l3*m1*t13*t24*t29*t86*3.0 - l1*l2*l3*m2*t14*t24*t29*t84*(9.0 / 4.0) - l1*l2*l3*m3*t13*t24*t29*t84*(9.0 / 4.0) - l1*l2*l4*m4*t15*t22*t27*t84*(9.0 / 4.0) - l1*l2*l3*m1*t14*t24*t29*t86*3.0 - l1*l2*l3*m2*t13*t24*t29*t86*(9.0 / 2.0) - l1*l2*l3*m2*t15*t24*t29*t84*(9.0 / 4.0) - l1*l2*l3*m3*t14*t24*t29*t84*(9.0 / 4.0) - l1*l2*l3*m1*t15*t24*t29*t86*3.0 - l1*l2*l3*m2*t14*t24*t29*t86*(9.0 / 2.0) - l1*l2*l3*m2*t16*t24*t29*t84*(9.0 / 4.0) - l1*l2*l3*m3*t15*t24*t29*t84*(9.0 / 4.0) - l1*l2*l3*m1*t16*t24*t29*t86*3.0 - l1*l2*l3*m2*t15*t24*t29*t86*(9.0 / 2.0) - l1*l2*l3*m3*t16*t24*t29*t84*(9.0 / 4.0) - l1*l2*l3*m2*t16*t24*t29*t86*(9.0 / 2.0) + l1*l2*l4*m4*t13*t22*t27*t171*(9.0 / 4.0) + l1*l2*l3*m2*t13*t24*t29*t171*(9.0 / 4.0) + l1*l2*l4*m4*t14*t22*t27*t171*(9.0 / 4.0) + l1*l2*l3*m2*t14*t24*t29*t171*(9.0 / 4.0) + l1*l2*l3*m3*t13*t24*t29*t171*(9.0 / 4.0) + l1*l2*l4*m4*t15*t22*t27*t171*(9.0 / 4.0) + l1*l2*l3*m2*t15*t24*t29*t171*(9.0 / 4.0) + l1*l2*l3*m3*t14*t24*t29*t171*(9.0 / 4.0) + l1*l2*l3*m2*t16*t24*t29*t171*(9.0 / 4.0) + l1*l2*l3*m3*t15*t24*t29*t171*(9.0 / 4.0) + l1*l2*l3*m3*t16*t24*t29*t171*(9.0 / 4.0) + l1*l2*l3*m2*t13*t24*t29*t188*(3.0 / 2.0) + l1*l2*l3*m2*t14*t24*t29*t188*(3.0 / 2.0) + l1*l2*l3*m2*t15*t24*t29*t188*(3.0 / 2.0) + l1*l2*l3*m2*t16*t24*t29*t188*(3.0 / 2.0) + l1*l2*l4*t9*t13*t22*t25*t28*4.5*10.0 + l1*l2*l3*t9*t13*t24*t25*t28*1.5*10.0 + l1*l2*l4*t9*t13*t22*t26*t28*1.8*10.0 + l1*l2*l4*t9*t14*t22*t25*t28*4.5*10.0 + l1*l2*l3*t9*t13*t24*t26*t28*1.8*10.0 + l1*l2*l3*t9*t14*t24*t25*t28*1.5*10.0 + l1*l2*l4*t9*t14*t22*t26*t28*1.8*10.0 + l1*l2*l4*t9*t15*t22*t25*t28*4.5*10.0 + l1*l2*l3*t9*t14*t24*t26*t28*1.8*10.0 + l1*l2*l3*t9*t15*t24*t25*t28*1.5*10.0 + l1*l2*l4*t9*t15*t22*t26*t28*1.8*10.0 + l1*l2*l3*t9*t15*t24*t26*t28*1.8*10.0 + l1*l2*l3*t9*t16*t24*t25*t28*1.5*10.0 + l1*l2*l3*t9*t16*t24*t26*t28*1.8*10.0 - l1*l2*l4*t13*t22*t25*t28*t84*(2.7*10.0 / 2.0) - l1*l2*l3*t13*t24*t25*t28*t84*(9.0 / 2.0) - l1*l2*l4*t13*t22*t26*t28*t84*9.0 - l1*l2*l4*t14*t22*t25*t28*t84*(2.7*10.0 / 2.0) - l1*l2*l3*t13*t24*t26*t28*t84*9.0 - l1*l2*l3*t14*t24*t25*t28*t84*(9.0 / 2.0) - l1*l2*l4*t14*t22*t26*t28*t84*9.0 - l1*l2*l4*t15*t22*t25*t28*t84*(2.7*10.0 / 2.0) - l1*l2*l3*t14*t24*t26*t28*t84*9.0 - l1*l2*l3*t15*t24*t25*t28*t84*(9.0 / 2.0) - l1*l2*l4*t15*t22*t26*t28*t84*9.0 - l1*l2*l3*t15*t24*t26*t28*t84*9.0 - l1*l2*l3*t16*t24*t25*t28*t84*(9.0 / 2.0) - l1*l2*l3*t16*t24*t26*t28*t84*9.0 + l1*l2*l4*t13*t22*t25*t28*t171*(2.7*10.0 / 2.0) + l1*l2*l3*t13*t24*t25*t28*t171*(9.0 / 2.0) + l1*l2*l4*t13*t22*t26*t28*t171*9.0 + l1*l2*l4*t14*t22*t25*t28*t171*(2.7*10.0 / 2.0) + l1*l2*l3*t13*t24*t26*t28*t171*9.0 + l1*l2*l3*t14*t24*t25*t28*t171*(9.0 / 2.0) + l1*l2*l4*t14*t22*t26*t28*t171*9.0 + l1*l2*l4*t15*t22*t25*t28*t171*(2.7*10.0 / 2.0) + l1*l2*l3*t14*t24*t26*t28*t171*9.0 + l1*l2*l3*t15*t24*t25*t28*t171*(9.0 / 2.0) + l1*l2*l4*t15*t22*t26*t28*t171*9.0 + l1*l2*l3*t15*t24*t26*t28*t171*9.0 + l1*l2*l3*t16*t24*t25*t28*t171*(9.0 / 2.0) + l1*l2*l3*t16*t24*t26*t28*t171*9.0 - l1*l3*m1*t8*t13*t19*t23*t29*1.2*10.0 - l1*l3*m1*t8*t14*t19*t23*t29*1.2*10.0 - l1*l3*m2*t8*t13*t19*t23*t29*1.5*10.0 - l1*l3*m2*t8*t14*t19*t23*t29*1.5*10.0 - l1*l2*m1*t13*t21*t23*t29*t37*6.0 - l1*l2*m1*t14*t21*t23*t29*t37*6.0 - l1*l2*m2*t13*t21*t23*t29*t37*9.0 - l1*l2*m1*t15*t21*t23*t29*t37*6.0 - l1*l2*m2*t14*t21*t23*t29*t37*9.0 - l1*l2*m2*t15*t21*t23*t29*t37*9.0 - l2*l3*m1*t13*t17*t23*t29*t42*6.0 - l2*l3*m2*t13*t17*t23*t29*t42*3.0 + l1*l3*m2*t13*t19*t23*t29*t82*3.0 + l1*l3*m2*t14*t19*t23*t29*t82*3.0 + l2*l3*m1*t13*t17*t23*t29*t106*6.0 + l2*l3*m2*t13*t17*t23*t29*t106*9.0 + l1*l2*m2*t13*t21*t23*t29*t133*3.0 + l1*l2*m2*t14*t21*t23*t29*t133*3.0 + l1*l2*m2*t15*t21*t23*t29*t133*3.0 - l1*l3*t8*t13*t19*t23*t25*t28*(4.5*10.0 / 4.0) - l1*l3*t8*t14*t19*t23*t25*t28*(4.5*10.0 / 4.0) + l1*l2*t13*t21*t23*t25*t28*t38*(4.5*10.0 / 2.0) + l1*l2*t13*t21*t23*t26*t28*t38*(2.7*10.0 / 2.0) + l1*l2*t14*t21*t23*t25*t28*t38*(4.5*10.0 / 2.0) + l1*l2*t14*t21*t23*t26*t28*t38*(2.7*10.0 / 2.0) + l1*l2*t15*t21*t23*t25*t28*t38*(4.5*10.0 / 2.0) + l2*l3*t13*t17*t23*t25*t28*t42*(1.5*10.0 / 4.0) + l1*l2*t15*t21*t23*t26*t28*t38*(2.7*10.0 / 2.0) + l1*l2*t16*t21*t23*t25*t28*t38*(4.5*10.0 / 4.0) + l1*l4*t13*t19*t21*t25*t28*t43*(2.7*10.0 / 2.0) + l1*l2*t16*t21*t23*t26*t28*t38*(2.7*10.0 / 4.0) + l1*l4*t14*t19*t21*t25*t28*t43*(2.7*10.0 / 2.0) - l2*l4*t13*t17*t21*t25*t28*t61*(9.0 / 2.0) + l1*l3*t13*t19*t23*t25*t28*t82*(1.5*10.0 / 4.0) + l1*l3*t14*t19*t23*t25*t28*t82*(1.5*10.0 / 4.0) + l1*l3*t13*t19*t23*t25*t28*t85*(2.7*10.0 / 4.0) + l1*l3*t14*t19*t23*t25*t28*t85*(2.7*10.0 / 4.0) + l2*l3*t13*t17*t23*t25*t28*t106*(4.5*10.0 / 4.0) - l1*l4*t13*t19*t21*t25*t28*t109*(2.7*10.0 / 2.0) - l1*l4*t14*t19*t21*t25*t28*t109*(2.7*10.0 / 2.0) - l2*l3*t13*t17*t23*t25*t28*t122*(9.0 / 4.0) - l1*l4*t13*t19*t21*t25*t28*t123*(9.0 / 2.0) - l1*l4*t14*t19*t21*t25*t28*t123*(9.0 / 2.0) - l1*l2*t13*t21*t23*t25*t28*t134*(2.7*10.0 / 4.0) - l1*l2*t13*t21*t23*t26*t28*t134*(2.7*10.0 / 4.0) - l1*l2*t14*t21*t23*t25*t28*t134*(2.7*10.0 / 4.0) - l1*l2*t14*t21*t23*t26*t28*t134*(2.7*10.0 / 4.0) - l1*l2*t15*t21*t23*t25*t28*t134*(2.7*10.0 / 4.0) - l1*l2*t15*t21*t23*t26*t28*t134*(2.7*10.0 / 4.0) - l1*l2*t16*t21*t23*t25*t28*t134*(2.7*10.0 / 8.0) - l1*l2*t16*t21*t23*t26*t28*t134*(2.7*10.0 / 8.0) + l2*l4*t13*t17*t21*t25*t28*t149*(9.0 / 2.0) + l2*l4*t13*t17*t21*t25*t28*t150*(2.7*10.0 / 2.0) + l2*l4*t13*t17*t21*t25*t28*t151*(2.7*10.0 / 2.0) + l1*l2*t13*t21*t23*t25*t28*t172*(2.7*10.0 / 4.0) + l1*l2*t13*t21*t23*t26*t28*t172*(2.7*10.0 / 4.0) + l1*l2*t14*t21*t23*t25*t28*t172*(2.7*10.0 / 4.0) + l1*l2*t14*t21*t23*t26*t28*t172*(2.7*10.0 / 4.0) + l1*l2*t15*t21*t23*t25*t28*t172*(2.7*10.0 / 4.0) + l1*l2*t15*t21*t23*t26*t28*t172*(2.7*10.0 / 4.0) + l1*l2*t16*t21*t23*t25*t28*t172*(2.7*10.0 / 8.0) + l1*l2*t16*t21*t23*t26*t28*t172*(2.7*10.0 / 8.0) - l1*l3*t13*t19*t23*t25*t28*t187*(9.0 / 4.0) - l1*l3*t14*t19*t23*t25*t28*t187*(9.0 / 4.0) + l2*l3*t13*t17*t23*t25*t28*t198*(2.7*10.0 / 4.0) + l1*l4*t13*t19*t21*t25*t28*t199*(9.0 / 2.0) + l1*l4*t14*t19*t21*t25*t28*t199*(9.0 / 2.0) + dth1*dth2*l1*l2*l3*m1*t9*t24*t29*6.0 + dth1*dth2*l1*l2*l4*m4*t9*t22*t27*9.0 + dth1*dth2*l1*l2*l3*m2*t9*t24*t29*1.5*10.0 + dth1*dth3*l1*l2*l3*m1*t9*t24*t29*6.0 + dth1*dth3*l1*l2*l4*m4*t9*t22*t27*9.0 + dth1*dth2*l1*l2*l3*m3*t9*t24*t29*9.0 + dth1*dth3*l1*l2*l3*m2*t9*t24*t29*1.5*10.0 + dth1*dth4*l1*l2*l3*m1*t9*t24*t29*6.0 + dth2*dth3*l1*l2*l3*m1*t9*t24*t29*6.0 + dth2*dth3*l1*l2*l4*m4*t9*t22*t27*9.0 + dth1*dth3*l1*l2*l3*m3*t9*t24*t29*9.0 + dth1*dth4*l1*l2*l3*m2*t9*t24*t29*1.5*10.0 + dth2*dth3*l1*l2*l3*m2*t9*t24*t29*1.5*10.0 + dth2*dth4*l1*l2*l3*m1*t9*t24*t29*6.0 + dth1*dth4*l1*l2*l3*m3*t9*t24*t29*9.0 + dth2*dth3*l1*l2*l3*m3*t9*t24*t29*9.0 + dth2*dth4*l1*l2*l3*m2*t9*t24*t29*1.5*10.0 + dth3*dth4*l1*l2*l3*m1*t9*t24*t29*6.0 + dth2*dth4*l1*l2*l3*m3*t9*t24*t29*9.0 + dth3*dth4*l1*l2*l3*m2*t9*t24*t29*1.5*10.0 + dth3*dth4*l1*l2*l3*m3*t9*t24*t29*9.0 - dth1*dth2*l1*l2*l4*m4*t22*t27*t84*(9.0 / 2.0) - dth1*dth2*l1*l2*l3*m2*t24*t29*t84*(9.0 / 2.0) - dth1*dth3*l1*l2*l4*m4*t22*t27*t84*(9.0 / 2.0) - dth1*dth2*l1*l2*l3*m1*t24*t29*t86*6.0 - dth1*dth2*l1*l2*l3*m3*t24*t29*t84*(9.0 / 2.0) - dth1*dth3*l1*l2*l3*m2*t24*t29*t84*(9.0 / 2.0) - dth2*dth3*l1*l2*l4*m4*t22*t27*t84*(9.0 / 2.0) - dth1*dth2*l1*l2*l3*m2*t24*t29*t86*9.0 - dth1*dth3*l1*l2*l3*m1*t24*t29*t86*6.0 - dth1*dth3*l1*l2*l3*m3*t24*t29*t84*(9.0 / 2.0) - dth1*dth4*l1*l2*l3*m2*t24*t29*t84*(9.0 / 2.0) - dth2*dth3*l1*l2*l3*m2*t24*t29*t84*(9.0 / 2.0) - dth1*dth3*l1*l2*l3*m2*t24*t29*t86*9.0 - dth1*dth4*l1*l2*l3*m1*t24*t29*t86*6.0 - dth1*dth4*l1*l2*l3*m3*t24*t29*t84*(9.0 / 2.0) - dth2*dth3*l1*l2*l3*m1*t24*t29*t86*6.0 - dth2*dth3*l1*l2*l3*m3*t24*t29*t84*(9.0 / 2.0) - dth2*dth4*l1*l2*l3*m2*t24*t29*t84*(9.0 / 2.0) - dth1*dth4*l1*l2*l3*m2*t24*t29*t86*9.0 - dth2*dth3*l1*l2*l3*m2*t24*t29*t86*9.0 - dth2*dth4*l1*l2*l3*m1*t24*t29*t86*6.0 - dth2*dth4*l1*l2*l3*m3*t24*t29*t84*(9.0 / 2.0) - dth3*dth4*l1*l2*l3*m2*t24*t29*t84*(9.0 / 2.0) - dth2*dth4*l1*l2*l3*m2*t24*t29*t86*9.0 - dth3*dth4*l1*l2*l3*m1*t24*t29*t86*6.0 - dth3*dth4*l1*l2*l3*m3*t24*t29*t84*(9.0 / 2.0) - dth3*dth4*l1*l2*l3*m2*t24*t29*t86*9.0 + dth1*dth2*l1*l2*l4*m4*t22*t27*t171*(9.0 / 2.0) + dth1*dth2*l1*l2*l3*m2*t24*t29*t171*(9.0 / 2.0) + dth1*dth3*l1*l2*l4*m4*t22*t27*t171*(9.0 / 2.0) + dth1*dth2*l1*l2*l3*m3*t24*t29*t171*(9.0 / 2.0) + dth1*dth3*l1*l2*l3*m2*t24*t29*t171*(9.0 / 2.0) + dth2*dth3*l1*l2*l4*m4*t22*t27*t171*(9.0 / 2.0) + dth1*dth3*l1*l2*l3*m3*t24*t29*t171*(9.0 / 2.0) + dth1*dth4*l1*l2*l3*m2*t24*t29*t171*(9.0 / 2.0) + dth2*dth3*l1*l2*l3*m2*t24*t29*t171*(9.0 / 2.0) + dth1*dth4*l1*l2*l3*m3*t24*t29*t171*(9.0 / 2.0) + dth2*dth3*l1*l2*l3*m3*t24*t29*t171*(9.0 / 2.0) + dth2*dth4*l1*l2*l3*m2*t24*t29*t171*(9.0 / 2.0) + dth2*dth4*l1*l2*l3*m3*t24*t29*t171*(9.0 / 2.0) + dth3*dth4*l1*l2*l3*m2*t24*t29*t171*(9.0 / 2.0) + dth3*dth4*l1*l2*l3*m3*t24*t29*t171*(9.0 / 2.0) + dth1*dth2*l1*l2*l3*m2*t24*t29*t188*3.0 + dth1*dth3*l1*l2*l3*m2*t24*t29*t188*3.0 + dth1*dth4*l1*l2*l3*m2*t24*t29*t188*3.0 + dth2*dth3*l1*l2*l3*m2*t24*t29*t188*3.0 + dth2*dth4*l1*l2*l3*m2*t24*t29*t188*3.0 + dth3*dth4*l1*l2*l3*m2*t24*t29*t188*3.0 + dth1*dth2*l1*l2*l4*t9*t22*t25*t28*9.0*10.0 + dth1*dth2*l1*l2*l3*t9*t24*t25*t28*3.0*10.0 + dth1*dth2*l1*l2*l4*t9*t22*t26*t28*3.6*10.0 + dth1*dth3*l1*l2*l4*t9*t22*t25*t28*9.0*10.0 + dth1*dth2*l1*l2*l3*t9*t24*t26*t28*3.6*10.0 + dth1*dth3*l1*l2*l3*t9*t24*t25*t28*3.0*10.0 + dth1*dth3*l1*l2*l4*t9*t22*t26*t28*3.6*10.0 + dth2*dth3*l1*l2*l4*t9*t22*t25*t28*9.0*10.0 + dth1*dth3*l1*l2*l3*t9*t24*t26*t28*3.6*10.0 + dth1*dth4*l1*l2*l3*t9*t24*t25*t28*3.0*10.0 + dth2*dth3*l1*l2*l3*t9*t24*t25*t28*3.0*10.0 + dth2*dth3*l1*l2*l4*t9*t22*t26*t28*3.6*10.0 + dth1*dth4*l1*l2*l3*t9*t24*t26*t28*3.6*10.0 + dth2*dth3*l1*l2*l3*t9*t24*t26*t28*3.6*10.0 + dth2*dth4*l1*l2*l3*t9*t24*t25*t28*3.0*10.0 + dth2*dth4*l1*l2*l3*t9*t24*t26*t28*3.6*10.0 + dth3*dth4*l1*l2*l3*t9*t24*t25*t28*3.0*10.0 + dth3*dth4*l1*l2*l3*t9*t24*t26*t28*3.6*10.0 - dth1*dth2*l1*l2*l4*t22*t25*t28*t84*2.7*10.0 - dth1*dth2*l1*l2*l3*t24*t25*t28*t84*9.0 - dth1*dth2*l1*l2*l4*t22*t26*t28*t84*1.8*10.0 - dth1*dth3*l1*l2*l4*t22*t25*t28*t84*2.7*10.0 - dth1*dth2*l1*l2*l3*t24*t26*t28*t84*1.8*10.0 - dth1*dth3*l1*l2*l3*t24*t25*t28*t84*9.0 - dth1*dth3*l1*l2*l4*t22*t26*t28*t84*1.8*10.0 - dth2*dth3*l1*l2*l4*t22*t25*t28*t84*2.7*10.0 - dth1*dth3*l1*l2*l3*t24*t26*t28*t84*1.8*10.0 - dth1*dth4*l1*l2*l3*t24*t25*t28*t84*9.0 - dth2*dth3*l1*l2*l3*t24*t25*t28*t84*9.0 - dth2*dth3*l1*l2*l4*t22*t26*t28*t84*1.8*10.0 - dth1*dth4*l1*l2*l3*t24*t26*t28*t84*1.8*10.0 - dth2*dth3*l1*l2*l3*t24*t26*t28*t84*1.8*10.0 - dth2*dth4*l1*l2*l3*t24*t25*t28*t84*9.0 - dth2*dth4*l1*l2*l3*t24*t26*t28*t84*1.8*10.0 - dth3*dth4*l1*l2*l3*t24*t25*t28*t84*9.0 - dth3*dth4*l1*l2*l3*t24*t26*t28*t84*1.8*10.0 + dth1*dth2*l1*l2*l4*t22*t25*t28*t171*2.7*10.0 + dth1*dth2*l1*l2*l3*t24*t25*t28*t171*9.0 + dth1*dth2*l1*l2*l4*t22*t26*t28*t171*1.8*10.0 + dth1*dth3*l1*l2*l4*t22*t25*t28*t171*2.7*10.0 + dth1*dth2*l1*l2*l3*t24*t26*t28*t171*1.8*10.0 + dth1*dth3*l1*l2*l3*t24*t25*t28*t171*9.0 + dth1*dth3*l1*l2*l4*t22*t26*t28*t171*1.8*10.0 + dth2*dth3*l1*l2*l4*t22*t25*t28*t171*2.7*10.0 + dth1*dth3*l1*l2*l3*t24*t26*t28*t171*1.8*10.0 + dth1*dth4*l1*l2*l3*t24*t25*t28*t171*9.0 + dth2*dth3*l1*l2*l3*t24*t25*t28*t171*9.0 + dth2*dth3*l1*l2*l4*t22*t26*t28*t171*1.8*10.0 + dth1*dth4*l1*l2*l3*t24*t26*t28*t171*1.8*10.0 + dth2*dth3*l1*l2*l3*t24*t26*t28*t171*1.8*10.0 + dth2*dth4*l1*l2*l3*t24*t25*t28*t171*9.0 + dth2*dth4*l1*l2*l3*t24*t26*t28*t171*1.8*10.0 + dth3*dth4*l1*l2*l3*t24*t25*t28*t171*9.0 + dth3*dth4*l1*l2*l3*t24*t26*t28*t171*1.8*10.0 - dth1*dth2*l1*l3*m1*t8*t19*t23*t29*2.4*10.0 - dth1*dth2*l1*l3*m2*t8*t19*t23*t29*3.0*10.0 - dth1*dth2*l1*l2*m1*t21*t23*t29*t37*1.2*10.0 - dth1*dth2*l1*l2*m2*t21*t23*t29*t37*1.8*10.0 - dth1*dth3*l1*l2*m1*t21*t23*t29*t37*1.2*10.0 - dth1*dth3*l1*l2*m2*t21*t23*t29*t37*1.8*10.0 - dth2*dth3*l1*l2*m1*t21*t23*t29*t37*1.2*10.0 - dth2*dth3*l1*l2*m2*t21*t23*t29*t37*1.8*10.0 + dth1*dth2*l1*l3*m2*t19*t23*t29*t82*6.0 + dth1*dth2*l1*l2*m2*t21*t23*t29*t133*6.0 + dth1*dth3*l1*l2*m2*t21*t23*t29*t133*6.0 + dth2*dth3*l1*l2*m2*t21*t23*t29*t133*6.0 + l1*l2*l3*l4*m1*m2*m4*t5*u3*2.4*10.0 - l1*l2*l3*l4*m1*m2*m4*t5*u4*4.8*10.0 + l1*l2*l3*l4*m1*m3*m4*t5*u3*5.4*10.0 - l1*l2*l3*l4*m1*m3*m4*t5*u4*1.08*100.0 + l1*l2*l3*l4*m2*m3*m4*t5*u3*1.35*100.0 - l1*l2*l3*l4*m2*m3*m4*t5*u4*2.7*100.0 - l1*l2*l3*l4*m2*m3*m4*t75*u3*(8.1*10.0 / 2.0) - l1*l2*l3*l4*m1*m3*m4*t77*u3*1.8*10.0 + l1*l2*l3*l4*m2*m3*m4*t75*u4*8.1*10.0 + l1*l2*l3*l4*m1*m3*m4*t77*u4*3.6*10.0 - l1*l2*l3*l4*m2*m3*m4*t77*u3*2.7*10.0 + l1*l2*l3*l4*m2*m3*m4*t77*u4*5.4*10.0 - l1*l2*l3*l4*m2*m3*m4*t169*u3*(8.1*10.0 / 2.0) + l1*l2*l3*l4*m2*m3*m4*t169*u4*8.1*10.0 + l1*l2*l3*l4*m2*m3*m4*t182*u3*9.0 - l1*l2*l3*l4*m2*m3*m4*t182*u4*1.8*10.0 - g*l1*l2*l3*m1*m2*t23*t28*t60*(5.0 / 8.0) - g*l1*l2*l3*m1*m4*t23*t26*t60*3.0 - g*l1*l2*l3*m1*m3*t23*t28*t60*(4.5*10.0 / 8.0) - g*l1*l2*l3*m2*m4*t23*t26*t60*3.0 + g*l1*l2*l3*m3*m4*t23*t25*t60*(3.0 / 2.0) - g*l1*l2*l3*m2*m3*t23*t28*t60*(4.5*10.0 / 8.0) + g*l1*l2*l4*m1*m2*t21*t28*t91*(3.0 / 4.0) - g*l1*l2*l4*m1*m4*t21*t26*t91*(3.0 / 8.0) + g*l1*l2*l4*m1*m3*t21*t28*t91*(3.0 / 4.0) - g*l1*l2*l4*m2*m4*t21*t26*t91*(3.0 / 8.0) - g*l1*l2*l4*m3*m4*t21*t25*t91*(3.0 / 8.0) + g*l1*l2*l4*m2*m3*t21*t28*t91*(3.0 / 4.0) + g*l1*l2*l3*m1*m2*t23*t28*t127*(3.0 / 8.0) + g*l1*l2*l3*m1*m3*t23*t28*t127*(9.0 / 8.0) + g*l1*l2*l3*m2*m3*t23*t28*t127*(9.0 / 8.0) + g*l1*l2*l3*m1*m2*t23*t28*t145*(1.5*10.0 / 8.0) + g*l1*l2*l3*m1*m4*t23*t26*t145*3.0 - g*l1*l2*l3*m1*m2*t23*t28*t146*(4.5*10.0 / 8.0) + g*l1*l2*l3*m1*m3*t23*t28*t145*(4.5*10.0 / 8.0) - g*l1*l2*l3*m1*m4*t23*t26*t146*9.0 + g*l1*l2*l3*m2*m4*t23*t26*t145*9.0 + g*l1*l2*l3*m3*m4*t23*t25*t145*(9.0 / 2.0) - g*l1*l2*l3*m1*m2*t23*t28*t147*(1.5*10.0 / 8.0) - g*l1*l2*l3*m1*m3*t23*t28*t146*(1.35*100.0 / 8.0) - g*l1*l2*l3*m1*m4*t23*t26*t147*9.0 + g*l1*l2*l3*m2*m3*t23*t28*t145*(1.35*100.0 / 8.0) - g*l1*l2*l3*m2*m4*t23*t26*t146*9.0 - g*l1*l2*l3*m3*m4*t23*t25*t146*(9.0 / 2.0) - g*l1*l2*l3*m1*m3*t23*t28*t147*(1.35*100.0 / 8.0) - g*l1*l2*l3*m2*m3*t23*t28*t146*(1.35*100.0 / 8.0) - g*l1*l2*l3*m2*m4*t23*t26*t147*3.0 + g*l1*l2*l3*m3*m4*t23*t25*t147*(3.0 / 2.0) - g*l1*l2*l3*m2*m3*t23*t28*t147*(4.5*10.0 / 8.0) - g*l1*l2*l4*m1*m2*t21*t28*t165*(3.0 / 4.0) - g*l1*l2*l4*m1*m4*t21*t26*t165*(9.0 / 8.0) + g*l1*l2*l4*m1*m2*t21*t28*t166*(9.0 / 4.0) - g*l1*l2*l4*m1*m3*t21*t28*t165*(9.0 / 4.0) + g*l1*l2*l4*m1*m4*t21*t26*t166*(9.0 / 8.0) - g*l1*l2*l4*m2*m4*t21*t26*t165*(9.0 / 8.0) + g*l1*l2*l4*m3*m4*t21*t25*t165*(9.0 / 8.0) + g*l1*l2*l4*m1*m2*t21*t28*t167*(2.7*10.0 / 4.0) + g*l1*l2*l4*m1*m3*t21*t28*t166*(9.0 / 4.0) - g*l1*l2*l4*m1*m4*t21*t26*t167*(9.0 / 8.0) - g*l1*l2*l4*m2*m3*t21*t28*t165*(9.0 / 4.0) + g*l1*l2*l4*m2*m4*t21*t26*t166*(2.7*10.0 / 8.0) + g*l1*l2*l4*m3*m4*t21*t25*t166*(2.7*10.0 / 8.0) + g*l1*l2*l4*m1*m2*t21*t28*t168*(9.0 / 4.0) + g*l1*l2*l4*m1*m3*t21*t28*t167*(9.0 / 4.0) - g*l1*l2*l4*m1*m4*t21*t26*t168*(9.0 / 8.0) + g*l1*l2*l4*m2*m3*t21*t28*t166*(2.7*10.0 / 4.0) - g*l1*l2*l4*m2*m4*t21*t26*t167*(9.0 / 8.0) + g*l1*l2*l4*m3*m4*t21*t25*t167*(9.0 / 8.0) + g*l1*l2*l4*m1*m3*t21*t28*t168*(9.0 / 4.0) + g*l1*l2*l4*m2*m3*t21*t28*t167*(9.0 / 4.0) - g*l1*l2*l4*m2*m4*t21*t26*t168*(3.0 / 8.0) - g*l1*l2*l4*m3*m4*t21*t25*t168*(3.0 / 8.0) + g*l1*l2*l4*m2*m3*t21*t28*t168*(3.0 / 4.0) + g*l1*l2*l3*m1*m2*t23*t28*t205*(2.7*10.0 / 8.0) + g*l1*l2*l3*m1*m2*t23*t28*t206*(9.0 / 8.0) + g*l1*l2*l3*m1*m3*t23*t28*t205*(2.7*10.0 / 8.0) + g*l1*l2*l3*m1*m3*t23*t28*t206*(2.7*10.0 / 8.0) + g*l1*l2*l3*m2*m3*t23*t28*t205*(2.7*10.0 / 8.0) + g*l1*l2*l3*m2*m3*t23*t28*t206*(9.0 / 8.0) - g*l1*l2*l4*m1*m2*t21*t28*t218*(9.0 / 4.0) + g*l1*l2*l4*m1*m4*t21*t26*t218*(3.0 / 8.0) - g*l1*l2*l4*m1*m2*t21*t28*t219*(2.7*10.0 / 4.0) - g*l1*l2*l4*m1*m3*t21*t28*t218*(3.0 / 4.0) - g*l1*l2*l4*m1*m4*t21*t26*t219*(2.7*10.0 / 8.0) + g*l1*l2*l4*m2*m4*t21*t26*t218*(9.0 / 8.0) - g*l1*l2*l4*m3*m4*t21*t25*t218*(9.0 / 8.0) + g*l1*l2*l4*m1*m2*t21*t28*t220*(9.0 / 4.0) - g*l1*l2*l4*m1*m3*t21*t28*t219*(2.7*10.0 / 4.0) + g*l1*l2*l4*m1*m4*t21*t26*t220*(2.7*10.0 / 8.0) - g*l1*l2*l4*m2*m3*t21*t28*t218*(9.0 / 4.0) - g*l1*l2*l4*m2*m4*t21*t26*t219*(2.7*10.0 / 8.0) - g*l1*l2*l4*m3*m4*t21*t25*t219*(2.7*10.0 / 8.0) + g*l1*l2*l4*m1*m3*t21*t28*t220*(2.7*10.0 / 4.0) - g*l1*l2*l4*m2*m3*t21*t28*t219*(2.7*10.0 / 4.0) + g*l1*l2*l4*m2*m4*t21*t26*t220*(9.0 / 8.0) - g*l1*l2*l4*m3*m4*t21*t25*t220*(9.0 / 8.0) - g*l1*l2*l3*m1*m2*t23*t28*t221*(9.0 / 8.0) + g*l1*l2*l4*m2*m3*t21*t28*t220*(9.0 / 4.0) - g*l1*l2*l3*m1*m3*t23*t28*t221*(9.0 / 8.0) - g*l1*l2*l3*m2*m3*t23*t28*t221*(2.7*10.0 / 8.0) - dth1*dth2*l1*l3*t8*t19*t23*t25*t28*(4.5*10.0 / 2.0) + dth1*dth2*l1*l2*t21*t23*t25*t28*t38*4.5*10.0 + dth1*dth2*l1*l2*t21*t23*t26*t28*t38*2.7*10.0 + dth1*dth3*l1*l2*t21*t23*t25*t28*t38*4.5*10.0 + dth1*dth3*l1*l2*t21*t23*t26*t28*t38*2.7*10.0 + dth1*dth4*l1*l2*t21*t23*t25*t28*t38*(4.5*10.0 / 2.0) + dth2*dth3*l1*l2*t21*t23*t25*t28*t38*4.5*10.0 + dth1*dth2*l1*l4*t19*t21*t25*t28*t43*2.7*10.0 + dth1*dth4*l1*l2*t21*t23*t26*t28*t38*(2.7*10.0 / 2.0) + dth2*dth3*l1*l2*t21*t23*t26*t28*t38*2.7*10.0 + dth2*dth4*l1*l2*t21*t23*t25*t28*t38*(4.5*10.0 / 2.0) + dth2*dth4*l1*l2*t21*t23*t26*t28*t38*(2.7*10.0 / 2.0) + dth3*dth4*l1*l2*t21*t23*t25*t28*t38*(4.5*10.0 / 2.0) + dth3*dth4*l1*l2*t21*t23*t26*t28*t38*(2.7*10.0 / 2.0) + dth1*dth2*l1*l3*t19*t23*t25*t28*t82*(1.5*10.0 / 2.0) + dth1*dth2*l1*l3*t19*t23*t25*t28*t85*(2.7*10.0 / 2.0) - dth1*dth2*l1*l4*t19*t21*t25*t28*t109*2.7*10.0 - dth1*dth2*l1*l4*t19*t21*t25*t28*t123*9.0 - dth1*dth2*l1*l2*t21*t23*t25*t28*t134*(2.7*10.0 / 2.0) - dth1*dth2*l1*l2*t21*t23*t26*t28*t134*(2.7*10.0 / 2.0) - dth1*dth3*l1*l2*t21*t23*t25*t28*t134*(2.7*10.0 / 2.0) - dth1*dth3*l1*l2*t21*t23*t26*t28*t134*(2.7*10.0 / 2.0) - dth1*dth4*l1*l2*t21*t23*t25*t28*t134*(2.7*10.0 / 4.0) - dth2*dth3*l1*l2*t21*t23*t25*t28*t134*(2.7*10.0 / 2.0) - dth1*dth4*l1*l2*t21*t23*t26*t28*t134*(2.7*10.0 / 4.0) - dth2*dth3*l1*l2*t21*t23*t26*t28*t134*(2.7*10.0 / 2.0) - dth2*dth4*l1*l2*t21*t23*t25*t28*t134*(2.7*10.0 / 4.0) - dth2*dth4*l1*l2*t21*t23*t26*t28*t134*(2.7*10.0 / 4.0) - dth3*dth4*l1*l2*t21*t23*t25*t28*t134*(2.7*10.0 / 4.0) - dth3*dth4*l1*l2*t21*t23*t26*t28*t134*(2.7*10.0 / 4.0) + dth1*dth2*l1*l2*t21*t23*t25*t28*t172*(2.7*10.0 / 2.0) + dth1*dth2*l1*l2*t21*t23*t26*t28*t172*(2.7*10.0 / 2.0) + dth1*dth3*l1*l2*t21*t23*t25*t28*t172*(2.7*10.0 / 2.0) + dth1*dth3*l1*l2*t21*t23*t26*t28*t172*(2.7*10.0 / 2.0) + dth1*dth4*l1*l2*t21*t23*t25*t28*t172*(2.7*10.0 / 4.0) + dth2*dth3*l1*l2*t21*t23*t25*t28*t172*(2.7*10.0 / 2.0) + dth1*dth4*l1*l2*t21*t23*t26*t28*t172*(2.7*10.0 / 4.0) + dth2*dth3*l1*l2*t21*t23*t26*t28*t172*(2.7*10.0 / 2.0) + dth2*dth4*l1*l2*t21*t23*t25*t28*t172*(2.7*10.0 / 4.0) + dth2*dth4*l1*l2*t21*t23*t26*t28*t172*(2.7*10.0 / 4.0) + dth3*dth4*l1*l2*t21*t23*t25*t28*t172*(2.7*10.0 / 4.0) + dth3*dth4*l1*l2*t21*t23*t26*t28*t172*(2.7*10.0 / 4.0) - dth1*dth2*l1*l3*t19*t23*t25*t28*t187*(9.0 / 2.0) + dth1*dth2*l1*l4*t19*t21*t25*t28*t199*9.0 + l1*l2*l4*m1*m2*t9*t13*t22*t28*2.4*10.0 + l1*l2*l4*m1*m4*t9*t13*t22*t26*9.0 + l1*l2*l3*m1*m2*t9*t13*t24*t28*8.0 + l1*l2*l4*m1*m2*t9*t14*t22*t28*2.4*10.0 + l1*l2*l4*m1*m3*t9*t13*t22*t28*3.0*10.0 + l1*l2*l4*m1*m4*t9*t14*t22*t26*9.0 + l1*l2*l4*m2*m4*t9*t13*t22*t26*(4.5*10.0 / 2.0) + l1*l2*l4*m3*m4*t9*t13*t22*t25*1.5*10.0 + l1*l2*l3*m1*m2*t9*t14*t24*t28*8.0 + l1*l2*l3*m1*m3*t9*t13*t24*t28*1.8*10.0 + l1*l2*l4*m1*m2*t9*t15*t22*t28*2.4*10.0 + l1*l2*l4*m1*m3*t9*t14*t22*t28*3.0*10.0 + l1*l2*l4*m1*m4*t9*t15*t22*t26*9.0 + l1*l2*l4*m2*m3*t9*t13*t22*t28*7.5*10.0 + l1*l2*l4*m2*m4*t9*t14*t22*t26*(4.5*10.0 / 2.0) + l1*l2*l4*m3*m4*t9*t14*t22*t25*1.5*10.0 + l1*l2*l3*m1*m2*t9*t15*t24*t28*8.0 + l1*l2*l3*m1*m3*t9*t14*t24*t28*1.8*10.0 + l1*l2*l3*m2*m3*t9*t13*t24*t28*4.5*10.0 + l1*l2*l4*m1*m3*t9*t15*t22*t28*3.0*10.0 + l1*l2*l4*m2*m3*t9*t14*t22*t28*7.5*10.0 + l1*l2*l4*m2*m4*t9*t15*t22*t26*(4.5*10.0 / 2.0) + l1*l2*l4*m3*m4*t9*t15*t22*t25*1.5*10.0 + l1*l2*l3*m1*m2*t9*t16*t24*t28*8.0 + l1*l2*l3*m1*m3*t9*t15*t24*t28*1.8*10.0 + l1*l2*l3*m2*m3*t9*t14*t24*t28*4.5*10.0 + l1*l2*l4*m2*m3*t9*t15*t22*t28*7.5*10.0 + l1*l2*l3*m1*m3*t9*t16*t24*t28*1.8*10.0 + l1*l2*l3*m2*m3*t9*t15*t24*t28*4.5*10.0 + l1*l2*l3*m2*m3*t9*t16*t24*t28*4.5*10.0 - l1*l2*l4*m2*m4*t13*t22*t26*t84*(2.7*10.0 / 4.0) - l1*l2*l4*m3*m4*t13*t22*t25*t84*(9.0 / 2.0) - l1*l2*l4*m1*m4*t13*t22*t26*t86*3.0 - l1*l2*l4*m2*m3*t13*t22*t28*t84*(4.5*10.0 / 2.0) - l1*l2*l4*m2*m4*t14*t22*t26*t84*(2.7*10.0 / 4.0) - l1*l2*l4*m3*m4*t14*t22*t25*t84*(9.0 / 2.0) - l1*l2*l3*m2*m3*t13*t24*t28*t84*(2.7*10.0 / 2.0) - l1*l2*l4*m1*m3*t13*t22*t28*t86*6.0 - l1*l2*l4*m1*m4*t14*t22*t26*t86*3.0 - l1*l2*l4*m2*m3*t14*t22*t28*t84*(4.5*10.0 / 2.0) - l1*l2*l4*m2*m4*t13*t22*t26*t86*(9.0 / 2.0) - l1*l2*l4*m2*m4*t15*t22*t26*t84*(2.7*10.0 / 4.0) - l1*l2*l4*m3*m4*t15*t22*t25*t84*(9.0 / 2.0) - l1*l2*l3*m1*m3*t13*t24*t28*t86*6.0 - l1*l2*l3*m2*m3*t14*t24*t28*t84*(2.7*10.0 / 2.0) - l1*l2*l4*m1*m3*t14*t22*t28*t86*6.0 - l1*l2*l4*m1*m4*t15*t22*t26*t86*3.0 - l1*l2*l4*m2*m3*t13*t22*t28*t86*9.0 - l1*l2*l4*m2*m3*t15*t22*t28*t84*(4.5*10.0 / 2.0) - l1*l2*l4*m2*m4*t14*t22*t26*t86*(9.0 / 2.0) - l1*l2*l3*m1*m3*t14*t24*t28*t86*6.0 - l1*l2*l3*m2*m3*t13*t24*t28*t86*9.0 - l1*l2*l3*m2*m3*t15*t24*t28*t84*(2.7*10.0 / 2.0) - l1*l2*l4*m1*m3*t15*t22*t28*t86*6.0 - l1*l2*l4*m2*m3*t14*t22*t28*t86*9.0 - l1*l2*l4*m2*m4*t15*t22*t26*t86*(9.0 / 2.0) - l1*l2*l3*m1*m3*t15*t24*t28*t86*6.0 - l1*l2*l3*m2*m3*t14*t24*t28*t86*9.0 - l1*l2*l3*m2*m3*t16*t24*t28*t84*(2.7*10.0 / 2.0) - l1*l2*l4*m2*m3*t15*t22*t28*t86*9.0 - l1*l2*l3*m1*m3*t16*t24*t28*t86*6.0 - l1*l2*l3*m2*m3*t15*t24*t28*t86*9.0 - l1*l2*l3*m2*m3*t16*t24*t28*t86*9.0 + l1*l2*l4*m2*m4*t13*t22*t26*t171*(2.7*10.0 / 4.0) + l1*l2*l4*m3*m4*t13*t22*t25*t171*(9.0 / 2.0) + l1*l2*l4*m2*m3*t13*t22*t28*t171*(4.5*10.0 / 2.0) + l1*l2*l4*m2*m4*t14*t22*t26*t171*(2.7*10.0 / 4.0) + l1*l2*l4*m3*m4*t14*t22*t25*t171*(9.0 / 2.0) + l1*l2*l3*m2*m3*t13*t24*t28*t171*(2.7*10.0 / 2.0) + l1*l2*l4*m2*m3*t14*t22*t28*t171*(4.5*10.0 / 2.0) + l1*l2*l4*m2*m4*t15*t22*t26*t171*(2.7*10.0 / 4.0) + l1*l2*l4*m3*m4*t15*t22*t25*t171*(9.0 / 2.0) + l1*l2*l3*m2*m3*t14*t24*t28*t171*(2.7*10.0 / 2.0) + l1*l2*l4*m2*m3*t15*t22*t28*t171*(4.5*10.0 / 2.0) + l1*l2*l3*m2*m3*t15*t24*t28*t171*(2.7*10.0 / 2.0) + l1*l2*l3*m2*m3*t16*t24*t28*t171*(2.7*10.0 / 2.0) + l1*l2*l4*m2*m4*t13*t22*t26*t188*(3.0 / 2.0) + l1*l2*l4*m2*m3*t13*t22*t28*t188*3.0 + l1*l2*l4*m2*m4*t14*t22*t26*t188*(3.0 / 2.0) + l1*l2*l3*m2*m3*t13*t24*t28*t188*3.0 + l1*l2*l4*m2*m3*t14*t22*t28*t188*3.0 + l1*l2*l4*m2*m4*t15*t22*t26*t188*(3.0 / 2.0) + l1*l2*l3*m2*m3*t14*t24*t28*t188*3.0 + l1*l2*l4*m2*m3*t15*t22*t28*t188*3.0 + l1*l2*l3*m2*m3*t15*t24*t28*t188*3.0 + l1*l2*l3*m2*m3*t16*t24*t28*t188*3.0 - l1*l3*m1*m2*t8*t13*t19*t23*t28*1.0*10.0 - l1*l3*m1*m4*t8*t13*t19*t23*t26*2.4*10.0 - l1*l3*m1*m2*t8*t14*t19*t23*t28*1.0*10.0 - l1*l3*m1*m3*t8*t13*t19*t23*t28*4.5*10.0 - l1*l3*m1*m4*t8*t14*t19*t23*t26*2.4*10.0 - l1*l3*m2*m4*t8*t13*t19*t23*t26*3.0*10.0 - l1*l3*m3*m4*t8*t13*t19*t23*t25*9.0 - l1*l3*m1*m3*t8*t14*t19*t23*t28*4.5*10.0 - l1*l3*m2*m3*t8*t13*t19*t23*t28*(2.25*100.0 / 4.0) - l1*l3*m2*m4*t8*t14*t19*t23*t26*3.0*10.0 - l1*l3*m3*m4*t8*t14*t19*t23*t25*9.0 - l1*l3*m2*m3*t8*t14*t19*t23*t28*(2.25*100.0 / 4.0) - l1*l2*m1*m4*t13*t21*t23*t26*t37*6.0 + l1*l2*m1*m2*t13*t21*t23*t28*t38*1.2*10.0 - l1*l2*m1*m3*t13*t21*t23*t28*t37*1.5*10.0 - l1*l2*m1*m4*t14*t21*t23*t26*t37*6.0 - l1*l2*m2*m4*t13*t21*t23*t26*t37*9.0 + l1*l2*m1*m2*t14*t21*t23*t28*t38*1.2*10.0 + l1*l2*m1*m3*t13*t21*t23*t28*t38*1.8*10.0 - l1*l2*m1*m3*t14*t21*t23*t28*t37*1.5*10.0 - l1*l2*m1*m4*t15*t21*t23*t26*t37*6.0 - l1*l2*m2*m3*t13*t21*t23*t28*t37*(4.5*10.0 / 2.0) - l1*l2*m2*m4*t14*t21*t23*t26*t37*9.0 + l1*l2*m1*m2*t15*t21*t23*t28*t38*1.2*10.0 + l1*l2*m1*m3*t14*t21*t23*t28*t38*1.8*10.0 - l1*l2*m1*m3*t15*t21*t23*t28*t37*1.5*10.0 + l1*l2*m2*m3*t13*t21*t23*t28*t38*4.5*10.0 - l1*l2*m2*m3*t14*t21*t23*t28*t37*(4.5*10.0 / 2.0) - l1*l2*m2*m4*t15*t21*t23*t26*t37*9.0 - l2*l3*m1*m2*t13*t17*t23*t28*t42*(5.0 / 2.0) - l2*l3*m1*m4*t13*t17*t23*t26*t42*1.2*10.0 + l1*l2*m1*m2*t16*t21*t23*t28*t38*6.0 + l1*l2*m1*m3*t15*t21*t23*t28*t38*1.8*10.0 + l1*l2*m2*m3*t14*t21*t23*t28*t38*4.5*10.0 - l1*l2*m2*m3*t15*t21*t23*t28*t37*(4.5*10.0 / 2.0) + l1*l4*m1*m2*t13*t19*t21*t28*t43*1.2*10.0 - l1*l4*m1*m4*t13*t19*t21*t26*t43*3.0 - l2*l3*m1*m3*t13*t17*t23*t28*t42*(4.5*10.0 / 2.0) - l2*l3*m2*m4*t13*t17*t23*t26*t42*6.0 + l2*l3*m3*m4*t13*t17*t23*t25*t42*3.0 + l1*l2*m1*m3*t16*t21*t23*t28*t38*9.0 + l1*l2*m2*m3*t15*t21*t23*t28*t38*4.5*10.0 + l1*l4*m1*m2*t14*t19*t21*t28*t43*1.2*10.0 + l1*l4*m1*m3*t13*t19*t21*t28*t43*6.0 - l1*l4*m1*m4*t14*t19*t21*t26*t43*3.0 - l1*l4*m2*m4*t13*t19*t21*t26*t43*(1.5*10.0 / 4.0) + l1*l4*m3*m4*t13*t19*t21*t25*t43*(9.0 / 4.0) - l2*l3*m2*m3*t13*t17*t23*t28*t42*(4.5*10.0 / 4.0) + l1*l2*m2*m3*t16*t21*t23*t28*t38*(4.5*10.0 / 2.0) + l1*l4*m1*m3*t14*t19*t21*t28*t43*6.0 + l1*l4*m2*m3*t13*t19*t21*t28*t43*(1.5*10.0 / 2.0) - l1*l4*m2*m4*t14*t19*t21*t26*t43*(1.5*10.0 / 4.0) + l1*l4*m3*m4*t14*t19*t21*t25*t43*(9.0 / 4.0) + l1*l4*m2*m3*t14*t19*t21*t28*t43*(1.5*10.0 / 2.0) + l2*l4*m1*m2*t13*t17*t21*t28*t61*3.0 - l2*l4*m1*m4*t13*t17*t21*t26*t61*(3.0 / 2.0) + l2*l4*m1*m3*t13*t17*t21*t28*t61*3.0 - l2*l4*m2*m4*t13*t17*t21*t26*t61*(3.0 / 4.0) - l2*l4*m3*m4*t13*t17*t21*t25*t61*(3.0 / 4.0) + l2*l4*m2*m3*t13*t17*t21*t28*t61*(3.0 / 2.0) + l1*l3*m2*m4*t13*t19*t23*t26*t82*6.0 + l1*l3*m3*m4*t13*t19*t23*t25*t82*3.0 + l1*l3*m2*m3*t13*t19*t23*t28*t82*(4.5*10.0 / 4.0) + l1*l3*m2*m4*t14*t19*t23*t26*t82*6.0 + l1*l3*m3*m4*t14*t19*t23*t25*t82*3.0 + l1*l3*m1*m2*t13*t19*t23*t28*t85*6.0 + l1*l3*m2*m3*t14*t19*t23*t28*t82*(4.5*10.0 / 4.0) + l1*l3*m1*m2*t14*t19*t23*t28*t85*6.0 + l1*l3*m1*m3*t13*t19*t23*t28*t85*9.0 + l1*l3*m1*m3*t14*t19*t23*t28*t85*9.0 + l1*l3*m2*m3*t13*t19*t23*t28*t85*(4.5*10.0 / 4.0) + l1*l3*m2*m3*t14*t19*t23*t28*t85*(4.5*10.0 / 4.0) + l2*l3*m1*m2*t13*t17*t23*t28*t106*(1.5*10.0 / 2.0) + l2*l3*m1*m4*t13*t17*t23*t26*t106*1.2*10.0 + l2*l3*m1*m3*t13*t17*t23*t28*t106*(4.5*10.0 / 2.0) + l2*l3*m2*m4*t13*t17*t23*t26*t106*1.8*10.0 + l2*l3*m3*m4*t13*t17*t23*t25*t106*9.0 + l2*l3*m2*m3*t13*t17*t23*t28*t106*(1.35*100.0 / 4.0) - l1*l4*m1*m2*t13*t19*t21*t28*t109*1.2*10.0 - l1*l4*m1*m4*t13*t19*t21*t26*t109*9.0 - l1*l4*m1*m2*t14*t19*t21*t28*t109*1.2*10.0 - l1*l4*m1*m3*t13*t19*t21*t28*t109*1.8*10.0 - l1*l4*m1*m4*t14*t19*t21*t26*t109*9.0 - l1*l4*m2*m4*t13*t19*t21*t26*t109*(4.5*10.0 / 4.0) - l1*l4*m3*m4*t13*t19*t21*t25*t109*(2.7*10.0 / 4.0) - l1*l4*m1*m3*t14*t19*t21*t28*t109*1.8*10.0 - l1*l4*m2*m3*t13*t19*t21*t28*t109*(4.5*10.0 / 2.0) - l1*l4*m2*m4*t14*t19*t21*t26*t109*(4.5*10.0 / 4.0) - l1*l4*m3*m4*t14*t19*t21*t25*t109*(2.7*10.0 / 4.0) - l1*l4*m2*m3*t14*t19*t21*t28*t109*(4.5*10.0 / 2.0) + l2*l3*m1*m2*t13*t17*t23*t28*t122*(3.0 / 2.0) + l2*l3*m1*m3*t13*t17*t23*t28*t122*(9.0 / 2.0) + l1*l4*m2*m4*t13*t19*t21*t26*t123*(3.0 / 4.0) - l1*l4*m3*m4*t13*t19*t21*t25*t123*(3.0 / 4.0) + l2*l3*m2*m3*t13*t17*t23*t28*t122*(9.0 / 4.0) - l1*l4*m2*m3*t13*t19*t21*t28*t123*(3.0 / 2.0) + l1*l4*m2*m4*t14*t19*t21*t26*t123*(3.0 / 4.0) - l1*l4*m3*m4*t14*t19*t21*t25*t123*(3.0 / 4.0) - l1*l4*m2*m3*t14*t19*t21*t28*t123*(3.0 / 2.0) + l1*l2*m2*m4*t13*t21*t23*t26*t133*3.0 + l1*l2*m2*m3*t13*t21*t23*t28*t133*(1.5*10.0 / 2.0) + l1*l2*m2*m4*t14*t21*t23*t26*t133*3.0 - l1*l2*m1*m3*t13*t21*t23*t28*t135*3.0 - l1*l2*m2*m3*t13*t21*t23*t28*t134*(2.7*10.0 / 2.0) + l1*l2*m2*m3*t14*t21*t23*t28*t133*(1.5*10.0 / 2.0) + l1*l2*m2*m4*t15*t21*t23*t26*t133*3.0 - l1*l2*m1*m3*t14*t21*t23*t28*t135*3.0 - l1*l2*m2*m3*t13*t21*t23*t28*t135*(9.0 / 2.0) - l1*l2*m2*m3*t14*t21*t23*t28*t134*(2.7*10.0 / 2.0) + l1*l2*m2*m3*t15*t21*t23*t28*t133*(1.5*10.0 / 2.0) - l1*l2*m1*m3*t15*t21*t23*t28*t135*3.0 - l1*l2*m2*m3*t14*t21*t23*t28*t135*(9.0 / 2.0) - l1*l2*m2*m3*t15*t21*t23*t28*t134*(2.7*10.0 / 2.0) - l1*l2*m1*m3*t16*t21*t23*t28*t135*3.0 - l1*l2*m2*m3*t15*t21*t23*t28*t135*(9.0 / 2.0) - l1*l2*m2*m3*t16*t21*t23*t28*t134*(2.7*10.0 / 4.0) - l1*l2*m2*m3*t16*t21*t23*t28*t135*(9.0 / 2.0) - l2*l4*m1*m2*t13*t17*t21*t28*t149*3.0 - l2*l4*m1*m4*t13*t17*t21*t26*t149*(9.0 / 2.0) + l2*l4*m1*m2*t13*t17*t21*t28*t150*9.0 - l2*l4*m1*m3*t13*t17*t21*t28*t149*9.0 + l2*l4*m1*m4*t13*t17*t21*t26*t150*(9.0 / 2.0) - l2*l4*m2*m4*t13*t17*t21*t26*t149*(9.0 / 4.0) + l2*l4*m3*m4*t13*t17*t21*t25*t149*(9.0 / 4.0) + l2*l4*m1*m2*t13*t17*t21*t28*t151*9.0 + l2*l4*m1*m3*t13*t17*t21*t28*t150*9.0 - l2*l4*m1*m4*t13*t17*t21*t26*t151*(3.0 / 2.0) - l2*l4*m2*m3*t13*t17*t21*t28*t149*(9.0 / 2.0) + l2*l4*m2*m4*t13*t17*t21*t26*t150*(2.7*10.0 / 4.0) + l2*l4*m3*m4*t13*t17*t21*t25*t150*(2.7*10.0 / 4.0) + l2*l4*m1*m3*t13*t17*t21*t28*t151*3.0 + l2*l4*m2*m3*t13*t17*t21*t28*t150*(2.7*10.0 / 2.0) - l2*l4*m2*m4*t13*t17*t21*t26*t151*(9.0 / 4.0) + l2*l4*m3*m4*t13*t17*t21*t25*t151*(9.0 / 4.0) + l2*l4*m2*m3*t13*t17*t21*t28*t151*(9.0 / 2.0) + l1*l2*m2*m3*t13*t21*t23*t28*t172*(2.7*10.0 / 2.0) + l1*l2*m2*m3*t14*t21*t23*t28*t172*(2.7*10.0 / 2.0) + l1*l2*m2*m3*t15*t21*t23*t28*t172*(2.7*10.0 / 2.0) + l1*l2*m2*m3*t16*t21*t23*t28*t172*(2.7*10.0 / 4.0) - l1*l3*m2*m3*t13*t19*t23*t28*t187*(9.0 / 4.0) - l1*l3*m2*m3*t14*t19*t23*t28*t187*(9.0 / 4.0) + l2*l3*m1*m2*t13*t17*t23*t28*t198*(9.0 / 2.0) + l2*l3*m1*m3*t13*t17*t23*t28*t198*(9.0 / 2.0) + l1*l4*m2*m4*t13*t19*t21*t26*t199*(9.0 / 4.0) + l1*l4*m3*m4*t13*t19*t21*t25*t199*(9.0 / 4.0) + l2*l3*m2*m3*t13*t17*t23*t28*t198*(2.7*10.0 / 4.0) + l1*l4*m2*m3*t13*t19*t21*t28*t199*(9.0 / 2.0) + l1*l4*m2*m4*t14*t19*t21*t26*t199*(9.0 / 4.0) + l1*l4*m3*m4*t14*t19*t21*t25*t199*(9.0 / 4.0) + l1*l4*m2*m3*t14*t19*t21*t28*t199*(9.0 / 2.0) + l1*l2*m2*m3*t13*t21*t23*t28*t203*(3.0 / 2.0) + l1*l2*m2*m3*t14*t21*t23*t28*t203*(3.0 / 2.0) + l1*l2*m2*m3*t15*t21*t23*t28*t203*(3.0 / 2.0) + l1*l2*m2*m3*t16*t21*t23*t28*t203*(3.0 / 2.0) + dth1*dth2*l1*l2*l4*m1*m2*t9*t22*t28*4.8*10.0 + dth1*dth2*l1*l2*l4*m1*m4*t9*t22*t26*1.8*10.0 + dth1*dth2*l1*l2*l3*m1*m2*t9*t24*t28*1.6*10.0 + dth1*dth2*l1*l2*l4*m1*m3*t9*t22*t28*6.0*10.0 + dth1*dth2*l1*l2*l4*m2*m4*t9*t22*t26*4.5*10.0 + dth1*dth2*l1*l2*l4*m3*m4*t9*t22*t25*3.0*10.0 + dth1*dth3*l1*l2*l4*m1*m2*t9*t22*t28*4.8*10.0 + dth1*dth3*l1*l2*l4*m1*m4*t9*t22*t26*1.8*10.0 + dth1*dth2*l1*l2*l3*m1*m3*t9*t24*t28*3.6*10.0 + dth1*dth2*l1*l2*l4*m2*m3*t9*t22*t28*1.5*100.0 + dth1*dth3*l1*l2*l3*m1*m2*t9*t24*t28*1.6*10.0 + dth1*dth3*l1*l2*l4*m1*m3*t9*t22*t28*6.0*10.0 + dth1*dth3*l1*l2*l4*m2*m4*t9*t22*t26*4.5*10.0 + dth1*dth3*l1*l2*l4*m3*m4*t9*t22*t25*3.0*10.0 + dth2*dth3*l1*l2*l4*m1*m2*t9*t22*t28*4.8*10.0 + dth2*dth3*l1*l2*l4*m1*m4*t9*t22*t26*1.8*10.0 + dth1*dth2*l1*l2*l3*m2*m3*t9*t24*t28*9.0*10.0 + dth1*dth3*l1*l2*l3*m1*m3*t9*t24*t28*3.6*10.0 + dth1*dth3*l1*l2*l4*m2*m3*t9*t22*t28*1.5*100.0 + dth1*dth4*l1*l2*l3*m1*m2*t9*t24*t28*1.6*10.0 + dth2*dth3*l1*l2*l3*m1*m2*t9*t24*t28*1.6*10.0 + dth2*dth3*l1*l2*l4*m1*m3*t9*t22*t28*6.0*10.0 + dth2*dth3*l1*l2*l4*m2*m4*t9*t22*t26*4.5*10.0 + dth2*dth3*l1*l2*l4*m3*m4*t9*t22*t25*3.0*10.0 + dth1*dth3*l1*l2*l3*m2*m3*t9*t24*t28*9.0*10.0 + dth1*dth4*l1*l2*l3*m1*m3*t9*t24*t28*3.6*10.0 + dth2*dth3*l1*l2*l3*m1*m3*t9*t24*t28*3.6*10.0 + dth2*dth3*l1*l2*l4*m2*m3*t9*t22*t28*1.5*100.0 + dth2*dth4*l1*l2*l3*m1*m2*t9*t24*t28*1.6*10.0 + dth1*dth4*l1*l2*l3*m2*m3*t9*t24*t28*9.0*10.0 + dth2*dth3*l1*l2*l3*m2*m3*t9*t24*t28*9.0*10.0 + dth2*dth4*l1*l2*l3*m1*m3*t9*t24*t28*3.6*10.0 + dth3*dth4*l1*l2*l3*m1*m2*t9*t24*t28*1.6*10.0 + dth2*dth4*l1*l2*l3*m2*m3*t9*t24*t28*9.0*10.0 + dth3*dth4*l1*l2*l3*m1*m3*t9*t24*t28*3.6*10.0 + dth3*dth4*l1*l2*l3*m2*m3*t9*t24*t28*9.0*10.0 - dth1*dth2*l1*l2*l4*m2*m4*t22*t26*t84*(2.7*10.0 / 2.0) - dth1*dth2*l1*l2*l4*m3*m4*t22*t25*t84*9.0 - dth1*dth2*l1*l2*l4*m1*m4*t22*t26*t86*6.0 - dth1*dth2*l1*l2*l4*m2*m3*t22*t28*t84*4.5*10.0 - dth1*dth3*l1*l2*l4*m2*m4*t22*t26*t84*(2.7*10.0 / 2.0) - dth1*dth3*l1*l2*l4*m3*m4*t22*t25*t84*9.0 - dth1*dth2*l1*l2*l3*m2*m3*t24*t28*t84*2.7*10.0 - dth1*dth2*l1*l2*l4*m1*m3*t22*t28*t86*1.2*10.0 - dth1*dth2*l1*l2*l4*m2*m4*t22*t26*t86*9.0 - dth1*dth3*l1*l2*l4*m1*m4*t22*t26*t86*6.0 - dth1*dth3*l1*l2*l4*m2*m3*t22*t28*t84*4.5*10.0 - dth2*dth3*l1*l2*l4*m2*m4*t22*t26*t84*(2.7*10.0 / 2.0) - dth2*dth3*l1*l2*l4*m3*m4*t22*t25*t84*9.0 - dth1*dth2*l1*l2*l3*m1*m3*t24*t28*t86*1.2*10.0 - dth1*dth2*l1*l2*l4*m2*m3*t22*t28*t86*1.8*10.0 - dth1*dth3*l1*l2*l3*m2*m3*t24*t28*t84*2.7*10.0 - dth1*dth3*l1*l2*l4*m1*m3*t22*t28*t86*1.2*10.0 - dth1*dth3*l1*l2*l4*m2*m4*t22*t26*t86*9.0 - dth2*dth3*l1*l2*l4*m1*m4*t22*t26*t86*6.0 - dth2*dth3*l1*l2*l4*m2*m3*t22*t28*t84*4.5*10.0 - dth1*dth2*l1*l2*l3*m2*m3*t24*t28*t86*1.8*10.0 - dth1*dth3*l1*l2*l3*m1*m3*t24*t28*t86*1.2*10.0 - dth1*dth3*l1*l2*l4*m2*m3*t22*t28*t86*1.8*10.0 - dth1*dth4*l1*l2*l3*m2*m3*t24*t28*t84*2.7*10.0 - dth2*dth3*l1*l2*l3*m2*m3*t24*t28*t84*2.7*10.0 - dth2*dth3*l1*l2*l4*m1*m3*t22*t28*t86*1.2*10.0 - dth2*dth3*l1*l2*l4*m2*m4*t22*t26*t86*9.0 - dth1*dth3*l1*l2*l3*m2*m3*t24*t28*t86*1.8*10.0 - dth1*dth4*l1*l2*l3*m1*m3*t24*t28*t86*1.2*10.0 - dth2*dth3*l1*l2*l3*m1*m3*t24*t28*t86*1.2*10.0 - dth2*dth3*l1*l2*l4*m2*m3*t22*t28*t86*1.8*10.0 - dth2*dth4*l1*l2*l3*m2*m3*t24*t28*t84*2.7*10.0 - dth1*dth4*l1*l2*l3*m2*m3*t24*t28*t86*1.8*10.0 - dth2*dth3*l1*l2*l3*m2*m3*t24*t28*t86*1.8*10.0 - dth2*dth4*l1*l2*l3*m1*m3*t24*t28*t86*1.2*10.0 - dth3*dth4*l1*l2*l3*m2*m3*t24*t28*t84*2.7*10.0 - dth2*dth4*l1*l2*l3*m2*m3*t24*t28*t86*1.8*10.0 - dth3*dth4*l1*l2*l3*m1*m3*t24*t28*t86*1.2*10.0 - dth3*dth4*l1*l2*l3*m2*m3*t24*t28*t86*1.8*10.0 + dth1*dth2*l1*l2*l4*m2*m4*t22*t26*t171*(2.7*10.0 / 2.0) + dth1*dth2*l1*l2*l4*m3*m4*t22*t25*t171*9.0 + dth1*dth2*l1*l2*l4*m2*m3*t22*t28*t171*4.5*10.0 + dth1*dth3*l1*l2*l4*m2*m4*t22*t26*t171*(2.7*10.0 / 2.0) + dth1*dth3*l1*l2*l4*m3*m4*t22*t25*t171*9.0 + dth1*dth2*l1*l2*l3*m2*m3*t24*t28*t171*2.7*10.0 + dth1*dth3*l1*l2*l4*m2*m3*t22*t28*t171*4.5*10.0 + dth2*dth3*l1*l2*l4*m2*m4*t22*t26*t171*(2.7*10.0 / 2.0) + dth2*dth3*l1*l2*l4*m3*m4*t22*t25*t171*9.0 + dth1*dth3*l1*l2*l3*m2*m3*t24*t28*t171*2.7*10.0 + dth2*dth3*l1*l2*l4*m2*m3*t22*t28*t171*4.5*10.0 + dth1*dth4*l1*l2*l3*m2*m3*t24*t28*t171*2.7*10.0 + dth2*dth3*l1*l2*l3*m2*m3*t24*t28*t171*2.7*10.0 + dth2*dth4*l1*l2*l3*m2*m3*t24*t28*t171*2.7*10.0 + dth3*dth4*l1*l2*l3*m2*m3*t24*t28*t171*2.7*10.0 + dth1*dth2*l1*l2*l4*m2*m4*t22*t26*t188*3.0 + dth1*dth2*l1*l2*l4*m2*m3*t22*t28*t188*6.0 + dth1*dth3*l1*l2*l4*m2*m4*t22*t26*t188*3.0 + dth1*dth2*l1*l2*l3*m2*m3*t24*t28*t188*6.0 + dth1*dth3*l1*l2*l4*m2*m3*t22*t28*t188*6.0 + dth2*dth3*l1*l2*l4*m2*m4*t22*t26*t188*3.0 + dth1*dth3*l1*l2*l3*m2*m3*t24*t28*t188*6.0 + dth2*dth3*l1*l2*l4*m2*m3*t22*t28*t188*6.0 + dth1*dth4*l1*l2*l3*m2*m3*t24*t28*t188*6.0 + dth2*dth3*l1*l2*l3*m2*m3*t24*t28*t188*6.0 + dth2*dth4*l1*l2*l3*m2*m3*t24*t28*t188*6.0 + dth3*dth4*l1*l2*l3*m2*m3*t24*t28*t188*6.0 - (g*l1*l2*l3*m1*m2*m3*m4*t23*t60) / 2.0 + (g*l1*l2*l4*m1*m2*m3*m4*t21*t91) / 8.0 + g*l1*l2*l3*m1*m2*m3*m4*t23*t145*(3.0 / 2.0) - g*l1*l2*l3*m1*m2*m3*m4*t23*t146*(9.0 / 2.0) - g*l1*l2*l3*m1*m2*m3*m4*t23*t147*(3.0 / 2.0) - g*l1*l2*l4*m1*m2*m3*m4*t21*t165*(3.0 / 8.0) + g*l1*l2*l4*m1*m2*m3*m4*t21*t166*(9.0 / 8.0) + g*l1*l2*l4*m1*m2*m3*m4*t21*t167*(9.0 / 8.0) + g*l1*l2*l4*m1*m2*m3*m4*t21*t168*(3.0 / 8.0) - g*l1*l2*l4*m1*m2*m3*m4*t21*t218*(3.0 / 8.0) - g*l1*l2*l4*m1*m2*m3*m4*t21*t219*(2.7*10.0 / 8.0) + g*l1*l2*l4*m1*m2*m3*m4*t21*t220*(9.0 / 8.0) - dth1*dth2*l1*l3*m1*m2*t8*t19*t23*t28*2.0*10.0 - dth1*dth2*l1*l3*m1*m4*t8*t19*t23*t26*4.8*10.0 - dth1*dth2*l1*l3*m1*m3*t8*t19*t23*t28*9.0*10.0 - dth1*dth2*l1*l3*m2*m4*t8*t19*t23*t26*6.0*10.0 - dth1*dth2*l1*l3*m3*m4*t8*t19*t23*t25*1.8*10.0 - dth1*dth2*l1*l3*m2*m3*t8*t19*t23*t28*(2.25*100.0 / 2.0) - dth1*dth2*l1*l2*m1*m4*t21*t23*t26*t37*1.2*10.0 + dth1*dth2*l1*l2*m1*m2*t21*t23*t28*t38*2.4*10.0 - dth1*dth2*l1*l2*m1*m3*t21*t23*t28*t37*3.0*10.0 - dth1*dth2*l1*l2*m2*m4*t21*t23*t26*t37*1.8*10.0 - dth1*dth3*l1*l2*m1*m4*t21*t23*t26*t37*1.2*10.0 + dth1*dth2*l1*l2*m1*m3*t21*t23*t28*t38*3.6*10.0 - dth1*dth2*l1*l2*m2*m3*t21*t23*t28*t37*4.5*10.0 + dth1*dth3*l1*l2*m1*m2*t21*t23*t28*t38*2.4*10.0 - dth1*dth3*l1*l2*m1*m3*t21*t23*t28*t37*3.0*10.0 - dth1*dth3*l1*l2*m2*m4*t21*t23*t26*t37*1.8*10.0 - dth2*dth3*l1*l2*m1*m4*t21*t23*t26*t37*1.2*10.0 + dth1*dth2*l1*l2*m2*m3*t21*t23*t28*t38*9.0*10.0 + dth1*dth3*l1*l2*m1*m3*t21*t23*t28*t38*3.6*10.0 - dth1*dth3*l1*l2*m2*m3*t21*t23*t28*t37*4.5*10.0 + dth1*dth4*l1*l2*m1*m2*t21*t23*t28*t38*1.2*10.0 + dth2*dth3*l1*l2*m1*m2*t21*t23*t28*t38*2.4*10.0 - dth2*dth3*l1*l2*m1*m3*t21*t23*t28*t37*3.0*10.0 - dth2*dth3*l1*l2*m2*m4*t21*t23*t26*t37*1.8*10.0 + dth1*dth2*l1*l4*m1*m2*t19*t21*t28*t43*2.4*10.0 - dth1*dth2*l1*l4*m1*m4*t19*t21*t26*t43*6.0 + dth1*dth3*l1*l2*m2*m3*t21*t23*t28*t38*9.0*10.0 + dth1*dth4*l1*l2*m1*m3*t21*t23*t28*t38*1.8*10.0 + dth2*dth3*l1*l2*m1*m3*t21*t23*t28*t38*3.6*10.0 - dth2*dth3*l1*l2*m2*m3*t21*t23*t28*t37*4.5*10.0 + dth2*dth4*l1*l2*m1*m2*t21*t23*t28*t38*1.2*10.0 + dth1*dth2*l1*l4*m1*m3*t19*t21*t28*t43*1.2*10.0 - dth1*dth2*l1*l4*m2*m4*t19*t21*t26*t43*(1.5*10.0 / 2.0) + dth1*dth2*l1*l4*m3*m4*t19*t21*t25*t43*(9.0 / 2.0) + dth1*dth4*l1*l2*m2*m3*t21*t23*t28*t38*4.5*10.0 + dth2*dth3*l1*l2*m2*m3*t21*t23*t28*t38*9.0*10.0 + dth2*dth4*l1*l2*m1*m3*t21*t23*t28*t38*1.8*10.0 + dth3*dth4*l1*l2*m1*m2*t21*t23*t28*t38*1.2*10.0 + dth1*dth2*l1*l4*m2*m3*t19*t21*t28*t43*1.5*10.0 + dth2*dth4*l1*l2*m2*m3*t21*t23*t28*t38*4.5*10.0 + dth3*dth4*l1*l2*m1*m3*t21*t23*t28*t38*1.8*10.0 + dth3*dth4*l1*l2*m2*m3*t21*t23*t28*t38*4.5*10.0 + dth1*dth2*l1*l3*m2*m4*t19*t23*t26*t82*1.2*10.0 + dth1*dth2*l1*l3*m3*m4*t19*t23*t25*t82*6.0 + dth1*dth2*l1*l3*m2*m3*t19*t23*t28*t82*(4.5*10.0 / 2.0) + dth1*dth2*l1*l3*m1*m2*t19*t23*t28*t85*1.2*10.0 + dth1*dth2*l1*l3*m1*m3*t19*t23*t28*t85*1.8*10.0 + dth1*dth2*l1*l3*m2*m3*t19*t23*t28*t85*(4.5*10.0 / 2.0) - dth1*dth2*l1*l4*m1*m2*t19*t21*t28*t109*2.4*10.0 - dth1*dth2*l1*l4*m1*m4*t19*t21*t26*t109*1.8*10.0 - dth1*dth2*l1*l4*m1*m3*t19*t21*t28*t109*3.6*10.0 - dth1*dth2*l1*l4*m2*m4*t19*t21*t26*t109*(4.5*10.0 / 2.0) - dth1*dth2*l1*l4*m3*m4*t19*t21*t25*t109*(2.7*10.0 / 2.0) - dth1*dth2*l1*l4*m2*m3*t19*t21*t28*t109*4.5*10.0 + dth1*dth2*l1*l4*m2*m4*t19*t21*t26*t123*(3.0 / 2.0) - dth1*dth2*l1*l4*m3*m4*t19*t21*t25*t123*(3.0 / 2.0) - dth1*dth2*l1*l4*m2*m3*t19*t21*t28*t123*3.0 + dth1*dth2*l1*l2*m2*m4*t21*t23*t26*t133*6.0 + dth1*dth2*l1*l2*m2*m3*t21*t23*t28*t133*1.5*10.0 + dth1*dth3*l1*l2*m2*m4*t21*t23*t26*t133*6.0 - dth1*dth2*l1*l2*m1*m3*t21*t23*t28*t135*6.0 - dth1*dth2*l1*l2*m2*m3*t21*t23*t28*t134*2.7*10.0 + dth1*dth3*l1*l2*m2*m3*t21*t23*t28*t133*1.5*10.0 + dth2*dth3*l1*l2*m2*m4*t21*t23*t26*t133*6.0 - dth1*dth2*l1*l2*m2*m3*t21*t23*t28*t135*9.0 - dth1*dth3*l1*l2*m1*m3*t21*t23*t28*t135*6.0 - dth1*dth3*l1*l2*m2*m3*t21*t23*t28*t134*2.7*10.0 + dth2*dth3*l1*l2*m2*m3*t21*t23*t28*t133*1.5*10.0 - dth1*dth3*l1*l2*m2*m3*t21*t23*t28*t135*9.0 - dth1*dth4*l1*l2*m1*m3*t21*t23*t28*t135*6.0 - dth1*dth4*l1*l2*m2*m3*t21*t23*t28*t134*(2.7*10.0 / 2.0) - dth2*dth3*l1*l2*m1*m3*t21*t23*t28*t135*6.0 - dth2*dth3*l1*l2*m2*m3*t21*t23*t28*t134*2.7*10.0 - dth1*dth4*l1*l2*m2*m3*t21*t23*t28*t135*9.0 - dth2*dth3*l1*l2*m2*m3*t21*t23*t28*t135*9.0 - dth2*dth4*l1*l2*m1*m3*t21*t23*t28*t135*6.0 - dth2*dth4*l1*l2*m2*m3*t21*t23*t28*t134*(2.7*10.0 / 2.0) - dth2*dth4*l1*l2*m2*m3*t21*t23*t28*t135*9.0 - dth3*dth4*l1*l2*m1*m3*t21*t23*t28*t135*6.0 - dth3*dth4*l1*l2*m2*m3*t21*t23*t28*t134*(2.7*10.0 / 2.0) - dth3*dth4*l1*l2*m2*m3*t21*t23*t28*t135*9.0 + dth1*dth2*l1*l2*m2*m3*t21*t23*t28*t172*2.7*10.0 + dth1*dth3*l1*l2*m2*m3*t21*t23*t28*t172*2.7*10.0 + dth1*dth4*l1*l2*m2*m3*t21*t23*t28*t172*(2.7*10.0 / 2.0) + dth2*dth3*l1*l2*m2*m3*t21*t23*t28*t172*2.7*10.0 + dth2*dth4*l1*l2*m2*m3*t21*t23*t28*t172*(2.7*10.0 / 2.0) + dth3*dth4*l1*l2*m2*m3*t21*t23*t28*t172*(2.7*10.0 / 2.0) - dth1*dth2*l1*l3*m2*m3*t19*t23*t28*t187*(9.0 / 2.0) + dth1*dth2*l1*l4*m2*m4*t19*t21*t26*t199*(9.0 / 2.0) + dth1*dth2*l1*l4*m3*m4*t19*t21*t25*t199*(9.0 / 2.0) + dth1*dth2*l1*l4*m2*m3*t19*t21*t28*t199*9.0 + dth1*dth2*l1*l2*m2*m3*t21*t23*t28*t203*3.0 + dth1*dth3*l1*l2*m2*m3*t21*t23*t28*t203*3.0 + dth1*dth4*l1*l2*m2*m3*t21*t23*t28*t203*3.0 + dth2*dth3*l1*l2*m2*m3*t21*t23*t28*t203*3.0 + dth2*dth4*l1*l2*m2*m3*t21*t23*t28*t203*3.0 + dth3*dth4*l1*l2*m2*m3*t21*t23*t28*t203*3.0 + l1*l2*l4*m1*m2*m3*m4*t9*t13*t22*8.0 + l1*l2*l4*m1*m2*m3*m4*t9*t14*t22*8.0 + l1*l2*l4*m1*m2*m3*m4*t9*t15*t22*8.0 - l1*l3*m1*m2*m3*m4*t8*t13*t19*t23*8.0 - l1*l3*m1*m2*m3*m4*t8*t14*t19*t23*8.0 - l2*l3*m1*m2*m3*m4*t13*t17*t23*t42*2.0 + l1*l4*m1*m2*m3*m4*t13*t19*t21*t43*2.0 + l1*l4*m1*m2*m3*m4*t14*t19*t21*t43*2.0 + (l2*l4*m1*m2*m3*m4*t13*t17*t21*t61) / 2.0 + l2*l3*m1*m2*m3*m4*t13*t17*t23*t106*6.0 - l1*l4*m1*m2*m3*m4*t13*t19*t21*t109*6.0 - l1*l4*m1*m2*m3*m4*t14*t19*t21*t109*6.0 - l2*l4*m1*m2*m3*m4*t13*t17*t21*t149*(3.0 / 2.0) + l2*l4*m1*m2*m3*m4*t13*t17*t21*t150*(9.0 / 2.0) + l2*l4*m1*m2*m3*m4*t13*t17*t21*t151*(3.0 / 2.0) + dth1*dth2*l1*l2*l4*m1*m2*m3*m4*t9*t22*1.6*10.0 + dth1*dth3*l1*l2*l4*m1*m2*m3*m4*t9*t22*1.6*10.0 + dth2*dth3*l1*l2*l4*m1*m2*m3*m4*t9*t22*1.6*10.0 - dth1*dth2*l1*l3*m1*m2*m3*m4*t8*t19*t23*1.6*10.0 + dth1*dth2*l1*l4*m1*m2*m3*m4*t19*t21*t43*4.0 - dth1*dth2*l1*l4*m1*m2*m3*m4*t19*t21*t109*1.2*10.0)* - 2.4*10.0) / (m4*t23);

        index = index + num_threads;

        index1 = (grid_size[0] > 1) ? index : 0;
        index2 = (grid_size[1] > 1) ? index : 0;
        index3 = (grid_size[2] > 1) ? index : 0;
        index4 = (grid_size[3] > 1) ? index : 0;
        index5 = (grid_size[4] > 1) ? index : 0;
        index6 = (grid_size[5] > 1) ? index : 0;
        index7 = (grid_size[6] > 1) ? index : 0;
        index8 = (grid_size[7] > 1) ? index : 0;

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
                        * grid_size[4] * grid_size[5] * grid_size[6] * grid_size[7];
    
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
                        * grid_size[4] * grid_size[5] * grid_size[6] * grid_size[7];
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
    /* Declare all variables.*/
    // Inputs
    // GPU Arrays
    mxGPUArray const* X1;
    mxGPUArray const* X2;
    mxGPUArray const* X3;
    mxGPUArray const* X4;
    mxGPUArray const* dX1;
    mxGPUArray const* dX2;
    mxGPUArray const* dX3;
    mxGPUArray const* dX4;
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
    double* p_dX1;
    double* p_dX2;
    double* p_dX3;
    double* p_dX4;
    double const* p_in1; 
    double const* p_in2;
    double const* p_in3;
    double const* p_in4; 
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
//     mxGPUArray* k1_X1;
//     mxGPUArray* k1_X2;
//     mxGPUArray* k1_X3;
//     mxGPUArray* k1_X4;
    mxGPUArray* k1_dX1;
    mxGPUArray* k1_dX2;
    mxGPUArray* k1_dX3;
    mxGPUArray* k1_dX4;
    mxGPUArray* k2_X1;
    mxGPUArray* k2_X2;
    mxGPUArray* k2_X3;
    mxGPUArray* k2_X4;
    mxGPUArray* k2_dX1;
    mxGPUArray* k2_dX2;
    mxGPUArray* k2_dX3;
    mxGPUArray* k2_dX4;
    mxGPUArray* k3_X1;
    mxGPUArray* k3_X2;
    mxGPUArray* k3_X3;
    mxGPUArray* k3_X4;
    mxGPUArray* k3_dX1;
    mxGPUArray* k3_dX2;
    mxGPUArray* k3_dX3;
    mxGPUArray* k3_dX4;
    mxGPUArray* k4_X1;
    mxGPUArray* k4_X2;
    mxGPUArray* k4_X3;
    mxGPUArray* k4_X4;
    mxGPUArray* k4_dX1;
    mxGPUArray* k4_dX2;
    mxGPUArray* k4_dX3;
    mxGPUArray* k4_dX4;
    // Pointers for intermediate variables
    double* p_k1_X1;
    double* p_k1_X2;
    double* p_k1_X3;
    double* p_k1_X4;
    double* p_k1_dX1;
    double* p_k1_dX2;
    double* p_k1_dX3;
    double* p_k1_dX4;
    double* p_k2_X1;
    double* p_k2_X2;
    double* p_k2_X3;
    double* p_k2_X4;
    double* p_k2_dX1;
    double* p_k2_dX2;
    double* p_k2_dX3;
    double* p_k2_dX4;
    double* p_k3_X1;
    double* p_k3_X2;
    double* p_k3_X3;
    double* p_k3_X4;
    double* p_k3_dX1;
    double* p_k3_dX2;
    double* p_k3_dX3;
    double* p_k3_dX4;
    double* p_k4_X1;
    double* p_k4_X2;
    double* p_k4_X3;
    double* p_k4_X4;
    double* p_k4_dX1;
    double* p_k4_dX2;
    double* p_k4_dX3;
    double* p_k4_dX4;

    // Outputs
    mxGPUArray* X1n;
    mxGPUArray* X2n;
    mxGPUArray* X3n;
    mxGPUArray* X4n;
    mxGPUArray* dX1n;
    mxGPUArray* dX2n;
    mxGPUArray* dX3n;
    mxGPUArray* dX4n;
    // Pointers for outputs
    double* p_X1n;
    double* p_X2n;
    double* p_X3n;
    double* p_X4n;
    double* p_dX1n;
    double* p_dX2n;
    double* p_dX3n;
    double* p_dX4n;

    char const* const errId = "parallel:gpu:mexGPUExample:InvalidInput";
    char const* const errMsg = "Invalid input to MEX file.";

    /* Choose a reasonably sized number of threads for the block. */
    int const threadsPerBlock = 256;
    int const blocksPerGrid = 1024;

    /* Initialize the MathWorks GPU API. */
    mxInitGPU();

    /* Throw an error if the input is not a GPU array. */
    if ((nrhs != 26) || !(mxIsGPUArray(prhs[0])) || !(mxIsGPUArray(prhs[1])) || !(mxIsGPUArray(prhs[2])) || !(mxIsGPUArray(prhs[3])) 
                     || !(mxIsGPUArray(prhs[4])) || !(mxIsGPUArray(prhs[5])) || !(mxIsGPUArray(prhs[6])) || !(mxIsGPUArray(prhs[7]))
                     || !(mxIsGPUArray(prhs[8])) || !(mxIsGPUArray(prhs[9])) || !(mxIsGPUArray(prhs[10])) || !(mxIsGPUArray(prhs[11]))) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }

    X1 = mxGPUCreateFromMxArray(prhs[0]);
    X2 = mxGPUCreateFromMxArray(prhs[1]);
    X3 = mxGPUCreateFromMxArray(prhs[2]);
    X4 = mxGPUCreateFromMxArray(prhs[3]);
    dX1 = mxGPUCreateFromMxArray(prhs[4]);
    dX2 = mxGPUCreateFromMxArray(prhs[5]);
    dX3 = mxGPUCreateFromMxArray(prhs[6]);
    dX4 = mxGPUCreateFromMxArray(prhs[7]);
    in1 = mxGPUCreateFromMxArray(prhs[8]);
    in2 = mxGPUCreateFromMxArray(prhs[9]);
    in3 = mxGPUCreateFromMxArray(prhs[10]);
    in4 = mxGPUCreateFromMxArray(prhs[11]);
    grid_size = mxGPUCreateFromMxArray(prhs[22]);
    active_actions = mxGPUCreateFromMxArray(prhs[23]);
    
    p_m = mxGetDoubles(prhs[12]); 
    double const m1 = p_m[0];
    p_m = mxGetDoubles(prhs[13]); 
    double const m2 = p_m[0];
    p_m = mxGetDoubles(prhs[14]); 
    double const m3 = p_m[0];
    p_m = mxGetDoubles(prhs[15]); 
    double const m4 = p_m[0];

    p_l = mxGetDoubles(prhs[16]); 
    double const l1 = p_l[0];
    p_l = mxGetDoubles(prhs[17]); 
    double const l2 = p_l[0];
    p_l = mxGetDoubles(prhs[18]); 
    double const l3 = p_l[0];
    p_l = mxGetDoubles(prhs[19]); 
    double const l4 = p_l[0];

    p_g = mxGetDoubles(prhs[20]);
    double const g = p_g[0];
    
    p_dt = mxGetDoubles(prhs[21]);
    double const dt = p_dt[0];
    
    p_limits = mxGetDoubles(prhs[24]);
    
    p_x_dims_free = mxGetDoubles(prhs[25]);
    mwSize const* num_x_dims = mxGetDimensions(prhs[25]);
    if ((mxGetNumberOfDimensions(prhs[25]) != 2) || (num_x_dims[1] > 1)) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }
    
    /*
     * Verify that inputs are of appropriate type before extracting the pointer.
     */
    if ((mxGPUGetClassID(X1) != mxDOUBLE_CLASS) || (mxGPUGetClassID(X2) != mxDOUBLE_CLASS)
        || (mxGPUGetClassID(X3) != mxDOUBLE_CLASS) || (mxGPUGetClassID(X4) != mxDOUBLE_CLASS)
        || (mxGPUGetClassID(dX1) != mxDOUBLE_CLASS) || (mxGPUGetClassID(dX2) != mxDOUBLE_CLASS)
        || (mxGPUGetClassID(dX3) != mxDOUBLE_CLASS) || (mxGPUGetClassID(dX4) != mxDOUBLE_CLASS)        
        || (mxGPUGetClassID(in1) != mxDOUBLE_CLASS) || (mxGPUGetClassID(in2) != mxDOUBLE_CLASS)
        || (mxGPUGetClassID(in3) != mxDOUBLE_CLASS) || (mxGPUGetClassID(in4) != mxDOUBLE_CLASS)
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
    p_dX1 = (double*)(mxGPUGetDataReadOnly(dX1));
    p_dX2 = (double*)(mxGPUGetDataReadOnly(dX2));
    p_dX3 = (double*)(mxGPUGetDataReadOnly(dX3));
    p_dX4 = (double*)(mxGPUGetDataReadOnly(dX4));
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
    
    dX1n = mxGPUCopyGPUArray(dX1);
    p_dX1n = (double*)(mxGPUGetData(dX1n));
    
    dX2n = mxGPUCopyGPUArray(dX2);
    p_dX2n = (double*)(mxGPUGetData(dX2n));
    
    dX3n = mxGPUCopyGPUArray(dX3);
    p_dX3n = (double*)(mxGPUGetData(dX3n));
    
    dX4n = mxGPUCopyGPUArray(dX4);
    p_dX4n = (double*)(mxGPUGetData(dX4n));
    
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
            p_k1_X4 = p_dX4;
            step << <blocksPerGrid, threadsPerBlock >> > (p_X4n, p_X4, p_k1_X4, 0.5 * dt, p_grid_size);
            
        } else if (curr_free_dim == 5) {
            k1_dX1 = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(dX1), mxGPUGetDimensions(dX1), mxGPUGetClassID(dX1), mxGPUGetComplexity(dX1), MX_GPU_DO_NOT_INITIALIZE);
            p_k1_dX1 = (double*)(mxGPUGetData(k1_dX1));
            dyn5_mex_continuous << <blocksPerGrid, threadsPerBlock >> > (p_k1_dX1, p_X1, p_X2, p_X3, p_X4, p_dX1, p_dX2, p_dX3, p_dX4, 
                                                                         p_in1, p_in2, p_in3, p_in4,
                                                                         m1, m2, m3, m4, l1, l2, l3, l4, g, p_grid_size, p_active_actions);
            step << <blocksPerGrid, threadsPerBlock >> > (p_dX1n, p_dX1, p_k1_dX1, 0.5 * dt, p_grid_size);
            
        } else if (curr_free_dim == 6) {
            k1_dX2 = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(dX2), mxGPUGetDimensions(dX2), mxGPUGetClassID(dX2), mxGPUGetComplexity(dX2), MX_GPU_DO_NOT_INITIALIZE);
            p_k1_dX2 = (double*)(mxGPUGetData(k1_dX2));
            dyn6_mex_continuous << <blocksPerGrid, threadsPerBlock >> > (p_k1_dX2, p_X1, p_X2, p_X3, p_X4, p_dX1, p_dX2, p_dX3, p_dX4, 
                                                                         p_in1, p_in2, p_in3, p_in4,
                                                                         m1, m2, m3, m4, l1, l2, l3, l4, g, p_grid_size, p_active_actions);
            step << <blocksPerGrid, threadsPerBlock >> > (p_dX2n, p_dX2, p_k1_dX2, 0.5 * dt, p_grid_size);
            
        } else if (curr_free_dim == 7) {
            k1_dX3 = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(dX3), mxGPUGetDimensions(dX3), mxGPUGetClassID(dX3), mxGPUGetComplexity(dX3), MX_GPU_DO_NOT_INITIALIZE);
            p_k1_dX3 = (double*)(mxGPUGetData(k1_dX3));
            dyn7_mex_continuous << <blocksPerGrid, threadsPerBlock >> > (p_k1_dX3, p_X1, p_X2, p_X3, p_X4, p_dX1, p_dX2, p_dX3, p_dX4, 
                                                                         p_in1, p_in2, p_in3, p_in4,
                                                                         m1, m2, m3, m4, l1, l2, l3, l4, g, p_grid_size, p_active_actions);
            step << <blocksPerGrid, threadsPerBlock >> > (p_dX3n, p_dX3, p_k1_dX3, 0.5 * dt, p_grid_size);
            
        } else if (curr_free_dim == 8) {
            k1_dX4 = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(dX4), mxGPUGetDimensions(dX4), mxGPUGetClassID(dX4), mxGPUGetComplexity(dX4), MX_GPU_DO_NOT_INITIALIZE);
            p_k1_dX4 = (double*)(mxGPUGetData(k1_dX4));
            dyn8_mex_continuous << <blocksPerGrid, threadsPerBlock >> > (p_k1_dX4, p_X1, p_X2, p_X3, p_X4, p_dX1, p_dX2, p_dX3, p_dX4, 
                                                                         p_in1, p_in2, p_in3, p_in4,
                                                                         m1, m2, m3, m4, l1, l2, l3, l4, g, p_grid_size, p_active_actions);
            step << <blocksPerGrid, threadsPerBlock >> > (p_dX4n, p_dX4, p_k1_dX4, 0.5 * dt, p_grid_size);
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
            k2_X4 = mxGPUCopyGPUArray(dX4n);
            p_k2_X4 = (double*)(mxGPUGetData(k2_X4));
            
        } else if (curr_free_dim == 5) {
            k2_dX1 = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(dX1), mxGPUGetDimensions(dX1), mxGPUGetClassID(dX1), mxGPUGetComplexity(dX1), MX_GPU_DO_NOT_INITIALIZE);
            p_k2_dX1 = (double*)(mxGPUGetData(k2_dX1));
            dyn5_mex_continuous << <blocksPerGrid, threadsPerBlock >> > (p_k2_dX1, p_X1n, p_X2n, p_X3n, p_X4n, p_dX1n, p_dX2n, p_dX3n, p_dX4n, 
                                                                         p_in1, p_in2, p_in3, p_in4,
                                                                         m1, m2, m3, m4, l1, l2, l3, l4, g, p_grid_size, p_active_actions);
        } else if (curr_free_dim == 6) {
            k2_dX2 = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(dX2), mxGPUGetDimensions(dX2), mxGPUGetClassID(dX2), mxGPUGetComplexity(dX2), MX_GPU_DO_NOT_INITIALIZE);
            p_k2_dX2 = (double*)(mxGPUGetData(k2_dX2));
            dyn6_mex_continuous << <blocksPerGrid, threadsPerBlock >> > (p_k2_dX2, p_X1n, p_X2n, p_X3n, p_X4n, p_dX1n, p_dX2n, p_dX3n, p_dX4n, 
                                                                         p_in1, p_in2, p_in3, p_in4,
                                                                         m1, m2, m3, m4, l1, l2, l3, l4, g, p_grid_size, p_active_actions);            
        } else if (curr_free_dim == 7) {
            k2_dX3 = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(dX3), mxGPUGetDimensions(dX3), mxGPUGetClassID(dX3), mxGPUGetComplexity(dX3), MX_GPU_DO_NOT_INITIALIZE);
            p_k2_dX3 = (double*)(mxGPUGetData(k2_dX3));
            dyn7_mex_continuous << <blocksPerGrid, threadsPerBlock >> > (p_k2_dX3, p_X1n, p_X2n, p_X3n, p_X4n, p_dX1n, p_dX2n, p_dX3n, p_dX4n, 
                                                                         p_in1, p_in2, p_in3, p_in4,
                                                                         m1, m2, m3, m4, l1, l2, l3, l4, g, p_grid_size, p_active_actions); 
        } else if (curr_free_dim == 8) {
            k2_dX4 = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(dX4), mxGPUGetDimensions(dX4), mxGPUGetClassID(dX4), mxGPUGetComplexity(dX4), MX_GPU_DO_NOT_INITIALIZE);
            p_k2_dX4 = (double*)(mxGPUGetData(k2_dX4));
            dyn8_mex_continuous << <blocksPerGrid, threadsPerBlock >> > (p_k2_dX4, p_X1n, p_X2n, p_X3n, p_X4n, p_dX1n, p_dX2n, p_dX3n, p_dX4n, 
                                                                         p_in1, p_in2, p_in3, p_in4,
                                                                         m1, m2, m3, m4, l1, l2, l3, l4, g, p_grid_size, p_active_actions);
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
            step << <blocksPerGrid, threadsPerBlock >> > (p_dX1n, p_dX1, p_k2_dX1, 0.5 * dt, p_grid_size);
            
        } else if (curr_free_dim == 6) {
            step << <blocksPerGrid, threadsPerBlock >> > (p_dX2n, p_dX2, p_k2_dX2, 0.5 * dt, p_grid_size);
            
        } else if (curr_free_dim == 7) {
            step << <blocksPerGrid, threadsPerBlock >> > (p_dX3n, p_dX3, p_k2_dX3, 0.5 * dt, p_grid_size);
            
        } else if (curr_free_dim == 8) {
            step << <blocksPerGrid, threadsPerBlock >> > (p_dX4n, p_dX4, p_k2_dX4, 0.5 * dt, p_grid_size);
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
            k3_X4 = mxGPUCopyGPUArray(dX4n);
            p_k3_X4 = (double*)(mxGPUGetData(k3_X4));
            
        } else if (curr_free_dim == 5) {
            k3_dX1 = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(dX1), mxGPUGetDimensions(dX1), mxGPUGetClassID(dX1), mxGPUGetComplexity(dX1), MX_GPU_DO_NOT_INITIALIZE);
            p_k3_dX1 = (double*)(mxGPUGetData(k3_dX1));
            dyn5_mex_continuous << <blocksPerGrid, threadsPerBlock >> > (p_k3_dX1, p_X1n, p_X2n, p_X3n, p_X4n, p_dX1n, p_dX2n, p_dX3n, p_dX4n, 
                                                                         p_in1, p_in2, p_in3, p_in4,
                                                                         m1, m2, m3, m4, l1, l2, l3, l4, g, p_grid_size, p_active_actions);
        } else if (curr_free_dim == 6) {
            k3_dX2 = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(dX2), mxGPUGetDimensions(dX2), mxGPUGetClassID(dX2), mxGPUGetComplexity(dX2), MX_GPU_DO_NOT_INITIALIZE);
            p_k3_dX2 = (double*)(mxGPUGetData(k3_dX2));
            dyn6_mex_continuous << <blocksPerGrid, threadsPerBlock >> > (p_k3_dX2, p_X1n, p_X2n, p_X3n, p_X4n, p_dX1n, p_dX2n, p_dX3n, p_dX4n, 
                                                                         p_in1, p_in2, p_in3, p_in4,
                                                                         m1, m2, m3, m4, l1, l2, l3, l4, g, p_grid_size, p_active_actions);            
        } else if (curr_free_dim == 7) {
            k3_dX3 = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(dX3), mxGPUGetDimensions(dX3), mxGPUGetClassID(dX3), mxGPUGetComplexity(dX3), MX_GPU_DO_NOT_INITIALIZE);
            p_k3_dX3 = (double*)(mxGPUGetData(k3_dX3));
            dyn7_mex_continuous << <blocksPerGrid, threadsPerBlock >> > (p_k3_dX3, p_X1n, p_X2n, p_X3n, p_X4n, p_dX1n, p_dX2n, p_dX3n, p_dX4n, 
                                                                         p_in1, p_in2, p_in3, p_in4,
                                                                         m1, m2, m3, m4, l1, l2, l3, l4, g, p_grid_size, p_active_actions); 
        } else if (curr_free_dim == 8) {
            k3_dX4 = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(dX4), mxGPUGetDimensions(dX4), mxGPUGetClassID(dX4), mxGPUGetComplexity(dX4), MX_GPU_DO_NOT_INITIALIZE);
            p_k3_dX4 = (double*)(mxGPUGetData(k3_dX4));
            dyn8_mex_continuous << <blocksPerGrid, threadsPerBlock >> > (p_k3_dX4, p_X1n, p_X2n, p_X3n, p_X4n, p_dX1n, p_dX2n, p_dX3n, p_dX4n, 
                                                                         p_in1, p_in2, p_in3, p_in4,
                                                                         m1, m2, m3, m4, l1, l2, l3, l4, g, p_grid_size, p_active_actions);
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
            step << <blocksPerGrid, threadsPerBlock >> > (p_dX1n, p_dX1, p_k3_dX1, dt, p_grid_size);
            
        } else if (curr_free_dim == 6) {
            step << <blocksPerGrid, threadsPerBlock >> > (p_dX2n, p_dX2, p_k3_dX2, dt, p_grid_size);
            
        } else if (curr_free_dim == 7) {
            step << <blocksPerGrid, threadsPerBlock >> > (p_dX3n, p_dX3, p_k3_dX3, dt, p_grid_size);
            
        } else if (curr_free_dim == 8) {
            step << <blocksPerGrid, threadsPerBlock >> > (p_dX4n, p_dX4, p_k3_dX4, dt, p_grid_size);
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
            k4_X4 = mxGPUCopyGPUArray(dX4n);
            p_k4_X4 = (double*)(mxGPUGetData(k4_X4));
            
        } else if (curr_free_dim == 5) {
            k4_dX1 = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(dX1), mxGPUGetDimensions(dX1), mxGPUGetClassID(dX1), mxGPUGetComplexity(dX1), MX_GPU_DO_NOT_INITIALIZE);
            p_k4_dX1 = (double*)(mxGPUGetData(k4_dX1));
            dyn5_mex_continuous << <blocksPerGrid, threadsPerBlock >> > (p_k4_dX1, p_X1n, p_X2n, p_X3n, p_X4n, p_dX1n, p_dX2n, p_dX3n, p_dX4n, 
                                                                         p_in1, p_in2, p_in3, p_in4,
                                                                         m1, m2, m3, m4, l1, l2, l3, l4, g, p_grid_size, p_active_actions);
        } else if (curr_free_dim == 6) {
            k4_dX2 = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(dX2), mxGPUGetDimensions(dX2), mxGPUGetClassID(dX2), mxGPUGetComplexity(dX2), MX_GPU_DO_NOT_INITIALIZE);
            p_k4_dX2 = (double*)(mxGPUGetData(k4_dX2));
            dyn6_mex_continuous << <blocksPerGrid, threadsPerBlock >> > (p_k4_dX2, p_X1n, p_X2n, p_X3n, p_X4n, p_dX1n, p_dX2n, p_dX3n, p_dX4n, 
                                                                         p_in1, p_in2, p_in3, p_in4,
                                                                         m1, m2, m3, m4, l1, l2, l3, l4, g, p_grid_size, p_active_actions);            
        } else if (curr_free_dim == 7) {
            k4_dX3 = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(dX3), mxGPUGetDimensions(dX3), mxGPUGetClassID(dX3), mxGPUGetComplexity(dX3), MX_GPU_DO_NOT_INITIALIZE);
            p_k4_dX3 = (double*)(mxGPUGetData(k4_dX3));
            dyn7_mex_continuous << <blocksPerGrid, threadsPerBlock >> > (p_k4_dX3, p_X1n, p_X2n, p_X3n, p_X4n, p_dX1n, p_dX2n, p_dX3n, p_dX4n, 
                                                                         p_in1, p_in2, p_in3, p_in4,
                                                                         m1, m2, m3, m4, l1, l2, l3, l4, g, p_grid_size, p_active_actions); 
        } else if (curr_free_dim == 8) {
            k4_dX4 = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(dX4), mxGPUGetDimensions(dX4), mxGPUGetClassID(dX4), mxGPUGetComplexity(dX4), MX_GPU_DO_NOT_INITIALIZE);
            p_k4_dX4 = (double*)(mxGPUGetData(k4_dX4));
            dyn8_mex_continuous << <blocksPerGrid, threadsPerBlock >> > (p_k4_dX4, p_X1n, p_X2n, p_X3n, p_X4n, p_dX1n, p_dX2n, p_dX3n, p_dX4n, 
                                                                         p_in1, p_in2, p_in3, p_in4,
                                                                         m1, m2, m3, m4, l1, l2, l3, l4, g, p_grid_size, p_active_actions);
        } 
    }

    // RK4 Ouput
    for (int32_t ii=0; ii < num_x_dims[0]; ii++) {
        curr_free_dim = int32_t(p_x_dims_free[ii]);
        if (curr_free_dim == 1) {
            rk4_wbounds << <blocksPerGrid, threadsPerBlock >> > (p_X1n, p_X1, p_k1_X1, p_k2_X1, p_k3_X1, p_k4_X1, dt, p_limits[0], p_limits[8], p_grid_size);
            mxGPUDestroyGPUArray(k2_X1);
            mxGPUDestroyGPUArray(k3_X1);
            mxGPUDestroyGPUArray(k4_X1);
        } else if (curr_free_dim == 2) {
            rk4_wbounds << <blocksPerGrid, threadsPerBlock >> > (p_X2n, p_X2, p_k1_X2, p_k2_X2, p_k3_X2, p_k4_X2, dt, p_limits[1], p_limits[9], p_grid_size);
            mxGPUDestroyGPUArray(k2_X2);
            mxGPUDestroyGPUArray(k3_X2);
            mxGPUDestroyGPUArray(k4_X2);
        } else if (curr_free_dim == 3) {
            rk4_wbounds << <blocksPerGrid, threadsPerBlock >> > (p_X3n, p_X3, p_k1_X3, p_k2_X3, p_k3_X3, p_k4_X3, dt, p_limits[2], p_limits[10], p_grid_size);
            mxGPUDestroyGPUArray(k2_X3);
            mxGPUDestroyGPUArray(k3_X3);
            mxGPUDestroyGPUArray(k4_X3);
        } else if (curr_free_dim == 4) {
            rk4_wbounds << <blocksPerGrid, threadsPerBlock >> > (p_X4n, p_X4, p_k1_X4, p_k2_X4, p_k3_X4, p_k4_X4, dt, p_limits[3], p_limits[11], p_grid_size);
            mxGPUDestroyGPUArray(k2_X4);
            mxGPUDestroyGPUArray(k3_X4);
            mxGPUDestroyGPUArray(k4_X4);
        } else if (curr_free_dim == 5) {
            rk4_wbounds << <blocksPerGrid, threadsPerBlock >> > (p_dX1n, p_dX1, p_k1_dX1, p_k2_dX1, p_k3_dX1, p_k4_dX1, dt, p_limits[4], p_limits[12], p_grid_size);
            mxGPUDestroyGPUArray(k1_dX1);
            mxGPUDestroyGPUArray(k2_dX1);
            mxGPUDestroyGPUArray(k3_dX1);
            mxGPUDestroyGPUArray(k4_dX1);
        } else if (curr_free_dim == 6) {
            rk4_wbounds << <blocksPerGrid, threadsPerBlock >> > (p_dX2n, p_dX2, p_k1_dX2, p_k2_dX2, p_k3_dX2, p_k4_dX2, dt, p_limits[5], p_limits[13], p_grid_size);
            mxGPUDestroyGPUArray(k1_dX2);
            mxGPUDestroyGPUArray(k2_dX2);
            mxGPUDestroyGPUArray(k3_dX2);
            mxGPUDestroyGPUArray(k4_dX2);
        } else if (curr_free_dim == 7) {
            rk4_wbounds << <blocksPerGrid, threadsPerBlock >> > (p_dX3n, p_dX3, p_k1_dX3, p_k2_dX3, p_k3_dX3, p_k4_dX3, dt, p_limits[6], p_limits[14], p_grid_size);
            mxGPUDestroyGPUArray(k1_dX3);
            mxGPUDestroyGPUArray(k2_dX3);
            mxGPUDestroyGPUArray(k3_dX3);
            mxGPUDestroyGPUArray(k4_dX3);
        } else if (curr_free_dim == 8) {
            rk4_wbounds << <blocksPerGrid, threadsPerBlock >> > (p_dX4n, p_dX4, p_k1_dX4, p_k2_dX4, p_k3_dX4, p_k4_dX4, dt, p_limits[7], p_limits[15], p_grid_size);
            mxGPUDestroyGPUArray(k1_dX4);
            mxGPUDestroyGPUArray(k2_dX4);
            mxGPUDestroyGPUArray(k3_dX4);
            mxGPUDestroyGPUArray(k4_dX4);
        } 
    }
 
    /* Wrap the result up as a MATLAB gpuArray for return. */
    plhs[0] = mxGPUCreateMxArrayOnGPU(X1n);
    plhs[1] = mxGPUCreateMxArrayOnGPU(X2n);
    plhs[2] = mxGPUCreateMxArrayOnGPU(X3n);
    plhs[3] = mxGPUCreateMxArrayOnGPU(X4n);
    plhs[4] = mxGPUCreateMxArrayOnGPU(dX1n);
    plhs[5] = mxGPUCreateMxArrayOnGPU(dX2n);
    plhs[6] = mxGPUCreateMxArrayOnGPU(dX3n);
    plhs[7] = mxGPUCreateMxArrayOnGPU(dX4n);

    /*
     * The mxGPUArray pointers are host-side structures that refer to device
     * data. These must be destroyed before leaving the MEX function.
     */
    mxGPUDestroyGPUArray(X1);
    mxGPUDestroyGPUArray(X2);
    mxGPUDestroyGPUArray(X3);
    mxGPUDestroyGPUArray(X4);
    mxGPUDestroyGPUArray(dX1);
    mxGPUDestroyGPUArray(dX2);
    mxGPUDestroyGPUArray(dX3);
    mxGPUDestroyGPUArray(dX4);
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
    mxGPUDestroyGPUArray(dX1n);
    mxGPUDestroyGPUArray(dX2n);
    mxGPUDestroyGPUArray(dX3n);
    mxGPUDestroyGPUArray(dX4n);
}
