__global__ void calc_average4(double* Xi1, const double* Xi2, const double* Xi3, const double* Xi4, const double* V, const int query_grid_size,
    const int Nx1, const int Nx2, const int Nx3, const int Nx4,
    const double dx1, const double dx2, const double dx3, const double dx4,
    const double minx1, const double minx2, const double minx3, const double minx4, const int* corners_index)
{
    int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x * blockDim.y
        + (threadIdx.x + threadIdx.y * blockDim.x);
    int num_threads = blockDim.x * blockDim.y * gridDim.x * gridDim.y;

//     int query_grid_size = Nx1 * Nx2 * Nx3 * Nx4;
    int loc1, loc2, loc3, loc4, scalar_loc;
    double xi2, xi3, xi4;

    while (index < query_grid_size)
    {
        Xi1[index] = (Xi1[index] - minx1) / dx1;
        xi2 = (Xi2[index] - minx2) / dx2;
        xi3 = (Xi3[index] - minx3) / dx3;
        xi4 = (Xi4[index] - minx4) / dx4;

        loc1 = min(Nx1 - 2, int(floor(Xi1[index])));
        loc2 = min(Nx2 - 2, int(floor(xi2)));
        loc3 = min(Nx3 - 2, int(floor(xi3)));
        loc4 = min(Nx4 - 2, int(floor(xi4)));

        Xi1[index] = Xi1[index] - loc1; // Weights from the 0 corner
        xi2 = xi2 - loc2;
        xi3 = xi3 - loc3;
        xi4 = xi4 - loc4;

        scalar_loc = loc1 + Nx1 * (loc2 + Nx2 * (loc3 + Nx3 * loc4));

        Xi1[index] = (1 - Xi1[index]) * ((1 - xi2) * ((1 - xi3) * ((1 - xi4) * V[scalar_loc + corners_index[0]]
                                                                       + xi4 * V[scalar_loc + corners_index[1]])
                                                          + xi3 * ((1 - xi4) * V[scalar_loc + corners_index[2]]
                                                                       + xi4 * V[scalar_loc + corners_index[3]]))
                                             + xi2 * ((1 - xi3) * ((1 - xi4) * V[scalar_loc + corners_index[4]]
                                                                       + xi4 * V[scalar_loc + corners_index[5]])
                                                          + xi3 * ((1 - xi4) * V[scalar_loc + corners_index[6]]
                                                                       + xi4 * V[scalar_loc + corners_index[7]])))
                         + Xi1[index] * ((1 - xi2) * ((1 - xi3) * ((1 - xi4) * V[scalar_loc + corners_index[8]]
                                                                       + xi4 * V[scalar_loc + corners_index[9]])
                                                          + xi3 * ((1 - xi4) * V[scalar_loc + corners_index[10]]
                                                                       + xi4 * V[scalar_loc + corners_index[11]]))
                                             + xi2 * ((1 - xi3) * ((1 - xi4) * V[scalar_loc + corners_index[12]]
                                                                       + xi4 * V[scalar_loc + corners_index[13]])
                                                          + xi3 * ((1 - xi4) * V[scalar_loc + corners_index[14]]
                                                                       + xi4 * V[scalar_loc + corners_index[15]])));
        index = index + num_threads;
    }
}