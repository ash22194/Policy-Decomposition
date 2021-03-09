__global__ void calc_average6(double* Xi1, const double* Xi2, const double* Xi3, 
                              const double* Xi4, const double* Xi5, const double* Xi6, const double* V, const int query_grid_size,
                              const int Nx1, const int Nx2, const int Nx3, 
                              const int Nx4, const int Nx5, const int Nx6,
                              const double dx1, const double dx2, const double dx3, 
                              const double dx4, const double dx5, const double dx6,
                              const double minx1, const double minx2, const double minx3, 
                              const double minx4, const double minx5, const double minx6, const int* corners_index)
{
    int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x * blockDim.y
        + (threadIdx.x + threadIdx.y * blockDim.x);
    int num_threads = blockDim.x * blockDim.y * gridDim.x * gridDim.y;

//     int query_grid_size = Nx1 * Nx2 * Nx3 * Nx4 * Nx5 * Nx6;
    int loc1, loc2, loc3, loc4, loc5, loc6, scalar_loc;
    double xi2, xi3, xi4, xi5, xi6;

    while (index < query_grid_size)
    {
        Xi1[index] = (Xi1[index] - minx1) / dx1;
        xi2 = (Xi2[index] - minx2) / dx2;
        xi3 = (Xi3[index] - minx3) / dx3;
        xi4 = (Xi4[index] - minx4) / dx4;
        xi5 = (Xi5[index] - minx5) / dx5;
        xi6 = (Xi6[index] - minx6) / dx6;

        loc1 = min(Nx1 - 2, int(floor(Xi1[index])));
        loc2 = min(Nx2 - 2, int(floor(xi2)));
        loc3 = min(Nx3 - 2, int(floor(xi3)));
        loc4 = min(Nx4 - 2, int(floor(xi4)));
        loc5 = min(Nx5 - 2, int(floor(xi5)));
        loc6 = min(Nx6 - 2, int(floor(xi6)));

        Xi1[index] = Xi1[index] - loc1; // Weights from the 0 corner
        xi2 = xi2 - loc2;
        xi3 = xi3 - loc3;
        xi4 = xi4 - loc4;
        xi5 = xi5 - loc5;
        xi6 = xi6 - loc6;

        scalar_loc = loc1 + Nx1 * (loc2 + Nx2 * (loc3 + Nx3 * (loc4 + Nx4 * (loc5 + Nx5 * loc6))));

        Xi1[index] = (1 - Xi1[index]) * ((1 - xi2) * ((1 - xi3) * ((1 - xi4) * ((1 - xi5) * ((1 - xi6) * V[scalar_loc + corners_index[0]]
                                                                                                 + xi6 * V[scalar_loc + corners_index[1]])
                                                                                    + xi5 * ((1 - xi6) * V[scalar_loc + corners_index[2]]
                                                                                                 + xi6 * V[scalar_loc + corners_index[3]]))
                                                                       + xi4 * ((1 - xi5) * ((1 - xi6) * V[scalar_loc + corners_index[4]]
                                                                                                 + xi6 * V[scalar_loc + corners_index[5]])
                                                                                    + xi5 * ((1 - xi6) * V[scalar_loc + corners_index[6]]
                                                                                                 + xi6 * V[scalar_loc + corners_index[7]])))
                                                          + xi3 * ((1 - xi4) * ((1 - xi5) * ((1 - xi6) * V[scalar_loc + corners_index[8]]
                                                                                                 + xi6 * V[scalar_loc + corners_index[9]])
                                                                                    + xi5 * ((1 - xi6) * V[scalar_loc + corners_index[10]]
                                                                                                 + xi6 * V[scalar_loc + corners_index[11]]))
                                                                       + xi4 * ((1 - xi5) * ((1 - xi6) * V[scalar_loc + corners_index[12]]
                                                                                                 + xi6 * V[scalar_loc + corners_index[13]])
                                                                                    + xi5 * ((1 - xi6) * V[scalar_loc + corners_index[14]]
                                                                                                 + xi6 * V[scalar_loc + corners_index[15]]))))
                                             + xi2 * ((1 - xi3) * ((1 - xi4) * ((1 - xi5) * ((1 - xi6) * V[scalar_loc + corners_index[16]]
                                                                                                 + xi6 * V[scalar_loc + corners_index[17]])
                                                                                    + xi5 * ((1 - xi6) * V[scalar_loc + corners_index[18]]
                                                                                                 + xi6 * V[scalar_loc + corners_index[19]]))
                                                                       + xi4 * ((1 - xi5) * ((1 - xi6) * V[scalar_loc + corners_index[20]]
                                                                                                 + xi6 * V[scalar_loc + corners_index[21]])
                                                                                    + xi5 * ((1 - xi6) * V[scalar_loc + corners_index[22]]
                                                                                                 + xi6 * V[scalar_loc + corners_index[23]])))
                                                          + xi3 * ((1 - xi4) * ((1 - xi5) * ((1 - xi6) * V[scalar_loc + corners_index[24]]
                                                                                                 + xi6 * V[scalar_loc + corners_index[25]])
                                                                                    + xi5 * ((1 - xi6) * V[scalar_loc + corners_index[26]]
                                                                                                 + xi6 * V[scalar_loc + corners_index[27]]))
                                                                       + xi4 * ((1 - xi5) * ((1 - xi6) * V[scalar_loc + corners_index[28]]
                                                                                                 + xi6 * V[scalar_loc + corners_index[29]])
                                                                                    + xi5 * ((1 - xi6) * V[scalar_loc + corners_index[30]]
                                                                                                 + xi6 * V[scalar_loc + corners_index[31]])))))
                         + Xi1[index] * ((1 - xi2) * ((1 - xi3) * ((1 - xi4) * ((1 - xi5) * ((1 - xi6) * V[scalar_loc + corners_index[32]]
                                                                                                 + xi6 * V[scalar_loc + corners_index[33]])
                                                                                    + xi5 * ((1 - xi6) * V[scalar_loc + corners_index[34]]
                                                                                                 + xi6 * V[scalar_loc + corners_index[35]]))
                                                                       + xi4 * ((1 - xi5) * ((1 - xi6) * V[scalar_loc + corners_index[36]]
                                                                                                 + xi6 * V[scalar_loc + corners_index[37]])
                                                                                    + xi5 * ((1 - xi6) * V[scalar_loc + corners_index[38]]
                                                                                                 + xi6 * V[scalar_loc + corners_index[39]])))
                                                          + xi3 * ((1 - xi4) * ((1 - xi5) * ((1 - xi6) * V[scalar_loc + corners_index[40]]
                                                                                                 + xi6 * V[scalar_loc + corners_index[41]])
                                                                                    + xi5 * ((1 - xi6) * V[scalar_loc + corners_index[42]]
                                                                                                 + xi6 * V[scalar_loc + corners_index[43]]))
                                                                       + xi4 * ((1 - xi5) * ((1 - xi6) * V[scalar_loc + corners_index[44]]
                                                                                                 + xi6 * V[scalar_loc + corners_index[45]])
                                                                                    + xi5 * ((1 - xi6) * V[scalar_loc + corners_index[46]]
                                                                                                 + xi6 * V[scalar_loc + corners_index[47]]))))
                                             + xi2 * ((1 - xi3) * ((1 - xi4) * ((1 - xi5) * ((1 - xi6) * V[scalar_loc + corners_index[48]]
                                                                                                 + xi6 * V[scalar_loc + corners_index[49]])
                                                                                    + xi5 * ((1 - xi6) * V[scalar_loc + corners_index[50]]
                                                                                                 + xi6 * V[scalar_loc + corners_index[51]]))
                                                                       + xi4 * ((1 - xi5) * ((1 - xi6) * V[scalar_loc + corners_index[52]]
                                                                                                 + xi6 * V[scalar_loc + corners_index[53]])
                                                                                    + xi5 * ((1 - xi6) * V[scalar_loc + corners_index[54]]
                                                                                                 + xi6 * V[scalar_loc + corners_index[55]])))
                                                          + xi3 * ((1 - xi4) * ((1 - xi5) * ((1 - xi6) * V[scalar_loc + corners_index[56]]
                                                                                                 + xi6 * V[scalar_loc + corners_index[57]])
                                                                                    + xi5 * ((1 - xi6) * V[scalar_loc + corners_index[58]]
                                                                                                 + xi6 * V[scalar_loc + corners_index[59]]))
                                                                       + xi4 * ((1 - xi5) * ((1 - xi6) * V[scalar_loc + corners_index[60]]
                                                                                                 + xi6 * V[scalar_loc + corners_index[61]])
                                                                                    + xi5 * ((1 - xi6) * V[scalar_loc + corners_index[62]]
                                                                                                 + xi6 * V[scalar_loc + corners_index[63]])))));
        index = index + num_threads;
    }
}