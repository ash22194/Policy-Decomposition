__global__ void calc_average7(double* Xi1, const double* Xi2, const double* Xi3,
                            const double* Xi4, const double* Xi5, const double* Xi6, const double* Xi7, const double* V, const int query_grid_size,
                            const int Nx1, const int Nx2, const int Nx3,
                            const int Nx4, const int Nx5, const int Nx6, const int Nx7,
                            const double dx1, const double dx2, const double dx3,
                            const double dx4, const double dx5, const double dx6, const double dx7,
                            const double minx1, const double minx2, const double minx3,
                            const double minx4, const double minx5, const double minx6, const double minx7, const int* corners_index)
{
    int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x * blockDim.y
        + (threadIdx.x + threadIdx.y * blockDim.x);
    int num_threads = blockDim.x * blockDim.y * gridDim.x * gridDim.y;

//     int query_grid_size = Nx1 * Nx2 * Nx3 * Nx4 * Nx5 * Nx6 * Nx7;
    int loc1, loc2, loc3, loc4, loc5, loc6, loc7, scalar_loc;
    double xi2, xi3, xi4, xi5, xi6, xi7;

    while (index < query_grid_size)
    {
        Xi1[index] = (Xi1[index] - minx1) / dx1;
        xi2 = (Xi2[index] - minx2) / dx2;
        xi3 = (Xi3[index] - minx3) / dx3;
        xi4 = (Xi4[index] - minx4) / dx4;
        xi5 = (Xi5[index] - minx5) / dx5;
        xi6 = (Xi6[index] - minx6) / dx6;
        xi7 = (Xi7[index] - minx7) / dx7;

        loc1 = min(Nx1 - 2, int(floor(Xi1[index])));
        loc2 = min(Nx2 - 2, int(floor(xi2)));
        loc3 = min(Nx3 - 2, int(floor(xi3)));
        loc4 = min(Nx4 - 2, int(floor(xi4)));
        loc5 = min(Nx5 - 2, int(floor(xi5)));
        loc6 = min(Nx6 - 2, int(floor(xi6)));
        loc7 = min(Nx7 - 2, int(floor(xi7)));

        Xi1[index] = Xi1[index] - loc1; // Weights from the 0 corner
        xi2 = xi2 - loc2;
        xi3 = xi3 - loc3;
        xi4 = xi4 - loc4;
        xi5 = xi5 - loc5;
        xi6 = xi6 - loc6;
        xi7 = xi7 - loc7;

        scalar_loc = loc1 + Nx1 * (loc2 + Nx2 * (loc3 + Nx3 * (loc4 + Nx4 * (loc5 + Nx5 * (loc6 + Nx6 * loc7)))));

        Xi1[index] = (1 - Xi1[index]) * ((1 - xi2) * ((1 - xi3) * ((1 - xi4) * ((1 - xi5) * ((1 - xi6) * ((1 - xi7) * V[scalar_loc + corners_index[0]]
                                                                                                              + xi7 * V[scalar_loc + corners_index[1]])   
                                                                                                 + xi6 * ((1 - xi7) * V[scalar_loc + corners_index[2]]
                                                                                                              + xi7 * V[scalar_loc + corners_index[3]]))
                                                                                    + xi5 * ((1 - xi6) * ((1 - xi7) * V[scalar_loc + corners_index[4]]
                                                                                                              + xi7 * V[scalar_loc + corners_index[5]])
                                                                                                 + xi6 * ((1 - xi7) * V[scalar_loc + corners_index[6]]
                                                                                                              + xi7 * V[scalar_loc + corners_index[7]])))
                                                                       + xi4 * ((1 - xi5) * ((1 - xi6) * ((1 - xi7) * V[scalar_loc + corners_index[8]]
                                                                                                              + xi7 * V[scalar_loc + corners_index[9]])
                                                                                                 + xi6 * ((1 - xi7) * V[scalar_loc + corners_index[10]]
                                                                                                              + xi7 * V[scalar_loc + corners_index[11]]))
                                                                                    + xi5 * ((1 - xi6) * ((1 - xi7) * V[scalar_loc + corners_index[12]]
                                                                                                              + xi7 * V[scalar_loc + corners_index[13]])
                                                                                                 + xi6 * ((1 - xi7) * V[scalar_loc + corners_index[14]]
                                                                                                              + xi7 * V[scalar_loc + corners_index[15]]))))
                                                          + xi3 * ((1 - xi4) * ((1 - xi5) * ((1 - xi6) * ((1 - xi7) * V[scalar_loc + corners_index[16]]
                                                                                                              + xi7 * V[scalar_loc + corners_index[17]])
                                                                                                 + xi6 * ((1 - xi7) * V[scalar_loc + corners_index[18]]
                                                                                                              + xi7 * V[scalar_loc + corners_index[19]]))
                                                                                    + xi5 * ((1 - xi6) * ((1 - xi7) * V[scalar_loc + corners_index[20]]
                                                                                                              + xi7 * V[scalar_loc + corners_index[21]])
                                                                                                 + xi6 * ((1 - xi7) * V[scalar_loc + corners_index[22]]
                                                                                                              + xi7 * V[scalar_loc + corners_index[23]])))
                                                                       + xi4 * ((1 - xi5) * ((1 - xi6) * ((1 - xi7) * V[scalar_loc + corners_index[24]]
                                                                                                              + xi7 * V[scalar_loc + corners_index[25]])
                                                                                                 + xi6 * ((1 - xi7) * V[scalar_loc + corners_index[26]]
                                                                                                              + xi7 * V[scalar_loc + corners_index[27]]))
                                                                                    + xi5 * ((1 - xi6) * ((1 - xi7) * V[scalar_loc + corners_index[28]]
                                                                                                              + xi7 * V[scalar_loc + corners_index[29]])
                                                                                                 + xi6 * ((1 - xi7) * V[scalar_loc + corners_index[30]]
                                                                                                              + xi7 * V[scalar_loc + corners_index[31]])))))
                                             + xi2 * ((1 - xi3) * ((1 - xi4) * ((1 - xi5) * ((1 - xi6) * ((1 - xi7) * V[scalar_loc + corners_index[32]]
                                                                                                              + xi7 * V[scalar_loc + corners_index[33]])
                                                                                                 + xi6 * ((1 - xi7) * V[scalar_loc + corners_index[34]]
                                                                                                              + xi7 * V[scalar_loc + corners_index[35]]))
                                                                                    + xi5 * ((1 - xi6) * ((1 - xi7) * V[scalar_loc + corners_index[36]]
                                                                                                              + xi7 * V[scalar_loc + corners_index[37]])
                                                                                                 + xi6 * ((1 - xi7) * V[scalar_loc + corners_index[38]]
                                                                                                              + xi7 * V[scalar_loc + corners_index[39]])))
                                                                       + xi4 * ((1 - xi5) * ((1 - xi6) * ((1 - xi7) * V[scalar_loc + corners_index[40]]
                                                                                                              + xi7 * V[scalar_loc + corners_index[41]])
                                                                                                 + xi6 * ((1 - xi7) * V[scalar_loc + corners_index[42]]
                                                                                                              + xi7 * V[scalar_loc + corners_index[43]]))
                                                                                    + xi5 * ((1 - xi6) * ((1 - xi7) * V[scalar_loc + corners_index[44]]
                                                                                                              + xi7 * V[scalar_loc + corners_index[45]])
                                                                                                 + xi6 * ((1 - xi7) * V[scalar_loc + corners_index[46]]
                                                                                                              + xi7 * V[scalar_loc + corners_index[47]])))) 
                                                          + xi3 * ((1 - xi4) * ((1 - xi5) * ((1 - xi6) * ((1 - xi7) * V[scalar_loc + corners_index[48]]
                                                                                                              + xi7 * V[scalar_loc + corners_index[49]])
                                                                                                 + xi6 * ((1 - xi7) * V[scalar_loc + corners_index[50]]
                                                                                                              + xi7 * V[scalar_loc + corners_index[51]]))
                                                                                    + xi5 * ((1 - xi6) * ((1 - xi7) * V[scalar_loc + corners_index[52]]
                                                                                                              + xi7 * V[scalar_loc + corners_index[53]])
                                                                                                 + xi6 * ((1 - xi7) * V[scalar_loc + corners_index[54]]
                                                                                                              + xi7 * V[scalar_loc + corners_index[55]])))
                                                                       + xi4 * ((1 - xi5) * ((1 - xi6) * ((1 - xi7) * V[scalar_loc + corners_index[56]]
                                                                                                              + xi7 * V[scalar_loc + corners_index[57]])
                                                                                                 + xi6 * ((1 - xi7) * V[scalar_loc + corners_index[58]]
                                                                                                              + xi7 * V[scalar_loc + corners_index[59]]))
                                                                                    + xi5 * ((1 - xi6) * ((1 - xi7) * V[scalar_loc + corners_index[60]]                                                                                                              
                                                                                                              + xi7 * V[scalar_loc + corners_index[61]])
                                                                                                 + xi6 * ((1 - xi7) * V[scalar_loc + corners_index[62]]
                                                                                                              + xi7 * V[scalar_loc + corners_index[63]]))))))
                         + Xi1[index] * ((1 - xi2) * ((1 - xi3) * ((1 - xi4) * ((1 - xi5) * ((1 - xi6) * ((1 - xi7) * V[scalar_loc + corners_index[64]]
                                                                                                              + xi7 * V[scalar_loc + corners_index[65]])
                                                                                                 + xi6 * ((1 - xi7) * V[scalar_loc + corners_index[66]]
                                                                                                              + xi7 * V[scalar_loc + corners_index[67]]))
                                                                                    + xi5 * ((1 - xi6) * ((1 - xi7) * V[scalar_loc + corners_index[68]]                   
                                                                                                              + xi7 * V[scalar_loc + corners_index[69]])
                                                                                                 + xi6 * ((1 - xi7) * V[scalar_loc + corners_index[70]]
                                                                                                              + xi7 * V[scalar_loc + corners_index[71]])))
                                                                       + xi4 * ((1 - xi5) * ((1 - xi6) * ((1 - xi7) * V[scalar_loc + corners_index[72]]
                                                                                                              + xi7 * V[scalar_loc + corners_index[73]])
                                                                                                 + xi6 * ((1 - xi7) * V[scalar_loc + corners_index[74]]
                                                                                                              + xi7 * V[scalar_loc + corners_index[75]]))
                                                                                    + xi5 * ((1 - xi6) * ((1 - xi7) * V[scalar_loc + corners_index[76]]
                                                                                                              + xi7 * V[scalar_loc + corners_index[77]])
                                                                                                 + xi6 * ((1 - xi7) * V[scalar_loc + corners_index[78]]
                                                                                                              + xi7 * V[scalar_loc + corners_index[79]]))))
                                                          + xi3 * ((1 - xi4) * ((1 - xi5) * ((1 - xi6) * ((1 - xi7) * V[scalar_loc + corners_index[80]]
                                                                                                              + xi7 * V[scalar_loc + corners_index[81]])
                                                                                                 + xi6 * ((1 - xi7) * V[scalar_loc + corners_index[82]]
                                                                                                              + xi7 * V[scalar_loc + corners_index[83]]))
                                                                                    + xi5 * ((1 - xi6) * ((1 - xi7) * V[scalar_loc + corners_index[84]]
                                                                                                              + xi7 * V[scalar_loc + corners_index[85]])
                                                                                                 + xi6 * ((1 - xi7) * V[scalar_loc + corners_index[86]]
                                                                                                              + xi7 * V[scalar_loc + corners_index[87]])))
                                                                       + xi4 * ((1 - xi5) * ((1 - xi6) * ((1 - xi7) * V[scalar_loc + corners_index[88]]
                                                                                                              + xi7 * V[scalar_loc + corners_index[89]])
                                                                                                 + xi6 * ((1 - xi7) * V[scalar_loc + corners_index[90]]
                                                                                                              + xi7 * V[scalar_loc + corners_index[91]]))
                                                                                    + xi5 * ((1 - xi6) * ((1 - xi7) * V[scalar_loc + corners_index[92]]
                                                                                                              + xi7 * V[scalar_loc + corners_index[93]])
                                                                                                 + xi6 * ((1 - xi7) * V[scalar_loc + corners_index[94]]
                                                                                                              + xi7 * V[scalar_loc + corners_index[95]])))))
                                             + xi2 * ((1 - xi3) * ((1 - xi4) * ((1 - xi5) * ((1 - xi6) * ((1 - xi7) * V[scalar_loc + corners_index[96]]
                                                                                                              + xi7 * V[scalar_loc + corners_index[97]])
                                                                                                 + xi6 * ((1 - xi7) * V[scalar_loc + corners_index[98]]
                                                                                                              + xi7 * V[scalar_loc + corners_index[99]]))
                                                                                    + xi5 * ((1 - xi6) * ((1 - xi7) * V[scalar_loc + corners_index[100]]
                                                                                                              + xi7 * V[scalar_loc + corners_index[101]])
                                                                                                 + xi6 * ((1 - xi7) * V[scalar_loc + corners_index[102]]
                                                                                                              + xi7 * V[scalar_loc + corners_index[103]])))
                                                                       + xi4 * ((1 - xi5) * ((1 - xi6) * ((1 - xi7) * V[scalar_loc + corners_index[104]]
                                                                                                              + xi7 * V[scalar_loc + corners_index[105]])
                                                                                                 + xi6 * ((1 - xi7) * V[scalar_loc + corners_index[106]]
                                                                                                              + xi7 * V[scalar_loc + corners_index[107]]))
                                                                                    + xi5 * ((1 - xi6) * ((1 - xi7) * V[scalar_loc + corners_index[108]]
                                                                                                              + xi7 * V[scalar_loc + corners_index[109]])
                                                                                                 + xi6 * ((1 - xi7) * V[scalar_loc + corners_index[110]]
                                                                                                              + xi7 * V[scalar_loc + corners_index[111]]))))
                                                          + xi3 * ((1 - xi4) * ((1 - xi5) * ((1 - xi6) * ((1 - xi7) * V[scalar_loc + corners_index[112]]
                                                                                                              + xi7 * V[scalar_loc + corners_index[113]])
                                                                                                 + xi6 * ((1 - xi7) * V[scalar_loc + corners_index[114]]
                                                                                                              + xi7 * V[scalar_loc + corners_index[115]]))
                                                                                    + xi5 * ((1 - xi6) * ((1 - xi7) * V[scalar_loc + corners_index[116]]
                                                                                                              + xi7 * V[scalar_loc + corners_index[117]])
                                                                                                 + xi6 * ((1 - xi7) * V[scalar_loc + corners_index[118]]
                                                                                                              + xi7 * V[scalar_loc + corners_index[119]])))
                                                                       + xi4 * ((1 - xi5) * ((1 - xi6) * ((1 - xi7) * V[scalar_loc + corners_index[120]]
                                                                                                              + xi7 * V[scalar_loc + corners_index[121]])
                                                                                                 + xi6 * ((1 - xi7) * V[scalar_loc + corners_index[122]]
                                                                                                              + xi7 * V[scalar_loc + corners_index[123]]))
                                                                                    + xi5 * ((1 - xi6) * ((1 - xi7) * V[scalar_loc + corners_index[124]]
                                                                                                              + xi7 * V[scalar_loc + corners_index[125]])
                                                                                                 + xi6 * ((1 - xi7) * V[scalar_loc + corners_index[126]]
                                                                                                              + xi7 * V[scalar_loc + corners_index[127]]))))));
        index = index + num_threads;
    }
}