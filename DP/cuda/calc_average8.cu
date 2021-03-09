__global__ void calc_average8(double* Xi1, const double* Xi2, const double* Xi3, const double* Xi4, 
    const double* Xi5, const double* Xi6, const double* Xi7, const double* Xi8, const double* V, const int query_grid_size,
    const int Nx1, const int Nx2, const int Nx3, const int Nx4, 
    const int Nx5, const int Nx6, const int Nx7, const int Nx8,
    const double dx1, const double dx2, const double dx3, const double dx4, 
    const double dx5, const double dx6, const double dx7, const double dx8,
    const double minx1, const double minx2, const double minx3, const double minx4, 
    const double minx5, const double minx6, const double minx7, const double minx8, const int* corners_index)
{
    int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x * blockDim.y
        + (threadIdx.x + threadIdx.y * blockDim.x);
    int num_threads = blockDim.x * blockDim.y * gridDim.x * gridDim.y;

//     int query_grid_size = Nx1 * Nx2 * Nx3 * Nx4 * Nx5 * Nx6 * Nx7 * Nx8;
    int loc1, loc2, loc3, loc4, loc5, loc6, loc7, loc8, scalar_loc;
    double xi2, xi3, xi4, xi5, xi6, xi7, xi8;

    while (index < query_grid_size)
    {
        Xi1[index] = (Xi1[index] - minx1) / dx1;
        xi2 = (Xi2[index] - minx2) / dx2;
        xi3 = (Xi3[index] - minx3) / dx3;
        xi4 = (Xi4[index] - minx4) / dx4;
        xi5 = (Xi5[index] - minx5) / dx5;
        xi6 = (Xi6[index] - minx6) / dx6;
        xi7 = (Xi7[index] - minx7) / dx7;
        xi8 = (Xi8[index] - minx8) / dx8;

        loc1 = min(Nx1 - 2, int(floor(Xi1[index])));
        loc2 = min(Nx2 - 2, int(floor(xi2)));
        loc3 = min(Nx3 - 2, int(floor(xi3)));
        loc4 = min(Nx4 - 2, int(floor(xi4)));
        loc5 = min(Nx5 - 2, int(floor(xi5)));
        loc6 = min(Nx6 - 2, int(floor(xi6)));
        loc7 = min(Nx7 - 2, int(floor(xi7)));
        loc8 = min(Nx8 - 2, int(floor(xi8)));

        Xi1[index] = Xi1[index] - loc1; // Weights from the 0 corner
        xi2 = xi2 - loc2;
        xi3 = xi3 - loc3;
        xi4 = xi4 - loc4;
        xi5 = xi5 - loc5;
        xi6 = xi6 - loc6;
        xi7 = xi7 - loc7;
        xi8 = xi8 - loc8;

        scalar_loc = loc1 + Nx1 * (loc2 + Nx2 * (loc3 + Nx3 * (loc4 + Nx4 * (loc5 + Nx5 * (loc6 + Nx6 * (loc7 + Nx7 * loc8))))));

        Xi1[index] = (1 - Xi1[index]) * ((1 - xi2) * ((1 - xi3) * ((1 - xi4) * ((1 - xi5) * ((1 - xi6) * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[0]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[1]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[2]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[3]]))
                                                                                                 + xi6 * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[4]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[5]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[6]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[7]])))
                                                                                    + xi5 * ((1 - xi6) * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[8]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[9]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[10]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[11]]))
                                                                                                 + xi6 * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[12]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[13]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[14]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[15]]))))
                                                                       + xi4 * ((1 - xi5) * ((1 - xi6) * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[16]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[17]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[18]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[19]]))
                                                                                                 + xi6 * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[20]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[21]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[22]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[23]])))
                                                                                    + xi5 * ((1 - xi6) * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[24]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[25]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[26]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[27]]))
                                                                                                 + xi6 * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[28]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[29]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[30]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[31]])))))
                                                          + xi3 * ((1 - xi4) * ((1 - xi5) * ((1 - xi6) * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[32]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[33]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[34]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[35]]))
                                                                                                 + xi6 * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[36]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[37]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[38]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[39]])))
                                                                                    + xi5 * ((1 - xi6) * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[40]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[41]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[42]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[43]]))
                                                                                                 + xi6 * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[44]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[45]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[46]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[47]]))))
                                                                       + xi4 * ((1 - xi5) * ((1 - xi6) * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[48]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[49]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[50]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[51]]))
                                                                                                 + xi6 * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[52]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[53]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[54]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[55]])))
                                                                                    + xi5 * ((1 - xi6) * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[56]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[57]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[58]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[59]]))
                                                                                                 + xi6 * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[60]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[61]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[62]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[63]]))))))
                                             + xi2 * ((1 - xi3) * ((1 - xi4) * ((1 - xi5) * ((1 - xi6) * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[64]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[65]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[66]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[67]]))
                                                                                                 + xi6 * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[68]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[69]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[70]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[71]])))
                                                                                    + xi5 * ((1 - xi6) * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[72]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[73]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[74]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[75]]))
                                                                                                 + xi6 * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[76]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[77]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[78]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[79]]))))
                                                                       + xi4 * ((1 - xi5) * ((1 - xi6) * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[80]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[81]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[82]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[83]]))
                                                                                                 + xi6 * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[84]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[85]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[86]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[87]])))
                                                                                    + xi5 * ((1 - xi6) * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[88]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[89]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[90]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[91]]))
                                                                                                 + xi6 * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[92]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[93]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[94]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[95]])))))
                                                          + xi3 * ((1 - xi4) * ((1 - xi5) * ((1 - xi6) * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[96]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[97]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[98]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[99]]))
                                                                                                 + xi6 * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[100]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[101]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[102]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[103]])))
                                                                                    + xi5 * ((1 - xi6) * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[104]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[105]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[106]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[107]]))
                                                                                                 + xi6 * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[108]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[109]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[110]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[111]]))))
                                                                       + xi4 * ((1 - xi5) * ((1 - xi6) * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[112]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[113]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[114]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[115]]))
                                                                                                 + xi6 * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[116]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[117]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[118]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[119]])))
                                                                                    + xi5 * ((1 - xi6) * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[120]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[121]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[122]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[123]]))
                                                                                                 + xi6 * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[124]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[125]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[126]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[127]])))))))
                         + Xi1[index] * ((1 - xi2) * ((1 - xi3) * ((1 - xi4) * ((1 - xi5) * ((1 - xi6) * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[128]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[129]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[130]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[131]]))
                                                                                                 + xi6 * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[132]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[133]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[134]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[135]])))
                                                                                    + xi5 * ((1 - xi6) * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[136]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[137]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[138]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[139]]))
                                                                                                 + xi6 * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[140]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[141]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[142]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[143]]))))
                                                                       + xi4 * ((1 - xi5) * ((1 - xi6) * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[144]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[145]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[146]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[147]]))
                                                                                                 + xi6 * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[148]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[149]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[150]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[151]])))
                                                                                    + xi5 * ((1 - xi6) * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[152]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[153]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[154]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[155]]))
                                                                                                 + xi6 * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[156]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[157]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[158]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[159]])))))
                                                          + xi3 * ((1 - xi4) * ((1 - xi5) * ((1 - xi6) * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[160]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[161]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[162]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[163]]))
                                                                                                 + xi6 * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[164]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[165]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[166]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[167]])))
                                                                                    + xi5 * ((1 - xi6) * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[168]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[169]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[170]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[171]]))
                                                                                                 + xi6 * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[172]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[173]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[174]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[175]]))))
                                                                       + xi4 * ((1 - xi5) * ((1 - xi6) * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[176]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[177]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[178]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[179]]))
                                                                                                 + xi6 * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[180]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[181]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[182]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[183]])))
                                                                                    + xi5 * ((1 - xi6) * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[184]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[185]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[186]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[187]]))
                                                                                                 + xi6 * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[188]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[189]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[190]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[191]]))))))
                                             + xi2 * ((1 - xi3) * ((1 - xi4) * ((1 - xi5) * ((1 - xi6) * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[192]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[193]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[194]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[195]]))
                                                                                                 + xi6 * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[196]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[197]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[198]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[199]])))
                                                                                    + xi5 * ((1 - xi6) * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[200]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[201]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[202]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[203]]))
                                                                                                 + xi6 * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[204]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[205]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[206]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[207]]))))
                                                                       + xi4 * ((1 - xi5) * ((1 - xi6) * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[208]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[209]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[210]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[211]]))
                                                                                                 + xi6 * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[212]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[213]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[214]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[215]])))
                                                                                    + xi5 * ((1 - xi6) * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[216]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[217]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[218]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[219]]))
                                                                                                 + xi6 * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[220]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[221]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[222]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[223]])))))
                                                          + xi3 * ((1 - xi4) * ((1 - xi5) * ((1 - xi6) * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[224]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[225]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[226]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[227]]))
                                                                                                 + xi6 * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[228]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[229]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[230]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[231]])))
                                                                                    + xi5 * ((1 - xi6) * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[232]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[233]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[234]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[235]]))
                                                                                                 + xi6 * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[236]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[237]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[238]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[239]]))))
                                                                       + xi4 * ((1 - xi5) * ((1 - xi6) * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[240]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[241]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[242]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[243]]))
                                                                                                 + xi6 * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[244]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[245]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[246]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[247]])))
                                                                                    + xi5 * ((1 - xi6) * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[248]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[249]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[250]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[251]]))
                                                                                                 + xi6 * ((1 - xi7) * ((1 - xi8) * V[scalar_loc + corners_index[252]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[253]])
                                                                                                              + xi7 * ((1 - xi8) * V[scalar_loc + corners_index[254]]
                                                                                                                           + xi8 * V[scalar_loc + corners_index[255]])))))));
        index = index + num_threads;
    }
}