__global__ void calc_average1(double * Xi1, const double * V, const int query_grid_size, const int Nx1, const double dx1, const double minx1, const int * corners_index)
{
    int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x * blockDim.y 
                + (threadIdx.x + threadIdx.y * blockDim.x);
    int num_threads = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
//     int query_grid_size = Nx1;

    while (index < query_grid_size)
    {
        Xi1[index] = (Xi1[index] - minx1) / dx1;
        int scalar_loc = min(Nx1 - 2, int(floor(Xi1[index])));
        Xi1[index] = Xi1[index] - scalar_loc; // Weights from the 0 corner
        
        Xi1[index] = (1 - Xi1[index]) * V[scalar_loc + corners_index[0]] 
                       + (Xi1[index]) * V[scalar_loc + corners_index[1]];
        
        index = index + num_threads;
    }
        
}