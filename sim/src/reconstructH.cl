#include "reconstructH_types.h"

__kernel void ReconstructH (
    __global const float *H,
    __global float *Hr_u,
    __global float *Hr_v,
    ReconstructH_args args)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);
    const int nx = args.nx;
    const int ny = args.ny;

    if ((i > 0) && (i < (nx - 1)))
        Hr_u[i * ny + j] = 0.5 * (H[i * ny + j] + H[(i + 1) * ny + j]);
    if ((j > 0) && (j < (ny - 1)))
        Hr_v[i * ny + j] = 0.5 * (H[i * ny + j] + H[i * ny + j + 1]);

   // ### The above indexing doesn't work, since none of H, Hr_u, or Hr_v have width=nx and height=ny. ADJUST!
   // ### Also, consider avoiding the tests (thus saving time) by computing Hr_u and Hr_v in separate kernels
}
