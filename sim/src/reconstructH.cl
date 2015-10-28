__kernel void ReconstructH (
    __global const float *H,
    __global float *Hr_u,
    __global float *Hr_v)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);

    if ((i > 0) && (i < (NX - 1)))
        Hr_u[i * NY + j] = 0.5 * (H[i * NY + j] + H[(i + 1) * NY + j]);
    if ((j > 0) && (j < (NY - 1)))
        Hr_v[i * NY + j] = 0.5 * (H[i * NY + j] + H[i * NY + j + 1]);

   // ### The above indexing doesn't work, since none of H, Hr_u, or Hr_v have width=NX and height=NY. ADJUST!
   // ### Also, consider avoiding the tests (thus saving time) by computing Hr_u and Hr_v in separate kernels
}
