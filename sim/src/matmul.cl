__kernel void MatMul (
    __global const float *a,
    __global const float *b,
    __global float *ab)
{
    const int2 pos = {get_global_id(0), get_global_id(1)};

    float sum = 0.0;
    for (int i = 0; i < MATRIX_SIZE; ++i)
        sum += a[pos.x * MATRIX_SIZE + i] * b[i * MATRIX_SIZE + pos.y];

    ab[pos.x * MATRIX_SIZE + pos.y] = sum;
}
