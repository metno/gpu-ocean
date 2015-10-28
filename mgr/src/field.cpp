#include "field.h"

FieldInfo::FieldInfo()
    : nx(0)
    , ny(0)
    , dx(0)
    , dy(0)
{
    data.reset(new std::vector<float>());
}

FieldInfo::FieldInfo(const FieldPtr &data, int nx, int ny, float dx, float dy)
    : data(data)
    , nx(nx)
    , ny(ny)
    , dx(dx)
    , dy(dy)
{
}

/**
 * Returns a reference to the element at (i, j), where 0 <= i < nx and 0 <= j < ny.
 */
float &FieldInfo::operator()(int i, int j) const
{
    return data->at(i * ny + j);
}
