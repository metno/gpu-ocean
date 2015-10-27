#include "field.h"

FieldInfo::FieldInfo()
    : nx(-1)
    , ny(-1)
    , dx(-1)
    , dy(-1)
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
