#include "field.h"
#include <iostream>
#include <iomanip>

using namespace std;

FieldInfo::FieldInfo()
    : nx(0)
    , ny(0)
    , dx(0)
    , dy(0)
{
    data.reset(new std::vector<float>());
}

FieldInfo::FieldInfo(const FieldPtr &fp, int nx, int ny, float dx, float dy)
    : data(data)
    , nx(nx)
    , ny(ny)
    , dx(dx)
    , dy(dy)
{
}

FieldInfo::FieldInfo(std::vector<float> *field, int nx, int ny, float dx, float dy)
    : nx(nx)
    , ny(ny)
    , dx(dx)
    , dy(dy)
{
    data.reset(field);
}

/**
 * Returns a reference to the element at (i, j), where 0 <= i < nx and 0 <= j < ny.
 */
float &FieldInfo::operator()(int i, int j) const
{
    return data->at(i + j * nx);
}

bool FieldInfo::empty() const
{
    return data->empty();
}

void FieldInfo::dump(const string &title) const
{
    if (!title.empty())
        cerr << title << endl;
    cerr << "nx: " << nx << ", ny: " << ny << ", vector.size(): " << data->size() << endl << "values:" << endl;
    for (int x = 0; x < nx; ++x)
        for (int y = 0; y < ny; ++y)
            cerr << setw(2) << "x: " << x << ", y: " << setw(2) << y << ", val: " << (*this)(x, y) << endl;
}
