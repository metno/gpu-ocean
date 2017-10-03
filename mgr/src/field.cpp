#include "field.h"
#include <iostream>
#include <iomanip>
#include <cassert>

using namespace std;

Field2D::Field2D()
    : nx(0)
    , ny(0)
    , dx(0)
    , dy(0)
{
    data.reset(new std::vector<float>());
    validate();
}

Field2D::Field2D(const FieldPtr &data_, int nx_, int ny_, float dx_, float dy_)
    : data(data_)
    , nx(nx_)
    , ny(ny_)
    , dx(dx_)
    , dy(dy_)
{
    validate();
}

Field2D::Field2D(vector<float> *field_, int nx_, int ny_, float dx_, float dy_)
    : nx(nx_)
    , ny(ny_)
    , dx(dx_)
    , dy(dy_)
{
    data.reset(field_);
    validate();
}

Field2D::Field2D(const Field2D &other_)
    : nx(other_.nx)
    , ny(other_.ny)
    , dx(other_.dx)
    , dy(other_.dy)
{
    data.reset(new vector<float>(*other_.data.get()));
    validate();
}

void Field2D::validate() const
{
    assert(nx >= 0);
    assert(ny >= 0);
    assert(data->size() == nx * ny);
    assert((nx <= 0) || (dx > 0));
    assert((ny <= 0) || (dy > 0));
}

FieldPtr Field2D::getData() const
{
    return data;
}

int Field2D::getNx() const
{
    return nx;
}

int Field2D::getNy() const
{
    return ny;
}

float Field2D::getDx() const
{
    return dx;
}

float Field2D::getDy() const
{
    return dy;
}

float &Field2D::operator()(int i, int j) const
{
    return data->at(i + j * nx);
}

void Field2D::fill(float value)
{
    for (int i = 0; i < data->size(); ++i)
        data->at(i) = value;
}

void Field2D::fill(const Field2D &src)
{
    assert(data->size() == src.data->size());
    *data.get() = *src.data.get();
}

bool Field2D::empty() const
{
    return data->empty();
}

void Field2D::dump(const string &title) const
{
    if (!title.empty())
        cerr << title << endl;
    cerr << "nx: " << nx << ", ny: " << ny << ", vector.size(): " << data->size() << endl << "values:" << endl;
    for (int x = 0; x < nx; ++x)
        for (int y = 0; y < ny; ++y)
            cerr << setw(2) << "x: " << x << ", y: " << setw(2) << y << ", val: " << (*this)(x, y) << endl;
}
