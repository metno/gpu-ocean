#include "field.h"
#include <iostream>
#include <iomanip>
#include <cassert>

using namespace std;

Field2D::Field2D()
    : nx_(0)
    , ny_(0)
    , dx_(0)
    , dy_(0)
{
    data_.reset(new std::vector<float>());
    validate();
}

Field2D::Field2D(const FieldPtr &data, int nx, int ny, float dx, float dy)
    : data_(data)
    , nx_(nx)
    , ny_(ny)
    , dx_(dx)
    , dy_(dy)
{
    validate();
}

Field2D::Field2D(vector<float> *field, int nx, int ny, float dx, float dy)
    : nx_(nx)
    , ny_(ny)
    , dx_(dx)
    , dy_(dy)
{
    data_.reset(field);
    validate();
}

Field2D::Field2D(const Field2D &other)
    : nx_(other.nx_)
    , ny_(other.ny_)
    , dx_(other.dx_)
    , dy_(other.dy_)
{
    data_.reset(new vector<float>(*other.data_.get()));
    validate();
}

void Field2D::validate() const
{
    assert(nx_ >= 0);
    assert(ny_ >= 0);
    assert(data_->size() == nx_ * ny_);
    assert((nx_ <= 0) || (dx_ > 0));
    assert((ny_ <= 0) || (dy_ > 0));
}

FieldPtr Field2D::data() const
{
    return data_;
}

int Field2D::nx() const
{
    return nx_;
}

int Field2D::ny() const
{
    return ny_;
}

float Field2D::dx() const
{
    return dx_;
}

float Field2D::dy() const
{
    return dy_;
}

float &Field2D::operator()(int i, int j) const
{
    return data_->at(i + j * nx_);
}

void Field2D::fill(float value)
{
    for (int i = 0; i < data_->size(); ++i)
        data_->at(i) = value;
}

void Field2D::fill(const Field2D &src)
{
    assert(data_->size() == src.data_->size());
    *data_.get() = *src.data_.get();
}

bool Field2D::empty() const
{
    return data_->empty();
}

void Field2D::dump(const string &title) const
{
    if (!title.empty())
        cerr << title << endl;
    cerr << "nx: " << nx_ << ", ny: " << ny_ << ", vector.size(): " << data_->size() << endl << "values:" << endl;
    for (int x = 0; x < nx_; ++x)
        for (int y = 0; y < ny_; ++y)
            cerr << setw(2) << "x: " << x << ", y: " << setw(2) << y << ", val: " << (*this)(x, y) << endl;
}
