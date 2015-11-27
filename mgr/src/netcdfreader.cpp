#include "netcdfreader.h"
#include <boost/format.hpp>
#include <stdexcept>
#include <cassert>
#include <vector>

using namespace std;

struct NetCDFReader::NetCDFReaderImpl
{
    std::shared_ptr<NcFile> file;
    int nx;
    int ny;
    float width;
    float height;
    map<string, NcVar *> vars;
    NetCDFReaderImpl();
};

NetCDFReader::NetCDFReaderImpl::NetCDFReaderImpl()
    : nx(-1)
    , ny(-1)
    , width(-1)
    , height(-1)
{
}

NetCDFReader::NetCDFReader(const std::string &fname)
    :  pimpl(new NetCDFReaderImpl())
{
    // open file
    pimpl->file.reset(new NcFile(fname.c_str(), NcFile::ReadOnly));
    if (!pimpl->file->is_valid()) {
        stringstream ss;
        ss << "Failed to open '" << fname << "' for reading NetCDF.";
        throw runtime_error(ss.str());
    }


    // read global attributes
    map<string, NcAtt *> attrs;
    for (int i = 0; i < pimpl->file->num_atts(); ++i) {
        NcAtt *attr = pimpl->file->get_att(i);
        attrs[string(attr->name())] = attr;
    }

    // read mandatory grid dimensions
    // ... nx
    if ((attrs.count("nx") == 0) || (!attrs.at("nx")->is_valid()) || (attrs.at("nx")->as_int(0) < 2))
        throw runtime_error("failed to read nx as a valid NetCDF integer attribute >= 2");
    pimpl->nx = attrs.at("nx")->as_int(0);

    // ... ny
    if ((attrs.count("ny") == 0) || (!attrs.at("ny")->is_valid()) || (attrs.at("ny")->as_int(0) < 2))
        throw runtime_error("failed to read ny as a valid NetCDF integer attribute >= 2");
    pimpl->ny = attrs.at("ny")->as_int(0);

    // ... width
    if ((attrs.count("width") > 0) && attrs.at("width")->is_valid() && (attrs.at("width")->as_float(0) > 0))
        pimpl->width = attrs.at("width")->as_float(0);
    else if ((attrs.count("dx") > 0) && attrs.at("dx")->is_valid() && (attrs.at("dx")->as_float(0) > 0))
        pimpl->width = (pimpl->nx - 1) * attrs.at("dx")->as_float(0);
    else
        throw runtime_error("neither width nor dx readable as a valid NetCDF float attribute > 0");

    // ... height
    if ((attrs.count("height") > 0) && attrs.at("height")->is_valid() && (attrs.at("height")->as_float(0) > 0))
        pimpl->height = attrs.at("height")->as_float(0);
    else if ((attrs.count("dy") > 0) && attrs.at("dy")->is_valid() && (attrs.at("dy")->as_float(0) > 0))
        pimpl->height = (pimpl->ny - 1) * attrs.at("dy")->as_float(0);
    else
        throw runtime_error("neither height nor dy readable as a valid NetCDF float attribute > 0");


    // read variables
    for (int i = 0; i < pimpl->file->num_vars(); ++i) {
        NcVar *var = pimpl->file->get_var(i);
        pimpl->vars[string(var->name())] = var;
    }
}

NetCDFReader::~NetCDFReader()
{
    pimpl->file->sync();
    if (!pimpl->file->close()) {
        throw("Error: Couldn't close NetCDF file!");
    }
    pimpl->file.reset();
}

int NetCDFReader::nx() const
{
    return pimpl->nx;
}

int NetCDFReader::ny() const
{
    return pimpl->ny;
}

float NetCDFReader::width() const
{
    return pimpl->width;
}

float NetCDFReader::height() const
{
    return pimpl->height;
}

float NetCDFReader::dx() const
{
    assert(pimpl->nx > 1);
    return pimpl->width / (pimpl->nx - 1);
}

float NetCDFReader::dy() const
{
    assert(pimpl->ny > 1);
    return pimpl->height / (pimpl->ny - 1);
}

// Reads H.
FieldInfo NetCDFReader::H() const
{
    return read2DFloatField("H", pimpl->nx + 1, pimpl->ny + 1);
}

// Reads number of timestemps for eta.
long NetCDFReader::etaTimesteps() const
{
    return timesteps("eta");
}

// Returns eta for a given timestep (0 = first, -1 = last). The timestep is ignored if the field variable (NcVar) has only 2 dimensions.
FieldInfo NetCDFReader::eta(long timestep) const
{
    return read2DFloatField("eta", pimpl->nx + 1, pimpl->ny + 1, timestep);
}

// Reads number of timestemps for U.
long NetCDFReader::UTimesteps() const
{
    return timesteps("U");
}

// Returns U for a given timestep (0 = first, -1 = last). The timestep is ignored if the field variable (NcVar) has only 2 dimensions.
FieldInfo NetCDFReader::U(long timestep) const
{
    return read2DFloatField("U", pimpl->nx + 2, pimpl->ny - 1, timestep);
}

// Reads number of timestemps for V.
long NetCDFReader::VTimesteps() const
{
    return timesteps("V");
}

// Returns V for a given timestep (0 = first, -1 = last). The timestep is ignored if the field variable (NcVar) has only 2 dimensions.
FieldInfo NetCDFReader::V(long timestep) const
{
    return read2DFloatField("V", pimpl->nx - 1, pimpl->ny + 2, timestep);
}

/**
 * This function copies a 2D float field from file to memory. If the field variable (NcVar) has three dimensions, it is assumed that the third
 * dimension is time, and the 2D field of the last timestep is copied. Otherwise, the field variable must have two dimensions, and the
 * field is copied directly.
 * The function throws runtime_error if an error occurs.
 * @param name: Field name.
 * @param nx_exp: Expected size of X-dimension.
 * @param ny_exp: Expected size of Y-dimension.
 * @param timestep: Timestep (if applicable, i.e. if the field variable is 3D). The first and last timestep is indicated by 0 and -1 respectively.
 * @returns The FieldInfo object. NOTE: An empty object is returned if the field doesn't exist (which is not considered an error).
 */
FieldInfo NetCDFReader::read2DFloatField(const string &name, int nx_exp, int ny_exp, long timestep) const
{
    if (pimpl->vars.count(name) == 0)
        return FieldInfo();

    NcVar *var = pimpl->vars.at(name);

    if (var->type() != ncFloat)
        throw runtime_error(
                (boost::format("error in field %s: type (%d) not float (%d)") % name % var->type() % ncFloat).str());

    if (var->num_dims() == 2) {
        // assume field is not part of a time series

        const NcDim *dimy = var->get_dim(0);
        if (dimy->size() != ny_exp)
            throw runtime_error(
                    (boost::format("error in field %s (ndims=2): ny (%d) != %d") % name % dimy->size() % ny_exp).str());

        const NcDim *dimx = var->get_dim(1);
        if (dimx->size() != nx_exp)
            throw runtime_error(
                    (boost::format("error in field %s (ndims=2): nx (%d) != %d") % name % dimx->size() % nx_exp).str());

        vector<float> *data = new vector<float>(nx_exp * ny_exp);
        if (!var->get(data->data(), ny_exp, nx_exp))
            throw runtime_error((boost::format("error in field %s (ndims=2): failed to copy values") % name).str());

        return FieldInfo(data, nx_exp, ny_exp, dx(), dy());

    } else if (var->num_dims() == 3) {
        // assume field is part of a time series

        const NcDim *dimt = var->get_dim(0);
        if (dimt->name() != string("T"))
            throw runtime_error(
                    (boost::format("error in field %s (ndims=3): name of time dimension (%s) != T") % name % dimt->name()).str());
        const long actualTimestep = (timestep < 0) ? (dimt->size() - 1) : timestep;
        if (actualTimestep < 0 || actualTimestep >= dimt->size())
            throw runtime_error(
                    (boost::format("error in field %s (ndims=3): timestep (%d) outside available range ([0, %d])")
                     % name % actualTimestep % (dimt->size() - 1)).str());

        const NcDim *dimy = var->get_dim(1);
        if (dimy->size() != ny_exp)
            throw runtime_error(
                    (boost::format("error in field %s (ndims=3): ny (%d) != %d") % name % dimy->size() % ny_exp).str());

        const NcDim *dimx = var->get_dim(2);
        if (dimx->size() != nx_exp)
            throw runtime_error(
                    (boost::format("error in field %s (ndims=3): nx (%d) != %d") % name % dimx->size() % nx_exp).str());

        vector<float> *data = new vector<float>(nx_exp * ny_exp);
        var->set_cur(actualTimestep);
        if (!var->get(data->data(), 1, ny_exp, nx_exp))
            throw runtime_error((boost::format("error in field %s (ndims=3): failed to copy values") % name).str());

        return FieldInfo(data, nx_exp, ny_exp, dx(), dy());

    } else {
        throw runtime_error(
                (boost::format("error in field %s: # of dimensions (%d) neither 2 nor 3") % name % var->num_dims()).str());
    }

    return FieldInfo();
}

/**
 * Returns the number of timesteps of a 2D field.
 * The function throws runtime_error if an error occurs.
 * @param name: Field name.
 * @returns The number of timesteps (>= 0) of the 2D field if the field exists and the field variable (NcVar) has three dimensions. Otherwise -1.
 */
long NetCDFReader::timesteps(const string &name) const
{
    if (pimpl->vars.count(name) == 0)
        return -1;

    NcVar *var = pimpl->vars.at(name);

    if (var->num_dims() == 2)
        return -1;

    if (var->num_dims() == 3) {
        // assume field is part of a time series
        const NcDim *dimt = var->get_dim(0);
        if (dimt->name() != string("T"))
            throw runtime_error(
                    (boost::format("error in field %s (ndims=3): name of time dimension (%s) != T") % name % dimt->name()).str());
        return dimt->size();
    } else {
        throw runtime_error(
                (boost::format("error in field %s: # of dimensions (%d) neither 2 nor 3") % name % var->num_dims()).str());
    }

    return -1;
}
