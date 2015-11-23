#include "netcdfwriter.h"
#include <sstream>
#include <iomanip>
#include <vector>
#include <ctime>
#include <cstring>
#include <stdexcept>
#include <cassert>

using std::setw;
using std::setprecision;
using std::setfill;

struct NetCDFWriter::NetCDFWriterImpl
{
    std::shared_ptr<NcFile> file;

    struct {
        struct {
            NcDim *x;       //!< integer x coordinates
            NcDim *y;       //!<
            NcDim *x_ghost; //!< integer x coordinates, including ghost cells
            NcDim *y_ghost; //!<
            NcDim *x_half;  //!< x coordinates at center of cells
            NcDim *y_half;  //!<
            NcDim *x_half_ghost; //!< x coordinates at center of cells, including ghost cells
            NcDim *y_half_ghost; //!<
            NcDim *t;       //!< Time
        } dims;

        struct {
            NcVar *H;
            NcVar *eta;
            NcVar *U;
            NcVar *V;
            NcVar *x;
            NcVar *y;
            NcVar *x_ghost;
            NcVar *y_ghost;
            NcVar *x_half;
            NcVar *y_half;
            NcVar *x_half_ghost;
            NcVar *y_half_ghost;
            NcVar *t;        //!< time
        } vars;
    } layout;

    long timestepCounter; // internal timestep counter
    unsigned int nx;
    unsigned int ny;

    NetCDFWriterImpl();
};

NetCDFWriter::NetCDFWriterImpl::NetCDFWriterImpl()
    : timestepCounter(0)
{
}

NetCDFWriter::NetCDFWriter()
    : pimpl(new NetCDFWriterImpl())
{
    std::stringstream ss;
    time_t secs = time(0);
    tm *t = localtime(&secs);
    ss << "results_"
       << setw(2) << setfill('0') << setprecision(2) << t->tm_year + 1900 << "-"
       << setw(2) << setfill('0') << setprecision(2) << t->tm_mon + 1 << "-"
       << setw(2) << setfill('0') << setprecision(2) << t->tm_mday << "T"
       << setw(2) << setfill('0') << setprecision(2) << t->tm_hour << ":"
       << setw(2) << setfill('0') << setprecision(2) << t->tm_min << ":"
       << setw(2) << setfill('0') << setprecision(2) << t->tm_sec << ".nc";
    initFile(ss.str());
}

NetCDFWriter::NetCDFWriter(std::string fname)
    :  pimpl(new NetCDFWriterImpl())
{
    initFile(fname);
}

void NetCDFWriter::initFile(std::string fname)
{
    pimpl->file.reset(new NcFile(fname.c_str(), NcFile::New));
    if (!pimpl->file->is_valid()) {
        std::stringstream ss;
        ss << "Failed to create '" << fname << "'. Possible reasons: 1: The file already exists. 2: The disk is full." << std::endl;
        throw std::runtime_error(ss.str());
    }
    memset(&pimpl->layout, 0, sizeof(pimpl->layout));
}

NetCDFWriter::~NetCDFWriter()
{
    pimpl->file->sync();
    if (!pimpl->file->close()) {
        throw("Error: Couldn't close NetCDF file!");
    }
    pimpl->file.reset();
}

static void setSpatialVars(NcVar *var, int size, float delta, bool halfOffset = false)
{
    assert(size > 0);
    assert(delta > 0);
    const float offset = halfOffset ? 0.5 : 0;
    std::vector<float> tmp;
    tmp.resize(size);
    for (int i = 0; i < tmp.size(); ++i)
        tmp[i] = (i + offset) * delta;
    var->put(&tmp[0], tmp.size());
}

void NetCDFWriter::init(int nx, int ny, float dt, float dx, float dy, float f, float r, float *H, float *eta, float *U, float *V)
{
    pimpl->nx = nx;
    pimpl->ny = ny;

    // create dimensions
    pimpl->layout.dims.x = pimpl->file->add_dim("X", nx);
    pimpl->layout.dims.y = pimpl->file->add_dim("Y", ny);
    pimpl->layout.dims.x_ghost = pimpl->file->add_dim("X_ghost", nx + 2);
    pimpl->layout.dims.y_ghost = pimpl->file->add_dim("Y_ghost", ny + 2);
    pimpl->layout.dims.x_half = pimpl->file->add_dim("X_half", nx - 1);
    pimpl->layout.dims.y_half = pimpl->file->add_dim("Y_half", ny - 1);
    pimpl->layout.dims.x_half_ghost = pimpl->file->add_dim("X_half_ghost", nx + 1);
    pimpl->layout.dims.y_half_ghost = pimpl->file->add_dim("Y_half_ghost", ny + 1);
    pimpl->layout.dims.t = pimpl->file->add_dim("T");

    // create indexing variables
    pimpl->layout.vars.x = pimpl->file->add_var("X", ncFloat, pimpl->layout.dims.x);
    pimpl->layout.vars.y = pimpl->file->add_var("Y", ncFloat, pimpl->layout.dims.y);
    pimpl->layout.vars.x_ghost = pimpl->file->add_var("X_ghost", ncFloat, pimpl->layout.dims.x_ghost);
    pimpl->layout.vars.y_ghost = pimpl->file->add_var("Y_ghost", ncFloat, pimpl->layout.dims.y_ghost);
    pimpl->layout.vars.x_half = pimpl->file->add_var("X_half", ncFloat, pimpl->layout.dims.x_half);
    pimpl->layout.vars.y_half = pimpl->file->add_var("Y_half", ncFloat, pimpl->layout.dims.y_half);
    pimpl->layout.vars.x_half_ghost = pimpl->file->add_var("X_half_ghost", ncFloat, pimpl->layout.dims.x_half_ghost);
    pimpl->layout.vars.y_half_ghost = pimpl->file->add_var("Y_half_ghost", ncFloat, pimpl->layout.dims.y_half_ghost);
    pimpl->layout.vars.x->add_att("description", "Longitudal coordinate for values given at grid cell intersections");
    pimpl->layout.vars.y->add_att("description", "Latitudal coordinate for values given at grid cell intersections");
    pimpl->layout.vars.x_ghost->add_att("description", "Longitudal coordinate for values given at grid cell intersections, including ghost cells");
    pimpl->layout.vars.y_ghost->add_att("description", "Latitudal coordinate for values given at grid cell intersections, including ghost cells");
    pimpl->layout.vars.x_half->add_att("description", "Longitudal coordinate for values given at grid cell centers");
    pimpl->layout.vars.y_half->add_att("description", "Latitudal coordinate for values given at grid cell centers");
    pimpl->layout.vars.x_half_ghost->add_att("description", "Longitudal coordinate for values given at grid cell centers, including ghost cells");
    pimpl->layout.vars.y_half_ghost->add_att("description", "Latitudal coordinate for values given at grid cell centers, including ghost cells");

    pimpl->layout.vars.t = pimpl->file->add_var("T", ncFloat, pimpl->layout.dims.t);
    pimpl->layout.vars.t->add_att("description", "Time");

    // write parameters
    pimpl->file->add_att("nx", static_cast<int>(nx));
    pimpl->file->add_att("ny", static_cast<int>(ny));
    pimpl->file->add_att("dt", dt);
    pimpl->file->add_att("dx", dx);
    pimpl->file->add_att("dy", dy);
    pimpl->file->add_att("f", f);
    pimpl->file->add_att("r", r);

    // write contents of spatial variables
    setSpatialVars(pimpl->layout.vars.x, nx, dx);
    setSpatialVars(pimpl->layout.vars.y, ny, dy);
    setSpatialVars(pimpl->layout.vars.x_ghost, nx + 2, dx);
    setSpatialVars(pimpl->layout.vars.y_ghost, ny + 2, dy);
    setSpatialVars(pimpl->layout.vars.x_half, nx - 1, dx, true);
    setSpatialVars(pimpl->layout.vars.y_half, ny - 1, dy, true);
    setSpatialVars(pimpl->layout.vars.x_half_ghost, nx + 1, dx, true);
    setSpatialVars(pimpl->layout.vars.y_half_ghost, ny + 1, dy, true);

    pimpl->file->sync();

    // create initial condition variables
    pimpl->layout.vars.H = pimpl->file->add_var("H", ncFloat, pimpl->layout.dims.y_half_ghost, pimpl->layout.dims.x_half_ghost);
    pimpl->layout.vars.H->add_att("description", "Mean water depth");

    // create the timestep variables
    pimpl->layout.vars.eta = pimpl->file->add_var(
                "eta", ncFloat, pimpl->layout.dims.t, pimpl->layout.dims.y_half_ghost, pimpl->layout.dims.x_half_ghost);
    pimpl->layout.vars.U = pimpl->file->add_var(
                "U", ncFloat, pimpl->layout.dims.t, pimpl->layout.dims.y_half, pimpl->layout.dims.x_ghost);
    pimpl->layout.vars.V = pimpl->file->add_var(
                "V", ncFloat, pimpl->layout.dims.t, pimpl->layout.dims.y_ghost, pimpl->layout.dims.x_half);

    pimpl->layout.vars.eta->add_att("description", "Water elevation disturbances");
    pimpl->layout.vars.U->add_att("description", "Longitudal water discharge");
    pimpl->layout.vars.V->add_att("description", "Latitudal water discharge");

    // set compression
    nc_def_var_deflate(pimpl->file->id(), pimpl->layout.vars.H->id(), 1, 1, 2);
    nc_def_var_deflate(pimpl->file->id(), pimpl->layout.vars.eta->id(), 1, 1, 2);
    nc_def_var_deflate(pimpl->file->id(), pimpl->layout.vars.U->id(), 1, 1, 2);
    nc_def_var_deflate(pimpl->file->id(), pimpl->layout.vars.V->id(), 1, 1, 2);

    // write data
    pimpl->layout.vars.H->put(H, ny + 1, nx + 1);
    pimpl->layout.vars.eta->put(eta, 1, ny + 1, nx + 1);
    pimpl->layout.vars.U->put(U, 1, ny - 1, nx + 2);
    pimpl->layout.vars.V->put(V, 1, ny + 2, nx - 1);

    pimpl->file->sync();
}

void NetCDFWriter::writeTimestep(float *eta, float *U, float *V, float t)
{
    pimpl->layout.vars.t->set_cur(pimpl->timestepCounter);
    pimpl->layout.vars.t->put(&t, 1);

    pimpl->layout.vars.eta->set_cur(pimpl->timestepCounter, 0, 0);
    pimpl->layout.vars.eta->put(eta, 1, pimpl->ny + 1, pimpl->nx + 1);

    pimpl->layout.vars.U->set_cur(pimpl->timestepCounter, 0, 0);
    pimpl->layout.vars.U->put(U, 1, pimpl->ny - 1, pimpl->nx + 2);

    pimpl->layout.vars.V->set_cur(pimpl->timestepCounter, 0, 0);
    pimpl->layout.vars.V->put(V, 1, pimpl->ny + 2, pimpl->nx - 1);

    pimpl->file->sync();
    ++pimpl->timestepCounter;
}
