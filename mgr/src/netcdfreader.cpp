#include "netcdfreader.h"
#include <stdexcept>
#include <cassert>

using namespace std;

struct NetCDFReader::NetCDFReaderImpl
{
    std::shared_ptr<NcFile> file;

    int nx;
    int ny;
    float width;
    float height;
    FieldInfo H;
    FieldInfo eta;
    FieldInfo U;
    FieldInfo V;

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
        ss << "Failed to open '" << fname << "' for reading.";
        throw runtime_error(ss.str());
    }

    // 1: read nx and ny from input file (both values are MANDATORY, i.e. fatal error if not present!)
    // 2: assert that there are no time series in the file
    // 3: read H if present, assert that its dimension is [nx + 1, ny + 1]
    // 4: read eta if present, assert that its dimension is [nx + 1, ny + 1]
    // 5: read U if present, assert that its dimension is [nx + 2, ny - 1]
    // 6: read V if present, assert that its dimension is [nx - 1, ny + 2]

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

FieldInfo NetCDFReader::H() const
{
    return pimpl->H;
}

FieldInfo NetCDFReader::eta() const
{
    return pimpl->eta;
}

FieldInfo NetCDFReader::U() const
{
    return pimpl->U;
}

FieldInfo NetCDFReader::V() const
{
    return pimpl->V;
}
