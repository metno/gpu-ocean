#ifndef NETCDFREADER_H
#define NETCDFREADER_H

#include "field.h"
#include <netcdfcpp.h>
#include <memory>
#include <map>
#include <string>

class NetCDFReader {
public:
    NetCDFReader(const std::string &);
    ~NetCDFReader();

    int nx() const;
    int ny() const;
    float width() const;
    float height() const;
    float dx() const;
    float dy() const;

    FieldInfo H() const;

    long etaTimesteps() const;
    FieldInfo eta(long = -1) const;

    long UTimesteps() const;
    FieldInfo U(long = -1) const;

    long VTimesteps() const;
    FieldInfo V(long = -1) const;

private:
    struct NetCDFReaderImpl;
    NetCDFReaderImpl *pimpl;
    FieldInfo read2DFloatField(const std::string &, int, int, long = -1) const;
    long timesteps(const std::string &) const;
};

typedef std::shared_ptr<NetCDFReader> NetCDFReaderPtr;

#endif // NETCDFREADER_H
