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
    FieldInfo eta(long = -1) const;
    FieldInfo U(long = -1) const;
    FieldInfo V(long = -1) const;

private:
    struct NetCDFReaderImpl;
    NetCDFReaderImpl *pimpl;
    FieldInfo read2DFloatField(const std::string &, int, int, long = -1) const;
};

typedef std::shared_ptr<NetCDFReader> NetCDFReaderPtr;

#endif // NETCDFREADER_H
