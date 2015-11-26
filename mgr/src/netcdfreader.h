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
    FieldInfo eta() const;
    FieldInfo U() const;
    FieldInfo V() const;

private:
    struct NetCDFReaderImpl;
    NetCDFReaderImpl *pimpl;
    FieldInfo read2DFloatField(const std::map<std::string, NcVar *> &, const std::string &, int, int);
};

typedef std::shared_ptr<NetCDFReader> NetCDFReaderPtr;

#endif // NETCDFREADER_H
