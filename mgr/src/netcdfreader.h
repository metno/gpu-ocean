#ifndef NETCDFREADER_H
#define NETCDFREADER_H

#include "field.h"
#include <netcdfcpp.h>
#include <memory>

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
};

typedef std::shared_ptr<NetCDFReader> NetCDFReaderPtr;

#endif // NETCDFREADER_H
