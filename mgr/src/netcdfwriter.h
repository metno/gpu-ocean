#ifndef NETCDFWRITER_H
#define NETCDFWRITER_H

#include <netcdfcpp.h>
//#include <netcdf.h> // ### needed?
#include <boost/shared_ptr.hpp>

class NetCDFWriter {
public:
	NetCDFWriter();
    NetCDFWriter(std::string);
	~NetCDFWriter();

    void init(int, int, float, float, float, float, float, float *);

    void writeTimestep(float *, float *, float *, float);

private:
	void initFile(std::string filename);

    struct NetCDFWriterImpl;
    NetCDFWriterImpl *pimpl;
};

#endif // NETCDFWRITER_H
