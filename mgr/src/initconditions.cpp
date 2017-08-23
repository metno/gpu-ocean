#include "config.h"

#include "initconditions.h"
#ifdef mgr_USE_NETCDF
#include "netcdfreader.h"
#endif
#include <boost/format.hpp>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <memory>
#include <iostream>

using namespace std;

const unsigned int STEADY_FLOW_OVER_BUMP = 3;
const unsigned int IDEALISED_CIRCULAR_DAM = 4;

struct InitConditions::InitConditionsImpl
{
    int nx;
    int ny;
    float width;
    float height;
    Field2D waterElevationField;
    Field2D bathymetryField;
    Field2D H;
    Field2D eta;
    Field2D U;
    Field2D V;
    InitConditionsImpl();
};

InitConditions::InitConditionsImpl::InitConditionsImpl()
{
}

InitConditions::InitConditions()
    : pimpl(new InitConditionsImpl)
{
}

inline Field2D generateBathymetry(int no, int nx, int ny, float width, float height)
{
    const float dx = width / nx;
    const float dy = height / ny;

	if (nx <= 0 || ny <= 0) {
		stringstream log;
		log << "Invalid nx or ny: [" << nx << ", " << ny << "]." << endl;
		throw runtime_error(log.str());
	}

	cout << "Generating bathymetry: '";

    vector<float> *f = new vector<float>(nx * ny, 0.0f);

	switch (no)
	{
	case 0:
        cout << "flat";
        for (int i = 0; i < f->size(); ++i)
            f->at(i) = 0.0f;
		break;
	case 1:
		cout << "peaks";
#pragma omp parallel for
        for (int j = 0; j < ny; ++j) {
			float y = j * 6.0f/(float) ny - 3.0f;
            for (int i = 0; i < nx; ++i) {
                float x = i * 6.0f/nx - 3.0f;
				float value = 3.0f*(1-x)*(1-x) * exp(-(x*x) - (y-1)*(y-1))
                                - 10.0f * (x/5.0f - x*x*x - y*y*y*y*y) * exp(-(x*x) - (y*y))
                                - 1.0f/3.0f * exp(-(x+1)*(x+1) - (y*y));

                f->at(j * nx + i) = 0.1f*value;
			}
		}
		break;
	case 2:
		cout << "3 bumps";
#pragma omp parallel for
        for (int j=0; j < ny; ++j) {
            const float y = j / (float) ny;
            for (int i = 0; i < nx; ++i) {
                const float x = i / (float) nx;
                const int index = j * nx + i;
				if ((x-0.25f)*(x-0.25f)+(y-0.25f)*(y-0.25f)<0.01)
                    f->at(index) = 5.0f*(0.01f-(x-0.25f)*(x-0.25f)-(y-0.25f)*(y-0.25f));
				else if ((x-0.75f)*(x-0.75f)+(y-0.25f)*(y-0.25f)<0.01f)
                    f->at(index) = 5.0f*(0.01f-(x-0.75f)*(x-0.75f)-(y-0.25f)*(y-0.25f));
				else if ((x-0.25f)*(x-0.25f)+(y-0.75f)*(y-0.75f)<0.01f)
                    f->at(index) = 5.0f*(0.01f-(x-0.25f)*(x-0.25f)-(y-0.75f)*(y-0.75f));
				else
                    f->at(index) = 0.0f;
			}
		}
		break;
    case STEADY_FLOW_OVER_BUMP:
        cout << "Steady Flow Over Bump";
        // ### use std::min(width, height) instead of width below?
    {
        const float x1 = width * 0.32f;
        const float x2 = width * 0.48f;
        const float v1 = width * 0.008f;
        const float v2 = width * 0.002f;
        const float v3 = width * 0.4f;
#pragma omp parallel for
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const float x = i*dx;
                f->at(j * nx + i) = (x1 < x && x < x2)
                        ? (v1 - v2*(x-v3)*(x-v3))
                        : 0.0f;
            }
        }
    }
        break;
    case IDEALISED_CIRCULAR_DAM:
		cout << "idealized circular dam";
        for (int i = 0; i < f->size(); ++i)
            f->at(i) = 0.0f;
		break;
	default:
		cout << "Could not recognize " << no << " as a valid id." << endl;
		exit(-1);
	}

    cout << "' (" << nx << "x" << ny << " values)" << endl;

    return Field2D(f, nx, ny, dx, dy);
}

inline Field2D generateWaterElevation(int no, int nx, int ny, float width, float height)
{
    const float dx = width / nx;
    const float dy = height / ny;

	if (nx <= 0 || ny <= 0) {
		cout << "Invalid nx or ny: [" << nx << ", " << ny << "]." << endl;
		exit(-1);
	}

	cout << "Generating water elevation: '";

    vector<float> *f = new vector<float>(nx * ny, 0.0f);

	switch (no)
	{
	case 0:
		cout << "column_dry";
#pragma omp parallel for
        for (int j = 0; j < ny; ++j) {
			float y = (j+0.5f) / (float) ny - 0.5f;
            for (int i = 0; i < nx; ++i) {
                float x = (i+0.5f) / (float) nx - 0.5f;
                f->at(j * nx + i) = ((x*x)+(y*y)<0.01f) ? 1.0f : 0.0f;
			}
		}
		break;


	case 1:
		cout << "gaussian";
#pragma omp parallel for
        for (int j = 0; j < ny; ++j) {
			float y = (j+0.5f) / (float) ny - 0.5f;
            for (int i = 0; i < nx; ++i) {
				float x = (i+0.5f) / (float) nx - 0.5f;
				//if ((x*x)+(y*y)<0.01)
                    f->at(j * nx + i) = exp(-(x*x) - (y*y));
				//else
                //	f->at(j * nx + i) = 0.0f;
			}
		}
		break;



	case 2:
		cout << "zero";
#pragma omp parallel for
        for (int i = 0; i < f->size(); ++i)
            f->at(i) = 0.0f;
		break;


	case STEADY_FLOW_OVER_BUMP:
		cout << "Steady Flow Over Bump";
#pragma omp parallel for
        for (int i = 0; i < f->size(); ++i)
            f->at(i) = 1.0f;
		break;

	case IDEALISED_CIRCULAR_DAM:
		cout << "Idealised Circular Dam";
    {
        const float x1 = width * 0.5f;
        const float y1 = height * 0.5f;
        const float v1 = width * 0.1f;
        const float v2 = width * 0.065f;
#pragma omp parallel for
        for (int j = 0; j < ny; ++j) {
            float y = dy*(j+0.5f)-y1;
            for (int i = 0; i < nx; ++i) {
                float x = dx*(i+0.5f)-x1;
                f->at(j * nx + i) = (sqrt(x*x+y*y) < v2) ? v1 : 0.0f;
            }
        }
    }
        break;

	case 5:
		cout << "column_wet";
#pragma omp parallel for
        for (int j = 0; j < ny; ++j) {
			float y = (j+0.5f) / (float) ny - 0.5f;
            for (int i = 0; i < nx; ++i) {
				float x = (i+0.5f) / (float) nx - 0.5f;
                f->at(j * nx + i) = ((x*x)+(y*y)<0.01f) ? 1.5f : 1.0f;
			}
		}
		break;

    case 6:
        cout << "Wet";
            for (int i = 0; i < f->size(); ++i)
                f->at(i) = 0.3f;
        break;

	default:
		cout << "Could not recognize " << no << " as a valid id." << endl;
		exit(-1);
	}

    cout << "' (" << nx << "x" << ny << " values)" << endl;

    return Field2D(f, nx, ny, dx, dy);
}

inline Field2D generateH(int nx, int ny, float width, float height, Field2D B, float w=1.0f)
{
	if (nx <= 0 || ny <= 0) {
		cout << "Invalid nx or ny: [" << nx << ", " << ny << "]." << endl;
		exit(-1);
	}

	cout << "Generating sea surface mean depth (H): '";

    vector<float> *f = new vector<float>(nx * ny, 0.0f);

#pragma omp parallel for
    for (int i = 0; i < f->size(); ++i) {
        f->at(i) = w - B.getData()->at(i);
        if (f->at(i) < 0.0f) {
			cout << "Negative values in H are not allowed. (Increase global water elevation (w) or change bathymetry (B) input.)" << endl;
			exit(-1);
		}
	}

    cout << "' (" << nx << "x" << ny << " values)" << endl;

    return Field2D(f, nx, ny, width / nx, height / ny);
}

inline Field2D generateEta(int no, int nx, int ny, float width, float height)
{
    const float dx = width / nx;
    const float dy = height / ny;

	if (nx <= 0 || ny <= 0) {
		cout << "Invalid nx or ny: [" << nx << ", " << ny << "]." << endl;
		exit(-1);
	}

	cout << "Generating sea surface deviation (eta): '";

    vector<float> *f = new vector<float>(nx * ny, 0.0f);

	switch (no)
	{
	case 0:
		cout << "column";
#pragma omp parallel for
        for (int j = 0; j < ny; ++j) {
			float y = (j+0.5f) / (float) ny - 0.5f;
            for (int i = 0; i < nx; ++i) {
				float x = (i+0.5f) / (float) nx - 0.5f;
                f->at(j * nx + i) = ((x*x)+(y*y)<0.01f) ? 1.0f : 0.0f;
			}
		}
		break;

	case 1:
		cout << "gaussian";
#pragma omp parallel for
        for (int j = 0; j < ny; ++j) {
			float y = (j+0.5f) / (float) ny - 0.5f;
            for (int i = 0; i < nx; ++i) {
				float x = (i+0.5f) / (float) nx - 0.5f;
                f->at(j * nx + i) = exp(-(x*x) - (y*y));
			}
		}
		break;

	case 2:
		cout << "zero";
#pragma omp parallel for
        for (int i = 0; i < f->size(); ++i)
            f->at(i) = 0.0f;
		break;


	case STEADY_FLOW_OVER_BUMP:
		cout << "Steady Flow Over Bump";
#pragma omp parallel for
        for (int i = 0; i < f->size(); ++i)
            f->at(i) = 0.0f;
		break;

	case IDEALISED_CIRCULAR_DAM:
		cout << "Idealised Circular Dam";
    {
        const float x1 = width * 0.5f;
        const float y1 = height * 0.5f;
        const float v1 = width * 0.1f;
        const float v2 = width * 0.065f;
#pragma omp parallel for
        for (int j = 0; j < ny; ++j) {
            float y = dy*(j+0.5f)-y1;
            for (int i = 0; i < nx; ++i) {
                float x = dx*(i+0.5f)-x1;
                f->at(j * nx + i) = (sqrt(x*x+y*y) < v2) ? v1 : 0.0f;
            }
        }
    }
        break;

	default:
		cout << "Could not recognize " << no << " as a valid id." << endl;
		exit(-1);
	}

    cout << "' (" << nx << "x" << ny << " values)" << endl;

    return Field2D(f, nx, ny, dx, dy);
}

void InitConditions::init(const OptionsPtr &options)
{
    // read grid dimensions and fields from input file if available
    if (!options->inputFile().empty()) {
		#ifdef mgr_USE_NETCDF
        if (options->nx() != -1)
            throw runtime_error("error: nx specified outside of input file");
        if (options->ny() != -1)
            throw runtime_error("error: ny specified outside of input file");
        if (options->width() != -1)
            throw runtime_error("error: width specified outside of input file");
        if (options->height() != -1)
            throw runtime_error("error: height specified outside of input file");

        NetCDFReaderPtr fileReader(new NetCDFReader(options->inputFile()));
        pimpl->nx = fileReader->nx();
        pimpl->ny = fileReader->ny();
        pimpl->width = fileReader->width();
        pimpl->height = fileReader->height();

        // copy available fields from the file reader
        if (!fileReader->H().empty())
            pimpl->H = fileReader->H();
        if (!fileReader->eta().empty())
            pimpl->eta = fileReader->eta();
        if (!fileReader->U().empty())
            pimpl->U = fileReader->U();
        if (!fileReader->V().empty())
            pimpl->V = fileReader->V();
		#else
        throw runtime_error("error: NetCDF support not enabled!");
		#endif

    } else {
        // no input file available, so read grid dimensions from program options
        pimpl->nx = options->nx();
        pimpl->ny = options->ny();
        pimpl->width = options->width();
        pimpl->height = options->height();
    }

    // validate grid dimensions
    if (pimpl->nx < 2)
        throw runtime_error((boost::format("error: nx (%1%) < 2") % pimpl->nx).str());
    if (pimpl->ny < 2)
        throw runtime_error((boost::format("error: ny (%1%) < 2") % pimpl->ny).str());
    if (pimpl->width <= 0)
        throw runtime_error((boost::format("error: width (%1%) <= 0") % pimpl->width).str());
    if (pimpl->height <= 0)
        throw runtime_error((boost::format("error: height (%1%) <= 0") % pimpl->height).str());

    // generated H if necessary
    if (pimpl->H.empty()) {
        if (options->bathymetryNo() < 0)
            throw runtime_error(
                    (boost::format("error: H needs to be generated, but bathymetryNo (%1%) < 0") % options->bathymetryNo()).str());
        pimpl->bathymetryField = generateBathymetry(options->bathymetryNo(), pimpl->nx, pimpl->ny, pimpl->width, pimpl->height);
        float wGlobal = 1.0f;
        if (options->wGlobal() < 0)
        	cout << "warning: no initial global water elevation level given, using " << wGlobal << endl;
        else
        	wGlobal = options->wGlobal();
        pimpl->H = generateH(pimpl->nx, pimpl->ny, pimpl->width, pimpl->height, pimpl->bathymetryField, wGlobal);
    }

    // generate eta if necessary
    if (pimpl->eta.empty()) {
        if (options->etaNo() < 0)
            throw runtime_error(
                    (boost::format("error: eta needs to be generated, but etaNo (%1%) < 0") % options->etaNo()).str());
        pimpl->eta = generateEta(options->etaNo(), pimpl->nx, pimpl->ny, pimpl->width, pimpl->height);
    }

    /// TODO: We don't really need this field to initialize a simulation run for time being, but maybe later...
    if (options->waterElevationNo() >= 0)
        pimpl->waterElevationField = generateWaterElevation(options->waterElevationNo(), pimpl->nx, pimpl->ny, pimpl->width, pimpl->height);
}

int InitConditions::getNx() const
{
    return pimpl->nx;
}

int InitConditions::getNy() const
{
    return pimpl->ny;
}

float InitConditions::width() const
{
    return pimpl->width;
}

float InitConditions::height() const
{
    return pimpl->height;
}

float InitConditions::getDx() const
{
    assert(pimpl->nx > 1);
    return pimpl->width / pimpl->nx;
}

float InitConditions::getDy() const
{
    assert(pimpl->ny > 1);
    return pimpl->height / pimpl->ny;
}

Field2D InitConditions::waterElevationField() const
{
    return pimpl->waterElevationField;
}

Field2D InitConditions::bathymetryField() const
{
    return pimpl->bathymetryField;
}

Field2D InitConditions::H() const
{
    return pimpl->H;
}

Field2D InitConditions::eta() const
{
    return pimpl->eta;
}

Field2D InitConditions::U() const
{
    return pimpl->U;
}

Field2D InitConditions::V() const
{
    return pimpl->V;
}
