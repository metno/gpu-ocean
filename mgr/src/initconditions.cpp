#include "initconditions.h"

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
    FieldInfo waterElevationField;
    FieldInfo bathymetryField;
    InitConditionsImpl();
};

InitConditions::InitConditionsImpl::InitConditionsImpl()
{
}

InitConditions::InitConditions()
    : pimpl(new InitConditionsImpl)
{
}

InitConditions::FieldInfo::FieldInfo()
    : nx(-1)
    , ny(-1)
    , dx(-1)
    , dy(-1)
{
}

InitConditions::FieldInfo::FieldInfo(const FieldPtr &data, int nx, int ny, float dx, float dy)
    : data(data)
    , nx(nx)
    , ny(ny)
    , dx(dx)
    , dy(dy)
{
}

inline InitConditions::FieldInfo generateBathymetry(int no, int nx, int ny)
{
    float dx = -1;
    float dy = -1;

	if (nx <= 0 || ny <= 0) {
		stringstream log;
		log << "Invalid nx or ny: [" << nx << ", " << ny << "]." << endl;
		throw runtime_error(log.str());
	}

	cout << "Generating bathymetry: '";

	shared_ptr<vector<float> > f_;
	f_.reset(new vector<float>(nx+1, ny+1));
	vector<float> f = *f_;

	switch (no)
	{
	case 0:
		cout << "flat";
		for (unsigned int i=0; i<nx*ny; ++i)
			f[i] = 0.0f;
		break;
	case 1:
		cout << "peaks";
#pragma omp parallel for
		for (int j=0; j<static_cast<int>(ny); ++j) {
			float y = j * 6.0f/(float) ny - 3.0f;
			for (unsigned int i=0; i<nx; ++i) {
				float x = i * 6.0f/(float) nx - 3.0f;
				float value = 3.0f*(1-x)*(1-x) * exp(-(x*x) - (y-1)*(y-1))
								- 10.0f * (x/5.0f - x*x*x - y*y*y*y*y) * exp(-(x*x) - (y*y))
								- 1.0f/3.0f * exp(-(x+1)*(x+1) - (y*y));

				f[j*nx+i] = 0.1f*value;
			}
		}
		break;
	case 2:
		cout << "3 bumps";
#pragma omp parallel for
		for (int j=0; j<static_cast<int>(ny); ++j) {
			float y = j / (float) ny;
			for (unsigned int i=0; i<nx; ++i) {
				float x = i / (float) nx;
				if ((x-0.25f)*(x-0.25f)+(y-0.25f)*(y-0.25f)<0.01)
					f[j*nx+i] = 5.0f*(0.01f-(x-0.25f)*(x-0.25f)-(y-0.25f)*(y-0.25f));
				else if ((x-0.75f)*(x-0.75f)+(y-0.25f)*(y-0.25f)<0.01f)
					f[j*nx+i] = 5.0f*(0.01f-(x-0.75f)*(x-0.75f)-(y-0.25f)*(y-0.25f));
				else if ((x-0.25f)*(x-0.25f)+(y-0.75f)*(y-0.75f)<0.01f)
					f[j*nx+i] = 5.0f*(0.01f-(x-0.25f)*(x-0.25f)-(y-0.75f)*(y-0.75f));
				else
					f[j*nx+i] = 0.0f;
			}
		}
		break;
	case STEADY_FLOW_OVER_BUMP:
		cout << "Steady Flow Over Bump";
		dx = 25.0f / (float) nx;
		dy = 20.0f / (float) ny;
#pragma omp parallel for
		for (int j=0; j<static_cast<int>(ny); ++j) {
			for (unsigned int i = 0; i<nx; ++i) {
				float x = i*dx;
				if (8.0f < x && x < 12.0f) {
					f[j*nx+i] = 0.2f - 0.05f*(x-10.0f)*(x-10.0f);
				}
				else {
					f[j*nx+i] = 0.0f;
				}
			}
		}
		break;
	case IDEALISED_CIRCULAR_DAM:
		dx = 100.0f / (float) nx;
		dy = 100.0f / (float) ny;
		cout << "idealized circular dam";
		for (unsigned int i=0; i<nx*ny; ++i)
			f[i] = 0.0f;
		break;
	default:
		cout << "Could not recognize " << no << " as a valid id." << endl;
		exit(-1);
	}

	cout << "' (" << nx << "x" << ny << " values)" << endl;

    return InitConditions::FieldInfo(f_, nx, ny, dx, dy);
}

inline InitConditions::FieldInfo generateWaterElevation(int no, int nx, int ny)
{
    float dx = -1;
    float dy = -1;

	if (nx <= 0 || ny <= 0) {
		cout << "Invalid nx or ny: [" << nx << ", " << ny << "]." << endl;
		exit(-1);
	}

	cout << "Generating water elevation: '";

	shared_ptr<vector<float> > f_;
	f_.reset(new vector<float>(nx+1, ny+1));
	vector<float> f = *f_;

	switch (no)
	{
	case 0:
		cout << "column_dry";
#pragma omp parallel for
		for (int j=0; j<static_cast<int>(ny); ++j) {
			float y = (j+0.5f) / (float) ny - 0.5f;
			for (unsigned int i=0; i<nx; ++i) {
				float x = (i+0.5f) / (float) nx - 0.5f;
				if ((x*x)+(y*y)<0.01f)
					f[j*nx+i] = 1.0f;
				else
					f[j*nx+i] = 0.0f;
			}
		}
		break;


	case 1:
		cout << "gaussian";
#pragma omp parallel for
		for (int j=0; j<static_cast<int>(ny); ++j) {
			float y = (j+0.5f) / (float) ny - 0.5f;
			for (unsigned int i=0; i<nx; ++i) {
				float x = (i+0.5f) / (float) nx - 0.5f;
				//if ((x*x)+(y*y)<0.01)
					f[j*nx+i] = exp(-(x*x) - (y*y));
				//else
				//	f[j*nx+i] = 0.0f;
			}
		}
		break;



	case 2:
		cout << "zero";
#pragma omp parallel for
		for (int i=0; i<static_cast<int>(nx*ny); ++i)
			f[i] = 0.0f;
		break;


	case STEADY_FLOW_OVER_BUMP:
		cout << "Steady Flow Over Bump";
		dx = 25.0f / (float) nx;
		dy = 20.0f / (float) ny;
#pragma omp parallel for
		for (int i = 0; i<static_cast<int>(nx*ny); ++i)
			f[i] = 1.0f;
		break;

	case IDEALISED_CIRCULAR_DAM:
		cout << "Idealised Circular Dam";
		dx = 100.0f / (float) nx;
		dy = 100.0f / (float) ny;
#pragma omp parallel for
		for (int j=0; j<static_cast<int>(ny); ++j) {
			float y = dy*(j+0.5f)-50.0f;
			for (unsigned int i=0; i<nx; ++i) {
				float x = dx*(i+0.5f)-50.0f;
				if (sqrt(x*x+y*y) < 6.5f)
					f[j*nx+i] = 10.0f;
				else
					f[j*nx+i] = 0.0f;
			}
		}
		break;

	case 5:
		cout << "column_wet";
#pragma omp parallel for
		for (int j=0; j<static_cast<int>(ny); ++j) {
			float y = (j+0.5f) / (float) ny - 0.5f;
			for (unsigned int i=0; i<nx; ++i) {
				float x = (i+0.5f) / (float) nx - 0.5f;
				if ((x*x)+(y*y)<0.01f)
					f[j*nx+i] = 1.0f;
				else
					f[j*nx+i] = 0.1f;
			}
		}
		break;

    case 10:
        cout << "Wet";
            for (int i=0; i<static_cast<int>(nx*ny); ++i)
                f[i] = 0.3f;
        break;

	default:
		cout << "Could not recognize " << no << " as a valid id." << endl;
		exit(-1);
	}

	cout << "' (" << nx << "x" << ny << " values)" << endl;

    return InitConditions::FieldInfo(f_, nx, ny, dx, dy);
}

void InitConditions::init(const OptionsPtr &options)
{
	if(options->waterElevationNo() >= 0 && options->bathymetryNo() >= 0) {
		///XXX: Maybe we should move the data generation outside of this class?
		pimpl->waterElevationField = generateWaterElevation(options->waterElevationNo(), options->nx(), options->ny());
		pimpl->bathymetryField = generateBathymetry(options->bathymetryNo(), options->nx(), options->ny());
	}
}

InitConditions::FieldInfo InitConditions::waterElevationField() const
{
    return pimpl->waterElevationField;
}

InitConditions::FieldInfo InitConditions::bathymetryField() const
{
    return pimpl->bathymetryField;
}
