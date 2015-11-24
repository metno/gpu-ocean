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
    FieldInfo H;
    FieldInfo eta;
    InitConditionsImpl();
};

InitConditions::InitConditionsImpl::InitConditionsImpl()
{
}

InitConditions::InitConditions()
    : pimpl(new InitConditionsImpl)
{
}

inline FieldInfo generateBathymetry(int no, int nx, int ny, float width, float height)
{
    const float dx = width / (nx - 1);
    const float dy = height / (ny - 1);

	if (nx <= 0 || ny <= 0) {
		stringstream log;
		log << "Invalid nx or ny: [" << nx << ", " << ny << "]." << endl;
		throw runtime_error(log.str());
	}

	cout << "Generating bathymetry: '";

    vector<float> *f = new vector<float>((nx + 1) * (ny + 1));

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
        for (int j = 0; j <= ny; ++j) {
			float y = j * 6.0f/(float) ny - 3.0f;
            for (int i = 0; i <= nx; ++i) {
                float x = i * 6.0f/nx - 3.0f;
				float value = 3.0f*(1-x)*(1-x) * exp(-(x*x) - (y-1)*(y-1))
								- 10.0f * (x/5.0f - x*x*x - y*y*y*y*y) * exp(-(x*x) - (y*y))
								- 1.0f/3.0f * exp(-(x+1)*(x+1) - (y*y));

                f->at(j * (nx + 1) + i) = 0.1f*value;
			}
		}
		break;
	case 2:
		cout << "3 bumps";
#pragma omp parallel for
        for (int j=0; j <= ny; ++j) {
            const float y = j / (float) ny;
            for (int i = 0; i <= nx; ++i) {
                const float x = i / (float) nx;
                const int index = j * (nx + 1) + i;
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
        for (int j = 0; j <= ny; ++j) {
            for (int i = 0; i <= nx; ++i) {
                const float x = i*dx;
                f->at(j * (nx + 1) + i) = (x1 < x && x < x2)
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

    cout << "' (" << nx + 1 << "x" << ny + 1 << " values)" << endl;

    return FieldInfo(f, nx + 1, ny + 1, dx, dy);
}

inline FieldInfo generateWaterElevation(int no, int nx, int ny, float width, float height)
{
    const float dx = width / (nx - 1);
    const float dy = height / (ny - 1);

	if (nx <= 0 || ny <= 0) {
		cout << "Invalid nx or ny: [" << nx << ", " << ny << "]." << endl;
		exit(-1);
	}

	cout << "Generating water elevation: '";

    vector<float> *f = new vector<float>((nx + 1) * (ny + 1));

	switch (no)
	{
	case 0:
		cout << "column_dry";
#pragma omp parallel for
        for (int j = 0; j <= ny; ++j) {
			float y = (j+0.5f) / (float) ny - 0.5f;
            for (int i = 0; i <= nx; ++i) {
                float x = (i+0.5f) / (float) nx - 0.5f;
                f->at(j * (nx + 1) + i) = ((x*x)+(y*y)<0.01f) ? 1.0f : 0.0f;
			}
		}
		break;


	case 1:
		cout << "gaussian";
#pragma omp parallel for
        for (int j = 0; j <= ny; ++j) {
			float y = (j+0.5f) / (float) ny - 0.5f;
            for (int i = 0; i <= nx; ++i) {
				float x = (i+0.5f) / (float) nx - 0.5f;
				//if ((x*x)+(y*y)<0.01)
                    f->at(j * (nx + 1) + i) = exp(-(x*x) - (y*y));
				//else
                //	f->at(j * (nx + 1) + i) = 0.0f;
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
        for (int j = 0; j <= ny; ++j) {
            float y = dy*(j+0.5f)-y1;
            for (int i = 0; i <= nx; ++i) {
                float x = dx*(i+0.5f)-x1;
                f->at(j * (nx + 1) + i) = (sqrt(x*x+y*y) < v2) ? v1 : 0.0f;
            }
        }
    }
        break;

	case 5:
		cout << "column_wet";
#pragma omp parallel for
        for (int j = 0; j <= ny; ++j) {
			float y = (j+0.5f) / (float) ny - 0.5f;
            for (int i = 0; i <= nx; ++i) {
				float x = (i+0.5f) / (float) nx - 0.5f;
                f->at(j * (nx + 1) + i) = ((x*x)+(y*y)<0.01f) ? 1.5f : 1.0f;
			}
		}
		break;

    case 10:
        cout << "Wet";
            for (int i = 0; i < f->size(); ++i)
                f->at(i) = 0.3f;
        break;

	default:
		cout << "Could not recognize " << no << " as a valid id." << endl;
		exit(-1);
	}

    cout << "' (" << nx + 1 << "x" << ny + 1 << " values)" << endl;

    return FieldInfo(f, nx + 1, ny + 1, dx, dy);
}

inline FieldInfo generateH(int nx, int ny, float width, float height, FieldInfo B, float w=1.0f)
{
	if (nx <= 0 || ny <= 0) {
		cout << "Invalid nx or ny: [" << nx << ", " << ny << "]." << endl;
		exit(-1);
	}

	cout << "Generating sea surface mean depth (H): '";

    vector<float> *f = new vector<float>((nx + 1) * (ny + 1));

#pragma omp parallel for
    for (int i = 0; i < f->size(); ++i) {
        f->at(i) = w - B.data->at(i);
        if (f->at(i) < 0.0f) {
			cout << "Negative values in H are not allowed. (Increase global water elevation (w) or change bathymetry (B) input.)" << endl;
			exit(-1);
		}
	}

    cout << "' (" << nx + 1 << "x" << ny + 1 << " values)" << endl;

    return FieldInfo(f, nx + 1, ny + 1, width / (nx - 1), height / (ny - 1));
}

inline FieldInfo generateEta(int no, int nx, int ny, float width, float height)
{
    const float dx = width / (nx - 1);
    const float dy = height / (ny - 1);

	if (nx <= 0 || ny <= 0) {
		cout << "Invalid nx or ny: [" << nx << ", " << ny << "]." << endl;
		exit(-1);
	}

	cout << "Generating sea surface deviation (eta): '";

    vector<float> *f = new vector<float>((nx + 1) * (ny + 1));

	switch (no)
	{
	case 0:
		cout << "column";
#pragma omp parallel for
        for (int j = 0; j <= ny; ++j) {
			float y = (j+0.5f) / (float) ny - 0.5f;
            for (int i = 0; i <= nx; ++i) {
				float x = (i+0.5f) / (float) nx - 0.5f;
                f->at(j * (nx + 1) + i) = ((x*x)+(y*y)<0.01f) ? 1.0f : 0.0f;
			}
		}
		break;

	case 1:
		cout << "gaussian";
#pragma omp parallel for
        for (int j = 0; j <= ny; ++j) {
			float y = (j+0.5f) / (float) ny - 0.5f;
            for (int i = 0; i <= nx; ++i) {
				float x = (i+0.5f) / (float) nx - 0.5f;
                f->at(j * (nx + 1) + i) = exp(-(x*x) - (y*y));
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
        for (int j = 0; j <= ny; ++j) {
            float y = dy*(j+0.5f)-y1;
            for (int i = 0; i <= nx; ++i) {
                float x = dx*(i+0.5f)-x1;
                f->at(j * (nx + 1) + i) = (sqrt(x*x+y*y) < v2) ? v1 : 0.0f;
            }
        }
    }
        break;

	default:
		cout << "Could not recognize " << no << " as a valid id." << endl;
		exit(-1);
	}

    cout << "' (" << nx + 1 << "x" << ny + 1 << " values)" << endl;

    return FieldInfo(f, nx + 1, ny + 1, dx, dy);
}

void InitConditions::init(const OptionsPtr &options)
{
	float wGlobal = 1.0f;

	/// TODO: Maybe we should move the data generation outside of this class?
	if((options->waterElevationNo() >= 0 || options->etaNo() >= 0) && options->bathymetryNo() >= 0) {
		/// TODO: We don't really need this field to initialize a simulation run for time being, but maybe later...
		//pimpl->waterElevationField = generateWaterElevation(options->waterElevationNo(), options->nx(), options->ny(), options->width(), options->height());
		if(options->etaNo() < 0)
			cerr << "warning: etaNo must be used! (waterElevationNo is not implemented yet)\n";

		pimpl->bathymetryField = generateBathymetry(options->bathymetryNo(), options->nx(), options->ny(), options->width(), options->height());
		if(options->wGlobal() < 0)
			cout << "warning: global water elevation level value (wGlobal) not given! (using default: " << wGlobal << ")\n";
        pimpl->H = generateH(options->nx(), options->ny(), options->width(), options->height(), pimpl->bathymetryField, wGlobal);

        pimpl->eta = generateEta(options->etaNo(), options->nx(), options->ny(), options->width(), options->height());
    } else {
        cerr << "warning: at least one of waterElevationNo/etaNo and bathymetryNo is less than zero => bathymetryField, H, and eta not initialized!\n";
    }
}

FieldInfo InitConditions::waterElevationField() const
{
    return pimpl->waterElevationField;
}

FieldInfo InitConditions::bathymetryField() const
{
    return pimpl->bathymetryField;
}

FieldInfo InitConditions::H() const
{
    return pimpl->H;
}

FieldInfo InitConditions::eta() const
{
    return pimpl->eta;
}
