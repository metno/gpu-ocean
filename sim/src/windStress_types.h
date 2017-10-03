#ifndef WINDSTRESS_TYPES_H
#define WINDSTRESS_TYPES_H

typedef struct {
	int wind_stress_type;
	float tau0;
	float rho;
	float alpha;
	float xm;
	float Rc;
	float x0;
	float y0;
	float u0;
	float v0;
} windStress_args;

#endif // WINDSTRESS_TYPES_H
