#ifndef COMPUTEV_TYPES_H
#define COMPUTEV_TYPES_H

typedef struct {
	//Discretization parameters
    int nx;
    int ny;
    float dt;
    float dx;
    float dy;

    //Physical parameters
    float r; //< Bottom friction coefficient
    float f; //< Coriolis coefficient
    float g; //< Gravitational constant
} computeV_args;

#endif // COMPUTEV_TYPES_H
