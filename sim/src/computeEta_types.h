#ifndef COMPUTEETA_TYPES_H
#define COMPUTEETA_TYPES_H

typedef struct {
	//Discretization parameters
    int nx;
    int ny;
    float dt;
    float dx;
    float dy;

    //Physical parameters
    float g; //< Gravitational constant
    float f; //< Coriolis coefficient
    float r; //< Bottom friction coefficient
} computeEta_args;

#endif // COMPUTEETA_TYPES_H
