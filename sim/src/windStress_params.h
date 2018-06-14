#ifndef WINDSTRESS_PARAMS_H
#define WINDSTRESS_PARAMS_H

/**
 * Mapped to subclasses of WindStress in SWESimulators/WindStress.py
 * DO NOT make changes here without changing WindStress.py accordingly!
 */
typedef enum {NO_WIND,
    GENERIC_UNIFORM,
    ALONGSHORE_UNIFORM,
    ALONGSHORE_BELLSHAPED,
    MOVING_CYCLONE}
wind_stress_type;

/**
 * Mapped to Structure WIND_STRESS_PARAMS in SWESimulators/WindStress.py
 * DO NOT make changes here without changing WindStress.py accordingly!
 */
// packed structs should be used, but are broken in Beignet 1.1.0 :-(
// typedef struct __attribute__ ((packed)) WindStressParams {
typedef struct WindStressParams {
   wind_stress_type type;
   float tau0;
   float rho;
   float rho_air;
   float alpha;
   float xm;
   float Rc;
   float x0;
   float y0;
   float u0;
   float v0;
   float wind_speed;
   float wind_direction;
} wind_stress_params;

#endif // WINDSTRESS_PARAMS_H
