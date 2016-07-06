/*
This OpenCL kernel implements the Kurganov-Petrova numerical scheme 
for the shallow water equations, described in 
A. Kurganov & Guergana Petrova
A Second-Order Well-Balanced Positivity Preserving Central-Upwind
Scheme for the Saint-Venant System Communications in Mathematical
Sciences, 5 (2007), 133-160. 

Copyright (C) 2016  SINTEF ICT

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/





/**
 * @return min(a, b, c), {a, b, c} > 0
 *         max(a, b, c), {a, b, c} < 0
 *         0           , otherwise
 */
float minmod(float a, float b, float c) {
	return 0.25f
		*copysign(1.0f, a)
		*(copysign(1.0f, a) + copysign(1.0f, b))
		*(copysign(1.0f, b) + copysign(1.0f, c))
		*min( min(fabs(a), fabs(b)), fabs(c) );
}







/**
  * Reconstructs a slope using the minmod limiter based on three 
  * consecutive values
  */
float reconstructSlope(float left, float center, float right, float theta) {
    const float backward = center - left;
    const float central = (right - left) * 0.5f;
    const float forward = right - center;
    return minmod(theta*backward, central, theta*forward);
}








float windStressX(int wind_stress_type_,
                float dx_, float dy_, float dt_,
                float tau0_, float rho_, float alpha_, float xm_, float Rc_,
                float x0_, float y0_,
                float u0_, float v0_,
                float t_) {
    
    float X = 0.0f;
    
    switch (wind_stress_type_) {
    case 0: //UNIFORM_ALONGSHORE
        {
            const float y = (get_global_id(1)+0.5f)*dy_;
            X = tau0_/rho_ * exp(-alpha_*y);
        }
        break;
    case 1: //BELL_SHAPED_ALONGSHORE
        if (t_ <= 48.0f*3600.0f) {
            const float a = alpha_*((get_global_id(0)+0.5f)*dx_-xm_);
            const float aa = a*a;
            const float y = (get_global_id(1)+0.5f)*dy_;
            X = tau0_/rho_ * exp(-aa) * exp(-alpha_*y);
        }
        break;
    case 2: //MOVING_CYCLONE
        {
            const float x = (get_global_id(0))*dx_;
            const float y = (get_global_id(1)+0.5f)*dy_;
            const float a = (x-x0_-u0_*(t_+dt_));
            const float aa = a*a;
            const float b = (y-y0_-v0_*(t_+dt_));
            const float bb = b*b;
            const float r = sqrt(aa+bb);
            const float c = 1.0f - r/Rc_;
            const float xi = c*c;
            
            X = -(tau0_/rho_) * (b/Rc_) * exp(-0.5f*xi);
        }
        break;
    }

    return X;
}






float windStressY(int wind_stress_type_,
                float dx_, float dy_, float dt_,
                float tau0_, float rho_, float alpha_, float xm_, float Rc_,
                float x0_, float y0_,
                float u0_, float v0_,
                float t_) {
    float Y = 0.0f;
    
    switch (wind_stress_type_) {
    case 2: //MOVING_CYCLONE:
        {
            const float x = (get_global_id(0)+0.5f)*dx_; 
            const float y = (get_global_id(1))*dy_;
            const float a = (x-x0_-u0_*(t_+dt_));
            const float aa = a*a;
            const float b = (y-y0_-v0_*(t_+dt_));
            const float bb = b*b;
            const float r = sqrt(aa+bb);
            const float c = 1.0f - r/Rc_;
            const float xi = c*c;
            
            Y = (tau0_/rho_) * (a/Rc_) * exp(-0.5f*xi);
        }
        break;
    }

    return Y;
}







float3 F_func(const float3 Q, const float g) {
    float3 F;

    F.x = Q.y;                              //hu
    F.y = Q.y*Q.y / Q.x + 0.5f*g*Q.x*Q.x;   //hu*hu/h + 0.5f*g*h*h;
    F.z = Q.y*Q.z / Q.x;                    //hu*hv/h;

    return F;
}





/**
  * Central upwind flux function
  */
float3 CentralUpwindFlux(const float3 Qm, float3 Qp, const float g) {
    const float3 Fp = F_func(Qp, g);
    const float up = Qp.y / Qp.x;   // hu / h
    const float cp = sqrt(g*Qp.x); // sqrt(g*h)

    const float3 Fm = F_func(Qm, g);
    const float um = Qm.y / Qm.x;   // hu / h
    const float cm = sqrt(g*Qm.x); // sqrt(g*h)
    
    const float am = min(min(um-cm, up-cp), 0.0f); // largest negative wave speed
    const float ap = max(max(um+cm, up+cp), 0.0f); // largest positive wave speed
    
    return ((ap*Fm - am*Fp) + ap*am*(Qp-Qm))/(ap-am);
}










/**
  * Harten-Lax-van Leer with contact discontinuity (Toro 2001, p 180)
  */
float3 HLL_flux(const float3 Q_l, const float3 Q_r, const float g_) {    
    const float h_l = Q_l.x;
    const float h_r = Q_r.x;
    
    // Calculate velocities
    const float u_l = Q_l.y / h_l;
    const float u_r = Q_r.y / h_r;
    
    // Estimate the potential wave speeds
    const float c_l = sqrt(g_*h_l);
    const float c_r = sqrt(g_*h_r);
    
    // Compute h in the "star region", h^dagger
    const float h_dag = 0.5f * (h_l+h_r) - 0.25f * (u_r-u_l)*(h_l+h_r)/(c_l+c_r);
    
    const float q_l_tmp = sqrt(0.5f * ( (h_dag+h_l)*h_dag / (h_l*h_l) ) );
    const float q_r_tmp = sqrt(0.5f * ( (h_dag+h_r)*h_dag / (h_r*h_r) ) );
    
    const float q_l = (h_dag > h_l) ? q_l_tmp : 1.0f;
    const float q_r = (h_dag > h_r) ? q_r_tmp : 1.0f;
    
    // Compute wave speed estimates
    const float S_l = u_l - c_l*q_l;
    const float S_r = u_r + c_r*q_r;
    
    //Upwind selection
    if (S_l >= 0.0f) {
        return F_func(Q_l, g_);
    }
    else if (S_r <= 0.0f) {
        return F_func(Q_r, g_);
    }
    //Or estimate flux in the star region
    else {
        const float3 F_l = F_func(Q_l, g_);
        const float3 F_r = F_func(Q_r, g_);
        const float3 flux = (S_r*F_l - S_l*F_r + S_r*S_l*(Q_r - Q_l)) / (S_r-S_l);
        return flux;
    }
}









/**
  * Harten-Lax-van Leer with contact discontinuity (Toro 2001, p 181)
  */
float3 HLLC_flux(const float3 Q_l, const float3 Q_r, const float g_) {    
    const float h_l = Q_l.x;
    const float h_r = Q_r.x;
    
    // Calculate velocities
    const float u_l = Q_l.y / h_l;
    const float u_r = Q_r.y / h_r;
    
    // Estimate the potential wave speeds
    const float c_l = sqrt(g_*h_l);
    const float c_r = sqrt(g_*h_r);
    
    // Compute h in the "star region", h^dagger
    const float h_dag = 0.5f * (h_l+h_r) - 0.25f * (u_r-u_l)*(h_l+h_r)/(c_l+c_r);
    
    const float q_l_tmp = sqrt(0.5f * ( (h_dag+h_l)*h_dag / (h_l*h_l) ) );
    const float q_r_tmp = sqrt(0.5f * ( (h_dag+h_r)*h_dag / (h_r*h_r) ) );
    
    const float q_l = (h_dag > h_l) ? q_l_tmp : 1.0f;
    const float q_r = (h_dag > h_r) ? q_r_tmp : 1.0f;
    
    // Compute wave speed estimates
    const float S_l = u_l - c_l*q_l;
    const float S_r = u_r + c_r*q_r;
    const float S_star = ( S_l*h_r*(u_r - S_r) - S_r*h_l*(u_l - S_l) ) / ( h_r*(u_r - S_r) - h_l*(u_l - S_l) );
    
    const float3 F_l = F_func(Q_l, g_);
    const float3 F_r = F_func(Q_r, g_);
    
    //Upwind selection
    if (S_l >= 0.0f) {
        return F_l;
    }
    else if (S_r <= 0.0f) {
        return F_r;
    }
    //Or estimate flux in the "left star" region
    else if (S_l <= 0.0f && 0.0f <=S_star) {
        const float v_l = Q_l.z / h_l;
        const float3 Q_star_l = h_l * (S_l - u_l) / (S_l - S_star) * (float3)(1, S_star, v_l);
        const float3 flux = F_l + S_l*(Q_star_l - Q_l);
        return flux;
    }
    //Or estimate flux in the "righ star" region
    else if (S_star <= 0.0f && 0.0f <=S_r) {
        const float v_r = Q_r.z / h_r;
        const float3 Q_star_r = h_r * (S_r - u_r) / (S_r - S_star) * (float3)(1, S_star, v_r);
        const float3 flux = F_r + S_r*(Q_star_r - Q_r);
        return flux;
    }
    else {
        return -99999.9f; //Something wrong here
    }
}



/**
  * Superbee flux limiter for WAF.
  * Related to superbee limiter so that WAF_superbee(r, c) = 1 - (1-|c|)*superbee(r)
  * @param r_ the ratio of upwind change (see Toro 2001, p. 203/204)
  * @param c_ the courant number for wave k, dt*S_k/dx
  */
float WAF_superbee(float r_, float c_) {
    // r <= 0.0
    if (r_ <= 0.0f) { 
        return 1.0f;
    }
    // 0.0 <= r <= 1/2
    else if (r_ <= 0.5f) { 
        return 1.0f - 2.0f*(1.0f - fabs(c_))*r_;
    }
    // 1/2 <= r <= 1
    else if (r_ <= 1.0f) {
        return fabs(c_);
    }
    // 1 <= r <= 2
    else  if (r_ <= 2.0f) {
        return 1.0f - (1.0f - fabs(c_))*r_;
    }
    // r >= 2
    else {
        return 2*fabs(c_) - 1.0f;
    }
}




/**
  * Weighted average flux (Toro 2001, p 200) for interface {i+1/2}
  * @param r_ The flux limiter parameter (see Toro 2001, p. 203)
  * @param Q_l2 Q_{i-1}
  * @param Q_l1 Q_{i}
  * @param Q_r1 Q_{i+1}
  * @param Q_r2 Q_{i+2}
  */
float3 WAF_1D_flux(const float3 Q_l2, const float3 Q_l1, const float3 Q_r1, const float3 Q_r2, const float g_, const float dx_, const float dt_) {     
    const float h_l = Q_l1.x;
    const float h_r = Q_r1.x;
    
    const float h_l2 = Q_l2.x;
    const float h_r2 = Q_r2.x;
    
    // Calculate velocities
    const float u_l = Q_l1.y / h_l;
    const float u_r = Q_r1.y / h_r;
    
    const float v_l = Q_l1.z / h_l;
    const float v_r = Q_r1.z / h_r;
    
    const float v_l2 = Q_l2.z / h_l2;
    const float v_r2 = Q_r2.z / h_r2;
    
    // Estimate the potential wave speeds
    const float c_l = sqrt(g_*h_l);
    const float c_r = sqrt(g_*h_r);
    
    // Compute h in the "star region", h^dagger
    const float h_dag = 0.5f * (h_l+h_r) - 0.25f * (u_r-u_l)*(h_l+h_r)/(c_l+c_r);
    
    const float q_l_tmp = sqrt(0.5f * ( (h_dag+h_l)*h_dag / (h_l*h_l) ) );
    const float q_r_tmp = sqrt(0.5f * ( (h_dag+h_r)*h_dag / (h_r*h_r) ) );
    
    const float q_l = (h_dag > h_l) ? q_l_tmp : 1.0f;
    const float q_r = (h_dag > h_r) ? q_r_tmp : 1.0f;
    
    // Compute wave speed estimates
    const float S_l = u_l - c_l*q_l;
    const float S_r = u_r + c_r*q_r;
    const float S_star = ( S_l*h_r*(u_r - S_r) - S_r*h_l*(u_l - S_l) ) / ( h_r*(u_r - S_r) - h_l*(u_l - S_l) );
    
    const float3 Q_star_l = h_l * (S_l - u_l) / (S_l - S_star) * (float3)(1, S_star, v_l);
    const float3 Q_star_r = h_r * (S_r - u_r) / (S_r - S_star) * (float3)(1, S_star, v_r);
    
    // Estimate the fluxes in the four regions
    const float3 F_1 = F_func(Q_l1, g_);
    const float3 F_4 = F_func(Q_r1, g_);
    const float3 F_2 = F_1 + S_l*(Q_star_l - Q_l1);
    const float3 F_3 = F_4 + S_r*(Q_star_r - Q_r1);
    
    // Compute the courant numbers for the waves
    const float c_1 = S_l * dt_ / dx_;
    const float c_2 = S_star * dt_ / dx_;
    const float c_3 = S_r * dt_ / dx_;
    
    // Compute the "upwind change" vectors for the i-3/2 and i+3/2 interfaces
    // We use h for the tangential direction, and v for the normal direction
    const float dh = h_r - h_l;
    const float rh_m = (h_l - h_l2) / dh;
    const float rh_p = (h_r2 - h_r) / dh;
    
    const float dv = v_r - v_l;
    const float rv_m = (v_l - v_l2) / dv;
    const float rv_p = (v_r2 - v_r) / dv;
    
    // Compute the r parameters for the flux limiter
    // Note that you use h for h and hu, and v for hv component/equation
    const float rh_1 = (c_1 > 0.0f) ? rh_m : rh_p; 
    const float rv_1 = (c_1 > 0.0f) ? rv_m : rv_p; 
    
    const float rh_2 = (c_2 > 0.0f) ? rh_m : rh_p; 
    const float rv_2 = (c_2 > 0.0f) ? rv_m : rv_p; 
    
    const float rh_3 = (c_3 > 0.0f) ? rh_m : rh_p;
    const float rv_3 = (c_3 > 0.0f) ? rv_m : rv_p;
    
    // Compute the limiter
    const float3 A_1 = (float3)(WAF_superbee(rh_1, c_1), WAF_superbee(rh_1, c_1), WAF_superbee(rv_1, c_1));
    const float3 A_2 = (float3)(WAF_superbee(rh_2, c_2), WAF_superbee(rh_2, c_2), WAF_superbee(rv_2, c_2));
    const float3 A_3 = (float3)(WAF_superbee(rh_3, c_3), WAF_superbee(rh_3, c_3), WAF_superbee(rv_3, c_3));
        
    //Average the fluxes
    const float3 flux = 0.5f*( F_1 + F_4 )
                      - 0.5f*( sign(c_1) * A_1 * (F_2 - F_1) 
                             + sign(c_2) * A_2 * (F_3 - F_2)
                             + sign(c_3) * A_3 * (F_4 - F_3) );
    return flux;
}







/**
  * Lax-Friedrichs flux (Toro 2001, p 163)
  */
float3 LxF_1D_flux(const float3 Q_l, const float3 Q_r, const float g_, const float dx_, const float dt_) {
    const float3 F_l = F_func(Q_l, g_);
    const float3 F_r = F_func(Q_r, g_);
    
    return 0.5f*(F_l + F_r) + (Q_l - Q_r) * dx_ / (2.0f*dt_);
}



/**
  * Lax-Friedrichs extended to 2D
  */
float3 LxF_2D_flux(const float3 Q_l, const float3 Q_r, const float g_, const float dx_, const float dt_) {
    const float3 F_l = F_func(Q_l, g_);
    const float3 F_r = F_func(Q_r, g_);
    
    //Note numerical diffusion for 2D here (0.25)
    return 0.5f*(F_l + F_r) + (Q_l - Q_r) * dx_ / (4.0f*dt_);
}




/**
  * Richtmeyer / Two-step Lax-Wendroff flux (Toro 2001, p 164)
  */
float3 LxW2_1D_flux(const float3 Q_l, const float3 Q_r, const float g_, const float dx_, const float dt_) {
    const float3 F_l = F_func(Q_l, g_);
    const float3 F_r = F_func(Q_r, g_);
    
    const float3 Q_lw2 = 0.5f*(Q_l + Q_r) + (F_l - F_r)*dt_/(2.0f*dx_);
    
    return F_func(Q_lw2, g_);
}






/**
  * Godunovs centered scheme (Toro 2001, p 165)
  */
float3 GodC_1D_flux(const float3 Q_l, const float3 Q_r, const float g_, const float dx_, const float dt_) {
    const float3 F_l = F_func(Q_l, g_);
    const float3 F_r = F_func(Q_r, g_);
    
    const float3 Q_godc = 0.5f*(Q_l + Q_r) + (F_l - F_r)*dt_/dx_;
    
    return F_func(Q_godc, g_);
}
    

    
    
/**
  * First Ordered Centered (Toro 2001, p.163)
  */
float3 FORCE_1D_flux(const float3 Q_l, const float3 Q_r, const float g_, const float dx_, const float dt_) {
    const float3 F_lf = LxF_1D_flux(Q_l, Q_r, g_, dx_, dt_);
    const float3 F_lw2 = LxW2_1D_flux(Q_l, Q_r, g_, dx_, dt_);
    return 0.5f*(F_lf + F_lw2);
}





