#ifndef PROFILE_H
#define PROFILE_H

struct ProfileInfo {
    // kernel execution times in milliseconds:
    float time_computeU;
    float time_computeV;
    float time_computeEta;

    ProfileInfo()
        : time_computeU(-1)
        , time_computeV(-1)
        , time_computeEta(-1)
    {}
};

#endif // PROFILE_H
