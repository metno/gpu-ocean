#ifndef FIELD_H
#define FIELD_H

#include <memory>
#include <vector>

typedef std::shared_ptr<std::vector<float> > FieldPtr;

struct FieldInfo {
    FieldPtr data;
    int nx;
    int ny;
    float dx;
    float dy;
    FieldInfo();
    FieldInfo(const FieldPtr &, int, int, float, float);
    float &operator()(int, int) const;
};

#endif // FIELD_H
