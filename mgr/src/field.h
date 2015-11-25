#ifndef FIELD_H
#define FIELD_H

#include <memory>
#include <vector>
#include <string>

typedef std::shared_ptr<std::vector<float> > FieldPtr;

struct FieldInfo {
    FieldPtr data;
    int nx;
    int ny;
    float dx;
    float dy;
    FieldInfo();
    FieldInfo(const FieldPtr &, int, int, float, float);
    FieldInfo(std::vector<float> *, int, int, float, float);
    float &operator()(int, int) const;
    bool empty() const;
    void dump(const std::string & = std::string()) const;
};

#endif // FIELD_H
