#ifndef FIELD_H
#define FIELD_H

#include <memory>
#include <vector>
#include <string>

typedef std::shared_ptr<std::vector<float> > FieldPtr;

/**
 * This class represents a 2D field of floats.
 */
class Field2D {
public:

    /**
     * Constructs the object as an empty field.
     */
    Field2D();

    /**
     * Constructs the object with an initial field.
     */
    Field2D(const FieldPtr &fp_, int nx_, int ny_, float dx_, float dy_);

    /**
     * Constructs the object with an initial field.
     */
    Field2D(std::vector<float> *, int, int, float, float);

    /**
     * Copy-constructs the object.
     */
    Field2D(const Field2D &);

    /**
     * Returns the 1D representation.
     */
    FieldPtr getData() const;

    /**
     * Returns the number of grid points in the x dimension.
     */
    int getNx() const;

    /**
     * Returns the number of grid points in the y dimension.
     */
    int getNy() const;

    /**
     * Returns the width of a grid cell.
     */
    float getDx() const;

    /**
     * Returns the height of a grid cell.
     */
    float getDy() const;

    /**
     * Returns a reference to an element in a 2D field.
     * @param i: Index in x dimension; valid interval: [0, nx - 1]
     * @param j: Index in y dimension; valid interval: [0, ny - 1]
     * @returns Reference to element at i + j * nx;
     */
    float &operator()(int i, int j) const;

    /**
     * Fills all elements with a single value.
     */
    void fill(float value);

    /**
     * Fills all elements with values from corresponding elements in another field.
     */
    void fill(const Field2D &src);

    /**
     * Returns true iff the field contains no elements.
     */
    bool empty() const;

    /**
     * Prints the field to stderr. Useful for debugging.
     * @param title: If non-empty, this is printed first, followed by a newline.
     */
    void dump(const std::string &title = std::string()) const;

private:

    int nx;
    int ny;
    float dx;
    float dy;
    FieldPtr data;

    /**
     * Validates the field (verifies that the number of elements is nx * ny etc.).
     */
    void validate() const;
};

#endif // FIELD_H
