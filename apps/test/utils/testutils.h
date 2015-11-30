#ifndef TESTUTILS_H
#define TESTUTILS_H

#include "field.h"
#include <vector>
#include <string>
#include <utility>

std::pair<int, char **> createArgs(const std::vector<std::string> &);
std::pair<int, char **> createArgs(const std::string &);

// Checks if the vector of two fields are equal.
#define CHECK_VECTORS_EQUAL(f1, f2) \
    BOOST_CHECK(*(f1.data.get()) == *(f2.data.get())); // per-item comparision of the std::vector objects

// Checks if two fields are equal.
#define CHECK_FIELDS_EQUAL(f1, f2) \
    BOOST_CHECK_EQUAL(f1.nx, f2.nx); \
    BOOST_CHECK_EQUAL(f1.ny, f2.ny); \
    BOOST_CHECK_EQUAL(f1.dx, f2.dx); \
    BOOST_CHECK_EQUAL(f1.dy, f2.dy); \
    CHECK_VECTORS_EQUAL(f1, f2);

#endif // TESTUTILS_H
