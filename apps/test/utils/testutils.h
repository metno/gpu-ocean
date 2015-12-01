#ifndef TESTUTILS_H
#define TESTUTILS_H

#include "field.h"
#include <vector>
#include <string>
#include <utility>

/**
 * Converts a list of words to standard arguments.
 * @param words: A vector of words.
 * @return An standard (argc, argv) pair.
 * @note The argv vector is allocated on the heap using malloc and the caller is responsible for freeing this memory.
 * @note The argument "argv0" is automatically prepended to the output.
 */
std::pair<int, char **> createArgs(const std::vector<std::string> &words);

/**
 * Converts a string of words to standard arguments.
 * @param words: A string of words separated by whitespace.
 * @return An standard (argc, argv) pair.
 * @note The argv vector is allocated on the heap using malloc and the caller is responsible for freeing this memory.
 * @note The argument "argv0" is automatically prepended to the output.
 */
std::pair<int, char **> createArgs(const std::string &s);

/**
 * Checks (using BOOST_CHECK*) if the vector of two fields are equal (per-item comparision of the std::vector objects).
 * @param f1: First field (FieldInfo object)
 * @param f2: Second field (FieldInfo object)
 */
#define CHECK_VECTORS_EQUAL(f1, f2) \
    do { \
        BOOST_CHECK(*(f1.data.get()) == *(f2.data.get())); \
    } while (false)

/**
 * Checks (using BOOST_CHECK*) if two fields are equal.
 * @param f1: First field (FieldInfo object)
 * @param f2: Second field (FieldInfo object)
 */
#define CHECK_FIELDS_EQUAL(f1, f2) \
    do { \
        BOOST_CHECK_EQUAL(f1.nx, f2.nx); \
        BOOST_CHECK_EQUAL(f1.ny, f2.ny); \
        BOOST_CHECK_EQUAL(f1.dx, f2.dx); \
        BOOST_CHECK_EQUAL(f1.dy, f2.dy); \
        CHECK_VECTORS_EQUAL(f1, f2); \
    } while (false)

#endif // TESTUTILS_H
