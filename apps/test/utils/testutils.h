#ifndef TESTUTILS_H
#define TESTUTILS_H

#include <vector>
#include <string>
#include <utility>

std::pair<int, char **> createArgs(const std::vector<std::string> &);
std::pair<int, char **> createArgs(const std::string &);

#endif // TESTUTILS_H
