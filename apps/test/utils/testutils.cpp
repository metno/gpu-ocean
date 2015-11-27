#include "testutils.h"
#include <vector>
#include <boost/algorithm/string.hpp>

using namespace std;

// Returns an (argc, argv) pair from the words in a string. The word "argv0" is automatically prepended.
pair<int, char **> s2args(const string &s)
{
    vector<string> words;
    boost::split(words, s, boost::is_any_of("\t "));
    words.insert(words.begin(), "argv0");
    const int argc = words.size();
    char **argv = (char **)malloc(argc * sizeof(char *));
    for (int i = 0; i < words.size(); ++i) {
        *(argv + i) = (char *)malloc(words.at(i).size() + 1);
        strcpy(*(argv + i), words.at(i).c_str());
    }

    return make_pair(argc, argv);
}
