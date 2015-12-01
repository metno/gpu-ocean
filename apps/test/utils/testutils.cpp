#include "testutils.h"
#include <boost/algorithm/string.hpp>

using namespace std;

pair<int, char **> createArgs(const vector<string> &words)
{
    vector<string> w(words);
    w.insert(w.begin(), "argv0");
    const int argc = w.size();
    char **argv = (char **)malloc(argc * sizeof(char *));
    for (int i = 0; i < w.size(); ++i) {
        *(argv + i) = (char *)malloc(w.at(i).size() + 1);
        strcpy(*(argv + i), w.at(i).c_str());
    }

    return make_pair(argc, argv);
}

pair<int, char **> createArgs(const string &s)
{
    vector<string> words;
    boost::split(words, s, boost::is_any_of("\t "));
    return createArgs(words);
}
