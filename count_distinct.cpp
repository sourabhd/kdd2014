#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <string>
#include <fstream>
#include <tr1/unordered_set>

using namespace std;
using namespace tr1;

int main(int argc, char *argv[])
{
    time_t t1 = time(NULL);
    string fname = "essay_corpus.txt";
    ifstream ifs;
    ifs.open(fname.c_str());
    if (!ifs) {
        cerr << "Could not open " << fname.c_str() << endl;
        exit(0);
    }
    string word = "";
    unordered_set<string> S; 
    while (true) {
        if (ifs.eof()) break;
        word = "";
        getline(ifs, word, ' ');
        S.insert(word);
    }
    ifs.close();
    cout << "Number of distinct words : " << S.size() << endl;
    time_t t2 = time(NULL);
    cout << "Time taken : " << (t2-t1) << endl;

}

