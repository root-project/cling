#include <fstream>
#include <string>

void Overloads () {
  std::string s;
  std::ofstream("file.txt") << s << std::endl;
}
