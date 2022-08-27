#include <iostream>

namespace io = std;

int startup_magic_num{43210};

void startup0() {
  std::cout << "Startup file ran, magic # was " << startup_magic_num << '\n';
  startup_magic_num += 2;
}
