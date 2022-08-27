
const char* startup_magic_str = "AaBbCc__51";

void startup1() {
  std::cout << "Startup file ran, magic # was " << startup_magic_num << '\n';
  ++startup_magic_num;
}
