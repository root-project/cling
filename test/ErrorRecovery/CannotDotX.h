extern "C" int printf(const char* fmt, ...);

class MyClass {
  MyClass() { printf("MyClass ctor called!\n"); }
};
