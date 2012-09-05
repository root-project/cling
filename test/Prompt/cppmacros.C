// RUN: %cling %s | FileCheck %s
extern "C" int printf(const char* fmt, ...);
#define MYMACRO(v) \
   if (v) { \
      printf("string:%s\n", v);\
   }

void cppmacros() {
   MYMACRO("PARAM"); // CHECK: string:PARAM
}
