// RUN: %cling %s\(42\) | FileCheck %s
extern "C" int printf(const char* fmt, ...);
void args(int I, const char* S = "ArgString") {
   printf("I=%d\n", I); // CHECK: I=42
   printf("S=%s\n", S); // CHECK: S=ArgString
}
