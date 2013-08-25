// RUN: cat %s | %cling 2>&1 | FileCheck %s
// Test lookupFunctionArgs()
.rawInput
#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/LookupHelper.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdio>
#include <string>

using namespace std;
using namespace llvm;

void dumpDecl(const char* title, const clang::Decl* D) {
  printf("%s: 0x%lx\n", title, (unsigned long) D);
  std::string S;
  llvm::raw_string_ostream OS(S);
  dyn_cast<clang::NamedDecl>(D)
    ->getNameForDiagnostic(OS, D->getASTContext().getPrintingPolicy(),
                           /*Qualified=*/true);
  printf("%s name: %s\n", title, OS.str().c_str());
  fflush(stdout);
  D->print(llvm::errs());
}
.rawInput

//
//  We need to fetch the global scope declaration,
//  otherwise known as the translation unit decl.
//
const cling::LookupHelper& lookup = gCling->getLookupHelper();
const clang::Decl* G = lookup.findScope("");
printf("G: 0x%lx\n", (unsigned long) G);
//CHECK: G: 0x{{[1-9a-f][0-9a-f]*$}}

//
//  Some tools for printing.
//

.rawInput 1
void G_f() { int x = 1; }
void G_a(int v) { int x = v; }
void G_b(int vi, double vd) { int x = vi; double y = vd; }
void G_c(int vi, int vj) { int x = vi; int y = vj; }
void G_c(int vi, double vd) { int x = vi; double y = vd; }
template <class T> void G_d(T v) { T x = v; }
// Note: In CINT, looking up a class template specialization causes
//       instantiation, but looking up a function template specialization
//       does not, so we explicitly request the instantiations we are
//       going to lookup so they will be there to find.
template void G_d(int);
template void G_d(double);
namespace N {
void H_f() { int x = 1; }
void H_a(int v) { int x = v; }
void H_b(int vi, double vd) { int x = vi; double y = vd; }
void H_c(int vi, int vj) { int x = vi; int y = vj; }
void H_c(int vi, double vd) { int x = vi; double y = vd; }
template <class T> void H_d(T v) { T x = v; }
// Note: In CINT, looking up a class template specialization causes
//       instantiation, but looking up a function template specialization
//       does not, so we explicitly request the instantiations we are
//       going to lookup so they will be there to find.
template void H_d(int);
template void H_d(double);
} // namespace N

class B {
private:
   long m_B_i;
   double m_B_d;
   int* m_B_ip;
public:
   virtual ~B() { delete m_B_ip; m_B_ip = 0; }
   B() : m_B_i(0), m_B_d(0.0), m_B_ip(0) {}
   B(int vi, double vd) : m_B_i(vi), m_B_d(vd), m_B_ip(0) {}
   template <class T> B(T v) : m_B_i(0), m_B_d(0.0), m_B_ip(0) { m_B_i = (T) v; }
   template <class T> B(T* v) : m_B_i(0), m_B_d(0.0), m_B_ip(0) { m_B_i = (long) (T*) v; m_B_d = 1.0; }
   void B_f() { int x = 1; }
   void B_g(int v) { int x = v; }
   void B_h(int vi, double vd) { int x = vi; double y = vd; }
   void B_j(int vi, int vj) { int x = vi; int y = vj; }
   void B_j(int vi, double vd) { int x = vi; double y = vd; }
   template <class T> void B_k(T v) { T x = v; }
   void B_m(const int& v) { int y = v; }
   const long &B_n() const { return m_B_i; }
   long &B_n() { return m_B_i; }
   const long &B_o() const { return m_B_i; }
   long B_p(float) const { return 0; }
   int B_p(int) { return 0; }
   void* operator new(std::size_t sz) { return ::operator new(sz); }
   void* operator new(std::size_t sz, void* arena) { return arena; }
   void* operator new[](std::size_t sz) { return ::operator new[](sz); }
   void* operator new[](std::size_t sz, void* arena) { return arena; }
   void operator delete(void* vp) { ::operator delete(vp); }
   void operator delete(void* vp, void* arena) {}
   void operator delete[](void* vp) { ::operator delete[](vp); }
   void operator delete[](void* vp, void* arena) {}
   B& operator*() { return *this; }
   B operator+(B b) { return b; }
};
class A : public B {
private:
   int m_A_i;
   double m_A_d;
public:
   void A_f() { int x = 1; }
   void A_g(int v) { int x = v; }
   void A_h(int vi, double vd) { int x = vi; double y = vd; }
   void A_j(int vi, int vj) { int x = vi; int y = vj; }
   void A_j(int vi, double vd) { int x = vi; double y = vd; }
   template <class T> void A_k(T v) { T x = v; }
   void A_m(const int& v) { int y = v; }
   void* operator new(std::size_t sz) { return ::operator new(sz); }
   void* operator new(std::size_t sz, void* arena) { return arena; }
   void* operator new[](std::size_t sz) { return ::operator new[](sz); }
   void* operator new[](std::size_t sz, void* arena) { return arena; }
   void operator delete(void* vp) { ::operator delete(vp); }
   void operator delete(void* vp, void* arena) {}
   void operator delete[](void* vp) { ::operator delete[](vp); }
   void operator delete[](void* vp, void* arena) {}
   
   void A_n(B& b) { b.B_f(); }
   void A_n(const char *msg, int ndim = 0) { if (ndim) ++msg; }
};
// Note: In CINT, looking up a class template specialization causes
//       instantiation, but looking up a function template specialization
//       does not, so we explicitly request the instantiations we are
//       going to lookup so they will be there to find.
template void A::A_k(int);
template void A::A_k(double);
template void A::B_k(int);
template void A::B_k(double);
B b_obj;
B* b_ptr = &b_obj;
B* b_ary = new B[3];
char b_arena[sizeof(B)*10];
char b_ary_arena[256];
.rawInput 0



//
//  We need these class declarations.
//

const clang::Decl* class_A = lookup.findScope("A");
printf("class_A: 0x%lx\n", (unsigned long) class_A);
//CHECK: class_A: 0x{{[1-9a-f][0-9a-f]*$}}

const clang::Decl* class_B = lookup.findScope("B");
printf("class_B: 0x%lx\n", (unsigned long) class_B);
//CHECK-NEXT: class_B: 0x{{[1-9a-f][0-9a-f]*$}}



//
//  We need to fetch the namespace N declaration.
//

const clang::Decl* namespace_N = lookup.findScope("N");
printf("namespace_N: 0x%lx\n", (unsigned long) namespace_N);
//CHECK: namespace_N: 0x{{[1-9a-f][0-9a-f]*$}}



//
//  Test finding a global function taking no args.
//

const clang::FunctionDecl* G_f_args = lookup.findFunctionArgs(G, "G_f", "");
const clang::FunctionDecl* G_f_proto = lookup.findFunctionProto(G, "G_f", "");

printf("G_f_args: 0x%lx\n", (unsigned long) G_f_args);
//CHECK-NEXT: G_f_args: 0x{{[1-9a-f][0-9a-f]*$}}
G_f_args->print(llvm::errs());
//CHECK-NEXT: void G_f() {
//CHECK-NEXT:     int x = 1;
//CHECK-NEXT: }

printf("G_f_proto: 0x%lx\n", (unsigned long) G_f_proto);
//CHECK: G_f_proto: 0x{{[1-9a-f][0-9a-f]*$}}
G_f_proto->print(llvm::errs());
//CHECK-NEXT: void G_f() {
//CHECK-NEXT:     int x = 1;
//CHECK-NEXT: }



//
//  Test finding a global function taking a single int argument.
//

const clang::FunctionDecl* G_a_args = lookup.findFunctionArgs(G, "G_a", "0");
const clang::FunctionDecl* G_a_proto = lookup.findFunctionProto(G, "G_a", "int");

printf("G_a_args: 0x%lx\n", (unsigned long) G_a_args);
//CHECK: G_a_args: 0x{{[1-9a-f][0-9a-f]*$}}
G_a_args->print(llvm::errs());
//CHECK-NEXT: void G_a(int v) {
//CHECK-NEXT:     int x = v;
//CHECK-NEXT: }

printf("G_a_proto: 0x%lx\n", (unsigned long) G_a_proto);
//CHECK: G_a_proto: 0x{{[1-9a-f][0-9a-f]*$}}
G_a_proto->print(llvm::errs());
//CHECK-NEXT: void G_a(int v) {
//CHECK-NEXT:     int x = v;
//CHECK-NEXT: }



//
//  Test finding a global function taking an int and a double argument.
//

const clang::FunctionDecl* G_b_args = lookup.findFunctionArgs(G, "G_b", "0,0.0");
const clang::FunctionDecl* G_b_proto = lookup.findFunctionProto(G, "G_b", "int,double");

printf("G_b_args: 0x%lx\n", (unsigned long) G_b_args);
//CHECK: G_b_args: 0x{{[1-9a-f][0-9a-f]*$}}
G_b_args->print(llvm::errs());
//CHECK-NEXT: void G_b(int vi, double vd) {
//CHECK-NEXT:     int x = vi;
//CHECK-NEXT:     double y = vd;
//CHECK-NEXT: }

printf("G_b_proto: 0x%lx\n", (unsigned long) G_b_proto);
//CHECK: G_b_proto: 0x{{[1-9a-f][0-9a-f]*$}}
G_b_proto->print(llvm::errs());
//CHECK-NEXT: void G_b(int vi, double vd) {
//CHECK-NEXT:     int x = vi;
//CHECK-NEXT:     double y = vd;
//CHECK-NEXT: }



//
//  Test finding a global overloaded function.
//

const clang::FunctionDecl* G_c1_args = lookup.findFunctionArgs(G, "G_c", "0,0");
const clang::FunctionDecl* G_c1_proto = lookup.findFunctionProto(G, "G_c", "int,int");

printf("G_c1_args: 0x%lx\n", (unsigned long) G_c1_args);
//CHECK: G_c1_args: 0x{{[1-9a-f][0-9a-f]*$}}
G_c1_args->print(llvm::errs());
//CHECK-NEXT: void G_c(int vi, int vj) {
//CHECK-NEXT:     int x = vi;
//CHECK-NEXT:     int y = vj;
//CHECK-NEXT: }

printf("G_c1_proto: 0x%lx\n", (unsigned long) G_c1_proto);
//CHECK: G_c1_proto: 0x{{[1-9a-f][0-9a-f]*$}}
G_c1_proto->print(llvm::errs());
//CHECK-NEXT: void G_c(int vi, int vj) {
//CHECK-NEXT:     int x = vi;
//CHECK-NEXT:     int y = vj;
//CHECK-NEXT: }

const clang::FunctionDecl* G_c2_args = lookup.findFunctionArgs(G, "G_c", "0,0.0");
const clang::FunctionDecl* G_c2_proto = lookup.findFunctionProto(G, "G_c", "int,double");

printf("G_c2_args: 0x%lx\n", (unsigned long) G_c2_args);
//CHECK: G_c2_args: 0x{{[1-9a-f][0-9a-f]*$}}
G_c2_args->print(llvm::errs());
//CHECK-NEXT: void G_c(int vi, double vd) {
//CHECK-NEXT:     int x = vi;
//CHECK-NEXT:     double y = vd;
//CHECK-NEXT: }

printf("G_c2_proto: 0x%lx\n", (unsigned long) G_c2_proto);
//CHECK: G_c2_proto: 0x{{[1-9a-f][0-9a-f]*$}}
G_c2_proto->print(llvm::errs());
//CHECK-NEXT: void G_c(int vi, double vd) {
//CHECK-NEXT:     int x = vi;
//CHECK-NEXT:     double y = vd;
//CHECK-NEXT: }



//
//  Test finding simple global template instantiations.
//

const clang::FunctionDecl* G_d1_args = lookup.findFunctionArgs(G, "G_d<int>", "0");
const clang::FunctionDecl* G_d1_proto = lookup.findFunctionProto(G, "G_d<int>", "int");

printf("G_d1_args: 0x%lx\n", (unsigned long) G_d1_args);
//CHECK: G_d1_args: 0x{{[1-9a-f][0-9a-f]*$}}
G_d1_args->print(llvm::errs());
//CHECK-NEXT: void G_d(int v) {
//CHECK-NEXT:     int x = v;
//CHECK-NEXT: }

printf("G_d1_proto: 0x%lx\n", (unsigned long) G_d1_proto);
//CHECK: G_d1_proto: 0x{{[1-9a-f][0-9a-f]*$}}
G_d1_proto->print(llvm::errs());
//CHECK-NEXT: void G_d(int v) {
//CHECK-NEXT:     int x = v;
//CHECK-NEXT: }

const clang::FunctionDecl* G_d2_args = lookup.findFunctionArgs(G, "G_d<double>", "0.0");
const clang::FunctionDecl* G_d2_proto = lookup.findFunctionProto(G, "G_d<double>", "double");

printf("G_d2_args: 0x%lx\n", (unsigned long) G_d2_args);
//CHECK: G_d2_args: 0x{{[1-9a-f][0-9a-f]*$}}
G_d2_args->print(llvm::errs());
//CHECK-NEXT: void G_d(double v) {
//CHECK-NEXT:     double x = v;
//CHECK-NEXT: }

printf("G_d2_proto: 0x%lx\n", (unsigned long) G_d2_proto);
//CHECK: G_d2_proto: 0x{{[1-9a-f][0-9a-f]*$}}
G_d2_proto->print(llvm::errs());
//CHECK-NEXT: void G_d(double v) {
//CHECK-NEXT:     double x = v;
//CHECK-NEXT: }



//
//  Test finding a namespace function taking no args.
//

const clang::FunctionDecl* H_f_args = lookup.findFunctionArgs(namespace_N, "H_f", "");
const clang::FunctionDecl* H_f_proto = lookup.findFunctionProto(namespace_N, "H_f", "");

printf("H_f_args: 0x%lx\n", (unsigned long) H_f_args);
//CHECK: H_f_args: 0x{{[1-9a-f][0-9a-f]*$}}
H_f_args->print(llvm::errs());
//CHECK-NEXT: void H_f() {
//CHECK-NEXT:     int x = 1;
//CHECK-NEXT: }

printf("H_f_proto: 0x%lx\n", (unsigned long) H_f_proto);
//CHECK: H_f_proto: 0x{{[1-9a-f][0-9a-f]*$}}
H_f_proto->print(llvm::errs());
//CHECK-NEXT: void H_f() {
//CHECK-NEXT:     int x = 1;
//CHECK-NEXT: }



//
//  Test finding a namespace function taking a single int argument.
//

const clang::FunctionDecl* H_a_args = lookup.findFunctionArgs(namespace_N, "H_a", "0");
const clang::FunctionDecl* H_a_proto = lookup.findFunctionProto(namespace_N, "H_a", "int");

printf("H_a_args: 0x%lx\n", (unsigned long) H_a_args);
//CHECK: H_a_args: 0x{{[1-9a-f][0-9a-f]*$}}
H_a_args->print(llvm::errs());
//CHECK-NEXT: void H_a(int v) {
//CHECK-NEXT:     int x = v;
//CHECK-NEXT: }

printf("H_a_proto: 0x%lx\n", (unsigned long) H_a_proto);
//CHECK: H_a_proto: 0x{{[1-9a-f][0-9a-f]*$}}
H_a_proto->print(llvm::errs());
//CHECK-NEXT: void H_a(int v) {
//CHECK-NEXT:     int x = v;
//CHECK-NEXT: }



//
//  Test finding a namespace function taking an int and a double argument.
//

const clang::FunctionDecl* H_b_args = lookup.findFunctionArgs(namespace_N, "H_b", "0,0.0");
const clang::FunctionDecl* H_b_proto = lookup.findFunctionProto(namespace_N, "H_b", "int,double");

printf("H_b_args: 0x%lx\n", (unsigned long) H_b_args);
//CHECK: H_b_args: 0x{{[1-9a-f][0-9a-f]*$}}
H_b_args->print(llvm::errs());
//CHECK-NEXT: void H_b(int vi, double vd) {
//CHECK-NEXT:     int x = vi;
//CHECK-NEXT:     double y = vd;
//CHECK-NEXT: }

printf("H_b_proto: 0x%lx\n", (unsigned long) H_b_proto);
//CHECK: H_b_proto: 0x{{[1-9a-f][0-9a-f]*$}}
H_b_proto->print(llvm::errs());
//CHECK-NEXT: void H_b(int vi, double vd) {
//CHECK-NEXT:     int x = vi;
//CHECK-NEXT:     double y = vd;
//CHECK-NEXT: }



//
//  Test finding a namespace overloaded function.
//

const clang::FunctionDecl* H_c1_args = lookup.findFunctionArgs(namespace_N, "H_c", "0,0");
const clang::FunctionDecl* H_c1_proto = lookup.findFunctionProto(namespace_N, "H_c", "int,int");

printf("H_c1_args: 0x%lx\n", (unsigned long) H_c1_args);
//CHECK: H_c1_args: 0x{{[1-9a-f][0-9a-f]*$}}
H_c1_args->print(llvm::errs());
//CHECK-NEXT: void H_c(int vi, int vj) {
//CHECK-NEXT:     int x = vi;
//CHECK-NEXT:     int y = vj;
//CHECK-NEXT: }

printf("H_c1_proto: 0x%lx\n", (unsigned long) H_c1_proto);
//CHECK: H_c1_proto: 0x{{[1-9a-f][0-9a-f]*$}}
H_c1_proto->print(llvm::errs());
//CHECK-NEXT: void H_c(int vi, int vj) {
//CHECK-NEXT:     int x = vi;
//CHECK-NEXT:     int y = vj;
//CHECK-NEXT: }

const clang::FunctionDecl* H_c2_args = lookup.findFunctionArgs(namespace_N, "H_c", "0,0.0");
const clang::FunctionDecl* H_c2_proto = lookup.findFunctionProto(namespace_N, "H_c", "int,double");

printf("H_c2_args: 0x%lx\n", (unsigned long) H_c2_args);
//CHECK: H_c2_args: 0x{{[1-9a-f][0-9a-f]*$}}
H_c2_args->print(llvm::errs());
//CHECK-NEXT: void H_c(int vi, double vd) {
//CHECK-NEXT:     int x = vi;
//CHECK-NEXT:     double y = vd;
//CHECK-NEXT: }

printf("H_c2_proto: 0x%lx\n", (unsigned long) H_c2_proto);
//CHECK: H_c2_proto: 0x{{[1-9a-f][0-9a-f]*$}}
H_c2_proto->print(llvm::errs());
//CHECK-NEXT: void H_c(int vi, double vd) {
//CHECK-NEXT:     int x = vi;
//CHECK-NEXT:     double y = vd;
//CHECK-NEXT: }



//
//  Test finding simple namespace template instantiations.
//

const clang::FunctionDecl* H_d1_args = lookup.findFunctionArgs(namespace_N, "H_d<int>", "0");
const clang::FunctionDecl* H_d1_proto = lookup.findFunctionProto(namespace_N, "H_d<int>", "int");

printf("H_d1_args: 0x%lx\n", (unsigned long) H_d1_args);
//CHECK: H_d1_args: 0x{{[1-9a-f][0-9a-f]*$}}
H_d1_args->print(llvm::errs());
//CHECK-NEXT: void H_d(int v) {
//CHECK-NEXT:     int x = v;
//CHECK-NEXT: }

printf("H_d1_proto: 0x%lx\n", (unsigned long) H_d1_proto);
//CHECK: H_d1_proto: 0x{{[1-9a-f][0-9a-f]*$}}
H_d1_proto->print(llvm::errs());
//CHECK-NEXT: void H_d(int v) {
//CHECK-NEXT:     int x = v;
//CHECK-NEXT: }

const clang::FunctionDecl* H_d2_args = lookup.findFunctionArgs(namespace_N, "H_d<double>", "0.0");
const clang::FunctionDecl* H_d2_proto = lookup.findFunctionProto(namespace_N, "H_d<double>", "double");

printf("H_d2_args: 0x%lx\n", (unsigned long) H_d2_args);
//CHECK: H_d2_args: 0x{{[1-9a-f][0-9a-f]*$}}
H_d2_args->print(llvm::errs());
//CHECK-NEXT: void H_d(double v) {
//CHECK-NEXT:     double x = v;
//CHECK-NEXT: }

printf("H_d2_proto: 0x%lx\n", (unsigned long) H_d2_proto);
//CHECK: H_d2_proto: 0x{{[1-9a-f][0-9a-f]*$}}
H_d2_proto->print(llvm::errs());
//CHECK-NEXT: void H_d(double v) {
//CHECK-NEXT:     double x = v;
//CHECK-NEXT: }



//
//  Test finding a member function taking no args.
//

const clang::FunctionDecl* func_A_f_args = lookup.findFunctionArgs(class_A, "A_f", "");
const clang::FunctionDecl* func_A_f_proto = lookup.findFunctionProto(class_A, "A_f", "");

printf("func_A_f_args: 0x%lx\n", (unsigned long) func_A_f_args);
//CHECK: func_A_f_args: 0x{{[1-9a-f][0-9a-f]*$}}
func_A_f_args->print(llvm::errs());
//CHECK-NEXT: void A_f() {
//CHECK-NEXT:     int x = 1;
//CHECK-NEXT: }

printf("func_A_f_proto: 0x%lx\n", (unsigned long) func_A_f_proto);
//CHECK: func_A_f_proto: 0x{{[1-9a-f][0-9a-f]*$}}
func_A_f_proto->print(llvm::errs());
//CHECK-NEXT: void A_f() {
//CHECK-NEXT:     int x = 1;
//CHECK-NEXT: }



//
//  Test finding a member function taking an int arg.
//

const clang::FunctionDecl* func_A_g_args = lookup.findFunctionArgs(class_A, "A_g", "0");
const clang::FunctionDecl* func_A_g_proto = lookup.findFunctionProto(class_A, "A_g", "int");

printf("func_A_g_args: 0x%lx\n", (unsigned long) func_A_g_args);
//CHECK: func_A_g_args: 0x{{[1-9a-f][0-9a-f]*$}}
func_A_g_args->print(llvm::errs());
//CHECK-NEXT: void A_g(int v) {
//CHECK-NEXT:     int x = v;
//CHECK-NEXT: }

printf("func_A_g_proto: 0x%lx\n", (unsigned long) func_A_g_proto);
//CHECK: func_A_g_proto: 0x{{[1-9a-f][0-9a-f]*$}}
func_A_g_proto->print(llvm::errs());
//CHECK-NEXT: void A_g(int v) {
//CHECK-NEXT:     int x = v;
//CHECK-NEXT: }



//
//  Test finding a member function taking an int and a double argument.
//

const clang::FunctionDecl* func_A_h_args = lookup.findFunctionArgs(class_A, "A_h", "0,0.0");
const clang::FunctionDecl* func_A_h_proto = lookup.findFunctionProto(class_A, "A_h", "int,double");

printf("func_A_h_args: 0x%lx\n", (unsigned long) func_A_h_args);
//CHECK: func_A_h_args: 0x{{[1-9a-f][0-9a-f]*$}}
func_A_h_args->print(llvm::errs());
//CHECK-NEXT: void A_h(int vi, double vd) {
//CHECK-NEXT:     int x = vi;
//CHECK-NEXT:     double y = vd;
//CHECK-NEXT: }

printf("func_A_h_proto: 0x%lx\n", (unsigned long) func_A_h_proto);
//CHECK: func_A_h_proto: 0x{{[1-9a-f][0-9a-f]*$}}
func_A_h_proto->print(llvm::errs());
//CHECK-NEXT: void A_h(int vi, double vd) {
//CHECK-NEXT:     int x = vi;
//CHECK-NEXT:     double y = vd;
//CHECK-NEXT: }



//
//  Test finding an overloaded member function.
//

const clang::FunctionDecl* func_A_j1_args = lookup.findFunctionArgs(class_A, "A_j", "0,0");
const clang::FunctionDecl* func_A_j1_proto = lookup.findFunctionProto(class_A, "A_j", "int,int");

printf("func_A_j1_args: 0x%lx\n", (unsigned long) func_A_j1_args);
//CHECK: func_A_j1_args: 0x{{[1-9a-f][0-9a-f]*$}}
func_A_j1_args->print(llvm::errs());
//CHECK-NEXT: void A_j(int vi, int vj) {
//CHECK-NEXT:     int x = vi;
//CHECK-NEXT:     int y = vj;
//CHECK-NEXT: }

printf("func_A_j1_proto: 0x%lx\n", (unsigned long) func_A_j1_proto);
//CHECK: func_A_j1_proto: 0x{{[1-9a-f][0-9a-f]*$}}
func_A_j1_proto->print(llvm::errs());
//CHECK-NEXT: void A_j(int vi, int vj) {
//CHECK-NEXT:     int x = vi;
//CHECK-NEXT:     int y = vj;
//CHECK-NEXT: }

const clang::FunctionDecl* func_A_j2_args = lookup.findFunctionArgs(class_A, "A_j", "0,0.0");
const clang::FunctionDecl* func_A_j2_proto = lookup.findFunctionProto(class_A, "A_j", "int,double");

printf("func_A_j2_args: 0x%lx\n", (unsigned long) func_A_j2_args);
//CHECK: func_A_j2_args: 0x{{[1-9a-f][0-9a-f]*$}}
func_A_j2_args->print(llvm::errs());
//CHECK-NEXT: void A_j(int vi, double vd) {
//CHECK-NEXT:     int x = vi;
//CHECK-NEXT:     double y = vd;
//CHECK-NEXT: }

printf("func_A_j2_proto: 0x%lx\n", (unsigned long) func_A_j2_proto);
//CHECK: func_A_j2_proto: 0x{{[1-9a-f][0-9a-f]*$}}
func_A_j2_proto->print(llvm::errs());
//CHECK-NEXT: void A_j(int vi, double vd) {
//CHECK-NEXT:     int x = vi;
//CHECK-NEXT:     double y = vd;
//CHECK-NEXT: }



//
//  Test finding simple member function template instantiations.
//

const clang::FunctionDecl* func_A_k1_args = lookup.findFunctionArgs(class_A, "A_k<int>", "0");
const clang::FunctionDecl* func_A_k1_proto = lookup.findFunctionProto(class_A, "A_k<int>", "int");

printf("func_A_k1_args: 0x%lx\n", (unsigned long) func_A_k1_args);
//CHECK: func_A_k1_args: 0x{{[1-9a-f][0-9a-f]*$}}
func_A_k1_args->print(llvm::errs());
//CHECK-NEXT: void A_k(int v) {
//CHECK-NEXT:     int x = v;
//CHECK-NEXT: }

printf("func_A_k1_proto: 0x%lx\n", (unsigned long) func_A_k1_proto);
//CHECK: func_A_k1_proto: 0x{{[1-9a-f][0-9a-f]*$}}
func_A_k1_proto->print(llvm::errs());
//CHECK-NEXT: void A_k(int v) {
//CHECK-NEXT:     int x = v;
//CHECK-NEXT: }

const clang::FunctionDecl* func_A_k2_args = lookup.findFunctionArgs(class_A, "A_k<double>", "0.0");
const clang::FunctionDecl* func_A_k2_proto = lookup.findFunctionProto(class_A, "A_k<double>", "double");

printf("func_A_k2_args: 0x%lx\n", (unsigned long) func_A_k2_args);
//CHECK: func_A_k2_args: 0x{{[1-9a-f][0-9a-f]*$}}
func_A_k2_args->print(llvm::errs());
//CHECK-NEXT: void A_k(double v) {
//CHECK-NEXT:     double x = v;
//CHECK-NEXT: }

printf("func_A_k2_proto: 0x%lx\n", (unsigned long) func_A_k2_proto);
//CHECK: func_A_k2_proto: 0x{{[1-9a-f][0-9a-f]*$}}
func_A_k2_proto->print(llvm::errs());
//CHECK-NEXT: void A_k(double v) {
//CHECK-NEXT:     double x = v;
//CHECK-NEXT: }



//
//  Test finding a member function taking a const int reference arg.
//

const clang::FunctionDecl* func_A_m_args = lookup.findFunctionArgs(class_A, "A_m", "0");
const clang::FunctionDecl* func_A_m_proto = lookup.findFunctionProto(class_A, "A_m", "const int&");

printf("func_A_m_args: 0x%lx\n", (unsigned long) func_A_m_args);
//CHECK: func_A_m_args: 0x{{[1-9a-f][0-9a-f]*$}}
func_A_m_args->print(llvm::errs());
//CHECK-NEXT: void A_m(const int &v) {
//CHECK-NEXT:     int y = v;
//CHECK-NEXT: }

printf("func_A_m_proto: 0x%lx\n", (unsigned long) func_A_m_proto);
//CHECK: func_A_m_proto: 0x{{[1-9a-f][0-9a-f]*$}}
func_A_m_proto->print(llvm::errs());
//CHECK-NEXT: void A_m(const int &v) {
//CHECK-NEXT:     int y = v;
//CHECK-NEXT: }

//
//  Test finding a member function taking an obj reference arg.
//
const clang::FunctionDecl* func_A_n_args = lookup.findFunctionArgs(class_A, "A_n", "*(new B())");
const clang::FunctionDecl* func_A_n_proto = lookup.findFunctionProto(class_A, "A_n", "B&");

printf("func_A_n_args: 0x%lx\n", (unsigned long) func_A_n_args);
//CHECK: func_A_n_args: 0x{{[1-9a-f][0-9a-f]*$}}
func_A_n_args->print(llvm::errs());
//CHECK-NEXT: void A_n(B &b) {
//CHECK-NEXT:   b.B_f();
//CHECK-NEXT: }

printf("func_A_n_proto: 0x%lx\n", (unsigned long) func_A_n_proto);
//CHECK: func_A_n_proto: 0x{{[1-9a-f][0-9a-f]*$}}
func_A_n_proto->print(llvm::errs());
//CHECK-NEXT: void A_n(B &b) {
//CHECK-NEXT:   b.B_f();
//CHECK-NEXT: }

//
//  Test finding a member function taking with a default argument.
//
const clang::FunctionDecl* func_A_n2_args = lookup.findFunctionArgs(class_A, "A_n", "\"\"");
const clang::FunctionDecl* func_A_n2_proto = lookup.findFunctionProto(class_A, "A_n", "const char *");

printf("func_A_n2_args: 0x%lx\n", (unsigned long) func_A_n2_args);
//CHECK: func_A_n2_args: 0x{{[1-9a-f][0-9a-f]*$}}
func_A_n2_args->print(llvm::errs());
//CHECK-NEXT: void A_n(const char *msg, int ndim = 0) {
//CHECK-NEXT:    if (ndim) 
//CHECK-NEXT:       ++msg;
//CHECK-NEXT: }

printf("func_A_n2_proto: 0x%lx\n", (unsigned long) func_A_n2_proto);
//CHECK: func_A_n2_proto: 0x{{[1-9a-f][0-9a-f]*$}}
func_A_n2_proto->print(llvm::errs());
//CHECK-NEXT: void A_n(const char *msg, int ndim = 0) {
//CHECK-NEXT:    if (ndim) 
//CHECK-NEXT:       ++msg;
//CHECK-NEXT: }


//
//  Test finding a member function taking no args in a base class.
//

const clang::FunctionDecl* func_B_F_args = lookup.findFunctionArgs(class_A, "B_f", "");
const clang::FunctionDecl* func_B_F_proto = lookup.findFunctionProto(class_A, "B_f", "");

printf("func_B_F_args: 0x%lx\n", (unsigned long) func_B_F_args);
//CHECK: func_B_F_args: 0x{{[1-9a-f][0-9a-f]*$}}
func_B_F_args->print(llvm::errs());
//CHECK-NEXT: void B_f() {
//CHECK-NEXT:     int x = 1;
//CHECK-NEXT: }

printf("func_B_F_proto: 0x%lx\n", (unsigned long) func_B_F_proto);
//CHECK: func_B_F_proto: 0x{{[1-9a-f][0-9a-f]*$}}
func_B_F_proto->print(llvm::errs());
//CHECK-NEXT: void B_f() {
//CHECK-NEXT:     int x = 1;
//CHECK-NEXT: }



//
//  Test finding a member function taking an int arg in a base class.
//

const clang::FunctionDecl* func_B_G_args = lookup.findFunctionArgs(class_A, "B_g", "0");
const clang::FunctionDecl* func_B_G_proto = lookup.findFunctionProto(class_A, "B_g", "int");

printf("func_B_G_args: 0x%lx\n", (unsigned long) func_B_G_args);
//CHECK: func_B_G_args: 0x{{[1-9a-f][0-9a-f]*$}}
func_B_G_args->print(llvm::errs());
//CHECK-NEXT: void B_g(int v) {
//CHECK-NEXT:     int x = v;
//CHECK-NEXT: }

printf("func_B_G_proto: 0x%lx\n", (unsigned long) func_B_G_proto);
//CHECK: func_B_G_proto: 0x{{[1-9a-f][0-9a-f]*$}}
func_B_G_proto->print(llvm::errs());
//CHECK-NEXT: void B_g(int v) {
//CHECK-NEXT:     int x = v;
//CHECK-NEXT: }



//
//  Test finding a member function taking an int and a double argument
//  in a base class.
//

const clang::FunctionDecl* func_B_h_args = lookup.findFunctionArgs(class_A, "B_h", "0,0.0");
const clang::FunctionDecl* func_B_h_proto = lookup.findFunctionProto(class_A, "B_h", "int,double");

printf("func_B_h_args: 0x%lx\n", (unsigned long) func_B_h_args);
//CHECK: func_B_h_args: 0x{{[1-9a-f][0-9a-f]*$}}
func_B_h_args->print(llvm::errs());
//CHECK-NEXT: void B_h(int vi, double vd) {
//CHECK-NEXT:     int x = vi;
//CHECK-NEXT:     double y = vd;
//CHECK-NEXT: }

printf("func_B_h_proto: 0x%lx\n", (unsigned long) func_B_h_proto);
//CHECK: func_B_h_proto: 0x{{[1-9a-f][0-9a-f]*$}}
func_B_h_proto->print(llvm::errs());
//CHECK-NEXT: void B_h(int vi, double vd) {
//CHECK-NEXT:     int x = vi;
//CHECK-NEXT:     double y = vd;
//CHECK-NEXT: }


//
//  Test finding a member function taking an int and a double argument
//  in a base class using the preparse types.
//
llvm::SmallVector<clang::QualType, 4> types;
types.push_back(lookup.findType("int"));
types.push_back(lookup.findType("float"));
const clang::FunctionDecl* func_B_h_proto_type = lookup.findFunctionProto(class_A, "B_h", types);
types.pop_back();
types.push_back(lookup.findType("double"));
const clang::FunctionDecl* func_B_h_match_proto_type = lookup.matchFunctionProto(class_A, "B_h", types, false);

printf("func_B_h_proto_type: 0x%lx\n", (unsigned long) func_B_h_proto_type);
//CHECK: func_B_h_proto_type: 0x{{[1-9a-f][0-9a-f]*$}}
func_B_h_proto_type->print(llvm::errs());
//CHECK-NEXT: void B_h(int vi, double vd) {
//CHECK-NEXT:     int x = vi;
//CHECK-NEXT:     double y = vd;
//CHECK-NEXT: }

printf("func_B_h_match_proto_type: 0x%lx\n", (unsigned long) func_B_h_match_proto_type);
//CHECK: func_B_h_match_proto_type: 0x{{[1-9a-f][0-9a-f]*$}}
func_B_h_match_proto_type->print(llvm::errs());
//CHECK-NEXT: void B_h(int vi, double vd) {
//CHECK-NEXT:     int x = vi;
//CHECK-NEXT:     double y = vd;
//CHECK-NEXT: }


//
//  Test finding an overloaded member function in a base class.
//

const clang::FunctionDecl* func_B_j1_args = lookup.findFunctionArgs(class_A, "B_j", "0,0");
const clang::FunctionDecl* func_B_j1_proto = lookup.findFunctionProto(class_A, "B_j", "int,int");

printf("func_B_j1_args: 0x%lx\n", (unsigned long) func_B_j1_args);
//CHECK: func_B_j1_args: 0x{{[1-9a-f][0-9a-f]*$}}
func_B_j1_args->print(llvm::errs());
//CHECK-NEXT: void B_j(int vi, int vj) {
//CHECK-NEXT:     int x = vi;
//CHECK-NEXT:     int y = vj;
//CHECK-NEXT: }

printf("func_B_j1_proto: 0x%lx\n", (unsigned long) func_B_j1_proto);
//CHECK: func_B_j1_proto: 0x{{[1-9a-f][0-9a-f]*$}}
func_B_j1_proto->print(llvm::errs());
//CHECK-NEXT: void B_j(int vi, int vj) {
//CHECK-NEXT:     int x = vi;
//CHECK-NEXT:     int y = vj;
//CHECK-NEXT: }

const clang::FunctionDecl* func_B_j2_args = lookup.findFunctionArgs(class_A, "B_j", "0,0.0");
const clang::FunctionDecl* func_B_j2_proto = lookup.findFunctionProto(class_A, "B_j", "int,double");

printf("func_B_j2_args: 0x%lx\n", (unsigned long) func_B_j2_args);
//CHECK: func_B_j2_args: 0x{{[1-9a-f][0-9a-f]*$}}
func_B_j2_args->print(llvm::errs());
//CHECK-NEXT: void B_j(int vi, double vd) {
//CHECK-NEXT:     int x = vi;
//CHECK-NEXT:     double y = vd;
//CHECK-NEXT: }

printf("func_B_j2_proto: 0x%lx\n", (unsigned long) func_B_j2_proto);
//CHECK: func_B_j2_proto: 0x{{[1-9a-f][0-9a-f]*$}}
func_B_j2_proto->print(llvm::errs());
//CHECK-NEXT: void B_j(int vi, double vd) {
//CHECK-NEXT:     int x = vi;
//CHECK-NEXT:     double y = vd;
//CHECK-NEXT: }



//
//  Test finding simple member function template instantiations in a base class.
//

const clang::FunctionDecl* func_B_k1_args = lookup.findFunctionArgs(class_A, "B_k<int>", "0");
const clang::FunctionDecl* func_B_k1_proto = lookup.findFunctionProto(class_A, "B_k<int>", "int");

printf("func_B_k1_args: 0x%lx\n", (unsigned long) func_B_k1_args);
//CHECK: func_B_k1_args: 0x{{[1-9a-f][0-9a-f]*$}}
func_B_k1_args->print(llvm::errs());
//CHECK-NEXT: void B_k(int v) {
//CHECK-NEXT:     int x = v;
//CHECK-NEXT: }

printf("func_B_k1_proto: 0x%lx\n", (unsigned long) func_B_k1_proto);
//CHECK: func_B_k1_proto: 0x{{[1-9a-f][0-9a-f]*$}}
func_B_k1_proto->print(llvm::errs());
//CHECK-NEXT: void B_k(int v) {
//CHECK-NEXT:     int x = v;
//CHECK-NEXT: }

const clang::FunctionDecl* func_B_k2_args = lookup.findFunctionArgs(class_A, "B_k<double>", "0.0");
const clang::FunctionDecl* func_B_k2_proto = lookup.findFunctionProto(class_A, "B_k<double>", "double");

printf("func_B_k2_args: 0x%lx\n", (unsigned long) func_B_k2_args);
//CHECK: func_B_k2_args: 0x{{[1-9a-f][0-9a-f]*$}}
func_B_k2_args->print(llvm::errs());
//CHECK-NEXT: void B_k(double v) {
//CHECK-NEXT:     double x = v;
//CHECK-NEXT: }

printf("func_B_k2_proto: 0x%lx\n", (unsigned long) func_B_k2_proto);
//CHECK: func_B_k2_proto: 0x{{[1-9a-f][0-9a-f]*$}}
func_B_k2_proto->print(llvm::errs());
//CHECK-NEXT: void B_k(double v) {
//CHECK-NEXT:     double x = v;
//CHECK-NEXT: }

//
//  Test finding a member function taking a const int reference arg in a base class.
//

const clang::FunctionDecl* func_B_m_args = lookup.findFunctionArgs(class_A, "B_m", "0");
const clang::FunctionDecl* func_B_m_proto = lookup.findFunctionProto(class_A, "B_m", "const int&");

printf("func_B_m_args: 0x%lx\n", (unsigned long) func_B_m_args);
//CHECK: func_B_m_args: 0x{{[1-9a-f][0-9a-f]*$}}
func_B_m_args->print(llvm::errs());
//CHECK-NEXT: void B_m(const int &v) {
//CHECK-NEXT:     int y = v;
//CHECK-NEXT: }

printf("func_B_m_proto: 0x%lx\n", (unsigned long) func_B_m_proto);
//CHECK: func_B_m_proto: 0x{{[1-9a-f][0-9a-f]*$}}
func_B_m_proto->print(llvm::errs());
//CHECK-NEXT: void B_m(const int &v) {
//CHECK-NEXT:     int y = v;
//CHECK-NEXT: }


//
//  Test finding a member function that const or not
//

const clang::FunctionDecl* func_B_n_args = lookup.findFunctionArgs(class_A, "B_n", "", false);
const clang::FunctionDecl* func_B_n_proto = lookup.findFunctionProto(class_A, "B_n", "", false);

printf("func_B_n_args: 0x%lx\n", (unsigned long) func_B_n_args);
//CHECK: func_B_n_args: 0x{{[1-9a-f][0-9a-f]*$}}
func_B_n_args->print(llvm::errs());
//CHECK-NEXT: long &B_n() {
//CHECK-NEXT:     return this->m_B_i;
//CHECK-NEXT: }

printf("func_B_n_proto: 0x%lx\n", (unsigned long) func_B_n_proto);
//CHECK: func_B_n_proto: 0x{{[1-9a-f][0-9a-f]*$}}
func_B_n_proto->print(llvm::errs());
//CHECK-NEXT: long &B_n() {
//CHECK-NEXT:     return this->m_B_i;
//CHECK-NEXT: }

const clang::FunctionDecl* func_const_B_n_args = lookup.findFunctionArgs(class_A, "B_n", "", true);
const clang::FunctionDecl* func_const_B_n_proto = lookup.findFunctionProto(class_A, "B_n", "", true);
printf("func_const_B_n_args: 0x%lx\n", (unsigned long) func_const_B_n_args);
//CHECK: func_const_B_n_args: 0x{{[1-9a-f][0-9a-f]*$}}
func_const_B_n_args->print(llvm::errs());
//CHECK-NEXT: const long &B_n() const {
//CHECK-NEXT:     return this->m_B_i;
//CHECK-NEXT: }

printf("func_const_B_n_proto: 0x%lx\n", (unsigned long) func_const_B_n_proto);
//CHECK: func_const_B_n_proto: 0x{{[1-9a-f][0-9a-f]*$}}
func_const_B_n_proto->print(llvm::errs());
//CHECK-NEXT: const long &B_n() const {
//CHECK-NEXT:     return this->m_B_i;
//CHECK-NEXT: }

const clang::FunctionDecl* func_const_B_m_proto = lookup.findFunctionProto(class_A, "B_m", "const int&", true);
const clang::FunctionDecl* func_const_B_o_proto = lookup.findFunctionProto(class_A, "B_o", "", true);
printf("func_const_B_m_proto: 0x%lx\n", (unsigned long) func_const_B_m_proto);
//CHECK: func_const_B_m_proto: 0x0

printf("func_const_B_o_proto: 0x%lx\n", (unsigned long) func_const_B_o_proto);
//CHECK: func_const_B_o_proto: 0x{{[1-9a-f][0-9a-f]*$}}
func_const_B_o_proto->print(llvm::errs());
//CHECK-NEXT: const long &B_o() const {
//CHECK-NEXT:     return this->m_B_i; 
//CHECK-NEXT: }

// Test exact matches
const clang::FunctionDecl* func_const_B_p_proto = lookup.findFunctionProto(class_A, "B_p", "double", true);
printf("func_const_B_p_proto 1: 0x%lx\n", (unsigned long) func_const_B_p_proto);
//CHECK: func_const_B_p_proto 1: 0x{{[1-9a-f][0-9a-f]*$}}
func_const_B_p_proto->print(llvm::errs());
//CHECK-NEXT: long B_p(float) const {
//CHECK-NEXT:     return 0; 
//CHECK-NEXT: }

func_const_B_p_proto = lookup.matchFunctionProto(class_A, "B_p", "double", true);
printf("func_const_B_p_proto 2: 0x%lx\n", (unsigned long) func_const_B_p_proto);
//CHECK: func_const_B_p_proto 2: 0x0

func_const_B_p_proto = lookup.matchFunctionProto(class_A, "B_p", "float", true);
printf("func_const_B_p_proto 3: 0x%lx\n", (unsigned long) func_const_B_p_proto);
//CHECK: func_const_B_p_proto 3: 0x{{[1-9a-f][0-9a-f]*$}}
func_const_B_p_proto->print(llvm::errs());
//CHECK-NEXT: long B_p(float) const {
//CHECK-NEXT:     return 0; 
//CHECK-NEXT: }

func_const_B_p_proto = lookup.matchFunctionProto(class_A, "B_p", "float", false);
printf("func_const_B_p_proto 4: 0x%lx\n", (unsigned long) func_const_B_p_proto);
//CHECK: func_const_B_p_proto 4: 0x0

func_const_B_p_proto = lookup.matchFunctionProto(class_A, "B_p", "int", false);
printf("func_const_B_p_proto 5: 0x%lx\n", (unsigned long) func_const_B_p_proto);
//CHECK: func_const_B_p_proto 5: 0x{{[1-9a-f][0-9a-f]*$}}
func_const_B_p_proto->print(llvm::errs());
//CHECK-NEXT: int B_p(int) {
//CHECK-NEXT:     return 0; 
//CHECK-NEXT: }

func_const_B_p_proto = lookup.matchFunctionProto(class_A, "B_p", "int", true);
printf("func_const_B_p_proto 6: 0x%lx\n", (unsigned long) func_const_B_p_proto);
//CHECK: func_const_B_p_proto 6: 0x0

func_const_B_p_proto = lookup.matchFunctionProto(class_A, "B_p", "short", false);
printf("func_const_B_p_proto 6: 0x%lx\n", (unsigned long) func_const_B_p_proto);
//CHECK: func_const_B_p_proto 6: 0x0

func_const_B_p_proto = lookup.matchFunctionProto(class_A, "B_p", "long", false);
printf("func_const_B_p_proto 6: 0x%lx\n", (unsigned long) func_const_B_p_proto);
//CHECK: func_const_B_p_proto 6: 0x0

//
//  Test finding constructors.
//
//
const clang::FunctionDecl* func_B_ctr1_args = lookup.findFunctionArgs(class_B, "B", "");
const clang::FunctionDecl* func_B_ctr1_proto = lookup.findFunctionProto(class_B, "B", "");

printf("func_B_ctr1_args: 0x%lx\n", (unsigned long) func_B_ctr1_args);
//CHECK: func_B_ctr1_args: 0x{{[1-9a-f][0-9a-f]*$}}

func_B_ctr1_args->print(llvm::errs());
//CHECK-NEXT: B() : m_B_i(0), m_B_d(0.), m_B_ip(0) {
//CHECK-NEXT: }

printf("func_B_ctr1_proto: 0x%lx\n", (unsigned long) func_B_ctr1_proto);
//CHECK: func_B_ctr1_proto: 0x{{[1-9a-f][0-9a-f]*$}}

func_B_ctr1_proto->print(llvm::errs());
//CHECK-NEXT: B() : m_B_i(0), m_B_d(0.), m_B_ip(0) {
//CHECK-NEXT: }

const clang::FunctionDecl* func_B_ctr2_args = lookup.findFunctionArgs(class_B, "B", "0,0.0");
const clang::FunctionDecl* func_B_ctr2_proto = lookup.findFunctionProto(class_B, "B", "int,double");

printf("func_B_ctr2_args: 0x%lx\n", (unsigned long) func_B_ctr2_args);
//CHECK: func_B_ctr2_args: 0x{{[1-9a-f][0-9a-f]*$}}

func_B_ctr2_args->print(llvm::errs());
//CHECK-NEXT: B(int vi, double vd) : m_B_i(vi), m_B_d(vd), m_B_ip(0) {
//CHECK-NEXT: }

printf("func_B_ctr2_proto: 0x%lx\n", (unsigned long) func_B_ctr2_proto);
//CHECK: func_B_ctr2_proto: 0x{{[1-9a-f][0-9a-f]*$}}

func_B_ctr2_proto->print(llvm::errs());
//CHECK-NEXT: B(int vi, double vd) : m_B_i(vi), m_B_d(vd), m_B_ip(0) {
//CHECK-NEXT: }

B* force_B_char_ctr = new B('a');
const clang::FunctionDecl* func_B_ctr3_args = lookup.findFunctionArgs(class_B, "B", "'a'");
const clang::FunctionDecl* func_B_ctr3_proto = lookup.findFunctionProto(class_B, "B", "char");

dumpDecl("func_B_ctr3_args", func_B_ctr3_args);
//CHECK: func_B_ctr3_args: 0x{{[1-9a-f][0-9a-f]*$}}
//CHECK-NEXT: func_B_ctr3_args name: B::B<char>
//CHECK-NEXT:  {
//CHECK-NEXT:     this->m_B_i = (char)v; 
//CHECK-NEXT: }

dumpDecl("func_B_ctr3_proto", func_B_ctr3_proto);
//CHECK: func_B_ctr3_proto: 0x{{[1-9a-f][0-9a-f]*$}}
//CHECK-NEXT: func_B_ctr3_proto name: B::B<char>
//CHECK-NEXT:  {
//CHECK-NEXT:     this->m_B_i = (char)v; 
//CHECK-NEXT: }

B* force_B_char_ptr_ctr = new B((char*)0);
const clang::FunctionDecl* func_B_ctr4_args = lookup.findFunctionArgs(class_B, "B", "(char*)0");
const clang::FunctionDecl* func_B_ctr4_proto = lookup.findFunctionProto(class_B, "B", "char*");

dumpDecl("func_B_ctr4_args", func_B_ctr4_args);
//CHECK: func_B_ctr4_args: 0x{{[1-9a-f][0-9a-f]*$}}
//CHECK-NEXT: func_B_ctr4_args name: B::B<char>
//CHECK-NEXT: B(char *v) : m_B_i(0), m_B_d(0.), m_B_ip(0) {
//CHECK-NEXT:     this->m_B_i = (long)(char *)v;
//CHECK-NEXT:     this->m_B_d = 1.;
//CHECK-NEXT: }

dumpDecl("func_B_ctr4_proto", func_B_ctr4_proto);
//CHECK: func_B_ctr4_proto: 0x{{[1-9a-f][0-9a-f]*$}}
//CHECK-NEXT: func_B_ctr4_proto name: B::B<char>
//CHECK-NEXT: B(char *v) : m_B_i(0), m_B_d(0.), m_B_ip(0) {
//CHECK-NEXT:     this->m_B_i = (long)(char *)v;
//CHECK-NEXT:     this->m_B_d = 1.;
//CHECK-NEXT: }

printf("func_B_ctr4_proto has body: %d\n", func_B_ctr4_proto->hasBody());
//CHECK: func_B_ctr4_proto has body: 1

//
//  Test finding destructors.
//

const clang::FunctionDecl* func_B_dtr_args = lookup.findFunctionArgs(class_B, "~B", "");
const clang::FunctionDecl* func_B_dtr_proto = lookup.findFunctionProto(class_B, "~B", "");

dumpDecl("func_B_dtr_args", func_B_dtr_args);
//CHECK: func_B_dtr_args: 0x{{[1-9a-f][0-9a-f]*$}}
//CHECK-NEXT: func_B_dtr_args name: B::~B
//CHECK-NEXT: virtual void ~B() {
//CHECK-NEXT:     delete this->m_B_ip;
//CHECK-NEXT:     this->m_B_ip = 0;
//CHECK-NEXT: }

dumpDecl("func_B_dtr_proto", func_B_dtr_proto);
//CHECK: func_B_dtr_proto: 0x{{[1-9a-f][0-9a-f]*$}}
//CHECK-NEXT: func_B_dtr_proto name: B::~B
//CHECK-NEXT: virtual void ~B() {
//CHECK-NEXT:     delete this->m_B_ip;
//CHECK-NEXT:     this->m_B_ip = 0;
//CHECK-NEXT: }



//
//  Test finding free store operator new.
//

const clang::FunctionDecl* func_B_new_args = lookup.findFunctionArgs(class_B, "operator new", "sizeof(B)");
const clang::FunctionDecl* func_B_new_proto = lookup.findFunctionProto(class_B, "operator new", "std::size_t");

dumpDecl("func_B_new_args", func_B_new_args);
//CHECK: func_B_new_args: 0x{{[1-9a-f][0-9a-f]*$}}
//CHECK-NEXT: func_B_new_args name: B::operator new
//CHECK-NEXT: void *operator new(std::size_t sz) {
//CHECK-NEXT:     return ::operator new(sz);
//CHECK-NEXT: }

dumpDecl("func_B_new_proto", func_B_new_proto);
//CHECK: func_B_new_proto: 0x{{[1-9a-f][0-9a-f]*$}}
//CHECK-NEXT: func_B_new_proto name: B::operator new
//CHECK-NEXT: void *operator new(std::size_t sz) {
//CHECK-NEXT:     return ::operator new(sz);
//CHECK-NEXT: }

const clang::FunctionDecl* func_B_new_plcmt_args = lookup.findFunctionArgs(class_B, "operator new", "sizeof(B),((B*)&b_arena[0])+2");
const clang::FunctionDecl* func_B_new_plcmt_proto = lookup.findFunctionProto(class_B, "operator new", "std::size_t,void*");

dumpDecl("func_B_new_plcmt_args", func_B_new_plcmt_args);
//CHECK: func_B_new_plcmt_args: 0x{{[1-9a-f][0-9a-f]*$}}
//CHECK-NEXT: func_B_new_plcmt_args name: B::operator new
//CHECK-NEXT: void *operator new(std::size_t sz, void *arena) {
//CHECK-NEXT:     return arena;
//CHECK-NEXT: }

dumpDecl("func_B_new_plcmt_proto", func_B_new_plcmt_proto);
//CHECK: func_B_new_plcmt_proto: 0x{{[1-9a-f][0-9a-f]*$}}
//CHECK-NEXT: func_B_new_plcmt_proto name: B::operator new
//CHECK-NEXT: void *operator new(std::size_t sz, void *arena) {
//CHECK-NEXT:     return arena;
//CHECK-NEXT: }

const clang::FunctionDecl* func_B_new_ary_args = lookup.findFunctionArgs(class_B, "operator new[]", "sizeof(B)*3");
const clang::FunctionDecl* func_B_new_ary_proto = lookup.findFunctionProto(class_B, "operator new[]", "std::size_t");

dumpDecl("func_B_new_ary_args", func_B_new_ary_args);
//CHECK: func_B_new_ary_args: 0x{{[1-9a-f][0-9a-f]*$}}
//CHECK-NEXT: func_B_new_ary_args name: B::operator new[]
//CHECK-NEXT: void *operator new[](std::size_t sz) {
//CHECK-NEXT:     return ::operator new[](sz);
//CHECK-NEXT: }

dumpDecl("func_B_new_ary_proto", func_B_new_ary_proto);
//CHECK: func_B_new_ary_proto: 0x{{[1-9a-f][0-9a-f]*$}}
//CHECK-NEXT: func_B_new_ary_proto name: B::operator new[]
//CHECK-NEXT: void *operator new[](std::size_t sz) {
//CHECK-NEXT:     return ::operator new[](sz);
//CHECK-NEXT: }

const clang::FunctionDecl* func_B_new_ary_plcmt_args = lookup.findFunctionArgs(class_B, "operator new[]", "sizeof(B)*3,&b_ary_arena[0]");
const clang::FunctionDecl* func_B_new_ary_plcmt_proto = lookup.findFunctionProto(class_B, "operator new[]", "std::size_t,void*");

dumpDecl("func_B_new_ary_plcmt_args", func_B_new_ary_plcmt_args);
//CHECK: func_B_new_ary_plcmt_args: 0x{{[1-9a-f][0-9a-f]*$}}
//CHECK-NEXT: func_B_new_ary_plcmt_args name: B::operator new[]
//CHECK-NEXT: void *operator new[](std::size_t sz, void *arena) {
//CHECK-NEXT:     return arena;
//CHECK-NEXT: }

dumpDecl("func_B_new_ary_plcmt_proto", func_B_new_ary_plcmt_proto);
//CHECK: func_B_new_ary_plcmt_proto: 0x{{[1-9a-f][0-9a-f]*$}}
//CHECK-NEXT: func_B_new_ary_plcmt_proto name: B::operator new[]
//CHECK-NEXT: void *operator new[](std::size_t sz, void *arena) {
//CHECK-NEXT:     return arena;
//CHECK-NEXT: }

//
//  Test finding free store operator delete.
//

const clang::FunctionDecl* func_B_del_args = lookup.findFunctionArgs(class_B, "operator delete", "b_ptr");
const clang::FunctionDecl* func_B_del_proto = lookup.findFunctionProto(class_B, "operator delete", "void*");

dumpDecl("func_B_del_args", func_B_del_args);
//CHECK: func_B_del_args: 0x{{[1-9a-f][0-9a-f]*$}}
//CHECK-NEXT: func_B_del_args name: B::operator delete
//CHECK-NEXT: void operator delete(void *vp) {
//CHECK-NEXT:     ::operator delete(vp);
//CHECK-NEXT: }

dumpDecl("func_B_del_proto", func_B_del_proto);
//CHECK: func_B_del_proto: 0x{{[1-9a-f][0-9a-f]*$}}
//CHECK-NEXT: func_B_del_proto name: B::operator delete
//CHECK-NEXT: void operator delete(void *vp) {
//CHECK-NEXT:     ::operator delete(vp);
//CHECK-NEXT: }

const clang::FunctionDecl* func_B_del_plcmt_args = lookup.findFunctionArgs(class_B, "operator delete", "((B*)&b_arena[0])+2,&b_arena[0]");
const clang::FunctionDecl* func_B_del_plcmt_proto = lookup.findFunctionProto(class_B, "operator delete", "void*,void*");

dumpDecl("func_B_del_plcmt_args", func_B_del_plcmt_args);
//CHECK: func_B_del_plcmt_args: 0x{{[1-9a-f][0-9a-f]*$}}
//CHECK-NEXT: func_B_del_plcmt_args name: B::operator delete
//CHECK-NEXT: void operator delete(void *vp, void *arena) {
//CHECK-NEXT: }

dumpDecl("func_B_del_plcmt_proto", func_B_del_plcmt_proto);
//CHECK: func_B_del_plcmt_proto: 0x{{[1-9a-f][0-9a-f]*$}}
//CHECK-NEXT: func_B_del_plcmt_proto name: B::operator delete
//CHECK-NEXT: void operator delete(void *vp, void *arena) {
//CHECK-NEXT: }

const clang::FunctionDecl* func_B_del_ary_args = lookup.findFunctionArgs(class_B, "operator delete[]", "b_ary");
const clang::FunctionDecl* func_B_del_ary_proto = lookup.findFunctionProto(class_B, "operator delete[]", "void*");

dumpDecl("func_B_del_ary_args", func_B_del_ary_args);
//CHECK: func_B_del_ary_args: 0x{{[1-9a-f][0-9a-f]*$}}
//CHECK-NEXT: func_B_del_ary_args name: B::operator delete[]
//CHECK-NEXT: void operator delete[](void *vp) {
//CHECK-NEXT:     ::operator delete[](vp);
//CHECK-NEXT: }

dumpDecl("func_B_del_ary_proto", func_B_del_ary_proto);
//CHECK: func_B_del_ary_proto: 0x{{[1-9a-f][0-9a-f]*$}}
//CHECK-NEXT: func_B_del_ary_proto name: B::operator delete[]
//CHECK-NEXT: void operator delete[](void *vp) {
//CHECK-NEXT:     ::operator delete[](vp);
//CHECK-NEXT: }

const clang::FunctionDecl* func_B_del_ary_plcmt_args = lookup.findFunctionArgs(class_B, "operator delete[]", "(B*)b_arena[3],&b_arena[0]");
const clang::FunctionDecl* func_B_del_ary_plcmt_proto = lookup.findFunctionProto(class_B, "operator delete[]", "void*,void*");

dumpDecl("func_B_del_ary_plcmt_args", func_B_del_ary_plcmt_args);
//CHECK: func_B_del_ary_plcmt_args: 0x{{[1-9a-f][0-9a-f]*$}}
//CHECK-NEXT: func_B_del_ary_plcmt_args name: B::operator delete[]
//CHECK-NEXT: void operator delete[](void *vp, void *arena) {
//CHECK-NEXT: }

dumpDecl("func_B_del_ary_plcmt_proto", func_B_del_ary_plcmt_proto);
//CHECK: func_B_del_ary_plcmt_proto: 0x{{[1-9a-f][0-9a-f]*$}}
//CHECK-NEXT: func_B_del_ary_plcmt_proto name: B::operator delete[]
//CHECK-NEXT: void operator delete[](void *vp, void *arena) {
//CHECK-NEXT: }



//
//  Test finding unary member operator.
//

const clang::FunctionDecl* func_B_star_args = lookup.findFunctionArgs(class_B, "operator*", "");
const clang::FunctionDecl* func_B_star_proto = lookup.findFunctionProto(class_B, "operator*", "");

dumpDecl("func_B_star_args", func_B_star_args);
//CHECK: func_B_star_args: 0x{{[1-9a-f][0-9a-f]*$}}
//CHECK-NEXT: func_B_star_args name: B::operator*
//CHECK-NEXT: B &operator*() {
//CHECK-NEXT:     return *this;
//CHECK-NEXT: }

dumpDecl("func_B_star_proto", func_B_star_proto);
//CHECK: func_B_star_proto: 0x{{[1-9a-f][0-9a-f]*$}}
//CHECK-NEXT: func_B_star_proto name: B::operator*
//CHECK-NEXT: B &operator*() {
//CHECK-NEXT:     return *this;
//CHECK-NEXT: }



//
//  Test finding binary member operator.
//

const clang::FunctionDecl* func_B_plus_args = lookup.findFunctionArgs(class_B, "operator+", "b_obj");
const clang::FunctionDecl* func_B_plus_proto = lookup.findFunctionProto(class_B, "operator+", "B");

dumpDecl("func_B_plus_args", func_B_plus_args);
//CHECK: func_B_plus_args: 0x{{[1-9a-f][0-9a-f]*$}}
//CHECK-NEXT: func_B_plus_args name: B::operator+
//CHECK-NEXT: B operator+(B b) {
//CHECK-NEXT:     return b;
//CHECK-NEXT: }

dumpDecl("func_B_plus_proto", func_B_plus_proto);
//CHECK: func_B_plus_proto: 0x{{[1-9a-f][0-9a-f]*$}}
//CHECK-NEXT: func_B_plus_proto name: B::operator+
//CHECK-NEXT: B operator+(B b) {
//CHECK-NEXT:     return b;
//CHECK-NEXT: }


//
//  One final check to make sure we are at the right line in the output.
//

"abc"
//CHECK: (const char [4]) "abc"

//
//  Cleanup.
//

delete[] b_ary;
.q
