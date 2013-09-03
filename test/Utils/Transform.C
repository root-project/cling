// RUN: cat %s | %cling | FileCheck %s

// The test verifies the expected behavior in cling::utils::Transform class,
// which is supposed to provide different transformation of AST nodes and types.

#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/LookupHelper.h"
#include "cling/Utils/AST.h"
#include "clang/AST/Type.h"
#include "clang/AST/ASTContext.h"
#include "llvm/ADT/SmallSet.h"
#include "clang/Sema/Sema.h"

.rawInput 1

#include <vector>
#include <iostream>

typedef double Double32_t;
typedef int Int_t;
typedef long Long_t;
typedef Int_t* IntPtr_t;
typedef Int_t& IntRef_t;

template <typename T> class A {};
template <typename T, typename U> class B {};
template <typename T, typename U> class C {};
typedef C<A<B<Double32_t, Int_t> >, Double32_t > CTD;
typedef C<A<B<const Double32_t, const Int_t> >, Double32_t > CTDConst;

template <typename key, typename value, typename compare_operation = std::less<key>, typename alloc = std::allocator<std::pair<const key, value> > > class cmap { key fKey; const value fValue; alloc fAlloc; };
   // : public std::map<key, value, compare_operation, alloc> {

template <typename key, typename value = const key> class mypair { public: key fKey; value fValue; };

#include <string>

namespace Details {
  class Impl {};
}

// To insure instantiation.
// typedef std::pair<Details::Impl,std::vector<Details::Impl> > details_pairs;

namespace NS {
  template <typename T, int size = 0> class ArrayType {};
  template <typename T> class Array {};
  
  template <typename T> class Container {
  public:
    class Content {};
    typedef T Value_t;
    typedef Content Content_t;
    typedef ::Details::Impl Impl_t;
  };
  
  template <typename T> class TDataPoint {};
  typedef TDataPoint<float> TDataPointF;
  typedef TDataPoint<Double32_t> TDataPointD32;
}

// Anonymous namespace
namespace {
  class InsideAnonymous {
  };
}

using namespace std;

class Embedded_objects {
public:
  enum Eenum {
    kEnumConst=16
  };
  
  class EmbeddedClasses;
  typedef EmbeddedClasses EmbeddedTypedef; 
  
  class EmbeddedClasses {
  public:
    class Embedded1 {};
    class Embedded2 {};
    class Embedded3 {};
    class Embedded4 {};
    class Embedded5 {};
    class Embedded6 {};

  };
  EmbeddedClasses m_embedded;
  EmbeddedClasses::Embedded1 m_emb1;
  EmbeddedClasses::Embedded2 m_emb2;
  EmbeddedClasses::Embedded3 m_emb3;
  
  EmbeddedTypedef::Embedded4 m_emb4;
  Embedded_objects::EmbeddedClasses::Embedded5 m_emb5;
  Embedded_objects::EmbeddedTypedef::Embedded6 m_emb6;
  typedef std::vector<int> vecint;
  vecint::iterator m_iter;
  const Eenum m_enum;
  typedef vector<int> vecint2;
  vecint2::iterator m_iter2;
};

namespace NS1 {
  namespace NS2 {
    namespace NS3 {
      class Point {};
      class Inner3 {
      public:
        Point p1;
        NS3::Point p2;
        ::NS1::NS2::NS3::Point p3;
      };
    }
  }
}

.rawInput 0

const cling::LookupHelper& lookup = gCling->getLookupHelper();
const clang::ASTContext& Ctx = gCling->getSema().getASTContext();
cling::utils::Transform::Config transConfig;

transConfig.m_toSkip.insert(lookup.findType("Double32_t").getTypePtr());
using namespace std;
transConfig.m_toSkip.insert(lookup.findType("string").getTypePtr());
transConfig.m_toSkip.insert(lookup.findType("std::string").getTypePtr());

const clang::Type* t = 0;
const clang::TypedefType *td = 0;
clang::QualType QT;
using namespace cling::utils;

// Test the behavior on a simple class
lookup.findScope("Details::Impl", &t);
QT = clang::QualType(t, 0);
//QT.getAsString().c_str()
Transform::GetPartiallyDesugaredType(Ctx, QT, transConfig).getAsString().c_str()
// CHECK: (const char *) "Details::Impl"

// Test the behavior for a class inside an anonymous namespace
lookup.findScope("InsideAnonymous", &t);
QT = clang::QualType(t, 0);
//QT.getAsString().c_str()c
Transform::GetPartiallyDesugaredType(Ctx, QT, transConfig).getAsString().c_str()
// CHECK: (const char *) "class <anonymous>::InsideAnonymous"

// The above result is not quite want we want, so the client must using 
// the following:
// The scope suppression is required for getting rid of the anonymous part of the name of a class defined in an anonymous namespace.
// This gives us more control vs not using the clang::ElaboratedType and relying on the Policy.SuppressUnwrittenScope which would
// strip both the anonymous and the inline namespace names (and we probably do not want the later to be suppressed).
clang::PrintingPolicy Policy(Ctx.getPrintingPolicy());
Policy.SuppressTagKeyword = true; // Never get the class or struct keyword
Policy.SuppressScope = true;      // Force the scope to be coming from a clang::ElaboratedType.
std::string name;
Transform::GetPartiallyDesugaredType(Ctx, QT, transConfig).getAsStringInternal(name,Policy);
name.c_str()
// CHECK: (const char *) "InsideAnonymous"

// Test desugaring pointers types:
QT = lookup.findType("Int_t*");
Transform::GetPartiallyDesugaredType(Ctx, QT, transConfig).getAsString().c_str()
// CHECK:(const char *) "int *"

QT = lookup.findType("const IntPtr_t*");
Transform::GetPartiallyDesugaredType(Ctx, QT, transConfig).getAsString().c_str()
// CHECK:(const char *) "int *const *"


// Test desugaring reference (both r- or l- value) types:
QT = lookup.findType("const IntPtr_t&");
Transform::GetPartiallyDesugaredType(Ctx, QT, transConfig).getAsString().c_str()
// CHECK:(const char *) "int *const &"

//TODO: QT = lookup.findType("IntPtr_t[32]");

// To do: findType does not return the const below:
// Test desugaring reference (both r- or l- value) types:
//QT = lookup.findType("const IntRef_t");
//Transform::GetPartiallyDesugaredType(Ctx, QT, transConfig).getAsString().c_str()
// should print:(const char *) "int &const"

// Test desugaring reference (both r- or l- value) types:
QT = lookup.findType("IntRef_t");
Transform::GetPartiallyDesugaredType(Ctx, QT, transConfig).getAsString().c_str()
// CHECK:(const char *) "int &"


//Desugar template parameters:
lookup.findScope("A<B<Double32_t, Int_t*> >", &t);
QT = clang::QualType(t, 0);
Transform::GetPartiallyDesugaredType(Ctx, QT, transConfig).getAsString().c_str()
// CHECK:(const char *) "A<B<Double32_t, int *> >"

lookup.findScope("A<B<Double32_t, std::size_t*> >", &t);
QT = clang::QualType(t, 0);
Transform::GetPartiallyDesugaredType(Ctx, QT, transConfig).getAsString().c_str()
// CHECK:(const char *) "A<B<Double32_t, unsigned {{long|int}} *> >"

lookup.findScope("CTD", &t);
QT = clang::QualType(t, 0);
Transform::GetPartiallyDesugaredType(Ctx, QT, transConfig).getAsString().c_str()
// CHECK: (const char *) "C<A<B<Double32_t, int> >, Double32_t>"

lookup.findScope("CTDConst", &t);
QT = clang::QualType(t, 0);
Transform::GetPartiallyDesugaredType(Ctx, QT, transConfig).getAsString().c_str()
// CHECK: (const char *) "C<A<B<const Double32_t, const int> >, Double32_t>"

lookup.findScope("std::pair<const std::string,int>", &t);
QT = clang::QualType(t, 0);
Transform::GetPartiallyDesugaredType(Ctx, QT, transConfig).getAsString().c_str()
// CHECK: (const char *) "std::pair<const std::string, int>"

lookup.findScope("NS::Array<NS::ArrayType<double> >", &t);
QT = clang::QualType(t, 0);
Transform::GetPartiallyDesugaredType(Ctx, QT, transConfig).getAsString().c_str()
// CHECK: (const char *) "NS::Array<NS::ArrayType<double> >"

lookup.findScope("NS::Array<NS::ArrayType<Double32_t> >", &t);
QT = clang::QualType(t, 0);
Transform::GetPartiallyDesugaredType(Ctx, QT, transConfig).getAsString().c_str()
// CHECK: (const char *) "NS::Array<NS::ArrayType<Double32_t> >"

lookup.findScope("NS::Container<Long_t>::Content", &t);
QT = clang::QualType(t, 0);
Transform::GetPartiallyDesugaredType(Ctx, QT, transConfig).getAsString().c_str()
// CHECK: (const char *) "NS::Container<long>::Content"

QT = lookup.findType("NS::Container<Long_t>::Value_t");
Transform::GetPartiallyDesugaredType(Ctx, QT, transConfig).getAsString().c_str()
// CHECK: (const char *) "long"

lookup.findScope("NS::Container<Long_t>::Content_t", &t);
QT = clang::QualType(t, 0);
Transform::GetPartiallyDesugaredType(Ctx, QT, transConfig).getAsString().c_str()
// CHECK: (const char *) "NS::Container<long>::Content"

lookup.findScope("NS::Container<Long_t>::Impl_t", &t);
QT = clang::QualType(t, 0);
Transform::GetPartiallyDesugaredType(Ctx, QT, transConfig).getAsString().c_str()
// CHECK: (const char *) "Details::Impl"

lookup.findScope("NS::Container<Double32_t>::Content", &t);
QT = clang::QualType(t, 0);
Transform::GetPartiallyDesugaredType(Ctx, QT, transConfig).getAsString().c_str()
// CHECK: (const char *) "NS::Container<Double32_t>::Content"

QT = lookup.findType("NS::Container<Double32_t>::Value_t");
Transform::GetPartiallyDesugaredType(Ctx, QT, transConfig).getAsString().c_str()
// CHECK: (const char *) "double"
// Really we would want it to say Double32_t but oh well.

lookup.findScope("NS::Container<Double32_t>::Content_t", &t);
QT = clang::QualType(t, 0);
Transform::GetPartiallyDesugaredType(Ctx, QT, transConfig).getAsString().c_str()
// CHECK: (const char *) "NS::Container<Double32_t>::Content"

lookup.findScope("NS::Container<Double32_t>::Impl_t", &t);
QT = clang::QualType(t, 0);
Transform::GetPartiallyDesugaredType(Ctx, QT, transConfig).getAsString().c_str()
// CHECK: (const char *) "Details::Impl"

lookup.findScope("NS::TDataPointF", &t);
QT = clang::QualType(t, 0);
Transform::GetPartiallyDesugaredType(Ctx, QT, transConfig).getAsString().c_str()
// CHECK: (const char *) "NS::TDataPoint<float>"

lookup.findScope("NS::TDataPointD32", &t);
QT = clang::QualType(t, 0);
Transform::GetPartiallyDesugaredType(Ctx, QT, transConfig).getAsString().c_str()
// CHECK: (const char *) "NS::TDataPoint<Double32_t>"

lookup.findScope("vector<Details::Impl>::value_type", &t);
QT = clang::QualType(t, 0);
Transform::GetPartiallyDesugaredType(Ctx, QT, transConfig).getAsString().c_str()
// CHECK: (const char *) "Details::Impl"

lookup.findScope("vector<Details::Impl>::iterator", &t);
QT = clang::QualType(t, 0);
Transform::GetPartiallyDesugaredType(Ctx, QT, transConfig).getAsString().c_str()
// CHECK: (const char *) "std::vector<Details::Impl>::iterator"

lookup.findScope("vector<Details::Impl>::const_iterator", &t);
QT = clang::QualType(t, 0);
td = QT->getAs<clang::TypedefType>();
clang::TypedefNameDecl *tdDecl = td->getDecl();
QT = Ctx.getTypedefType(tdDecl);
Transform::GetPartiallyDesugaredType(Ctx, QT, transConfig, true).getAsString().c_str()
// CHECK: (const char *) "std::vector<Details::Impl, std::allocator<Details::Impl> >::const_iterator"

const clang::Decl*decl=lookup.findScope("Embedded_objects",&t);
if (decl) {
  const clang::CXXRecordDecl *cxxdecl
    = llvm::dyn_cast<clang::CXXRecordDecl>(decl);
  if (cxxdecl) {
    clang::DeclContext::decl_iterator iter = cxxdecl->decls_begin();
    while ( *iter ) {
      const clang::Decl *mdecl = *iter;
      if (const clang::ValueDecl *vd = llvm::dyn_cast<clang::ValueDecl>(mdecl)) {
        clang::QualType vdType = vd->getType();
        name.clear();
        Transform::GetPartiallyDesugaredType(Ctx,vdType,transConfig).getAsStringInternal(name,Policy);
        std::cout << name.c_str() << std::endl;
      }
      ++iter;
    }
  }
}
// CHECK: Embedded_objects::EmbeddedClasses
// CHECK: Embedded_objects::EmbeddedClasses::Embedded1
// CHECK: Embedded_objects::EmbeddedClasses::Embedded2
// CHECK: Embedded_objects::EmbeddedClasses::Embedded3
// CHECK: Embedded_objects::EmbeddedClasses::Embedded4
// CHECK: Embedded_objects::EmbeddedClasses::Embedded5
// CHECK: Embedded_objects::EmbeddedClasses::Embedded6
// CHECK: std::vector<int>::iterator
// CHECK: const Embedded_objects::Eenum
// CHECK: std::vector<int>::iterator

// In the partial desugaring add support for the case where we have a type
// that point to an already completely desugared template instantiation in
// which case the type is a RecordDecl rather than a TemplateInstantationType
decl = lookup.findScope("std::pair<Details::Impl,std::vector<Details::Impl> >",&t);
QT = clang::QualType(t, 0);
std::cout << Transform::GetPartiallyDesugaredType(Ctx, QT, transConfig).getAsString().c_str() << std::endl;
// CHECK: std::pair<Details::Impl, std::vector<Details::Impl> >

if (const clang::RecordDecl *rdecl = llvm::dyn_cast_or_null<clang::RecordDecl>(decl)) {
  clang::RecordDecl::field_iterator field_iter = rdecl->field_begin();
  // For some reason we can not call field_end:
  // cling: root/interpreter/llvm/src/tools/clang/lib/CodeGen/CGCall.cpp:1839: void checkArgMatches(llvm::Value*, unsigned int&, llvm::FunctionType*): Assertion `Elt->getType() == FTy->getParamType(ArgNo)' failed.
  // so just 'guess' the size
  int i = 0;
  while( i < 2 )  {
    name.clear();
    clang::QualType fdType = field_iter->getType();
    Transform::GetPartiallyDesugaredType(Ctx,fdType,transConfig).getAsStringInternal(name,Policy);
    std::cout << name.c_str() << std::endl;
    ++field_iter;
    ++i;
  }
}
// CHECK: Details::Impl
// CHECK: std::vector<Details::Impl, std::allocator<Details::Impl> >


decl=lookup.findScope("NS1::NS2::NS3::Inner3",&t);
if (decl) {
  const clang::CXXRecordDecl *cxxdecl
  = llvm::dyn_cast<clang::CXXRecordDecl>(decl);
  if (cxxdecl) {
    clang::DeclContext::decl_iterator iter = cxxdecl->decls_begin();
    while ( *iter ) {
      const clang::Decl *mdecl = *iter;
      if (const clang::ValueDecl *vd = llvm::dyn_cast<clang::ValueDecl>(mdecl)) {
        clang::QualType vdType = vd->getType();
        name.clear();
        Transform::GetPartiallyDesugaredType(Ctx,vdType,transConfig).getAsStringInternal(name,Policy);
        std::cout << name.c_str() << std::endl;
      }
      ++iter;
    }
  }
}
// CHECK: NS1::NS2::NS3::Point
// CHECK: NS1::NS2::NS3::Point
// CHECK: NS1::NS2::NS3::Point

decl = lookup.findScope("cmap<volatile int,volatile int>",&t);
QT = clang::QualType(t, 0);
std::cout << Transform::GetPartiallyDesugaredType(Ctx, QT, transConfig).getAsString().c_str() << std::endl;
if (const clang::RecordDecl *rdecl = llvm::dyn_cast_or_null<clang::RecordDecl>(decl)) {
  QT = clang::QualType(rdecl->getTypeForDecl(), 0);
  std::cout << Transform::GetPartiallyDesugaredType(Ctx, QT, transConfig).getAsString().c_str() << std::endl;
  clang::RecordDecl::field_iterator field_iter = rdecl->field_begin();
  // For some reason we can not call field_end:
  // cling: root/interpreter/llvm/src/tools/clang/lib/CodeGen/CGCall.cpp:1839: void checkArgMatches(llvm::Value*, unsigned int&, llvm::FunctionType*): Assertion `Elt->getType() == FTy->getParamType(ArgNo)' failed.
  // so just 'guess' the size
  int i = 0;
  while( i < 2 )  {
    name.clear();
    clang::QualType fdType = field_iter->getType();
    Transform::GetPartiallyDesugaredType(Ctx,fdType,transConfig).getAsStringInternal(name,Policy);
    std::cout << name.c_str() << std::endl;
    ++field_iter;
    ++i;
  }
}
// CHECK: cmap<volatile int, volatile int>
// CHECK: cmap<volatile int, volatile int, std::less<volatile int>, std::allocator<std::pair<const volatile int, volatile int> > >
// CHECK: volatile int
// CHECK: const volatile int
