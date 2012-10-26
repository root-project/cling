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

#include <string>

namespace Details {
   class Impl {};
}

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

.rawInput 0

const cling::LookupHelper& lookup = gCling->getLookupHelper();

const clang::ASTContext& Ctx = gCling->getSema().getASTContext();
llvm::SmallSet<const clang::Type*, 4> skip;
skip.insert(lookup.findType("Double32_t").getTypePtr());
const clang::Type* t = 0;
clang::QualType QT;
using namespace cling::utils;

// Test the behavior on a simple class
lookup.findScope("Details::Impl", &t);
QT = clang::QualType(t, 0);
//QT.getAsString().c_str()
Transform::GetPartiallyDesugaredType(Ctx, QT, skip).getAsString().c_str()
// CHECK: (const char * const) "Details::Impl"

// Test the behavior for a class inside an anonymous namespace
lookup.findScope("InsideAnonymous", &t);
QT = clang::QualType(t, 0);
//QT.getAsString().c_str()c
Transform::GetPartiallyDesugaredType(Ctx, QT, skip).getAsString().c_str()
// CHECK: (const char * const) "class <anonymous>::InsideAnonymous"

// The above result is not quite want we want, so the client must using 
// the following:
// The scope suppression is required for getting rid of the anonymous part of the name of a class defined in an anonymous namespace.
// This gives us more control vs not using the clang::ElaboratedType and relying on the Policy.SuppressUnwrittenScope which would
// strip both the anonymous and the inline namespace names (and we probably do not want the later to be suppressed).
clang::PrintingPolicy Policy(Ctx.getPrintingPolicy());
Policy.SuppressTagKeyword = true; // Never get the class or struct keyword
Policy.SuppressScope = true;      // Force the scope to be coming from a clang::ElaboratedType.
std::string name;
Transform::GetPartiallyDesugaredType(Ctx, QT, skip).getAsStringInternal(name,Policy);
name.c_str()
// CHECK: (const char * const) "InsideAnonymous"

// Test desugaring pointers types:
QT = lookup.findType("Int_t*");
Transform::GetPartiallyDesugaredType(Ctx, QT, skip).getAsString().c_str()
// CHECK:(const char * const) "int *"

QT = lookup.findType("const IntPtr_t*");
Transform::GetPartiallyDesugaredType(Ctx, QT, skip).getAsString().c_str()
// CHECK:(const char * const) "int *const *"


// Test desugaring reference (both r- or l- value) types:
QT = lookup.findType("const IntPtr_t&");
Transform::GetPartiallyDesugaredType(Ctx, QT, skip).getAsString().c_str()
// CHECK:(const char * const) "int *const &"

//TODO: QT = lookup.findType("IntPtr_t[32]");

// To do: findType does not return the const below:
// Test desugaring reference (both r- or l- value) types:
//QT = lookup.findType("const IntRef_t");
//Transform::GetPartiallyDesugaredType(Ctx, QT, skip).getAsString().c_str()
// should print:(const char * const) "int &const"

// Test desugaring reference (both r- or l- value) types:
QT = lookup.findType("IntRef_t");
Transform::GetPartiallyDesugaredType(Ctx, QT, skip).getAsString().c_str()
// CHECK:(const char * const) "int &"


//Desugar template parameters:
lookup.findScope("A<B<Double32_t, Int_t*> >", &t);
QT = clang::QualType(t, 0);
Transform::GetPartiallyDesugaredType(Ctx, QT, skip).getAsString().c_str()
// CHECK:(const char * const) "A<B<Double32_t, int *> >"

lookup.findScope("CTD", &t);
QT = clang::QualType(t, 0);
Transform::GetPartiallyDesugaredType(Ctx, QT, skip).getAsString().c_str()
// CHECK: (const char * const) "C<A<B<Double32_t, int> >, Double32_t>"

lookup.findScope("CTDConst", &t);
QT = clang::QualType(t, 0);
Transform::GetPartiallyDesugaredType(Ctx, QT, skip).getAsString().c_str()
// CHECK: (const char * const) "C<A<B<const Double32_t, const int> >, Double32_t>"

lookup.findScope("std::pair<const std::string,int>", &t);
QT = clang::QualType(t, 0);
Transform::GetPartiallyDesugaredType(Ctx, QT, skip).getAsString().c_str()
// CHECK: (const char * const) "std::pair<const std::string, int>"

lookup.findScope("NS::Array<NS::ArrayType<double> >", &t);
QT = clang::QualType(t, 0);
Transform::GetPartiallyDesugaredType(Ctx, QT, skip).getAsString().c_str()
// CHECK: (const char * const) "NS::Array<NS::ArrayType<double> >"

lookup.findScope("NS::Array<NS::ArrayType<Double32_t> >", &t);
QT = clang::QualType(t, 0);
Transform::GetPartiallyDesugaredType(Ctx, QT, skip).getAsString().c_str()
// CHECK: (const char * const) "NS::Array<NS::ArrayType<Double32_t> >"

lookup.findScope("NS::Container<Long_t>::Content", &t);
QT = clang::QualType(t, 0);
Transform::GetPartiallyDesugaredType(Ctx, QT, skip).getAsString().c_str()
// CHECK: (const char * const) "NS::Container<long>::Content"

QT = lookup.findType("NS::Container<Long_t>::Value_t");
Transform::GetPartiallyDesugaredType(Ctx, QT, skip).getAsString().c_str()
// CHECK: (const char * const) "long"

lookup.findScope("NS::Container<Long_t>::Content_t", &t);
QT = clang::QualType(t, 0);
Transform::GetPartiallyDesugaredType(Ctx, QT, skip).getAsString().c_str()
// CHECK: (const char * const) "NS::Container<long>::Content"

lookup.findScope("NS::Container<Long_t>::Impl_t", &t);
QT = clang::QualType(t, 0);
Transform::GetPartiallyDesugaredType(Ctx, QT, skip).getAsString().c_str()
// CHECK: (const char * const) "::Details::Impl"
// Note it should probably return "Details::Impl"

lookup.findScope("NS::Container<Double32_t>::Content", &t);
QT = clang::QualType(t, 0);
Transform::GetPartiallyDesugaredType(Ctx, QT, skip).getAsString().c_str()
// CHECK: (const char * const) "NS::Container<Double32_t>::Content"

QT = lookup.findType("NS::Container<Double32_t>::Value_t");
Transform::GetPartiallyDesugaredType(Ctx, QT, skip).getAsString().c_str()
// CHECK: (const char * const) "double"
// Really we would want it to say Double32_t but oh well.

lookup.findScope("NS::Container<Double32_t>::Content_t", &t);
QT = clang::QualType(t, 0);
Transform::GetPartiallyDesugaredType(Ctx, QT, skip).getAsString().c_str()
// CHECK: (const char * const) "NS::Container<Double32_t>::Content"

lookup.findScope("NS::Container<Double32_t>::Impl_t", &t);
QT = clang::QualType(t, 0);
Transform::GetPartiallyDesugaredType(Ctx, QT, skip).getAsString().c_str()
// CHECK: (const char * const) "::Details::Impl"
// Note it should probably return "Details::Impl"

lookup.findScope("NS::TDataPointF", &t);
QT = clang::QualType(t, 0);
Transform::GetPartiallyDesugaredType(Ctx, QT, skip).getAsString().c_str()
// CHECK: (const char * const) "NS::TDataPoint<float>"

lookup.findScope("NS::TDataPointD32", &t);
QT = clang::QualType(t, 0);
Transform::GetPartiallyDesugaredType(Ctx, QT, skip).getAsString().c_str()
// CHECK: (const char * const) "NS::TDataPoint<Double32_t>"
