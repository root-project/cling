//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vvasilev@cern.ch>
//------------------------------------------------------------------------------

#include "cling/Interpreter/InterpreterCallbacks.h"

#include "cling/Interpreter/DynamicLookupExternalSemaSource.h"
#include "cling/Interpreter/Interpreter.h"

#include "clang/Sema/Sema.h"

using namespace clang;

namespace cling {

  // pin the vtable here
  InterpreterExternalSemaSource::~InterpreterExternalSemaSource() {}

  bool InterpreterExternalSemaSource::LookupUnqualified(LookupResult& R, 
                                                        Scope* S) {
    if (m_Callbacks)
      return m_Callbacks->LookupObject(R, S);
    
    return false;
  }

  InterpreterCallbacks::InterpreterCallbacks(Interpreter* interp,
                                             InterpreterExternalSemaSource* IESS)
    : m_Interpreter(interp),  m_SemaExternalSource(IESS) {
    if (!IESS)
      m_SemaExternalSource.reset(new InterpreterExternalSemaSource(this));
    m_Interpreter->getSema().addExternalSource(m_SemaExternalSource.get());

  }

  // pin the vtable here
  InterpreterCallbacks::~InterpreterCallbacks() {
    // FIXME: we have to remove the external source at destruction time. Needs
    // further tweaks of the patch in clang. This will be done later once the 
    // patch is in clang's mainline.
  }

  bool InterpreterCallbacks::LookupObject(LookupResult&, Scope*) {
    return false;
  }

} // end namespace cling

// TODO: Make the build system in the testsuite aware how to build that class
// and extract it out there again.
#include "DynamicLookup.h"
#include "cling/Utils/AST.h"

#include "clang/Sema/Lookup.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/Scope.h"
namespace cling {
namespace test {
  TestProxy* Tester = 0;

  extern "C" int printf(const char* fmt, ...);
  TestProxy::TestProxy(){}
  int TestProxy::Draw(){ return 12; }
  const char* TestProxy::getVersion(){ return "Interpreter.cpp"; }

  int TestProxy::Add10(int num) { return num + 10;}

  int TestProxy::Add(int a, int b) {
    return a + b;
  }

  void TestProxy::PrintString(std::string s) { printf("%s\n", s.c_str()); }

  bool TestProxy::PrintArray(int a[], size_t size) {
    for (unsigned i = 0; i < size; ++i)
      printf("%i", a[i]);

    printf("%s", "\n");

    return true;
  }

  void TestProxy::PrintArray(float a[][5], size_t size) {
    for (unsigned i = 0; i < size; ++i)
      for (unsigned j = 0; j < 5; ++j)
        printf("%i", (int)a[i][j]);

    printf("%s", "\n");
  }

  void TestProxy::PrintArray(int a[][4][5], size_t size) {
    for (unsigned i = 0; i < size; ++i)
      for (unsigned j = 0; j < 4; ++j)
        for (unsigned k = 0; k < 5; ++k)
          printf("%i", a[i][j][k]);

    printf("%s", "\n");
  }

  SymbolResolverCallback::SymbolResolverCallback(Interpreter* interp)
    : InterpreterCallbacks(interp, new DynamicIDHandler(this)), m_TesterDecl(0) {
    m_Interpreter->process("cling::test::Tester = new cling::test::TestProxy();");
  }

  SymbolResolverCallback::~SymbolResolverCallback() { }

  bool SymbolResolverCallback::LookupObject(LookupResult& R, Scope* S) {
    if (!IsDynamicLookup(R, S))
      return false;
    // We should react only on empty lookup result.
    if (!R.empty())
      return false;

    // Only for demo resolve all unknown objects to cling::test::Tester
    if (!m_TesterDecl) {
      clang::Sema& SemaRef = m_Interpreter->getSema();
      clang::NamespaceDecl* NSD = utils::Lookup::Namespace(&SemaRef, "cling");
      NSD = utils::Lookup::Namespace(&SemaRef, "test", NSD);
      m_TesterDecl = utils::Lookup::Named(&SemaRef, "Tester", NSD);
    }
    assert (m_TesterDecl && "Tester not found!");
    R.addDecl(m_TesterDecl);
    return true;
  }

  bool SymbolResolverCallback::IsDynamicLookup(LookupResult& R, Scope* S) {
    if (R.getLookupKind() != Sema::LookupOrdinaryName) return false;
    if (R.isForRedeclaration()) return false;
    // FIXME: Figure out better way to handle:
    // C++ [basic.lookup.classref]p1:
    //   In a class member access expression (5.2.5), if the . or -> token is
    //   immediately followed by an identifier followed by a <, the
    //   identifier must be looked up to determine whether the < is the
    //   beginning of a template argument list (14.2) or a less-than operator.
    //   The identifier is first looked up in the class of the object
    //   expression. If the identifier is not found, it is then looked up in
    //   the context of the entire postfix-expression and shall name a class
    //   or function template.
    //
    // We want to ignore object(.|->)member<template>
    if (R.getSema().PP.LookAhead(0).getKind() == tok::less)
      // TODO: check for . or -> in the cached token stream
      return false;

    for (Scope* DepScope = S; DepScope; DepScope = DepScope->getParent()) {
      if (DeclContext* Ctx = static_cast<DeclContext*>(DepScope->getEntity())) {
        return !Ctx->isDependentContext();
      }
    }

    return true;
  }


} // end test
} // end cling
