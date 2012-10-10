//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vvasilev@cern.ch>
//------------------------------------------------------------------------------

#include "cling/Interpreter/InterpreterCallbacks.h"

#include "cling/Interpreter/Interpreter.h"

#include "clang/Sema/Sema.h"

using namespace clang;

namespace cling {

  // pin the vtable here
  InterpreterExternalSemaSource::~InterpreterExternalSemaSource() {}

  bool InterpreterExternalSemaSource::LookupUnqualified(LookupResult& R, 
                                                        Scope* S) {
    if (m_Callbacks && m_Callbacks->isEnabled())
      return m_Callbacks->LookupObject(R, S);
    
    return false;
  }

  InterpreterCallbacks::InterpreterCallbacks(Interpreter* interp, bool enabled,
                                             InterpreterExternalSemaSource* IESS)
    : m_Interpreter(interp), m_Enabled(enabled), m_SemaExternalSource(IESS) {
    if (!IESS)
      m_SemaExternalSource.reset(new InterpreterExternalSemaSource());
    m_SemaExternalSource->setCallbacks(this);
  }

  // pin the vtable here
  InterpreterCallbacks::~InterpreterCallbacks() {}

  bool InterpreterCallbacks::LookupObject(LookupResult&, Scope*) {
    return false;
  }

} // end namespace cling

// TODO: Make the build system in the testsuite aware how to build that class
// and extract it out there again.
#include "cling/Utils/AST.h"
#include "clang/Sema/Lookup.h"
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

  SymbolResolverCallback::SymbolResolverCallback(Interpreter* interp, 
                                                 bool enabled,
                                             InterpreterExternalSemaSource* IESS)
    : InterpreterCallbacks(interp, enabled, IESS), m_TesterDecl(0) {
    m_Interpreter->process("cling::test::Tester = new cling::test::TestProxy();");
  }

  SymbolResolverCallback::~SymbolResolverCallback(){}

  bool SymbolResolverCallback::LookupObject(LookupResult& R, Scope* S) {
    // Only for demo resolve all unknown objects to cling::test::Tester
    if (m_Enabled) {
      if (!m_TesterDecl) {
        clang::Sema& S = m_Interpreter->getSema();
        clang::NamespaceDecl* NSD = utils::Lookup::Namespace(&S, "cling");
        NSD = utils::Lookup::Namespace(&S, "test", NSD);
        m_TesterDecl = utils::Lookup::Named(&S, "Tester", NSD);
      }
      assert (m_TesterDecl && "Tester not found!");
      R.addDecl(m_TesterDecl);
      return true;
    }
    return false;
  }

} // end test
} // end cling
