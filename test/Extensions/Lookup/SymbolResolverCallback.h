//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id: InterpreterCallbacks.h 39299 2011-05-20 12:53:30Z vvassilev $
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_TEST_SYMBOL_RESOLVER_CALLBACK
#define CLING_TEST_SYMBOL_RESOLVER_CALLBACK

#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/InterpreterCallbacks.h"

#include "cling/Utils/AST.h"

#include "clang/Frontend/CompilerInstance.h"
#include "clang/Sema/Lookup.h"

namespace cling {
  namespace test {
    extern "C" int printf(const char* fmt, ...);
    class TestProxy {
    public:
      TestProxy(){}
      int Draw(){ return 12; }
      const char* getVersion(){ return "Interpreter.cpp"; }

      int Add10(int num) { return num + 10;}
      void PrintString(std::string s) { printf("%s\n", s.c_str()); }
      bool PrintArray(int a[], size_t size) {
        for (unsigned i = 0; i < size; ++i)
          printf("%i", a[i]);

        printf("%s", "\n");

        return true;
      }

      int Add(int a, int b) {
        return a + b;
      }

      void PrintArray(float a[][5], size_t size) {
        for (unsigned i = 0; i < size; ++i)
          for (unsigned j = 0; j < 5; ++j)
            printf("%i", (int)a[i][j]);

        printf("%s", "\n");
      }

      void PrintArray(int a[][4][5], size_t size) {
        for (unsigned i = 0; i < size; ++i)
          for (unsigned j = 0; j < 4; ++j)
            for (unsigned k = 0; k < 5; ++k)
              printf("%i", a[i][j][k]);

        printf("%s", "\n");
      }

    };

    TestProxy* Tester = 0;

    class SymbolResolverCallback: public cling::InterpreterCallbacks {
    private:
      clang::NamedDecl* m_TesterDecl;
    public:
      SymbolResolverCallback(Interpreter* interp, bool enabled = false)
        : InterpreterCallbacks(interp, enabled), m_TesterDecl(0) {
        m_Interpreter->process("cling::test::Tester = new cling::test::TestProxy();");
      }
      ~SymbolResolverCallback(){}

      bool LookupObject(clang::LookupResult& R, clang::Scope* S) {
        // Only for demo resolve all unknown objects to cling::test::Tester
        if (m_Enabled) {
          if (!m_TesterDecl) {
            clang::Sema& S = m_Interpreter->getCI()->getSema();
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
    };
  } // end test
} // end cling

#endif // CLING_TEST_SYMBOL_RESOLVER_CALLBACK
