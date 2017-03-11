//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Baozeng Ding <sploving1@gmail.com>
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "cling/Interpreter/Exception.h"

#include "cling/Interpreter/InterpreterCallbacks.h"
#include "cling/Interpreter/Interpreter.h"
#include "cling/Utils/Output.h"
#include "cling/Utils/Validation.h"

#include "clang/Frontend/CompilerInstance.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaDiagnostic.h"

extern "C" {
/// Throw an InvalidDerefException if the Arg pointer is invalid.
///\param Interp: The interpreter that has compiled the code.
///\param Expr: The expression corresponding determining the pointer value.
///\param Arg: The pointer to be checked.
///\returns void*, const-cast from Arg, to reduce the complexity in the
/// calling AST nodes, at the expense of possibly doing a
/// T* -> const void* -> const_cast<void*> -> T* round trip.
void* cling_runtime_internal_throwIfInvalidPointer(void* Interp, void* Expr,
                                                   const void* Arg) {

  const clang::Expr* const E = (const clang::Expr*)Expr;

  // The isValidAddress function return true even when the pointer is
  // null thus the checks have to be done before returning successfully from the
  // function in this specific order.
  if (!Arg) {
    cling::Interpreter* I = (cling::Interpreter*)Interp;
    clang::Sema& S = I->getCI()->getSema();
    // Print a nice backtrace.
    I->getCallbacks()->PrintStackTrace();
    throw cling::InvalidDerefException(&S, E,
          cling::InvalidDerefException::DerefType::NULL_DEREF);
  } else if (!cling::utils::isAddressValid(Arg)) {
    cling::Interpreter* I = (cling::Interpreter*)Interp;
    clang::Sema& S = I->getCI()->getSema();
    // Print a nice backtrace.
    I->getCallbacks()->PrintStackTrace();
    throw cling::InvalidDerefException(&S, E,
          cling::InvalidDerefException::DerefType::INVALID_MEM);
  }
  return const_cast<void*>(Arg);
}
}

namespace cling {
  InterpreterException::InterpreterException(const std::string& What) :
    std::runtime_error(What), m_Sema(nullptr) {}
  InterpreterException::InterpreterException(const char* What, clang::Sema* S) :
    std::runtime_error(What), m_Sema(S) {}

  bool InterpreterException::diagnose() const { return false; }
  InterpreterException::~InterpreterException() LLVM_NOEXCEPT {}


  InvalidDerefException::InvalidDerefException(clang::Sema* S,
                                               const clang::Expr* E,
                                               DerefType type)
    : InterpreterException(type == INVALID_MEM  ?
      "Trying to access a pointer that points to an invalid memory address." :
      "Trying to dereference null pointer or trying to call routine taking "
      "non-null arguments", S),
    m_Arg(E), m_Type(type) {}

  InvalidDerefException::~InvalidDerefException() LLVM_NOEXCEPT {}

  bool InvalidDerefException::diagnose() const {
    // Construct custom diagnostic: warning for invalid memory address;
    // no equivalent in clang.
    if (m_Type == cling::InvalidDerefException::DerefType::INVALID_MEM) {
      clang::DiagnosticsEngine& Diags = m_Sema->getDiagnostics();
      unsigned DiagID =
        Diags.getCustomDiagID(clang::DiagnosticsEngine::Warning,
                                 "invalid memory pointer passed to a callee:");
      Diags.Report(m_Arg->getLocStart(), DiagID) << m_Arg->getSourceRange();
    }
    else
      m_Sema->Diag(m_Arg->getLocStart(), clang::diag::warn_null_arg)
        << m_Arg->getSourceRange();
    return true;
  }

  CompilationException::CompilationException(const std::string& Reason) :
    InterpreterException(Reason) {}

  CompilationException::~CompilationException() LLVM_NOEXCEPT {}

  void CompilationException::throwingHandler(void * /*user_data*/,
                                             const std::string& reason,
                                             bool /*gen_crash_diag*/) {
    throw cling::CompilationException(reason);
  }

  static void ReportException(const char* Msg, const char* What = nullptr) {
    cling::errs() << ">>> " << Msg;
    if (What) cling::errs() << ": '" << What << "'";
    cling::errs() << ".\n";
  }

  bool InterpreterException::ReportErr(void* Data, const std::exception* E) {
    if (E) {
      const InterpreterException* IE =
          dynamic_cast<const InterpreterException*>(E);
      if (IE && IE->diagnose())
        return true;

      ReportException(IE ? "Caught an interpreter exception"
                         : "Caught a std::exception",
                      E->what());
    } else
      ReportException("Caught an unkown exception");
    return true;
  }

  static bool NoReport(void*, const std::exception*) {
    return true;
  }

  void InterpreterException::RunLoop(bool (*Proc)(void* Data), void* Data,
                                     ErrorHandler OnError) {
    bool Run = true;
    if (!OnError) OnError = &NoReport;

    while (Run) {
      try {
        Run = Proc(Data);
      } catch (const InterpreterException& E) {
        Run = OnError(Data, &E);
      } catch (const std::exception& E) {
        Run = OnError(Data, &E);
      } catch (...) {
        Run = OnError(Data, nullptr);
      }
    }
  }

  namespace internal {
    void TestExceptions(intptr_t Throw) {
     struct LocalObj {};
     switch (Throw) {
        case -1:  throw LocalObj();
        case  1:  throw std::exception();
        case  2:  throw std::logic_error("std::logic_error");
        case  3:  throw std::runtime_error("std::runtime_error");
        case  4:  throw std::out_of_range("std::out_of_range");
        case  5:  throw std::bad_alloc();
        case  6:  throw "c-string";
        case  7:  throw std::string("std::string");

        case 10:  throw bool(true);
        case 11:  throw float(1.41421354);
        case 12:  throw double(3.14159265);

        case -8:  throw int8_t(-8);
        case  8:  throw uint8_t(8);
        case -16: throw int16_t(-16);
        case  16: throw uint16_t(16);
        case -32: throw int32_t(-32);
        case  32: throw uint32_t(32);
        case -64: throw int64_t(-64);
        case  64: throw uint64_t(64);
        default:  break;
     }
    }
  }
} // end namespace cling
