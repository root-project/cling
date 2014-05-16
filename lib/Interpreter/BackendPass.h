//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vvasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_TRANSACTION_BACKENDPASS_H
#define CLING_TRANSACTION_BACKENDPASS_H

#include "llvm/ADT/OwningPtr.h"

#include "TransactionTransformer.h"

namespace clang {
  class CodeGenOptions;
  class DiagnosticsEngine;
  class LangOptions;
  class TargetOptions;
}
namespace llvm {
  namespace legacy {
    class FunctionPassManager;
    class PassManager;
  }
  using legacy::FunctionPassManager;
  using legacy::PassManager;
  class Module;
  class Target;
  class TargetMachine;
}

namespace cling {

  ///\brief Run the backend passes. A stripped-down, streaming version
  /// of what's used by clang's BackendUtil. Implements no CodeGenPasses and
  /// no PerModulePasses.
  class BackendPass: public TransactionTransformer {
    llvm::Module* m_Module;
    llvm::OwningPtr<llvm::FunctionPassManager> m_PerFunctionPasses;
    llvm::OwningPtr<llvm::PassManager> m_PerModulePasses;
    llvm::OwningPtr<llvm::TargetMachine> m_TM;

  public:
    ///\brief Initializes the backend pass adaptor.
    ///
    ///\param[in] S - The semantic analysis object.
    ///\param[in] M - The module to run the passes on.
    ///\param[in] Diags - Diagnostics engine to be use by the passes.
    ///\param[in] TOpts - Current target options.
    ///\param[in] LangOpts - Current language options.
    ///\param[in] CodeGenOpts - Current CodeGen options.
    ///
    BackendPass(clang::Sema* S, llvm::Module* M,
                clang::DiagnosticsEngine& Diags,
                const clang::TargetOptions& TOpts,
                const clang::LangOptions& LangOpts,
                const clang::CodeGenOptions& CodeGenOpts);
    virtual ~BackendPass();

  protected:
    ///\brief Transforms the current transaction.
    ///
    void Transform() override;

    void CreatePasses(const clang::LangOptions& LangOpts,
                      const clang::CodeGenOptions& CodeGenOpts);
    /// CreateTargetMachine - Generates the TargetMachine.
    /// Returns Null if it is unable to create the target machine.
    llvm::TargetMachine*
    CreateTargetMachine(clang::DiagnosticsEngine& Diags,
                        const clang::TargetOptions &TOpts,
                        const clang::LangOptions& LangOpts,
                        const clang::CodeGenOptions& CodeGenOpts);

    llvm::FunctionPassManager *getPerFunctionPasses();
    llvm::PassManager *getPerModulePasses();
  };
} // end namespace cling
#endif // CLING_TRANSACTION_TRANSFORMER_H
