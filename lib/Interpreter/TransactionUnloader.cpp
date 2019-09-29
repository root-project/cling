//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vvasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "TransactionUnloader.h"

#include "IncrementalExecutor.h"
#include "DeclUnloader.h"

#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/Transaction.h"

#include "clang/AST/Decl.h"
#include "clang/AST/DependentDiagnostic.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Sema/Sema.h"

using namespace clang;

namespace cling {
  bool TransactionUnloader::unloadDeclarations(Transaction* T,
                                               DeclUnloader& DeclU) {
    bool Successful = true;

    for (Transaction::const_reverse_iterator I = T->rdecls_begin(),
           E = T->rdecls_end(); I != E; ++I) {
      const Transaction::ConsumerCallInfo& Call = I->m_Call;
      const DeclGroupRef& DGR = (*I).m_DGR;

      if (Call == Transaction::kCCIHandleVTable)
        continue;
      // The non templated classes come through HandleTopLevelDecl and
      // HandleTagDeclDefinition, this is why we need to filter.
      if (Call == Transaction::kCCIHandleTagDeclDefinition)
      if (const CXXRecordDecl* D
        = dyn_cast<CXXRecordDecl>(DGR.getSingleDecl()))
      if (D->getTemplateSpecializationKind() == TSK_Undeclared)
        continue;

      if (Call == Transaction::kCCINone)
        m_Interp->unload(*(*T->rnested_begin()));

      for (DeclGroupRef::const_iterator
             Di = DGR.end() - 1, E = DGR.begin() - 1; Di != E; --Di) {
        // Get rid of the declaration. If the declaration has name we should
        // heal the lookup tables as well
        Successful = DeclU.UnloadDecl(*Di) && Successful;
#ifndef NDEBUG
        assert(Successful && "Cannot handle that yet!");
#endif
      }
    }
    assert(T->rnested_begin() == T->rnested_end()
           && "nested transactions mismatch");
    return Successful;
  }

  bool TransactionUnloader::unloadFromPreprocessor(Transaction* T,
                                                   DeclUnloader& DeclU) {
    bool Successful = true;
    for (Transaction::const_reverse_macros_iterator MI = T->rmacros_begin(),
           ME = T->rmacros_end(); MI != ME; ++MI) {
      // Get rid of the macro definition
      Successful = DeclU.UnloadMacro(*MI) && Successful;
#ifndef NDEBUG
      assert(Successful && "Cannot handle that yet!");
#endif
    }
    return Successful;
  }

  bool TransactionUnloader::unloadDeserializedDeclarations(Transaction* T,
                                                   DeclUnloader& DeclU) {
    //FIXME: Terrible hack, we *must* get rid of parseForModule by implementing
    // a header file generator in cling.
    bool Successful = true;
    for (Transaction::const_reverse_iterator I = T->deserialized_rdecls_begin(),
           E = T->deserialized_rdecls_end(); I != E; ++I) {
      const DeclGroupRef& DGR = (*I).m_DGR;
      for (DeclGroupRef::const_iterator
             Di = DGR.end() - 1, E = DGR.begin() - 1; Di != E; --Di) {
        // UnloadDecl() shall unload decls that came through `parseForModule()',
        // but not those that came from the PCH.
        Successful = DeclU.UnloadDecl(*Di) && Successful;
#ifndef NDEBUG
        assert(Successful && "Cannot handle that yet!");
#endif
      }
    }
    return Successful;
  }

  bool TransactionUnloader::RevertTransaction(Transaction* T) {

    bool Successful = true;
    if (getExecutor() && T->getModule()) {
      Successful = getExecutor()->unloadModule(T->getModule()) && Successful;

      // Cleanup the module from unused global values.
      // if (T->getModule()) {
      //   llvm::ModulePass* globalDCE = llvm::createGlobalDCEPass();
      //   globalDCE->runOnModule(*T->getModule());
      // }

      Successful = unloadModule(T->getModule()) && Successful;
    }

    // Clean up the pending instantiations
    m_Sema->PendingInstantiations.clear();
    m_Sema->PendingLocalImplicitInstantiations.clear();

    DeclUnloader DeclU(m_Sema, m_CodeGen, T);
    Successful = unloadDeclarations(T, DeclU) && Successful;
    Successful = unloadDeserializedDeclarations(T, DeclU) && Successful;
    Successful = unloadFromPreprocessor(T, DeclU) && Successful;

#ifndef NDEBUG
    //FIXME: Move the nested transaction marker out of the decl lists and
    // reenable this assertion.
    //size_t DeclSize = std::distance(T->decls_begin(), T->decls_end());
    //if (T->getCompilationOpts().CodeGenerationForModule)
    //  assert (!DeclSize && "No parsed decls must happen in parse for module");
#endif

    if (Successful)
      T->setState(Transaction::kRolledBack);
    else
      T->setState(Transaction::kRolledBackWithErrors);

    // Release the input_line_X file unless verifying diagnostics.
    if (!m_Interp->getCI()->getDiagnosticOpts().VerifyDiagnostics)
      m_Sema->getSourceManager().invalidateCache(T->getBufferFID());

    return Successful;
  }

  bool TransactionUnloader::UnloadDecl(Decl* D) {
    return cling::UnloadDecl(m_Sema, m_CodeGen, D);
  }

  bool
  TransactionUnloader::unloadModule(const std::shared_ptr<llvm::Module>& M) {
    for (auto& Func: M->functions())
      m_CodeGen->forgetGlobal(&Func);
    for (auto& Glob: M->globals())
      m_CodeGen->forgetGlobal(&Glob);
    return true;
  }
} // end namespace cling
