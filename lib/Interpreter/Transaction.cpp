//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vvasielv@cern.ch>
//------------------------------------------------------------------------------

#include "cling/Interpreter/Transaction.h"

#include "cling/Utils/AST.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/PrettyPrinter.h"

using namespace clang;

namespace cling {

  Transaction::~Transaction() {
    for (size_t i = 0; i < m_NestedTransactions.size(); ++i)
      delete m_NestedTransactions[i];
  }

  void Transaction::append(DelayCallInfo DCI) {
    // for (const_iterator I = decls_begin(), E = decls_end(); I != E; ++I) {
    //   if (DGR.isNull() || (*I).getAsOpaquePtr() == DGR.getAsOpaquePtr())
    //     return;
    // }
    // register the wrapper if any.

    if (getState() == kCommitting) {
      // We are committing and getting enw decls in.
      // Move them into a sub transaction that will be processed
      // recursively at the end of of commitTransaction.
      Transaction* subTransactionWhileCommitting = 0;
      if (hasNestedTransactions()
          && m_NestedTransactions.back()->getState() == kUnknown)
        subTransactionWhileCommitting = m_NestedTransactions.back();
      else {
        // FIXME: is this correct (Axel says "yes")
        CompilationOptions Opts(getCompilationOpts());
        Opts.DeclarationExtraction = 0;
        Opts.ValuePrinting = CompilationOptions::VPDisabled;
        Opts.ResultEvaluation = 0;
        Opts.DynamicScoping = 0;
        subTransactionWhileCommitting
          = new Transaction(Opts, getModule());
        addNestedTransaction(subTransactionWhileCommitting);
      }
      subTransactionWhileCommitting->append(DCI);
      return;
    }

    if (DCI.m_DGR.isSingleDecl()) {
      if (FunctionDecl* FD = dyn_cast<FunctionDecl>(DCI.m_DGR.getSingleDecl()))
        if (utils::Analyze::IsWrapper(FD)) {
          assert(!m_WrapperFD && "Two wrappers in one transaction?");
          m_WrapperFD = FD;
        }
    }
    m_DeclQueue.push_back(DCI);
  }

  void Transaction::append(clang::DeclGroupRef DGR) {
    append(DelayCallInfo(DGR, kCCIHandleTopLevelDecl));
  }

  void Transaction::append(Decl* D) {
    append(DeclGroupRef(D));
  }

  void Transaction::dump() const {
    if (!size())
      return;

    ASTContext& C = getFirstDecl().getSingleDecl()->getASTContext();
    PrintingPolicy Policy = C.getPrintingPolicy();
    Policy.DumpSourceManager = &C.getSourceManager();
    print(llvm::errs(), Policy, /*Indent*/0, /*PrintInstantiation*/true);
  }

  void Transaction::dumpPretty() const {
    if (!size())
      return;
    ASTContext* C = 0;
    if (m_WrapperFD)
      C = &(m_WrapperFD->getASTContext());
    if (!getFirstDecl().isNull())
      C = &(getFirstDecl().getSingleDecl()->getASTContext());
      
    PrintingPolicy Policy(C->getLangOpts());
    print(llvm::errs(), Policy, /*Indent*/0, /*PrintInstantiation*/true);
  }

  void Transaction::print(llvm::raw_ostream& Out, const PrintingPolicy& Policy,
                          unsigned Indent, bool PrintInstantiation) const {
    int nestedT = 0;
    for (const_iterator I = decls_begin(), E = decls_end(); I != E; ++I) {
      if (I->m_DGR.isNull()) {
        assert(hasNestedTransactions() && "DGR is null even if no nesting?");
        // print the nested decl
        Out<< "\n";
        Out<<"+====================================================+\n";
        Out<<"        Nested Transaction" << nestedT << "           \n";
        Out<<"+====================================================+\n";
        m_NestedTransactions[nestedT++]->print(Out, Policy, Indent, 
                                               PrintInstantiation);
        Out<< "\n";
        Out<<"+====================================================+\n";
        Out<<"          End Transaction" << nestedT << "            \n";
        Out<<"+====================================================+\n";
      }
      for (DeclGroupRef::const_iterator J = I->m_DGR.begin(), 
             L = I->m_DGR.end(); J != L; ++J)
        if (*J)
          (*J)->print(Out, Policy, Indent, PrintInstantiation);
        else
          Out << "<<NULL DECL>>";
    }
  }

} // end namespace cling
