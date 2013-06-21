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
#include "llvm/Support/raw_ostream.h"

using namespace clang;

namespace cling {

  Transaction::~Transaction() {
    if (hasNestedTransactions())
      for (size_t i = 0; i < m_NestedTransactions->size(); ++i) {
        assert((*m_NestedTransactions)[i]->getState() == kCommitted 
               && "All nested transactions must be committed!");
        delete (*m_NestedTransactions)[i];
      }
  }

  void Transaction::removeNestedTransaction(Transaction* nested) {
    assert(hasNestedTransactions() && "Does not contain nested transactions");
    int nestedPos = -1;
    for (size_t i = 0; i < m_NestedTransactions->size(); ++i)
      if ((*m_NestedTransactions)[i] == nested) {
        nestedPos = i;
        break;
      }
    assert(nestedPos > -1 && "Not found!?");
    m_NestedTransactions->erase(m_NestedTransactions->begin() + nestedPos);
    // We need to remove the marker too.
    for (size_t i = 0; i < size(); ++i) {
      if ((*this)[i].m_DGR.isNull())
        --nestedPos;
      if (!nestedPos) {
        erase(i);
        break;
      }
    }
  }

  void Transaction::reset() {
    assert(empty() && "The transaction must be empty.");
    if (Transaction* parent = getParent())
      parent->removeNestedTransaction(this);
    m_Parent = 0;
    m_State = kCollecting;
    m_IssuedDiags = kNone;
    m_Opts = CompilationOptions();
    //m_Module = 0; <- NOTE: we want to reuse the empty module
    m_WrapperFD = 0;
    m_Next = 0;
  }

 
  void Transaction::append(DelayCallInfo DCI) {
    assert(!DCI.m_DGR.isNull() && "Appending null DGR?!");
    assert(getState() == kCollecting
           && "Cannot append declarations in current state.");
    forceAppend(DCI);
  }

  void Transaction::forceAppend(DelayCallInfo DCI) {
    assert(!DCI.m_DGR.isNull() && "Appending null DGR?!");

    // Lazy create the container on first append.
    if (!m_DeclQueue)
      m_DeclQueue.reset(new DeclQueue());

#ifdef NDEBUG
    // Check for duplicates
    for (size_t i = 0, e = m_DeclQueue->size(); i < e; ++i)
      assert((*m_DeclsQueue)[i] != DCI && "Duplicates?!");
#endif

    if (!DCI.m_DGR.isNull() && getState() == kCommitting) {
      // We are committing and getting new decls in.
      // Move them into a sub transaction that will be processed
      // recursively at the end of of commitTransaction.
      Transaction* subTransactionWhileCommitting = 0;
      if (hasNestedTransactions()
          && m_NestedTransactions->back()->getState() == kCollecting)
        subTransactionWhileCommitting = m_NestedTransactions->back();
      else {
        // FIXME: is this correct (Axel says "yes")
        CompilationOptions Opts(getCompilationOpts());
        Opts.DeclarationExtraction = 0;
        Opts.ValuePrinting = CompilationOptions::VPDisabled;
        Opts.ResultEvaluation = 0;
        Opts.DynamicScoping = 0;
        subTransactionWhileCommitting = new Transaction(Opts);
        addNestedTransaction(subTransactionWhileCommitting);
      }
      subTransactionWhileCommitting->append(DCI);
      return;
    }
    bool checkForWrapper = !m_WrapperFD;
    assert(checkForWrapper = true && "Check for wrappers with asserts");
    // register the wrapper if any.
    if (checkForWrapper && !DCI.m_DGR.isNull() && DCI.m_DGR.isSingleDecl()) {
      if (FunctionDecl* FD = dyn_cast<FunctionDecl>(DCI.m_DGR.getSingleDecl()))
        if (utils::Analyze::IsWrapper(FD)) {
          assert(!m_WrapperFD && "Two wrappers in one transaction?");
          m_WrapperFD = FD;
        }
    }
    m_DeclQueue->push_back(DCI);
  }

  void Transaction::append(clang::DeclGroupRef DGR) {
    append(DelayCallInfo(DGR, kCCIHandleTopLevelDecl));
  }

  void Transaction::append(Decl* D) {
    append(DeclGroupRef(D));
  }

  void Transaction::forceAppend(Decl* D) {
    forceAppend(DelayCallInfo(DeclGroupRef(D), kCCIHandleTopLevelDecl));
  }
  
  void Transaction::erase(size_t pos) {
    assert(!empty() && "Erasing from an empty transaction.");
    m_DeclQueue->erase(decls_begin() + pos);
  }

  void Transaction::dump() const {
    if (!size())
      return;

    ASTContext& C = getFirstDecl().getSingleDecl()->getASTContext();
    PrintingPolicy Policy = C.getPrintingPolicy();
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
        (*m_NestedTransactions)[nestedT++]->print(Out, Policy, Indent, 
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

  void Transaction::printStructure(size_t nindent) const {
    static const char* const stateNames[kNumStates] = {
      "Collecting",
      "kCompleted",
      "Committing",
      "RolledBack",
      "RolledBackWithErrors",
      "Committed"
    };
    std::string indent(nindent, ' ');
    llvm::errs() << indent << "Transaction @" << this << ": \n";
    for (const_nested_iterator I = nested_begin(), E = nested_end(); 
         I != E; ++I) {
      (*I)->printStructure(nindent + 3);
    }
    llvm::errs() << indent << " state: " << stateNames[getState()] << ", "
                 << size() << " decl groups, ";
    if (hasNestedTransactions())
      llvm::errs() << m_NestedTransactions->size();
    else
      llvm::errs() << "0";

    llvm::errs() << " nested transactions\n"
                 << indent << " wrapper: " << m_WrapperFD
                 << ", parent: " << m_Parent
                 << ", next: " << m_Next << "\n";
  }

} // end namespace cling
