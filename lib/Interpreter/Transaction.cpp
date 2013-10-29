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

  Transaction::Transaction(ASTContext& C) : m_ASTContext(C) {
    Initialize(C);
  }

  Transaction::Transaction(const CompilationOptions& Opts, ASTContext& C)
    : m_ASTContext(C) {
    Initialize(C);
    m_Opts = Opts; // intentional copy.
  }

  void Transaction::Initialize(ASTContext& C) {
    m_NestedTransactions.reset(0);
    m_Parent = 0; 
    m_State = kCollecting;
    m_IssuedDiags = kNone;
    m_Opts = CompilationOptions();
    m_Module = 0; 
    m_WrapperFD = 0;
    m_Next = 0;
    //m_ASTContext = C;
  }

  Transaction::~Transaction() {
    if (hasNestedTransactions())
      for (size_t i = 0; i < m_NestedTransactions->size(); ++i) {
        assert((*m_NestedTransactions)[i]->getState() == kCommitted 
               && "All nested transactions must be committed!");
        delete (*m_NestedTransactions)[i];
      }
  }

  void Transaction::addNestedTransaction(Transaction* nested) {
    // Create lazily the list
    if (!m_NestedTransactions)
      m_NestedTransactions.reset(new NestedTransactions());

    nested->setParent(this);
    // Leave a marker in the parent transaction, where the nested transaction
    // started.
    DelayCallInfo marker(clang::DeclGroupRef(), Transaction::kCCINone);
    m_DeclQueue.push_back(marker);
    m_NestedTransactions->push_back(nested);
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
    int markerPos = -1;
    for (size_t i = 0; i < size(); ++i) {
      if ((*this)[i].m_DGR.isNull() && (*this)[i].m_Call == kCCINone) {
        ++markerPos;
        if (nestedPos == markerPos) {
          erase(m_DeclQueue.begin() + i);
          break;
        }
      }
    }
    if (!m_NestedTransactions->size())
      m_NestedTransactions.reset(0);
  }

  void Transaction::reset() {
    assert((empty() || getState() == kRolledBack) 
           && "The transaction must be empty.");
    if (Transaction* parent = getParent())
      parent->removeNestedTransaction(this);
    m_Parent = 0;
    m_State = kCollecting;
    m_IssuedDiags = kNone;
    m_Opts = CompilationOptions();
    m_NestedTransactions.reset(0); // FIXME: leaks the nested transactions.
    m_Module = 0;
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
    assert((getState() == kCollecting || getState() == kCompleted)
           && "Must not be");

    bool checkForWrapper = !m_WrapperFD;

#ifndef NDEBUG
    // Check for duplicates
    for (size_t i = 0, e = m_DeclQueue.size(); i < e; ++i) {
      DelayCallInfo &oldDCI (m_DeclQueue[i]);
      // FIXME: This is possible bug in clang, which will instantiate one and 
      // the same CXXStaticMemberVar several times. This happens when there are
      // two dependent expressions and the first uses another declaration from 
      // the redeclaration chain. This will force Sema in to instantiate the
      // definition (usually the most recent decl in the chain) and then the
      // second expression might referece the definition (which was already)
      // instantiated, but Sema seems not to keep track of these kinds of
      // instantiations, even though the points of instantiation are the same!
      //
      // This should be investigated further when we merge with newest clang.
      // This is triggered by running the roottest: ./root/io/newstl
      if (oldDCI.m_Call == kCCIHandleCXXStaticMemberVarInstantiation)
        continue;
      // It is possible to have duplicate calls to HandleVTable with the same
      // declaration, because each time Sema believes a vtable is used it emits
      // that callback. 
      // For reference (clang::CodeGen::CodeGenModule::EmitVTable).
      if (oldDCI.m_Call != kCCIHandleVTable)
        assert(oldDCI != DCI && "Duplicates?!");
    }
    // We want to assert there is only one wrapper per transaction.
    checkForWrapper = true;
#endif
    
    // register the wrapper if any.
    if (checkForWrapper && !DCI.m_DGR.isNull() && DCI.m_DGR.isSingleDecl()) {
      if (FunctionDecl* FD = dyn_cast<FunctionDecl>(DCI.m_DGR.getSingleDecl())){
        if (checkForWrapper && utils::Analyze::IsWrapper(FD)) {
          assert(!m_WrapperFD && "Two wrappers in one transaction?");
          m_WrapperFD = FD;
        }
      }
    }
    
    if (comesFromASTReader(DCI.m_DGR))
      m_DeserializedDeclQueue.push_back(DCI);
    else
      m_DeclQueue.push_back(DCI);
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
  
  void Transaction::erase(iterator pos) {
    assert(!empty() && "Erasing from an empty transaction.");
    if (!pos->m_DGR.isNull() && m_WrapperFD == *pos->m_DGR.begin())
      m_WrapperFD = 0;
    m_DeclQueue.erase(pos);
  }

  void Transaction::DelayCallInfo::dump() const {
    PrintingPolicy Policy((LangOptions()));
    print(llvm::errs(), Policy, /*Indent*/0, /*PrintInstantiation*/true);
  }

  void Transaction::DelayCallInfo::print(llvm::raw_ostream& Out, 
                                         const PrintingPolicy& Policy,
                                         unsigned Indent, 
                                         bool PrintInstantiation, 
                                    llvm::StringRef prependInfo /*=""*/) const {
    static const char* const stateNames[Transaction::kCCINumStates] = {
      "kCCINone",
      "kCCIHandleTopLevelDecl",
      "kCCIHandleInterestingDecl",
      "kCCIHandleTagDeclDefinition",
      "kCCIHandleVTable",
      "kCCIHandleCXXImplicitFunctionInstantiation",
      "kCCIHandleCXXStaticMemberVarInstantiation",
    };
    assert((sizeof(stateNames) /sizeof(void*)) == Transaction::kCCINumStates 
           && "Missing states?");
    if (!prependInfo.empty()) {
      Out.changeColor(llvm::raw_ostream::RED);
      Out << prependInfo;
      Out.resetColor();
      Out << ", ";
    }
    Out.changeColor(llvm::raw_ostream::BLUE);
    Out << stateNames[m_Call]; 
    Out.changeColor(llvm::raw_ostream::GREEN);
    Out << " <- ";
    Out.resetColor();
    for (DeclGroupRef::const_iterator I = m_DGR.begin(), E = m_DGR.end(); 
         I != E; ++I) {
        if (*I)
          (*I)->print(Out, Policy, Indent, PrintInstantiation);
        else
          Out << "<<NULL DECL>>";
        Out << '\n';
    }
  }

  void Transaction::dump() const {
    const ASTContext& C = getASTContext();
    PrintingPolicy Policy = C.getPrintingPolicy();
    print(llvm::errs(), Policy, /*Indent*/0, /*PrintInstantiation*/true);
  }

  void Transaction::dumpPretty() const {
    const ASTContext& C = getASTContext();      
    PrintingPolicy Policy(C.getLangOpts());
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
      I->print(Out, Policy, Indent, PrintInstantiation);
    }

    // Print the deserialized decls if any.
    for (const_iterator I = deserialized_decls_begin(), 
           E = deserialized_decls_end(); I != E; ++I) {
      
      assert(!I->m_DGR.isNull() && "Must not contain null DGR.");
      I->print(Out, Policy, Indent, PrintInstantiation, "Deserialized");
    }
  }

  void Transaction::printStructure(size_t nindent) const {
    static const char* const stateNames[kNumStates] = {
      "Collecting",
      "kCompleted",
      "RolledBack",
      "RolledBackWithErrors",
      "Committed"
    };
    assert((sizeof(stateNames) / sizeof(void*)) == kNumStates 
           && "Missing a state to print.");
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

  void Transaction::printStructureBrief(size_t nindent /*=0*/) const {
    std::string indent(nindent, ' ');
    llvm::errs() << indent << "<cling::Transaction* " << this 
                 << " isEmpty=" << empty();
    llvm::errs() << " isCommitted=" << (getState() == kCommitted);
    llvm::errs() <<"> \n";

    for (const_nested_iterator I = nested_begin(), E = nested_end(); 
         I != E; ++I) {
      llvm::errs() << indent << "`";
      (*I)->printStructureBrief(nindent + 3);
    }
  }

  bool Transaction::comesFromASTReader(DeclGroupRef DGR) const {
    assert(!DGR.isNull() && "DeclGroupRef is Null!");
    if (getCompilationOpts().CodeGenerationForModule)
      return true;

    // Take the first/only decl in the group.
    Decl* D = *DGR.begin();
    return D->isFromASTFile();
  }

} // end namespace cling
