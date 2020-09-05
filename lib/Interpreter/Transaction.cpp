//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vvasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "cling/Interpreter/Transaction.h"
#include "cling/Utils/AST.h"
#include "cling/Utils/Output.h"
#include "IncrementalExecutor.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/Lex/MacroInfo.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/Sema.h"

#include "llvm/IR/Module.h"

using namespace clang;

namespace cling {

  Transaction::Transaction(Sema& S) : m_Sema(S) {
    Initialize(S);
  }

  Transaction::Transaction(const CompilationOptions& Opts, Sema& S)
    : m_Sema(S) {
    Initialize(S);
    m_Opts = Opts; // intentional copy.
  }

  void Transaction::Initialize(Sema& S) {
    m_NestedTransactions.reset(0);
    m_Parent = 0;
    m_State = kCollecting;
    m_IssuedDiags = kNone;
    m_Opts = CompilationOptions();
    m_DefinitionShadowNS = 0;
    m_Module = 0;
    m_WrapperFD = 0;
    m_Next = 0;
    //m_Sema = S;
    m_BufferFID = FileID(); // sets it to invalid.
    m_Exe = 0;
  }

  Transaction::~Transaction() {
    // FIXME: Enable this once we have a good control on the ownership.
    //assert(m_Module.use_count() <= 1 && "There is still a reference!");
    if (hasNestedTransactions())
      for (size_t i = 0; i < m_NestedTransactions->size(); ++i) {
        assert(((*m_NestedTransactions)[i]->getState() == kCommitted
                || (*m_NestedTransactions)[i]->getState() == kRolledBack)
               && "All nested transactions must be committed!");
        delete (*m_NestedTransactions)[i];
      }
  }

  void Transaction::setDefinitionShadowNS(clang::NamespaceDecl* NS) {
    assert(!m_DefinitionShadowNS && "Transaction has a __cling_N5xxx NS?");
    m_DefinitionShadowNS = NS;
    // Ensure `NS` is unloaded from the AST on transaction rollback, e.g. '.undo X'
    append(static_cast<clang::Decl*>(NS));
  }

  NamedDecl* Transaction::containsNamedDecl(llvm::StringRef name) const {
    for (auto I = decls_begin(), E = decls_end(); I != E; ++I) {
      for (auto DI : I->m_DGR) {
        if (NamedDecl* ND = dyn_cast<NamedDecl>(DI)) {
          if (name.equals(ND->getNameAsString()))
            return ND;
        }
      }
    }
    // Not found yet, peek inside extern "C" declarations
    for (auto I = decls_begin(), E = decls_end(); I != E; ++I) {
      for (auto DI : I->m_DGR) {
        if (LinkageSpecDecl* LSD = dyn_cast<LinkageSpecDecl>(DI)) {
          for (Decl* DI : LSD->decls()) {
            if (NamedDecl* ND = dyn_cast<NamedDecl>(DI)) {
              if (name.equals(ND->getNameAsString()))
                return ND;
            }
          }
        }
      }
    }
    return nullptr;
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
    for (iterator I = decls_begin(), E = decls_end(); I != E; ++I) {
      if (I->m_DGR.isNull() && I->m_Call == kCCINone) {
        ++markerPos;
        if (nestedPos == markerPos) {
          erase(I); // Safe because of the break stmt.
          break;
        }
      }
    }
    if (!m_NestedTransactions->size())
      m_NestedTransactions.reset(0);
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
      if (oldDCI.m_Call != kCCIHandleVTable
          && oldDCI.m_Call != kCCIHandleCXXImplicitFunctionInstantiation)
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

  void Transaction::append(MacroDirectiveInfo MDE) {
    assert(MDE.m_II && "Appending null IdentifierInfo?!");
    assert(MDE.m_MD && "Appending null MacroDirective?!");
    assert(getState() == kCollecting
           && "Cannot append declarations in current state.");

#ifndef NDEBUG
    if (size_t i = m_MacroDirectiveInfoQueue.size()) {
      // Check for duplicates
      do {
        MacroDirectiveInfo &prevDir (m_MacroDirectiveInfoQueue[--i]);
        if (prevDir == MDE) {
          const UndefMacroDirective* A =
                                        dyn_cast<UndefMacroDirective>(MDE.m_MD);
          const UndefMacroDirective* B =
                                    dyn_cast<UndefMacroDirective>(prevDir.m_MD);
          // Allow undef to follow def and vice versa, but that is all.
          assert((A ? B==nullptr : B!=nullptr) && "Duplicates");
          // Has previously been checked prior to here, so were done.
          break;
        }
      } while (i != 0);
    }
#endif

    m_MacroDirectiveInfoQueue.push_back(MDE);
  }

  unsigned Transaction::getUniqueID() const {
    return m_BufferFID.getHashValue();
  }

  void Transaction::erase(iterator pos) {
    assert(!empty() && "Erasing from an empty transaction.");
    if (!pos->m_DGR.isNull() && m_WrapperFD == *pos->m_DGR.begin())
      m_WrapperFD = 0;
    m_DeclQueue.erase(pos);
  }

  void Transaction::DelayCallInfo::dump() const {
    PrintingPolicy Policy((LangOptions()));
    print(cling::log(), Policy, /*Indent*/0, /*PrintInstantiation*/true);
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
      "kCCICompleteTentativeDefinition",
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

  void Transaction::MacroDirectiveInfo::dump(const clang::Preprocessor& PP) const {
    print(cling::log(), PP);
  }

  void Transaction::MacroDirectiveInfo::print(llvm::raw_ostream& Out,
                                              const clang::Preprocessor& PP) const {
    PP.printMacro(this->m_II, this->m_MD, Out);
  }

  void Transaction::dump() const {
    const ASTContext& C = m_Sema.getASTContext();
    PrintingPolicy Policy = C.getPrintingPolicy();
    print(cling::log(), Policy, /*Indent*/0, /*PrintInstantiation*/true);
  }

  void Transaction::dumpPretty() const {
    const ASTContext& C = m_Sema.getASTContext();
    PrintingPolicy Policy(C.getLangOpts());
    print(cling::log(), Policy, /*Indent*/0, /*PrintInstantiation*/true);
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

    for (Transaction::const_reverse_macros_iterator MI = rmacros_begin(),
           ME = rmacros_end(); MI != ME; ++MI) {
      MI->print(Out, m_Sema.getPreprocessor());
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
    cling::log() << indent << "Transaction @" << this << ": \n";
    for (const_nested_iterator I = nested_begin(), E = nested_end();
         I != E; ++I) {
      (*I)->printStructure(nindent + 3);
    }
    cling::log() << indent << " state: " << stateNames[getState()]
                 << " decl groups, ";
    if (hasNestedTransactions())
      cling::log() << m_NestedTransactions->size();
    else
      cling::log() << "0";

    cling::log() << " nested transactions\n"
                 << indent << " wrapper: " << m_WrapperFD
                 << ", parent: " << m_Parent
                 << ", next: " << m_Next << "\n";
  }

  void Transaction::printStructureBrief(size_t nindent /*=0*/) const {
    std::string indent(nindent, ' ');
    cling::log() << indent << "<cling::Transaction* " << this
                 << " isEmpty=" << empty();
    cling::log() << " isCommitted=" << (getState() == kCommitted);
    cling::log() <<"> \n";

    for (const_nested_iterator I = nested_begin(), E = nested_end();
         I != E; ++I) {
      cling::log() << indent << "`";
      (*I)->printStructureBrief(nindent + 3);
    }
  }

  bool Transaction::comesFromASTReader(DeclGroupRef DGR) const {
    assert(!DGR.isNull() && "DeclGroupRef is Null!");
    if (getCompilationOpts().CodeGenerationForModule)
      return true;

    // Take the first/only decl in the group.
    Decl* D = *DGR.begin();

    // If D is from an AST file, we can return true.
    if (D->isFromASTFile()) return true;

    if (m_Sema.getASTContext().getLangOpts().Modules) {
      // If we currently compile a module and the decl is from a submodule that
      // we are currently compiling, then we also pretend it's from a AST file.
      // If we don't do that than our duplicate check in forceAppend will fail
      // when we try to generate a module that has multiple submodules that
      // textually include the same declaration (which will cause multiple
      // entries of the same merged decl to be in this list).
      if (D->getOwningModule()) {
        StringRef CurrentModule =
            D->getASTContext().getLangOpts().CurrentModule;
        StringRef DeclModule = D->getOwningModule()->getTopLevelModuleName();
        return CurrentModule == DeclModule;
      }
    }

    return false;
  }

  SourceLocation
  Transaction::getSourceStart(const clang::SourceManager& SM) const {
    // Children can have invalid BufferIDs. In that case use the parent's.
    if (m_BufferFID.isInvalid() && m_Parent)
      return m_Parent->getSourceStart(SM);
    return SM.getLocForStartOfFile(m_BufferFID);
  }

} // end namespace cling
