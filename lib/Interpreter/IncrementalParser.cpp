//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Axel Naumann <axel@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "IncrementalParser.h"

#include "AutoSynthesizer.h"
#include "BackendPasses.h"
#include "CheckEmptyTransactionTransformer.h"
#include "ClingPragmas.h"
#include "DeclCollector.h"
#include "DeclExtractor.h"
#include "DynamicLookup.h"
#include "IncrementalExecutor.h"
#include "NullDerefProtectionTransformer.h"
#include "ValueExtractionSynthesizer.h"
#include "TransactionPool.h"
#include "ASTTransformer.h"
#include "TransactionUnloader.h"
#include "ValuePrinterSynthesizer.h"
#include "cling/Interpreter/CIFactory.h"
#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/InterpreterCallbacks.h"
#include "cling/Interpreter/Transaction.h"

#include "clang/AST/Attr.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclGroup.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/FileManager.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "clang/Parse/Parser.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "clang/Serialization/ASTWriter.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CrashRecoveryContext.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_os_ostream.h"

#include <iostream>
#include <stdio.h>
#include <sstream>

// Include the necessary headers to interface with the Windows registry and
// environment.
#ifdef _MSC_VER
  #define WIN32_LEAN_AND_MEAN
  #define NOGDI
  #define NOMINMAX
  #include <Windows.h>
  #include <sstream>
  #define popen _popen
  #define pclose _pclose
  #pragma comment(lib, "Advapi32.lib")
#endif

using namespace clang;

namespace {
  ///\brief Check the compile-time C++ ABI version vs the run-time ABI version,
  /// a mismatch could cause havoc. Reports if ABI versions differ.
  static void CheckABICompatibility(clang::CompilerInstance* CI) {
#ifdef __GLIBCXX__
# define CLING_CXXABIV __GLIBCXX__
# define CLING_CXXABIS "__GLIBCXX__"
#elif _LIBCPP_VERSION
# define CLING_CXXABIV _LIBCPP_VERSION
# define CLING_CXXABIS "_LIBCPP_VERSION"
#elif defined (_MSC_VER)
    // For MSVC we do not use CLING_CXXABI*
#else
# define CLING_CXXABIV -1 // intentionally invalid macro name
# define CLING_CXXABIS "-1" // intentionally invalid macro name
    llvm::errs()
      << "Warning in cling::CIFactory::createCI():\n  "
      "C++ ABI check not implemented for this standard library\n";
    return;
#endif
#ifdef _MSC_VER
    HKEY regVS;
    int VSVersion = (_MSC_VER / 100) - 6;
    std::stringstream subKey;
    subKey << "VisualStudio.DTE." << VSVersion << ".0";
    if (RegOpenKeyEx(HKEY_CLASSES_ROOT, subKey.str().c_str(), 0, KEY_READ, &regVS) == ERROR_SUCCESS) {
      RegCloseKey(regVS);
    }
    else {
      llvm::errs()
        << "Warning in cling::CIFactory::createCI():\n  "
        "Possible C++ standard library mismatch, compiled with Visual Studio v"
        << VSVersion << ".0,\n"
        "but this version of Visual Studio was not found in your system's registry.\n";
    }
#else

  struct EarlyReturnWarn {
    bool shouldWarn = true;
    ~EarlyReturnWarn() {
      if (shouldWarn) {
        llvm::errs()
          << "Warning in cling::IncrementalParser::CheckABICompatibility():\n  "
             "Possible C++ standard library mismatch, compiled with "
             CLING_CXXABIS " v" << CLING_CXXABIV
          << " but extraction of runtime standard library version failed.\n";
      }
    }
  } warnAtReturn;
  clang::Preprocessor& PP = CI->getPreprocessor();
  clang::IdentifierInfo* II = PP.getIdentifierInfo(CLING_CXXABIS);
  if (!II)
    return;
  const clang::DefMacroDirective* MD
    = llvm::dyn_cast<clang::DefMacroDirective>(PP.getMacroDirective(II));
  if (!MD)
    return;
  const clang::MacroInfo* MI = MD->getMacroInfo();
  if (!MI || MI->getNumTokens() != 1)
    return;
  const clang::Token& Tok = *MI->tokens_begin();
  if (!Tok.isLiteral())
    return;
  if (!Tok.getLength() || !Tok.getLiteralData())
    return;

  std::string cxxabivStr;
  {
     llvm::raw_string_ostream cxxabivStrStrm(cxxabivStr);
     cxxabivStrStrm << CLING_CXXABIV;
  }
  llvm::StringRef tokStr(Tok.getLiteralData(), Tok.getLength());

  warnAtReturn.shouldWarn = false;
  if (!tokStr.equals(cxxabivStr)) {
    llvm::errs()
      << "Warning in cling::IncrementalParser::CheckABICompatibility():\n  "
        "C++ ABI mismatch, compiled with "
        CLING_CXXABIS " v" << CLING_CXXABIV
      << " running with v" << tokStr << "\n";
  }
#endif
#undef CLING_CXXABIV
#undef CLING_CXXABIS
  }
} // unnamed namespace

namespace cling {
  IncrementalParser::IncrementalParser(Interpreter* interp,
                                       int argc, const char* const *argv,
                                       const char* llvmdir):
    m_Interpreter(interp), m_Consumer(0), m_ModuleNo(0) {

    CompilerInstance* CI = CIFactory::createCI("", argc, argv, llvmdir);
    assert(CI && "CompilerInstance is (null)!");

    m_Consumer = dyn_cast<DeclCollector>(&CI->getSema().getASTConsumer());
    assert(m_Consumer && "Expected ChainedConsumer!");

    m_CI.reset(CI);

    if (CI->getFrontendOpts().ProgramAction != clang::frontend::ParseSyntaxOnly){
      m_CodeGen.reset(CreateLLVMCodeGen(CI->getDiagnostics(), "cling-module-0",
                                        CI->getCodeGenOpts(),
                                        *m_Interpreter->getLLVMContext()
                                        ));
      m_Consumer->setContext(this, m_CodeGen.get());
    } else {
      m_Consumer->setContext(this, 0);
    }

    initializeVirtualFile();

    // Add transformers to the IncrementalParser, which owns them
    Sema* TheSema = &CI->getSema();
    // Register the AST Transformers
    typedef std::unique_ptr<ASTTransformer> ASTTPtr_t;
    std::vector<ASTTPtr_t> ASTTransformers;
    ASTTransformers.emplace_back(new AutoSynthesizer(TheSema));
    ASTTransformers.emplace_back(new EvaluateTSynthesizer(TheSema));

    typedef std::unique_ptr<WrapperTransformer> WTPtr_t;
    std::vector<WTPtr_t> WrapperTransformers;
    WrapperTransformers.emplace_back(new ValuePrinterSynthesizer(TheSema, 0));
    WrapperTransformers.emplace_back(new DeclExtractor(TheSema));
    WrapperTransformers.emplace_back(new ValueExtractionSynthesizer(TheSema));
    WrapperTransformers.emplace_back(new NullDerefProtectionTransformer(TheSema));
    WrapperTransformers.emplace_back(new CheckEmptyTransactionTransformer(TheSema));

    m_Consumer->SetTransformers(std::move(ASTTransformers),
                                std::move(WrapperTransformers));
  }

  void
  IncrementalParser::Initialize(llvm::SmallVectorImpl<ParseResultTransaction>&
                                result) {
    m_TransactionPool.reset(new TransactionPool(getCI()->getSema()));
    if (hasCodeGenerator()) {
      getCodeGenerator()->Initialize(getCI()->getASTContext());
      m_BackendPasses.reset(new BackendPasses(getCI()->getCodeGenOpts(),
                                              getCI()->getTargetOpts(),
                                              getCI()->getLangOpts()));
    }

    CompilationOptions CO;
    CO.DeclarationExtraction = 0;
    CO.ValuePrinting = CompilationOptions::VPDisabled;
    CO.CodeGeneration = hasCodeGenerator();

    // pull in PCHs
    const std::string& PCHFileName
      = m_CI->getInvocation ().getPreprocessorOpts().ImplicitPCHInclude;
    if (!PCHFileName.empty()) {
      Transaction* CurT = beginTransaction(CO);
      m_CI->createPCHExternalASTSource(PCHFileName,
                                       true /*DisablePCHValidation*/,
                                       true /*AllowPCHWithCompilerErrors*/,
                                       0 /*DeserializationListener*/,
                                       true /*OwnsDeserializationListener*/);
      result.push_back(endTransaction(CurT));
    }

    Transaction* CurT = beginTransaction(CO);
    Sema* TheSema = &m_CI->getSema();
    Preprocessor& PP = m_CI->getPreprocessor();
    addClingPragmas(*m_Interpreter);
    m_Parser.reset(new Parser(PP, *TheSema,
                              false /*skipFuncBodies*/));
    PP.EnterMainSourceFile();
    // Initialize the parser after we have entered the main source file.
    m_Parser->Initialize();
    // Perform initialization that occurs after the parser has been initialized
    // but before it parses anything. Initializes the consumers too.
    // No - already done by m_Parser->Initialize().
    // TheSema->Initialize();

    ExternalASTSource *External = TheSema->getASTContext().getExternalSource();
    if (External)
      External->StartTranslationUnit(m_Consumer);

    if (m_CI->getLangOpts().CPlusPlus) {
      // <new> is needed by the ValuePrinter so it's a good thing to include it.
      // We need to include it to determine the version number of the standard
      // library implementation.
      ParseInternal("#include <new>");
      // That's really C++ ABI compatibility. C has other problems ;-)
      CheckABICompatibility(m_CI.get());
    }

    // DO NOT commit the transactions here: static initialization in these
    // transactions requires gCling through local_cxa_atexit(), but that has not
    // been defined yet!
    ParseResultTransaction PRT = endTransaction(CurT);
    result.push_back(PRT);
  }

  const Transaction* IncrementalParser::getCurrentTransaction() const {
    return m_Consumer->getTransaction();
  }

  SourceLocation IncrementalParser::getLastMemoryBufferEndLoc() const {
    const SourceManager& SM = getCI()->getSourceManager();
    SourceLocation Result = SM.getLocForStartOfFile(m_VirtualFileID);
    return Result.getLocWithOffset(m_MemoryBuffers.size() + 1);
  }

  IncrementalParser::~IncrementalParser() {
    const Transaction* T = getFirstTransaction();
    const Transaction* nextT = 0;
    while (T) {
      assert((T->getState() == Transaction::kCommitted
              || T->getState() == Transaction::kRolledBackWithErrors
              || T->getState() == Transaction::kNumStates // reset from the pool
              || T->getState() == Transaction::kRolledBack)
             && "Not committed?");
      nextT = T->getNext();
      delete T;
      T = nextT;
    }
  }

  void IncrementalParser::addTransaction(Transaction* T) {
    if (!T->isNestedTransaction() && T != getLastTransaction()) {
      if (getLastTransaction())
        m_Transactions.back()->setNext(T);
      m_Transactions.push_back(T);
    }
  }


  Transaction* IncrementalParser::beginTransaction(const CompilationOptions&
                                                   Opts) {
    Transaction* OldCurT = m_Consumer->getTransaction();
    Transaction* NewCurT = m_TransactionPool->takeTransaction();
    NewCurT->setCompilationOpts(Opts);
    // If we are in the middle of transaction and we see another begin
    // transaction - it must be nested transaction.
    if (OldCurT && OldCurT != NewCurT
        && (OldCurT->getState() == Transaction::kCollecting
            || OldCurT->getState() == Transaction::kCompleted)) {
      OldCurT->addNestedTransaction(NewCurT); // takes the ownership
    }

    m_Consumer->setTransaction(NewCurT);
    return NewCurT;
  }

  IncrementalParser::ParseResultTransaction
  IncrementalParser::endTransaction(Transaction* T) {
    assert(T && "Null transaction!?");
    assert(T->getState() == Transaction::kCollecting);

#ifndef NDEBUG
    if (T->hasNestedTransactions()) {
      for(Transaction::const_nested_iterator I = T->nested_begin(),
            E = T->nested_end(); I != E; ++I)
        assert((*I)->isCompleted() && "Nested transaction not completed!?");
    }
#endif

    T->setState(Transaction::kCompleted);

    DiagnosticsEngine& Diags = getCI()->getSema().getDiagnostics();

    //TODO: Make the enum orable.
    EParseResult ParseResult = kSuccess;

    if (Diags.hasErrorOccurred() || Diags.hasFatalErrorOccurred()
        || T->getIssuedDiags() == Transaction::kErrors) {
      T->setIssuedDiags(Transaction::kErrors);
      ParseResult = kFailed;
    } else if (Diags.getNumWarnings() > 0) {
      T->setIssuedDiags(Transaction::kWarnings);
      ParseResult = kSuccessWithWarnings;
    }

    if (ParseResult != kSuccess) {
      // Now that we have captured the error, reset the Diags.
      Diags.Reset(/*soft=*/true);
      Diags.getClient()->clear();
    }

    // Empty transaction send it back to the pool.
    if (T->empty()) {
      assert((!m_Consumer->getTransaction()
              || (m_Consumer->getTransaction() == T))
             && "Cannot release different T");
      // If a nested transaction the active one should be its parent
      // from now on. FIXME: Merge conditional with commitTransaction
      if (T->isNestedTransaction())
        m_Consumer->setTransaction(T->getParent());
      else
        m_Consumer->setTransaction((Transaction*)0);

      m_TransactionPool->releaseTransaction(T);
      return ParseResultTransaction(nullptr, ParseResult);
    }

    addTransaction(T);
    return ParseResultTransaction(T, ParseResult);
  }

  void IncrementalParser::commitTransaction(ParseResultTransaction PRT) {
    Transaction* T = PRT.getPointer();
    if (!T)
      return;

    assert(T->isCompleted() && "Transaction not ended!?");
    assert(T->getState() != Transaction::kCommitted
           && "Committing an already committed transaction.");
    assert((T->getIssuedDiags() == Transaction::kErrors || !T->empty())
           && "Valid Transactions must not be empty;");

    // If committing a nested transaction the active one should be its parent
    // from now on.
    if (T->isNestedTransaction())
      m_Consumer->setTransaction(T->getParent());

    // Check for errors...
    if (T->getIssuedDiags() == Transaction::kErrors) {
      // Make module visible to TransactionUnloader.
      bool MustStartNewModule = false;
      if (!T->isNestedTransaction() && hasCodeGenerator()) {
        MustStartNewModule = true;
        std::unique_ptr<llvm::Module> M(getCodeGenerator()->ReleaseModule());

        if (M) {
          T->setModule(std::move(M));
        }
      }
      rollbackTransaction(T);

      if (MustStartNewModule) {
        // Create a new module.
        std::string ModuleName;
        {
          llvm::raw_string_ostream strm(ModuleName);
          strm << "cling-module-" << ++m_ModuleNo;
        }
        getCodeGenerator()->StartModule(ModuleName,
                                        *m_Interpreter->getLLVMContext(),
                                        getCI()->getCodeGenOpts());
      }
      return;
    }

    if (T->hasNestedTransactions()) {
      Transaction* TopmostParent = T->getTopmostParent();
      EParseResult PR = kSuccess;
      if (TopmostParent->getIssuedDiags() == Transaction::kErrors)
        PR = kFailed;
      else if (TopmostParent->getIssuedDiags() == Transaction::kWarnings)
        PR = kSuccessWithWarnings;

      for (Transaction::const_nested_iterator I = T->nested_begin(),
            E = T->nested_end(); I != E; ++I)
        if ((*I)->getState() != Transaction::kCommitted)
          commitTransaction(ParseResultTransaction(*I, PR));
    }

    // If there was an error coming from the transformers.
    if (T->getIssuedDiags() == Transaction::kErrors) {
      rollbackTransaction(T);
      return;
    }

    // Here we expect a template instantiation. We need to open the transaction
    // that we are currently work with.
    {
      Transaction* prevConsumerT = m_Consumer->getTransaction();
      m_Consumer->setTransaction(T);
      Transaction* nestedT = beginTransaction(CompilationOptions());
      // Pull all template instantiations in that came from the consumers.
      getCI()->getSema().PerformPendingInstantiations();
      ParseResultTransaction PRT = endTransaction(nestedT);
      commitTransaction(PRT);
      m_Consumer->setTransaction(prevConsumerT);
    }
    m_Consumer->HandleTranslationUnit(getCI()->getASTContext());


    // The static initializers might run anything and can thus cause more
    // decls that need to end up in a transaction. But this one is done
    // with CodeGen...
    if (T->getCompilationOpts().CodeGeneration && hasCodeGenerator()) {
      Transaction* prevConsumerT = m_Consumer->getTransaction();
      m_Consumer->setTransaction(T);
      codeGenTransaction(T);
      transformTransactionIR(T);
      T->setState(Transaction::kCommitted);
      if (!T->getParent()) {
        if (m_Interpreter->executeTransaction(*T)
            >= Interpreter::kExeFirstError) {
          // Roll back on error in initializers
          //assert(0 && "Error on inits.");
          rollbackTransaction(T);
          T->setState(Transaction::kRolledBackWithErrors);
          return;
        }
      }
      m_Consumer->setTransaction(prevConsumerT);
    }
    T->setState(Transaction::kCommitted);

    if (InterpreterCallbacks* callbacks = m_Interpreter->getCallbacks())
      callbacks->TransactionCommitted(*T);

  }

  void IncrementalParser::markWholeTransactionAsUsed(Transaction* T) const {
    ASTContext& C = m_CI->getASTContext();
    for (Transaction::const_iterator I = T->decls_begin(), E = T->decls_end();
         I != E; ++I) {
      // Copy DCI; it might get relocated below.
      Transaction::DelayCallInfo DCI = *I;
      // FIXME: implement for multiple decls in a DGR.
      assert(DCI.m_DGR.isSingleDecl());
      Decl* D = DCI.m_DGR.getSingleDecl();
      if (!D->hasAttr<clang::UsedAttr>())
        D->addAttr(::new (D->getASTContext())
                   clang::UsedAttr(D->getSourceRange(), D->getASTContext(),
                                   0/*AttributeSpellingListIndex*/));
    }
    for (Transaction::iterator I = T->deserialized_decls_begin(),
           E = T->deserialized_decls_end(); I != E; ++I) {
      // FIXME: implement for multiple decls in a DGR.
      assert(I->m_DGR.isSingleDecl());
      Decl* D = I->m_DGR.getSingleDecl();
      if (!D->hasAttr<clang::UsedAttr>())
        D->addAttr(::new (C) clang::UsedAttr(D->getSourceRange(), C,
                                   0/*AttributeSpellingListIndex*/));
    }
  }

  void IncrementalParser::emitTransaction(Transaction* T) {
    for (auto DI = T->decls_begin(), DE = T->decls_end(); DI != DE; ++DI)
      m_Consumer->HandleTopLevelDecl(DI->m_DGR);
  }

  void IncrementalParser::codeGenTransaction(Transaction* T) {
    // codegen the transaction
    assert(T->getCompilationOpts().CodeGeneration && "CodeGen turned off");
    assert(T->getState() == Transaction::kCompleted && "Must be completed");
    assert(hasCodeGenerator() && "No CodeGen");

    // Could trigger derserialization of decls.
    Transaction* deserT = beginTransaction(CompilationOptions());


    // Commit this transaction first - T might need symbols from it, so
    // trigger emission of weak symbols by providing use.
    ParseResultTransaction PRT = endTransaction(deserT);
    commitTransaction(PRT);

    // This llvm::Module is done; finalize it and pass it to the execution
    // engine.
    if (!T->isNestedTransaction() && hasCodeGenerator()) {
      // The initializers are emitted to the symbol "_GLOBAL__sub_I_" + filename.
      // Make that unique!
      ASTContext& Context = getCI()->getASTContext();
      SourceManager &SM = Context.getSourceManager();
      const FileEntry *MainFile = SM.getFileEntryForID(SM.getMainFileID());
      FileEntry* NcMainFile = const_cast<FileEntry*>(MainFile);
      // Hack to temporarily set the file entry's name to a unique name.
      assert(MainFile->getName() == *(const char**)NcMainFile
         && "FileEntry does not start with the name");
      const char* &FileName = *(const char**)NcMainFile;
      const char* OldName = FileName;
      std::string ModName = getCodeGenerator()->GetModule()->getName().str();
      FileName = ModName.c_str();

      deserT = beginTransaction(CompilationOptions());
      // Reset the module builder to clean up global initializers, c'tors, d'tors
      getCodeGenerator()->HandleTranslationUnit(Context);
      FileName = OldName;
      commitTransaction(endTransaction(deserT));

      std::unique_ptr<llvm::Module> M(getCodeGenerator()->ReleaseModule());

      if (M) {
        m_Interpreter->addModule(M.get());
        T->setModule(std::move(M));
      }

      // Create a new module.
      std::string ModuleName;
      {
        llvm::raw_string_ostream strm(ModuleName);
        strm << "cling-module-" << ++m_ModuleNo;
      }
      getCodeGenerator()->StartModule(ModuleName,
                                      *m_Interpreter->getLLVMContext(),
                                      getCI()->getCodeGenOpts());
    }
  }

  bool IncrementalParser::transformTransactionIR(Transaction* T) {
    // Transform IR
    bool success = true;
    if (!success)
      rollbackTransaction(T);
    if (m_BackendPasses && T->getModule())
      m_BackendPasses->runOnModule(*T->getModule());
    return success;
  }

  void IncrementalParser::rollbackTransaction(Transaction* T) {
    assert(T && "Must have value");
    // We can revert the most recent transaction or a nested transaction of a
    // transaction that is not in the middle of the transaction collection
    // (i.e. at the end or not yet added to the collection at all).
    assert(!T->getTopmostParent()->getNext() &&
           "Can not revert previous transactions");
    assert((T->getState() != Transaction::kRolledBack ||
            T->getState() != Transaction::kRolledBackWithErrors) &&
           "Transaction already rolled back.");
    if (m_Interpreter->getOptions().ErrorOut)
      return;

    TransactionUnloader U(&getCI()->getSema(), m_CodeGen.get());

    if (U.RevertTransaction(T))
      T->setState(Transaction::kRolledBack);
    else
      T->setState(Transaction::kRolledBackWithErrors);

    if (!T->getParent()) {
      // Remove from the queue
      m_Transactions.pop_back();
      if (!m_Transactions.empty())
        m_Transactions.back()->setNext(0);
    }
    //m_TransactionPool->releaseTransaction(T);
  }

  std::vector<const Transaction*> IncrementalParser::getAllTransactions() {
    std::vector<const Transaction*> result(m_Transactions.size());
    const cling::Transaction* T = getFirstTransaction();
    while (T) {
      result.push_back(T);
      T = T->getNext();
    }
    return result;
  }

  // Each input line is contained in separate memory buffer. The SourceManager
  // assigns sort-of invalid FileID for each buffer, i.e there is no FileEntry
  // for the MemoryBuffer's FileID. That in turn is problem because invalid
  // SourceLocations are given to the diagnostics. Thus the diagnostics cannot
  // order the overloads, for example
  //
  // Our work-around is creating a virtual file, which doesn't exist on the disk
  // with enormous size (no allocation is done). That file has valid FileEntry
  // and so on... We use it for generating valid SourceLocations with valid
  // offsets so that it doesn't cause any troubles to the diagnostics.
  //
  // +---------------------+
  // | Main memory buffer  |
  // +---------------------+
  // |  Virtual file SLoc  |
  // |    address space    |<-----------------+
  // |         ...         |<------------+    |
  // |         ...         |             |    |
  // |         ...         |<----+       |    |
  // |         ...         |     |       |    |
  // +~~~~~~~~~~~~~~~~~~~~~+     |       |    |
  // |     input_line_1    | ....+.......+..--+
  // +---------------------+     |       |
  // |     input_line_2    | ....+.....--+
  // +---------------------+     |
  // |          ...        |     |
  // +---------------------+     |
  // |     input_line_N    | ..--+
  // +---------------------+
  //
  void IncrementalParser::initializeVirtualFile() {
    SourceManager& SM = getCI()->getSourceManager();
    m_VirtualFileID = SM.getMainFileID();
    assert(!m_VirtualFileID.isInvalid() && "No VirtualFileID created?");
  }

  IncrementalParser::ParseResultTransaction
  IncrementalParser::Compile(llvm::StringRef input,
                             const CompilationOptions& Opts) {
    Transaction* CurT = beginTransaction(Opts);
    EParseResult ParseRes = ParseInternal(input);

    if (ParseRes == kSuccessWithWarnings)
      CurT->setIssuedDiags(Transaction::kWarnings);
    else if (ParseRes == kFailed)
      CurT->setIssuedDiags(Transaction::kErrors);

    ParseResultTransaction PRT = endTransaction(CurT);
    commitTransaction(PRT);

    return PRT;
  }

  IncrementalParser::ParseResultTransaction
  IncrementalParser::Parse(llvm::StringRef input,
                           const CompilationOptions& Opts) {
    Transaction* CurT = beginTransaction(Opts);
    ParseInternal(input);
    return endTransaction(CurT);
  }

  // Add the input to the memory buffer, parse it, and add it to the AST.
  IncrementalParser::EParseResult
  IncrementalParser::ParseInternal(llvm::StringRef input) {
    if (input.empty()) return IncrementalParser::kSuccess;

    Sema& S = getCI()->getSema();

    const CompilationOptions& CO
       = m_Consumer->getTransaction()->getCompilationOpts();

    assert(!(S.getLangOpts().Modules
             && CO.CodeGenerationForModule)
           && "CodeGenerationForModule to be removed once PCMs are available!");

    // Recover resources if we crash before exiting this method.
    llvm::CrashRecoveryContextCleanupRegistrar<Sema> CleanupSema(&S);

    Preprocessor& PP = m_CI->getPreprocessor();
    if (!PP.getCurrentLexer()) {
       PP.EnterSourceFile(m_CI->getSourceManager().getMainFileID(),
                          0, SourceLocation());
    }
    assert(PP.isIncrementalProcessingEnabled() && "Not in incremental mode!?");
    PP.enableIncrementalProcessing();

    std::ostringstream source_name;
    source_name << "input_line_" << (m_MemoryBuffers.size() + 1);

    // Create an uninitialized memory buffer, copy code in and append "\n"
    size_t InputSize = input.size(); // don't include trailing 0
    // MemBuffer size should *not* include terminating zero
    std::unique_ptr<llvm::MemoryBuffer>
      MB(llvm::MemoryBuffer::getNewUninitMemBuffer(InputSize + 1,
                                                   source_name.str()));
    char* MBStart = const_cast<char*>(MB->getBufferStart());
    memcpy(MBStart, input.data(), InputSize);
    memcpy(MBStart + InputSize, "\n", 2);

    SourceManager& SM = getCI()->getSourceManager();

    // Create SourceLocation, which will allow clang to order the overload
    // candidates for example
    SourceLocation NewLoc = getLastMemoryBufferEndLoc().getLocWithOffset(1);

    llvm::MemoryBuffer* MBNonOwn = MB.get();
    // Create FileID for the current buffer
    FileID FID = SM.createFileID(std::move(MB), SrcMgr::C_User,
                                 /*LoadedID*/0,
                                 /*LoadedOffset*/0, NewLoc);

    m_MemoryBuffers.push_back(std::make_pair(MBNonOwn, FID));

    PP.EnterSourceFile(FID, /*DirLookup*/0, NewLoc);
    m_Consumer->getTransaction()->setBufferFID(FID);

    DiagnosticsEngine& Diags = getCI()->getDiagnostics();

    bool IgnorePromptDiags = CO.IgnorePromptDiags;
    if (IgnorePromptDiags) {
      // Disable warnings which doesn't make sense when using the prompt
      // This gets reset with the clang::Diagnostics().Reset(/*soft*/=false)
      // using clang's API we simulate:
      // #pragma warning push
      // #pragma warning ignore ...
      // #pragma warning ignore ...
      // #pragma warning pop
      SourceLocation Loc = SM.getLocForStartOfFile(FID);
      Diags.pushMappings(Loc);
      // The source locations of #pragma warning ignore must be greater than
      // the ones from #pragma push

      auto setIgnore = [&](clang::diag::kind Diag) {
        Diags.setSeverity(Diag, diag::Severity::Ignored, SourceLocation());
      };

      setIgnore(clang::diag::warn_unused_expr);
      setIgnore(clang::diag::warn_unused_call);
      setIgnore(clang::diag::warn_unused_comparison);
      setIgnore(clang::diag::ext_return_has_expr);
    }
    auto setError = [&](clang::diag::kind Diag) {
      Diags.setSeverity(Diag, diag::Severity::Error, SourceLocation());
    };
    setError(clang::diag::warn_falloff_nonvoid_function);

    Sema::SavePendingInstantiationsRAII SavedPendingInstantiations(S);

    Parser::DeclGroupPtrTy ADecl;
    while (!m_Parser->ParseTopLevelDecl(ADecl)) {
      // If we got a null return and something *was* parsed, ignore it.  This
      // is due to a top-level semicolon, an action override, or a parse error
      // skipping something.
      if (Diags.hasErrorOccurred() || Diags.hasFatalErrorOccurred())
        m_Consumer->getTransaction()->setIssuedDiags(Transaction::kErrors);
      if (ADecl)
        m_Consumer->HandleTopLevelDecl(ADecl.get());
    };

#ifdef LLVM_ON_WIN32
    // Microsoft-specific:
    // Late parsed templates can leave unswallowed "macro"-like tokens.
    // They will seriously confuse the Parser when entering the next
    // source file. So lex until we are EOF.
    Token Tok;
    do {
      PP.Lex(Tok);
    } while (Tok.isNot(tok::eof));
#endif

#ifndef NDEBUG
    Token AssertTok;
    PP.Lex(AssertTok);
    assert(AssertTok.is(tok::eof) && "Lexer must be EOF when starting incremental parse!");
#endif

    if (IgnorePromptDiags) {
      SourceLocation Loc = SM.getLocForEndOfFile(m_MemoryBuffers.back().second);
      Diags.popMappings(Loc);
    }

    // Process any TopLevelDecls generated by #pragma weak.
    for (llvm::SmallVector<Decl*,2>::iterator I = S.WeakTopLevelDecls().begin(),
         E = S.WeakTopLevelDecls().end(); I != E; ++I) {
      m_Consumer->HandleTopLevelDecl(DeclGroupRef(*I));
    }

    if (m_Consumer->getTransaction()->getIssuedDiags() == Transaction::kErrors)
      return kFailed;
    else if (Diags.getNumWarnings())
      return kSuccessWithWarnings;

    return kSuccess;
  }

  void IncrementalParser::printTransactionStructure() const {
    for(size_t i = 0, e = m_Transactions.size(); i < e; ++i) {
      m_Transactions[i]->printStructureBrief();
    }
  }


} // namespace cling
