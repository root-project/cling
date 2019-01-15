//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Axel Naumann <axel@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "IncrementalParser.h"

#include "ASTTransformer.h"
#include "AutoSynthesizer.h"
#include "BackendPasses.h"
#include "CheckEmptyTransactionTransformer.h"
#include "ClingPragmas.h"
#include "DeclCollector.h"
#include "DeclExtractor.h"
#include "DefinitionShadower.h"
#include "DynamicLookup.h"
#include "IncrementalCUDADeviceCompiler.h"
#include "IncrementalExecutor.h"
#include "NullDerefProtectionTransformer.h"
#include "TransactionPool.h"
#include "ValueExtractionSynthesizer.h"
#include "ValuePrinterSynthesizer.h"
#include "cling/Interpreter/CIFactory.h"
#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/InterpreterCallbacks.h"
#include "cling/Interpreter/Transaction.h"
#include "cling/Utils/Diagnostics.h"
#include "cling/Utils/Output.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclGroup.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/FileManager.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Frontend/MultiplexConsumer.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Parse/Parser.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "clang/Serialization/ASTWriter.h"
#include "clang/Serialization/ASTReader.h"
#include "llvm/Support/Path.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CrashRecoveryContext.h"
#include "llvm/Support/MemoryBuffer.h"

#include <stdio.h>

using namespace clang;

namespace {

  ///\brief Check the compile-time C++ ABI version vs the run-time ABI version,
  /// a mismatch could cause havoc. Reports if ABI versions differ.
  static bool CheckABICompatibility(cling::Interpreter& Interp) {
#if defined(__GLIBCXX__)
    #define CLING_CXXABI_VERS       std::to_string(__GLIBCXX__)
    const char* CLING_CXXABI_NAME = "__GLIBCXX__";
    static constexpr bool CLING_CXXABI_BACKWARDCOMP = true;
#elif defined(_LIBCPP_VERSION)
    #define CLING_CXXABI_VERS       std::to_string(_LIBCPP_ABI_VERSION)
    const char* CLING_CXXABI_NAME = "_LIBCPP_ABI_VERSION";
    static constexpr bool CLING_CXXABI_BACKWARDCOMP = false;
#elif defined(_CRT_MSVCP_CURRENT)
    #define CLING_CXXABI_VERS        _CRT_MSVCP_CURRENT
    const char* CLING_CXXABI_NAME = "_CRT_MSVCP_CURRENT";
    static constexpr bool CLING_CXXABI_BACKWARDCOMP = false;
#else
    #error "Unknown platform for ABI check";
#endif

    const std::string CurABI = Interp.getMacroValue(CLING_CXXABI_NAME);
    if (CurABI == CLING_CXXABI_VERS)
      return true;
    if (CurABI.empty()) {
    cling::errs() <<
      "Warning in cling::IncrementalParser::CheckABICompatibility():\n"
      "  Failed to extract C++ standard library version.\n";
    }

    if (CLING_CXXABI_BACKWARDCOMP && CurABI < CLING_CXXABI_VERS) {
       // Backward compatible ABIs allow us to interpret old headers
       // against a newer stdlib.so.
       return true;
    }

    cling::errs() <<
      "Warning in cling::IncrementalParser::CheckABICompatibility():\n"
      "  Possible C++ standard library mismatch, compiled with "
      << CLING_CXXABI_NAME << " '" << CLING_CXXABI_VERS << "'\n"
      "  Extraction of runtime standard library version was: '"
      << CurABI << "'\n";

    return false;
  }

  class FilteringDiagConsumer : public cling::utils::DiagnosticsOverride {
    std::stack<bool> m_IgnorePromptDiags;

    void SyncDiagCountWithTarget() {
      NumWarnings = m_PrevClient.getNumWarnings();
      NumErrors = m_PrevClient.getNumErrors();
    }

    void BeginSourceFile(const LangOptions &LangOpts,
                         const Preprocessor *PP=nullptr) override {
      m_PrevClient.BeginSourceFile(LangOpts, PP);
    }

    void EndSourceFile() override {
      m_PrevClient.EndSourceFile();
      SyncDiagCountWithTarget();
    }

    void finish() override {
      m_PrevClient.finish();
      SyncDiagCountWithTarget();
    }

    void clear() override {
      m_PrevClient.clear();
      SyncDiagCountWithTarget();
    }

    bool IncludeInDiagnosticCounts() const override {
      return m_PrevClient.IncludeInDiagnosticCounts();
    }

    void HandleDiagnostic(DiagnosticsEngine::Level DiagLevel,
                          const Diagnostic &Info) override {
      if (Ignoring()) {
        if (Info.getID() == diag::warn_unused_expr
            || Info.getID() == diag::warn_unused_call
            || Info.getID() == diag::warn_unused_comparison)
          return; // ignore!
        if (Info.getID() == diag::warn_falloff_nonvoid_function) {
          DiagLevel = DiagnosticsEngine::Error;
        }
        if (Info.getID() == diag::ext_return_has_expr) {
          // An error that we need to suppress.
          auto Diags = const_cast<DiagnosticsEngine*>(Info.getDiags());
          assert(Diags->hasErrorOccurred() && "Expected ErrorOccurred");
          if (m_PrevClient.getNumErrors() == 0) { // first error
            Diags->Reset(true /*soft - only counts, not mappings*/);
          } // else we had other errors, too.
          return; // ignore!
        }
      }
      m_PrevClient.HandleDiagnostic(DiagLevel, Info);
      SyncDiagCountWithTarget();
    }

    bool Ignoring() const {
      return !m_IgnorePromptDiags.empty() && m_IgnorePromptDiags.top();
    }

  public:
    FilteringDiagConsumer(DiagnosticsEngine& Diags, bool Own) :
      DiagnosticsOverride(Diags, Own) {
    }

    struct RAAI {
      FilteringDiagConsumer& m_Client;
      RAAI(DiagnosticConsumer& F, bool Ignore) :
       m_Client(static_cast<FilteringDiagConsumer&>(F)) {
        m_Client.m_IgnorePromptDiags.push(Ignore);
      }
      ~RAAI() { m_Client.m_IgnorePromptDiags.pop(); }
    };
  };
} // unnamed namespace

static void HandlePlugins(CompilerInstance& CI,
                         std::vector<std::unique_ptr<ASTConsumer>>& Consumers) {
  // Copied from Frontend/FrontendAction.cpp.
  // FIXME: Remove when we switch to a tools-based cling driver.

  // If the FrontendPluginRegistry has plugins before loading any shared library
  // this means we have linked our plugins. This is useful when cling runs in
  // embedded mode (in a shared library). This is the only feasible way to have
  // plugins if cling is in a single shared library which is dlopen-ed with
  // RTLD_LOCAL. In that situation plugins can still find the cling, clang and
  // llvm symbols opened with local visibility.
  if (FrontendPluginRegistry::begin() == FrontendPluginRegistry::end()) {
    for (const std::string& Path : CI.getFrontendOpts().Plugins) {
      std::string Err;
      if (llvm::sys::DynamicLibrary::LoadLibraryPermanently(Path.c_str(), &Err))
        CI.getDiagnostics().Report(clang::diag::err_fe_unable_to_load_plugin)
          << Path << Err;
    }
    // If we are not statically linked, we should register the pragmas ourselves
    // because the dlopen happens after creating the clang::Preprocessor which
    // calls RegisterBuiltinPragmas.
    // FIXME: This can be avoided by refactoring our routine and moving it to
    // the CIFactory. This requires an abstraction which allows us to
    // conditionally create MultiplexingConsumers.

    // Copied from Lex/Pragma.cpp
    // Pragmas added by plugins
    for (PragmaHandlerRegistry::iterator it = PragmaHandlerRegistry::begin(),
           ie = PragmaHandlerRegistry::end(); it != ie; ++it)
      CI.getPreprocessor().AddPragmaHandler(it->instantiate().release());
  }

  for (auto it = clang::FrontendPluginRegistry::begin(),
         ie = clang::FrontendPluginRegistry::end();
       it != ie; ++it) {
    std::unique_ptr<clang::PluginASTAction> P(it->instantiate());

    PluginASTAction::ActionType PluginActionType = P->getActionType();
    assert(PluginActionType != clang::PluginASTAction::ReplaceAction);

    if (P->ParseArgs(CI, CI.getFrontendOpts().PluginArgs[it->getName()])) {
      std::unique_ptr<ASTConsumer> PluginConsumer
        = P->CreateASTConsumer(CI, /*InputFile*/ "");
      if (PluginActionType == clang::PluginASTAction::AddBeforeMainAction)
        Consumers.insert(Consumers.begin(), std::move(PluginConsumer));
      else
        Consumers.push_back(std::move(PluginConsumer));
    }
  }
}

namespace cling {
  IncrementalParser::IncrementalParser(Interpreter* interp, const char* llvmdir,
                                   const ModuleFileExtensions& moduleExtensions)
      : m_Interpreter(interp) {
    std::unique_ptr<cling::DeclCollector> consumer;
    consumer.reset(m_Consumer = new cling::DeclCollector());
    m_CI.reset(CIFactory::createCI("", interp->getOptions(), llvmdir,
                                   std::move(consumer), moduleExtensions));

    if (!m_CI) {
      cling::errs() << "Compiler instance could not be created.\n";
      return;
    }
    // Is the CompilerInstance being used to generate output only?
    if (m_Interpreter->getOptions().CompilerOpts.HasOutput)
      return;

    if (!m_Consumer) {
      cling::errs() << "No AST consumer available.\n";
      return;
    }


    std::vector<std::unique_ptr<ASTConsumer>> Consumers;
    HandlePlugins(*m_CI, Consumers);
    std::unique_ptr<ASTConsumer> WrappedConsumer;

    DiagnosticsEngine& Diag = m_CI->getDiagnostics();
    if (m_CI->getFrontendOpts().ProgramAction != frontend::ParseSyntaxOnly) {
      auto CG
        = std::unique_ptr<clang::CodeGenerator>(CreateLLVMCodeGen(Diag,
                                                               makeModuleName(),
                                                    m_CI->getHeaderSearchOpts(),
                                                    m_CI->getPreprocessorOpts(),
                                                         m_CI->getCodeGenOpts(),
                                               *m_Interpreter->getLLVMContext())
                                                );
      m_CodeGen = CG.get();
      assert(m_CodeGen);
      if (!Consumers.empty()) {
        Consumers.push_back(std::move(CG));
        WrappedConsumer.reset(new MultiplexConsumer(std::move(Consumers)));
      }
      else
        WrappedConsumer = std::move(CG);
    }

    // Initialize the DeclCollector and add callbacks keeping track of macros.
    m_Consumer->Setup(this, std::move(WrappedConsumer), m_CI->getPreprocessor());

    m_DiagConsumer.reset(new FilteringDiagConsumer(Diag, false));

    initializeVirtualFile();

    if(m_CI->getFrontendOpts().ProgramAction != frontend::ParseSyntaxOnly &&
      m_Interpreter->getOptions().CompilerOpts.CUDAHost){
        // Create temporary folder for all files, which the CUDA device compiler
        // will generate.
        llvm::SmallString<256> TmpPath;
        llvm::StringRef sep = llvm::sys::path::get_separator().data();
        llvm::sys::path::system_temp_directory(false, TmpPath);
        TmpPath.append(sep.data());
        TmpPath.append("cling-%%%%");
        TmpPath.append(sep.data());

        llvm::SmallString<256> TmpFolder;
        llvm::sys::fs::createUniqueFile(TmpPath.c_str(), TmpFolder);
        llvm::sys::fs::create_directory(TmpFolder);

        // The CUDA fatbin file is the connection beetween the CUDA device
        // compiler and the CodeGen of cling. The file will every time reused.
        if(getCI()->getCodeGenOpts().CudaGpuBinaryFileNames.empty())
          getCI()->getCodeGenOpts().CudaGpuBinaryFileNames.push_back(
            std::string(TmpFolder.c_str()) + "cling.fatbin");

        m_CUDACompiler.reset(
          new IncrementalCUDADeviceCompiler(TmpFolder.c_str(),
                                            m_CI->getCodeGenOpts().OptimizationLevel,
                                            m_Interpreter->getOptions(),
                                            *m_CI));
    }
  }

  bool
  IncrementalParser::Initialize(llvm::SmallVectorImpl<ParseResultTransaction>&
                                result, bool isChildInterpreter) {
    m_TransactionPool.reset(new TransactionPool);
    if (hasCodeGenerator())
      getCodeGenerator()->Initialize(getCI()->getASTContext());

    CompilationOptions CO = m_Interpreter->makeDefaultCompilationOpts();
    Transaction* CurT = beginTransaction(CO);
    Preprocessor& PP = m_CI->getPreprocessor();
    DiagnosticsEngine& Diags = m_CI->getSema().getDiagnostics();

    // Pull in PCH.
    const std::string& PCHFileName
      = m_CI->getInvocation().getPreprocessorOpts().ImplicitPCHInclude;
    if (!PCHFileName.empty()) {
      Transaction* PchT = beginTransaction(CO);
      DiagnosticErrorTrap Trap(Diags);
      m_CI->createPCHExternalASTSource(PCHFileName,
                                       true /*DisablePCHValidation*/,
                                       true /*AllowPCHWithCompilerErrors*/,
                                       0 /*DeserializationListener*/,
                                       true /*OwnsDeserializationListener*/);
      result.push_back(endTransaction(PchT));
      if (Trap.hasErrorOccurred()) {
        result.push_back(endTransaction(CurT));
        return false;
      }
    }

    addClingPragmas(*m_Interpreter);

    // Must happen after attaching the PCH, else PCH elements will end up
    // being lexed.
    PP.EnterMainSourceFile();

    Sema* TheSema = &m_CI->getSema();
    m_Parser.reset(new Parser(PP, *TheSema, false /*skipFuncBodies*/));

    // Initialize the parser after PP has entered the main source file.
    m_Parser->Initialize();

    ExternalASTSource *External = TheSema->getASTContext().getExternalSource();
    if (External)
      External->StartTranslationUnit(m_Consumer);

    // Start parsing the "main file" to warm up lexing (enter caching lex mode
    // for ParseInternal()'s call EnterSourceFile() to make sense.
    while (!m_Parser->ParseTopLevelDecl()) {}

    // If I belong to the parent Interpreter, am using C++, and -noruntime
    // wasn't given on command line, then #include <new> and check ABI
    if (!isChildInterpreter && m_CI->getLangOpts().CPlusPlus &&
        !m_Interpreter->getOptions().NoRuntime) {
      // <new> is needed by the ValuePrinter so it's a good thing to include it.
      // We need to include it to determine the version number of the standard
      // library implementation.
      ParseInternal("#include <new>");
      // That's really C++ ABI compatibility. C has other problems ;-)
      CheckABICompatibility(*m_Interpreter);
    }

    // DO NOT commit the transactions here: static initialization in these
    // transactions requires gCling through local_cxa_atexit(), but that has not
    // been defined yet!
    ParseResultTransaction PRT = endTransaction(CurT);
    result.push_back(PRT);
    return true;
  }

  bool IncrementalParser::isValid(bool initialized) const {
    return m_CI && m_CI->hasFileManager() && m_Consumer
           && !m_VirtualFileID.isInvalid()
           && (!initialized || (m_TransactionPool && m_Parser));
  }

  namespace {
    template <class T>
    struct Reversed {
      const T &m_orig;
      auto begin() -> decltype(m_orig.rbegin()) { return m_orig.rbegin(); }
      auto end() -> decltype (m_orig.rend()) { return m_orig.rend(); }
    };
    template <class T>
    Reversed<T> reverse(const T& orig) { return {orig}; }
  }

  const Transaction* IncrementalParser::getLastWrapperTransaction() const {
    if (auto *T = getCurrentTransaction())
      if (T->getWrapperFD())
        return T;

    for (auto T: reverse(m_Transactions))
      if (T->getWrapperFD())
        return T;
    return nullptr;
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
    Transaction* T = const_cast<Transaction*>(getFirstTransaction());
    while (T) {
      assert((T->getState() == Transaction::kCommitted
              || T->getState() == Transaction::kRolledBackWithErrors
              || T->getState() == Transaction::kNumStates // reset from the pool
              || T->getState() == Transaction::kRolledBack)
             && "Not committed?");
      const Transaction* nextT = T->getNext();
      m_TransactionPool->releaseTransaction(T, false);
      T = const_cast<Transaction*>(nextT);
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
    Transaction* NewCurT = m_TransactionPool->takeTransaction(m_CI->getSema());
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

    DiagnosticsEngine& Diag = getCI()->getSema().getDiagnostics();

    //TODO: Make the enum orable.
    EParseResult ParseResult = kSuccess;

    assert((Diag.hasFatalErrorOccurred() ? Diag.hasErrorOccurred() : true)
            && "Diag.hasFatalErrorOccurred without Diag.hasErrorOccurred !");

    if (Diag.hasErrorOccurred() || T->getIssuedDiags() == Transaction::kErrors) {
      T->setIssuedDiags(Transaction::kErrors);
      ParseResult = kFailed;
    } else if (Diag.getNumWarnings() > 0) {
      T->setIssuedDiags(Transaction::kWarnings);
      ParseResult = kSuccessWithWarnings;
    }

    // Empty transaction, send it back to the pool.
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

  std::string IncrementalParser::makeModuleName() {
    return std::string("cling-module-") + std::to_string(m_ModuleNo++);
  }

  llvm::Module* IncrementalParser::StartModule() {
    return getCodeGenerator()->StartModule(makeModuleName(),
                                           *m_Interpreter->getLLVMContext(),
                                           getCI()->getCodeGenOpts());
  }

  void IncrementalParser::commitTransaction(ParseResultTransaction& PRT,
                                            bool ClearDiagClient) {
    Transaction* T = PRT.getPointer();
    if (!T) {
      if (PRT.getInt() != kSuccess) {
        // Nothing has been emitted to Codegen, reset the Diags.
        DiagnosticsEngine& Diags = getCI()->getSema().getDiagnostics();
        Diags.Reset(/*soft=*/true);
        if (ClearDiagClient)
          Diags.getClient()->clear();
      }
      return;
    }

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
      // Module has been released from Codegen, reset the Diags now.
      DiagnosticsEngine& Diags = getCI()->getSema().getDiagnostics();
      Diags.Reset(/*soft=*/true);
      if (ClearDiagClient)
        Diags.getClient()->clear();

      PRT.setPointer(nullptr);
      PRT.setInt(kFailed);
      m_Interpreter->unload(*T);

      // Create a new module if necessary.
      if (MustStartNewModule)
        StartModule();

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
        if ((*I)->getState() != Transaction::kCommitted) {
          ParseResultTransaction PRT(*I, PR);
          commitTransaction(PRT);
        }
    }

    // If there was an error coming from the transformers.
    if (T->getIssuedDiags() == Transaction::kErrors) {
      m_Interpreter->unload(*T);
      return;
    }

    // Here we expect a template instantiation. We need to open the transaction
    // that we are currently work with.
    {
      Transaction* prevConsumerT = m_Consumer->getTransaction();
      m_Consumer->setTransaction(T);
      Transaction* nestedT = beginTransaction(T->getCompilationOpts());
      // Pull all template instantiations in that came from the consumers.
      getCI()->getSema().PerformPendingInstantiations();
#ifdef LLVM_ON_WIN32
      // Microsoft-specific:
      // Late parsed templates can leave unswallowed "macro"-like tokens.
      // They will seriously confuse the Parser when entering the next
      // source file. So lex until we are EOF.
      Token Tok;
      Tok.setKind(tok::eof);
      do {
        getCI()->getSema().getPreprocessor().Lex(Tok);
      } while (Tok.isNot(tok::eof));
#endif

      ParseResultTransaction nestedPRT = endTransaction(nestedT);
      commitTransaction(nestedPRT);
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
      T->setState(Transaction::kCommitted);
      if (!T->getParent()) {
        if (m_Interpreter->executeTransaction(*T)
            >= Interpreter::kExeFirstError) {
          // Roll back on error in initializers.
          // T maybe pointing to freed memory after this call:
          // Interpreter::unload
          //   IncrementalParser::deregisterTransaction
          //     TransactionPool::releaseTransaction
          m_Interpreter->unload(*T);
          return;
        }
      }
      m_Consumer->setTransaction(prevConsumerT);
    }
    T->setState(Transaction::kCommitted);

    {
      Transaction* prevConsumerT = m_Consumer->getTransaction();
      if (InterpreterCallbacks* callbacks = m_Interpreter->getCallbacks())
        callbacks->TransactionCommitted(*T);
      m_Consumer->setTransaction(prevConsumerT);
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
    deserT = PRT.getPointer();

    // This llvm::Module is done; finalize it and pass it to the execution
    // engine.
    if (!T->isNestedTransaction() && hasCodeGenerator()) {
      // The initializers are emitted to the symbol "_GLOBAL__sub_I_" + filename.
      // Make that unique!
      deserT = beginTransaction(CompilationOptions());
      // Reset the module builder to clean up global initializers, c'tors, d'tors
      getCodeGenerator()->HandleTranslationUnit(getCI()->getASTContext());
      auto PRT = endTransaction(deserT);
      commitTransaction(PRT);
      deserT = PRT.getPointer();

      std::unique_ptr<llvm::Module> M(getCodeGenerator()->ReleaseModule());

      if (M)
        T->setModule(std::move(M));

      if (T->getIssuedDiags() != Transaction::kNone) {
        // Module has been released from Codegen, reset the Diags now.
        DiagnosticsEngine& Diags = getCI()->getSema().getDiagnostics();
        Diags.Reset(/*soft=*/true);
        Diags.getClient()->clear();
      }

      // Create a new module.
      StartModule();
    }
  }

  void IncrementalParser::deregisterTransaction(Transaction& T) {
    if (&T == m_Consumer->getTransaction())
      m_Consumer->setTransaction(T.getParent());

    if (Transaction* Parent = T.getParent()) {
      Parent->removeNestedTransaction(&T);
      T.setParent(0);
    } else {
      // Remove from the queue
      assert(&T == m_Transactions.back() && "Out of order transaction removal");
      m_Transactions.pop_back();
      if (!m_Transactions.empty())
        m_Transactions.back()->setNext(0);
    }

    m_TransactionPool->releaseTransaction(&T);
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
    if (m_VirtualFileID.isInvalid())
      cling::errs() << "VirtualFileID could not be created.\n";
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

  // Add the input to the memory buffer, parse it, and add it to the AST.
  IncrementalParser::EParseResult
  IncrementalParser::ParseInternal(llvm::StringRef input) {
    if (input.empty()) return IncrementalParser::kSuccess;

    Sema& S = getCI()->getSema();

    const CompilationOptions& CO
       = m_Consumer->getTransaction()->getCompilationOpts();

    // Recover resources if we crash before exiting this method.
    llvm::CrashRecoveryContextCleanupRegistrar<Sema> CleanupSema(&S);

    Preprocessor& PP = m_CI->getPreprocessor();
    if (!PP.getCurrentLexer()) {
       PP.EnterSourceFile(m_CI->getSourceManager().getMainFileID(),
                          0, SourceLocation());
    }
    assert(PP.isIncrementalProcessingEnabled() && "Not in incremental mode!?");
    PP.enableIncrementalProcessing();

    smallstream source_name;
    source_name << "input_line_" << (m_MemoryBuffers.size() + 1);

    // Create an uninitialized memory buffer, copy code in and append "\n"
    size_t InputSize = input.size(); // don't include trailing 0
    // MemBuffer size should *not* include terminating zero
    std::unique_ptr<llvm::MemoryBuffer>
      MB(llvm::MemoryBuffer::getNewUninitMemBuffer(InputSize + 1,
                                                   source_name.str()));
    char* MBStart = const_cast<char*>(MB->getBufferStart());
    memcpy(MBStart, input.data(), InputSize);
    MBStart[InputSize] = '\n';

    SourceManager& SM = getCI()->getSourceManager();

    // Create SourceLocation, which will allow clang to order the overload
    // candidates for example
    SourceLocation NewLoc = getLastMemoryBufferEndLoc().getLocWithOffset(1);

    llvm::MemoryBuffer* MBNonOwn = MB.get();

    // Create FileID for the current buffer.
    FileID FID;
    // Create FileEntry and FileID for the current buffer.
    // Enabling the completion point only works on FileEntries.
    const clang::FileEntry* FE
      = SM.getFileManager().getVirtualFile(source_name.str(), InputSize,
                                           0 /* mod time*/);
    SM.overrideFileContents(FE, std::move(MB));
    FID = SM.createFileID(FE, NewLoc, SrcMgr::C_User);
    if (CO.CodeCompletionOffset != -1) {
      // The completion point is set one a 1-based line/column numbering.
      // It relies on the implementation to account for the wrapper extra line.
      PP.SetCodeCompletionPoint(FE, 1/* start point 1-based line*/,
                                CO.CodeCompletionOffset+1/* 1-based column*/);
    }

    m_MemoryBuffers.push_back(std::make_pair(MBNonOwn, FID));

    // NewLoc only used for diags.
    PP.EnterSourceFile(FID, /*DirLookup*/0, NewLoc);
    m_Consumer->getTransaction()->setBufferFID(FID);

    DiagnosticsEngine& Diags = getCI()->getDiagnostics();

    FilteringDiagConsumer::RAAI RAAITmp(*m_DiagConsumer, CO.IgnorePromptDiags);

    DiagnosticErrorTrap Trap(Diags);
    Sema::SavePendingInstantiationsRAII SavedPendingInstantiations(S);

    Parser::DeclGroupPtrTy ADecl;
    while (!m_Parser->ParseTopLevelDecl(ADecl)) {
      // If we got a null return and something *was* parsed, ignore it.  This
      // is due to a top-level semicolon, an action override, or a parse error
      // skipping something.
      if (Trap.hasErrorOccurred())
        m_Consumer->getTransaction()->setIssuedDiags(Transaction::kErrors);
      if (ADecl)
        m_Consumer->HandleTopLevelDecl(ADecl.get());
    };
    // If never entered the while block, there's a chance an error occured
    if (Trap.hasErrorOccurred())
      m_Consumer->getTransaction()->setIssuedDiags(Transaction::kErrors);

    if (CO.CodeCompletionOffset != -1) {
      assert((int)SM.getFileOffset(PP.getCodeCompletionLoc())
             == CO.CodeCompletionOffset
             && "Completion point wrongly set!");
      assert(PP.isCodeCompletionReached()
             && "Code completion set but not reached!");

      // Let's ignore this transaction:
      m_Consumer->getTransaction()->setIssuedDiags(Transaction::kErrors);

      return kSuccess;
    }

#ifdef LLVM_ON_WIN32
    // Microsoft-specific:
    // Late parsed templates can leave unswallowed "macro"-like tokens.
    // They will seriously confuse the Parser when entering the next
    // source file. So lex until we are EOF.
    Token Tok;
    Tok.setKind(tok::eof);
    do {
      PP.Lex(Tok);
    } while (Tok.isNot(tok::eof));
#endif

#ifndef NDEBUG
    Token AssertTok;
    PP.Lex(AssertTok);
    assert(AssertTok.is(tok::eof) && "Lexer must be EOF when starting incremental parse!");
#endif

    // Process any TopLevelDecls generated by #pragma weak.
    for (llvm::SmallVector<Decl*,2>::iterator I = S.WeakTopLevelDecls().begin(),
         E = S.WeakTopLevelDecls().end(); I != E; ++I) {
      m_Consumer->HandleTopLevelDecl(DeclGroupRef(*I));
    }

    if (m_Consumer->getTransaction()->getIssuedDiags() == Transaction::kErrors)
      return kFailed;
    else if (Diags.getNumWarnings())
      return kSuccessWithWarnings;

    if (!m_Interpreter->isInSyntaxOnlyMode() &&
        m_Interpreter->getOptions().CompilerOpts.CUDAHost)
      m_CUDACompiler->compileDeviceCode(input, m_Consumer->getTransaction());

    return kSuccess;
  }

  void IncrementalParser::printTransactionStructure() const {
    for(size_t i = 0, e = m_Transactions.size(); i < e; ++i) {
      m_Transactions[i]->printStructureBrief();
    }
  }

  void IncrementalParser::SetTransformers(bool isChildInterpreter) {
    // Add transformers to the IncrementalParser, which owns them
    Sema* TheSema = &m_CI->getSema();
    // if the interpreter compiles ptx code, some transformers should not used
    bool isCUDADevice = m_Interpreter->getOptions().CompilerOpts.CUDADevice;
    // Register the AST Transformers
    typedef std::unique_ptr<ASTTransformer> ASTTPtr_t;
    std::vector<ASTTPtr_t> ASTTransformers;
    ASTTransformers.emplace_back(new AutoSynthesizer(TheSema));
    ASTTransformers.emplace_back(new EvaluateTSynthesizer(TheSema));
    if (hasCodeGenerator() && !m_Interpreter->getOptions().NoRuntime) {
      // Don't protect against crashes if we cannot run anything.
      // cling might also be in a PCH-generation mode; don't inject our Sema
      // pointer into the PCH.
      if (!isCUDADevice)
        ASTTransformers.emplace_back(
            new NullDerefProtectionTransformer(m_Interpreter));
    }
    ASTTransformers.emplace_back(new DefinitionShadower(*TheSema, *m_Interpreter));

    typedef std::unique_ptr<WrapperTransformer> WTPtr_t;
    std::vector<WTPtr_t> WrapperTransformers;
    if (!m_Interpreter->getOptions().NoRuntime && !isCUDADevice)
      WrapperTransformers.emplace_back(new ValuePrinterSynthesizer(TheSema));
    WrapperTransformers.emplace_back(new DeclExtractor(TheSema));
    if (!m_Interpreter->getOptions().NoRuntime && !isCUDADevice)
      WrapperTransformers.emplace_back(new ValueExtractionSynthesizer(TheSema,
                                                           isChildInterpreter));
    WrapperTransformers.emplace_back(new CheckEmptyTransactionTransformer(TheSema));

    m_Consumer->SetTransformers(std::move(ASTTransformers),
                                std::move(WrapperTransformers));
  }


} // namespace cling
