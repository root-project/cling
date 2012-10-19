//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Axel Naumann <axel@cern.ch>
//------------------------------------------------------------------------------

#include "IncrementalParser.h"
#include "ASTDumper.h"
#include "ASTNodeEraser.h"
#include "DeclCollector.h"
#include "DeclExtractor.h"
#include "DynamicLookup.h"
#include "ReturnSynthesizer.h"
#include "ValuePrinterSynthesizer.h"
#include "cling/Interpreter/CIFactory.h"
#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/InterpreterCallbacks.h"
#include "cling/Interpreter/Transaction.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclGroup.h"
#include "clang/Basic/FileManager.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "clang/Parse/Parser.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Serialization/ASTWriter.h"

#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/Support/CrashRecoveryContext.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_os_ostream.h"

#include <ctime>
#include <iostream>
#include <stdio.h>
#include <sstream>

using namespace clang;

namespace cling {
  IncrementalParser::IncrementalParser(Interpreter* interp,
                                       int argc, const char* const *argv,
                                       const char* llvmdir):
    m_Interpreter(interp), m_Consumer(0) {

    CompilerInstance* CI
      = CIFactory::createCI(llvm::MemoryBuffer::getMemBuffer("", "CLING"),
                            argc, argv, llvmdir);
    assert(CI && "CompilerInstance is (null)!");

    m_Consumer = dyn_cast<DeclCollector>(&CI->getASTConsumer());
    assert(m_Consumer && "Expected ChainedConsumer!");

    m_CI.reset(CI);

    if (CI->getFrontendOpts().ProgramAction != clang::frontend::ParseSyntaxOnly){
      m_CodeGen.reset(CreateLLVMCodeGen(CI->getDiagnostics(), "cling input",
                                        CI->getCodeGenOpts(),
                                        *m_Interpreter->getLLVMContext()
                                        ));
    }

    CreateSLocOffsetGenerator();

    // Add transformers to the IncrementalParser, which owns them
    m_TTransformers.push_back(new EvaluateTSynthesizer(&CI->getSema()));

    m_TTransformers.push_back(new ValuePrinterSynthesizer(&CI->getSema(), 0));
    m_TTransformers.push_back(new ReturnSynthesizer(&CI->getSema()));
    m_TTransformers.push_back(new ASTDumper());
    m_TTransformers.push_back(new DeclExtractor(&getCI()->getSema()));

    m_Parser.reset(new Parser(CI->getPreprocessor(), CI->getSema(),
                              false /*skipFuncBodies*/));
    CI->getPreprocessor().EnterMainSourceFile();
    // Initialize the parser after we have entered the main source file.
    m_Parser->Initialize();
    // Perform initialization that occurs after the parser has been initialized
    // but before it parses anything. Initializes the consumers too.
    CI->getSema().Initialize();
  }

  IncrementalParser::~IncrementalParser() {
     if (hasCodeGenerator()) {
       getCodeGenerator()->ReleaseModule();
     }
     for (size_t i = 0; i < m_Transactions.size(); ++i)
       delete m_Transactions[i];

     for (size_t i = 0; i < m_TTransformers.size(); ++i)
       delete m_TTransformers[i];
  }

  // pin the vtable here since there is no point to create dedicated to that
  // cpp file.
  TransactionTransformer::~TransactionTransformer() {}

  void IncrementalParser::beginTransaction(const CompilationOptions& Opts) {
    llvm::Module* M = 0;
    if (hasCodeGenerator())
      M = getCodeGenerator()->GetModule();

    Transaction* NewCurT = new Transaction(Opts, M);
    Transaction* OldCurT = m_Consumer->getTransaction();
    m_Consumer->setTransaction(NewCurT);
    // If we are in the middle of transaction and we see another begin 
    // transaction - it must be nested transaction.
    if (OldCurT && !OldCurT->isCompleted()) {
      OldCurT->addNestedTransaction(NewCurT); // takes the ownership
      return;
    }

    getLastTransaction()->setNext(NewCurT);
    m_Transactions.push_back(NewCurT);
  }

  void IncrementalParser::endTransaction() const {
    Transaction* CurT = m_Consumer->getTransaction();
    CurT->setCompleted();
    const DiagnosticsEngine& Diags = getCI()->getSema().getDiagnostics();

    //TODO: Make the enum orable.
    if (Diags.getNumWarnings() > 0)
      CurT->setIssuedDiags(Transaction::kWarnings);

    if (Diags.hasErrorOccurred() || Diags.hasFatalErrorOccurred())
      CurT->setIssuedDiags(Transaction::kErrors);

      
    if (CurT->isNestedTransaction()) {
      assert(!CurT->getParent()->isCompleted() 
             && "Parent transaction completed!?");
      // FIXME: Not sure what I meant :) REVISIT
      //CurT = m_Consumer->getTransaction()->getParent();
    }
  }

  void IncrementalParser::commitCurrentTransaction() {
    Transaction* CurT = m_Consumer->getTransaction();
    assert(CurT->isCompleted() && "Transaction not ended!?");

    // Check for errors...
    if (CurT->getIssuedDiags() == Transaction::kErrors) {
      rollbackTransaction(CurT);
      return;
    }

    // We are sure it's safe to pipe it through the transformers
    for (size_t i = 0; i < m_TTransformers.size(); ++i)
      if (!m_TTransformers[i]->TransformTransaction(*CurT)) {
        // Roll back on error in a transformer
        rollbackTransaction(CurT);
        return;
      }

    // Pull all template instantiations in that came from the consumers.
    getCI()->getSema().PerformPendingInstantiations();

    m_Consumer->HandleTranslationUnit(getCI()->getASTContext());

    if (CurT->getCompilationOpts().CodeGeneration && hasCodeGenerator()) {
      // Reset the module builder to clean up global initializers, c'tors, d'tors
      getCodeGenerator()->Initialize(getCI()->getASTContext());

      // codegen the transaction
      for (Transaction::const_iterator I = CurT->decls_begin(), 
             E = CurT->decls_end(); I != E; ++I) {
        getCodeGenerator()->HandleTopLevelDecl(*I);
      }
      getCodeGenerator()->HandleTranslationUnit(getCI()->getASTContext());
      // run the static initializers that came from codegenning
      m_Interpreter->runStaticInitializersOnce();
    }

    CurT->setState(Transaction::kCommitted);
    InterpreterCallbacks* callbacks = m_Interpreter->getCallbacks();
    if (callbacks)
      callbacks->TransactionCommitted(*CurT);
  }

  void IncrementalParser::rollbackTransaction(Transaction* T) const {
    ASTNodeEraser NodeEraser(&getCI()->getSema());

    if (NodeEraser.RevertTransaction(T))
      T->setState(Transaction::kRolledBack);
    else
      T->setState(Transaction::kRolledBackWithErrors);
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
  void IncrementalParser::CreateSLocOffsetGenerator() {
    SourceManager& SM = getCI()->getSourceManager();
    FileManager& FM = SM.getFileManager();
    const FileEntry* FE
      = FM.getVirtualFile("InteractiveInputLineIncluder.h", 1U << 15U, time(0));
    m_VirtualFileID = SM.createFileID(FE, SourceLocation(), SrcMgr::C_User);

    assert(!m_VirtualFileID.isInvalid() && "No VirtualFileID created?");
  }

  IncrementalParser::EParseResult
  IncrementalParser::Compile(llvm::StringRef input,
                             const CompilationOptions& Opts) {

    beginTransaction(Opts);
    EParseResult Result = ParseInternal(input);
    endTransaction();

    // Check for errors coming from our custom consumers.
    DiagnosticConsumer& DClient = m_CI->getDiagnosticClient();
    DClient.BeginSourceFile(getCI()->getLangOpts(),
                            &getCI()->getPreprocessor());
    commitCurrentTransaction();

    DClient.EndSourceFile();
    m_CI->getDiagnostics().Reset();

    return Result;
  }

  Transaction* IncrementalParser::Parse(llvm::StringRef input,
                                        const CompilationOptions& Opts) {
    beginTransaction(Opts);
    ParseInternal(input);
    endTransaction();

    return getLastTransaction();
  }

  // Add the input to the memory buffer, parse it, and add it to the AST.
  IncrementalParser::EParseResult
  IncrementalParser::ParseInternal(llvm::StringRef input) {
    if (input.empty()) return IncrementalParser::kSuccess;

    PrettyStackTraceParserEntry CrashInfo(*m_Parser.get());

    // Recover resources if we crash before exiting this method.
    llvm::CrashRecoveryContextCleanupRegistrar<Parser> 
      CleanupParser(m_Parser.get());

    Sema& S = getCI()->getSema();
    // Recover resources if we crash before exiting this method.
    llvm::CrashRecoveryContextCleanupRegistrar<Sema> CleanupSema(&S);

    Preprocessor& PP = m_CI->getPreprocessor();
    DiagnosticConsumer& DClient = m_CI->getDiagnosticClient();

    if (!PP.getCurrentLexer()) {
       PP.EnterSourceFile(m_CI->getSourceManager().getMainFileID(),
                          0, SourceLocation());
    }
    assert(PP.isIncrementalProcessingEnabled() && "Not in incremental mode!?");
    PP.enableIncrementalProcessing();

    DClient.BeginSourceFile(m_CI->getLangOpts(), &PP);

    std::ostringstream source_name;
    source_name << "input_line_" << (m_MemoryBuffers.size() + 1);

    // Create an uninitialized memory buffer, copy code in and append "\n"
    size_t InputSize = input.size(); // don't include trailing 0
    // MemBuffer size should *not* include terminating zero
    llvm::MemoryBuffer* MB
      = llvm::MemoryBuffer::getNewUninitMemBuffer(InputSize + 1,
                                                  source_name.str());
    char* MBStart = const_cast<char*>(MB->getBufferStart());
    memcpy(MBStart, input.data(), InputSize);
    memcpy(MBStart + InputSize, "\n", 2);

    m_MemoryBuffers.push_back(MB);
    SourceManager& SM = getCI()->getSourceManager();

    // Create SourceLocation, which will allow clang to order the overload
    // candidates for example
    SourceLocation NewLoc = SM.getLocForStartOfFile(m_VirtualFileID);
    NewLoc = NewLoc.getLocWithOffset(m_MemoryBuffers.size() + 1);

    // Create FileID for the current buffer
    FileID FID = SM.createFileIDForMemBuffer(m_MemoryBuffers.back(),
                                             /*LoadedID*/0,
                                             /*LoadedOffset*/0, NewLoc);

    PP.EnterSourceFile(FID, /*DirLookup*/0, NewLoc);

    Parser::DeclGroupPtrTy ADecl;

    while (!m_Parser->ParseTopLevelDecl(ADecl)) {
      // If we got a null return and something *was* parsed, ignore it.  This
      // is due to a top-level semicolon, an action override, or a parse error
      // skipping something.
      if (ADecl)
        m_Consumer->HandleTopLevelDecl(ADecl.getAsVal<DeclGroupRef>());
    };

    // Process any TopLevelDecls generated by #pragma weak.
    for (llvm::SmallVector<Decl*,2>::iterator I = S.WeakTopLevelDecls().begin(),
           E = S.WeakTopLevelDecls().end(); I != E; ++I) {
      m_Consumer->HandleTopLevelDecl(DeclGroupRef(*I));
    }

    S.PerformPendingInstantiations();

    DClient.EndSourceFile();

    DiagnosticsEngine& Diag = S.getDiagnostics();
    if (Diag.hasErrorOccurred())
      return IncrementalParser::kFailed;
    else if (Diag.getNumWarnings())
      return IncrementalParser::kSuccessWithWarnings;

    return IncrementalParser::kSuccess;
  }

  void IncrementalParser::unloadTransaction(Transaction* T) {
    if (!T)
      T = getLastTransaction();

    assert(T->getState() == Transaction::kCommitted && 
           "Unloading not commited transaction?");
    assert(T->getModule() && 
           "Trying to uncodegen transaction taken in syntax only mode. ");

    ASTNodeEraser NodeEraser(&getCI()->getSema());
    NodeEraser.RevertTransaction(T);

    InterpreterCallbacks* callbacks = m_Interpreter->getCallbacks();
    if (callbacks)
      callbacks->TransactionUnloaded(*T);
  }

} // namespace cling
