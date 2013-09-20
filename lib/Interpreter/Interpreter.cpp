//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Lukasz Janyst <ljanyst@cern.ch>
//------------------------------------------------------------------------------

#include "cling/Interpreter/Interpreter.h"

#include "DynamicLookup.h"
#include "ExecutionContext.h"
#include "IncrementalParser.h"

#include "cling/Interpreter/CIFactory.h"
#include "cling/Interpreter/ClangInternalState.h"
#include "cling/Interpreter/CompilationOptions.h"
#include "cling/Interpreter/DynamicLibraryManager.h"
#include "cling/Interpreter/InterpreterCallbacks.h"
#include "cling/Interpreter/LookupHelper.h"
#include "cling/Interpreter/StoredValueRef.h"
#include "cling/Interpreter/Transaction.h"
#include "cling/Utils/AST.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Mangle.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/SourceManager.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/Utils.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Parse/ParseDiagnostic.h" // FIXME: remove this back-dependency!
// when clang is ready.
#include "clang/Parse/Parser.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaInternal.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <sstream>
#include <string>
#include <vector>

using namespace clang;

namespace {

  static cling::Interpreter::ExecutionResult
  ConvertExecutionResult(cling::ExecutionContext::ExecutionResult ExeRes) {
    switch (ExeRes) {
    case cling::ExecutionContext::kExeSuccess:
      return cling::Interpreter::kExeSuccess;
    case cling::ExecutionContext::kExeFunctionNotCompiled:
      return cling::Interpreter::kExeFunctionNotCompiled;
    case cling::ExecutionContext::kExeUnresolvedSymbols:
      return cling::Interpreter::kExeUnresolvedSymbols;
    default: break;
    }
    return cling::Interpreter::kExeSuccess;
  }
} // unnamed namespace

namespace cling {
  namespace runtime {
    namespace internal {
      // "Declared" to the JIT in RuntimeUniverse.h
      int local_cxa_atexit(void (*func) (void*), void* arg, void* dso,
                           void* interp) {
        Interpreter* cling = (cling::Interpreter*)interp;
        IncrementalParser* incrP = cling->m_IncrParser.get();
        // FIXME: Bind to the module symbols.
        Decl* lastTLD = incrP->getLastTransaction()->getLastDecl().getSingleDecl();
 
        int result = cling->m_ExecutionContext->CXAAtExit(func, arg, dso, lastTLD);
        return result;
      }
    } // end namespace internal
  } // end namespace runtime
}


namespace cling {
#if (!_WIN32)
  // "Declared" to the JIT in RuntimeUniverse.h
  namespace runtime {
    namespace internal {
      struct __trigger__cxa_atexit {
        ~__trigger__cxa_atexit();
      } /*S*/;
      __trigger__cxa_atexit::~__trigger__cxa_atexit() {
        if (std::getenv("bar") == (char*)-1) {
          llvm::errs() <<
            "UNEXPECTED cling::runtime::internal::__trigger__cxa_atexit\n";
        }
      }
    }
  }
#endif

  Interpreter::PushTransactionRAII::PushTransactionRAII(const Interpreter* i)
    : m_Interpreter(i) {
    CompilationOptions CO;
    CO.DeclarationExtraction = 0;
    CO.ValuePrinting = 0;
    CO.ResultEvaluation = 0;
    CO.DynamicScoping = 0;
    CO.Debug = 0;
    CO.CodeGeneration = 1;
    CO.CodeGenerationForModule = 0;

    m_Transaction = m_Interpreter->m_IncrParser->beginTransaction(CO);
  }

  Interpreter::PushTransactionRAII::~PushTransactionRAII() {
    pop();
  }

  void Interpreter::PushTransactionRAII::pop() const {
    if (Transaction* T 
        = m_Interpreter->m_IncrParser->endTransaction(m_Transaction)) {
      assert(T == m_Transaction && "Ended different transaction?");
      m_Interpreter->m_IncrParser->commitTransaction(T);
    }
  }  

  // This function isn't referenced outside its translation unit, but it
  // can't use the "static" keyword because its address is used for
  // GetMainExecutable (since some platforms don't support taking the
  // address of main, and some platforms can't implement GetMainExecutable
  // without being given the address of a function in the main executable).
  std::string GetExecutablePath(const char *Argv0) {
    // This just needs to be some symbol in the binary; C++ doesn't
    // allow taking the address of ::main however.
    void *MainAddr = (void*) (intptr_t) GetExecutablePath;
    return llvm::sys::fs::getMainExecutable(Argv0, MainAddr);
  }

  const Parser& Interpreter::getParser() const {
    return *m_IncrParser->getParser();
  }

  CodeGenerator* Interpreter::getCodeGenerator() const {
    return m_IncrParser->getCodeGenerator();
  }

  void Interpreter::unload() {
    m_IncrParser->unloadTransaction(0);
  }

  Interpreter::Interpreter(int argc, const char* const *argv,
                           const char* llvmdir /*= 0*/) :
    m_UniqueCounter(0), m_PrintAST(false), m_PrintIR(false), 
    m_DynamicLookupEnabled(false), m_RawInputEnabled(false) {

    m_LLVMContext.reset(new llvm::LLVMContext);
    std::vector<unsigned> LeftoverArgsIdx;
    m_Opts = InvocationOptions::CreateFromArgs(argc, argv, LeftoverArgsIdx);
    std::vector<const char*> LeftoverArgs;

    for (size_t I = 0, N = LeftoverArgsIdx.size(); I < N; ++I) {
      LeftoverArgs.push_back(argv[LeftoverArgsIdx[I]]);
    }

    m_DyLibManager.reset(new DynamicLibraryManager(getOptions()));

    m_IncrParser.reset(new IncrementalParser(this, LeftoverArgs.size(),
                                             &LeftoverArgs[0],
                                             llvmdir));
    Sema& SemaRef = getSema();
    m_LookupHelper.reset(new LookupHelper(new Parser(SemaRef.getPreprocessor(), 
                                                     SemaRef, 
                                                     /*SkipFunctionBodies*/false,
                                                     /*isTemp*/true), this));

    if (m_IncrParser->hasCodeGenerator()) {
      llvm::Module* theModule = m_IncrParser->getCodeGenerator()->GetModule();
      m_ExecutionContext.reset(new ExecutionContext(theModule));
    }

    m_IncrParser->Initialize();

    // Add path to interpreter's include files
    // Try to find the headers in the src folder first
#ifdef CLING_SRCDIR_INCL
    llvm::StringRef SrcIncl(CLING_SRCDIR_INCL);
    if (llvm::sys::fs::is_directory(SrcIncl))
      AddIncludePath(SrcIncl);
#endif

    llvm::SmallString<512> P(GetExecutablePath(argv[0]));
    if (!P.empty()) {
      // Remove /cling from foo/bin/clang
      llvm::StringRef ExeIncl = llvm::sys::path::parent_path(P);
      // Remove /bin   from foo/bin
      ExeIncl = llvm::sys::path::parent_path(ExeIncl);
      P.resize(ExeIncl.size() + 1);
      P[ExeIncl.size()] = 0;
      // Get foo/include
      llvm::sys::path::append(P, "include");
      if (llvm::sys::fs::is_directory(P.str()))
        AddIncludePath(P.str());
      else {
#ifdef CLING_INSTDIR_INCL
        llvm::StringRef InstIncl(CLING_INSTDIR_INCL);
        if (llvm::sys::fs::is_directory(InstIncl))
          AddIncludePath(InstIncl);
#endif
      }
    }

    // Enable incremental processing, which prevents the preprocessor destroying
    // the lexer on EOF token.
    getSema().getPreprocessor().enableIncrementalProcessing();

    handleFrontendOptions();

    // Tell the diagnostic client that we are entering file parsing mode.
    DiagnosticConsumer& DClient = getCI()->getDiagnosticClient();
    DClient.BeginSourceFile(getCI()->getLangOpts(),
                            &getCI()->getPreprocessor());

    if (getCI()->getLangOpts().CPlusPlus) {
      // Set up common declarations which are going to be available
      // only at runtime
      // Make sure that the universe won't be included to compile time by using
      // -D __CLING__ as CompilerInstance's arguments
#ifdef _WIN32
      // We have to use the #defined __CLING__ on windows first. 
      //FIXME: Find proper fix.
      declare("#ifdef __CLING__ \n#endif");  
#endif
      declare("#include \"cling/Interpreter/RuntimeUniverse.h\"");
      declare("#include \"cling/Interpreter/ValuePrinter.h\"");

      if (getCodeGenerator()) {
        // Set up the gCling variable if it can be used
        std::stringstream initializer;
        initializer << "namespace cling {namespace runtime { "
          "cling::Interpreter *gCling=(cling::Interpreter*)"
                    << (uintptr_t)this << ";} }";
        declare(initializer.str());
      }

      // Find cling::runtime::internal::local_cxa_atexit
      // We do not have an active transaction and that lookup might trigger
      // deserialization
      PushTransactionRAII pushedT(this);
      NamespaceDecl* NSD = utils::Lookup::Namespace(&getSema(), "cling");
      NSD = utils::Lookup::Namespace(&getSema(), "runtime");
      NSD = utils::Lookup::Namespace(&getSema(), "internal");
      NamedDecl* ND = utils::Lookup::Named(&getSema(), "local_cxa_atexit", NSD);
      std::string mangledName;
      maybeMangleDeclName(ND, mangledName);
      m_ExecutionContext->addSymbol(mangledName.c_str(),
                         (void*)(intptr_t)&runtime::internal::local_cxa_atexit);

    }
    else {
      declare("#include \"cling/Interpreter/CValuePrinter.h\"");
    }

  }

  Interpreter::~Interpreter() {
    getCI()->getDiagnostics().getClient()->EndSourceFile();
  }

  const char* Interpreter::getVersion() const {
    return "$Id$";
  }

  void Interpreter::handleFrontendOptions() {
    if (m_Opts.ShowVersion) {
      llvm::errs() << getVersion() << '\n';
    }
    if (m_Opts.Help) {
      m_Opts.PrintHelp();
    }
  }

  void Interpreter::AddIncludePath(llvm::StringRef incpath)
  {
    // Add the given path to the list of directories in which the interpreter
    // looks for include files. Only one path item can be specified at a
    // time, i.e. "path1:path2" is not supported.

    CompilerInstance* CI = getCI();
    HeaderSearchOptions& headerOpts = CI->getHeaderSearchOpts();
    const bool IsFramework = false;
    const bool IsSysRootRelative = true;
    headerOpts.AddPath(incpath, frontend::Angled, IsFramework,
                       IsSysRootRelative);

    Preprocessor& PP = CI->getPreprocessor();
    ApplyHeaderSearchOptions(PP.getHeaderSearchInfo(), headerOpts,
                                    PP.getLangOpts(),
                                    PP.getTargetInfo().getTriple());
  }

  void Interpreter::DumpIncludePath() {
    llvm::SmallVector<std::string, 100> IncPaths;
    GetIncludePaths(IncPaths, true /*withSystem*/, true /*withFlags*/);
    // print'em all
    for (unsigned i = 0; i < IncPaths.size(); ++i) {
      llvm::errs() << IncPaths[i] <<"\n";
    }
  }

  void Interpreter::storeInterpreterState(const std::string& name) const {
    ClangInternalState* state 
      = new ClangInternalState(getCI()->getASTContext(), name);
    m_StoredStates.push_back(state);
  }

  void Interpreter::compareInterpreterState(const std::string& name) const {
    ClangInternalState* state 
      = new ClangInternalState(getCI()->getASTContext(), name);    
    for (unsigned i = 0, e = m_StoredStates.size(); i != e; ++i) {
      if (m_StoredStates[i]->getName() == name) {
        state->compare(*m_StoredStates[i]);
        m_StoredStates.erase(m_StoredStates.begin() + i);
        break;
      }
    }
  }

  void Interpreter::printIncludedFiles(llvm::raw_ostream& Out) const {
    ClangInternalState::printIncludedFiles(Out, getCI()->getSourceManager());
  }


  // Adapted from clang/lib/Frontend/CompilerInvocation.cpp
  void Interpreter::GetIncludePaths(llvm::SmallVectorImpl<std::string>& incpaths,
                                   bool withSystem, bool withFlags) {
    const HeaderSearchOptions Opts(getCI()->getHeaderSearchOpts());

    if (withFlags && Opts.Sysroot != "/") {
      incpaths.push_back("-isysroot");
      incpaths.push_back(Opts.Sysroot);
    }

    /// User specified include entries.
    for (unsigned i = 0, e = Opts.UserEntries.size(); i != e; ++i) {
      const HeaderSearchOptions::Entry &E = Opts.UserEntries[i];
      if (E.IsFramework && E.Group != frontend::Angled)
        llvm::report_fatal_error("Invalid option set!");
      switch (E.Group) {
      case frontend::After:
        if (withFlags) incpaths.push_back("-idirafter");
        break;

      case frontend::Quoted:
        if (withFlags) incpaths.push_back("-iquote");
        break;

      case frontend::System:
        if (!withSystem) continue;
        if (withFlags) incpaths.push_back("-isystem");
        break;

      case frontend::IndexHeaderMap:
        if (!withSystem) continue;
        if (withFlags) incpaths.push_back("-index-header-map");
        if (withFlags) incpaths.push_back(E.IsFramework? "-F" : "-I");
        break;

      case frontend::CSystem:
        if (!withSystem) continue;
        if (withFlags) incpaths.push_back("-c-isystem");
        break;

      case frontend::ExternCSystem:
        if (!withSystem) continue;
        if (withFlags) incpaths.push_back("-extern-c-isystem");
        break;

      case frontend::CXXSystem:
        if (!withSystem) continue;
        if (withFlags) incpaths.push_back("-cxx-isystem");
        break;

      case frontend::ObjCSystem:
        if (!withSystem) continue;
        if (withFlags) incpaths.push_back("-objc-isystem");
        break;

      case frontend::ObjCXXSystem:
        if (!withSystem) continue;
        if (withFlags) incpaths.push_back("-objcxx-isystem");
        break;

      case frontend::Angled:
        if (withFlags) incpaths.push_back(E.IsFramework ? "-F" : "-I");
        break;
      }
      incpaths.push_back(E.Path);
    }

    if (withSystem && !Opts.ResourceDir.empty()) {
      if (withFlags) incpaths.push_back("-resource-dir");
      incpaths.push_back(Opts.ResourceDir);
    }
    if (withSystem && withFlags && !Opts.ModuleCachePath.empty()) {
      incpaths.push_back("-fmodule-cache-path");
      incpaths.push_back(Opts.ModuleCachePath);
    }
    if (withSystem && withFlags && !Opts.UseStandardSystemIncludes)
      incpaths.push_back("-nostdinc");
    if (withSystem && withFlags && !Opts.UseStandardCXXIncludes)
      incpaths.push_back("-nostdinc++");
    if (withSystem && withFlags && Opts.UseLibcxx)
      incpaths.push_back("-stdlib=libc++");
    if (withSystem && withFlags && Opts.Verbose)
      incpaths.push_back("-v");
  }

  CompilerInstance* Interpreter::getCI() const {
    return m_IncrParser->getCI();
  }

  const Sema& Interpreter::getSema() const {
    return getCI()->getSema();
  }

  Sema& Interpreter::getSema() {
    return getCI()->getSema();
  }

   llvm::ExecutionEngine* Interpreter::getExecutionEngine() const {
    return m_ExecutionContext->getExecutionEngine();
  }

  llvm::Module* Interpreter::getModule() const {
    return m_IncrParser->getCodeGenerator()->GetModule();
  }

  ///\brief Maybe transform the input line to implement cint command line
  /// semantics (declarations are global) and compile to produce a module.
  ///
  Interpreter::CompilationResult
  Interpreter::process(const std::string& input, StoredValueRef* V /* = 0 */,
                       Transaction** T /* = 0 */) {
    if (isRawInputEnabled() || !ShouldWrapInput(input))
      return declare(input, T);

    CompilationOptions CO;
    CO.DeclarationExtraction = 1;
    CO.ValuePrinting = CompilationOptions::VPAuto;
    CO.ResultEvaluation = (bool)V;
    CO.DynamicScoping = isDynamicLookupEnabled();
    CO.Debug = isPrintingAST();
    CO.IRDebug = isPrintingIR();

    if (EvaluateInternal(input, CO, V, T) == Interpreter::kFailure) {
      return Interpreter::kFailure;
    }

    return Interpreter::kSuccess;
  }

  Interpreter::CompilationResult 
  Interpreter::parse(const std::string& input, Transaction** T /*=0*/) const {
    CompilationOptions CO;
    CO.CodeGeneration = 0;
    CO.DeclarationExtraction = 0;
    CO.ValuePrinting = 0;
    CO.ResultEvaluation = 0;
    CO.DynamicScoping = isDynamicLookupEnabled();
    CO.Debug = isPrintingAST();
    CO.IRDebug = isPrintingIR();

    return DeclareInternal(input, CO, T);
  }

  Interpreter::CompilationResult
  Interpreter::loadModuleForHeader(const std::string& headerFile) {
    Preprocessor& PP = getCI()->getPreprocessor();
    //Copied from clang's PPDirectives.cpp
    bool isAngled = false;
    // Clang doc says:
    // "LookupFrom is set when this is a \#include_next directive, it specifies 
    // the file to start searching from."
    const DirectoryLookup* LookupFrom = 0;
    const DirectoryLookup* CurDir = 0;

    Module* module = 0;
    PP.LookupFile(headerFile, isAngled, LookupFrom, CurDir, /*SearchPath*/0,
                  /*RelativePath*/ 0, &module, /*SkipCache*/false);
    if (!module)
      return Interpreter::kFailure;

    SourceLocation fileNameLoc;
    ModuleIdPath path = std::make_pair(PP.getIdentifierInfo(module->Name),
                                       fileNameLoc);

    // Pretend that the module came from an inclusion directive, so that clang
    // will create an implicit import declaration to capture it in the AST.
    bool isInclude = true;
    SourceLocation includeLoc;
    if (getCI()->loadModule(includeLoc, path, Module::AllVisible, isInclude)) {
      // After module load we need to "force" Sema to generate the code for
      // things like dynamic classes.
      getSema().ActOnEndOfTranslationUnit();
      return Interpreter::kSuccess;
    }

    return Interpreter::kFailure;
  }

  Interpreter::CompilationResult
  Interpreter::parseForModule(const std::string& input) {
    CompilationOptions CO;
    CO.CodeGeneration = 1;
    CO.CodeGenerationForModule = 1;
    CO.DeclarationExtraction = 0;
    CO.ValuePrinting = 0;
    CO.ResultEvaluation = 0;
    CO.DynamicScoping = isDynamicLookupEnabled();
    CO.Debug = isPrintingAST();
    CO.IRDebug = isPrintingIR();
    
    // When doing parseForModule avoid warning about the user code
    // being loaded ... we probably might as well extend this to
    // ALL warnings ... but this will suffice for now (working
    // around a real bug in QT :().
    DiagnosticsEngine& Diag = getCI()->getDiagnostics();
    Diag.setDiagnosticMapping(clang::diag::warn_field_is_uninit,
                              clang::diag::MAP_IGNORE, SourceLocation());
    return DeclareInternal(input, CO);
  }
  
  Interpreter::CompilationResult
  Interpreter::declare(const std::string& input, Transaction** T/*=0 */) {
    CompilationOptions CO;
    CO.DeclarationExtraction = 0;
    CO.ValuePrinting = 0;
    CO.ResultEvaluation = 0;
    CO.DynamicScoping = isDynamicLookupEnabled();
    CO.Debug = isPrintingAST();
    CO.IRDebug = isPrintingIR();

    return DeclareInternal(input, CO, T);
  }

  Interpreter::CompilationResult
  Interpreter::evaluate(const std::string& input, StoredValueRef& V) {
    // Here we might want to enforce further restrictions like: Only one
    // ExprStmt can be evaluated and etc. Such enforcement cannot happen in the
    // worker, because it is used from various places, where there is no such
    // rule
    CompilationOptions CO;
    CO.DeclarationExtraction = 0;
    CO.ValuePrinting = 0;
    CO.ResultEvaluation = 1;

    return EvaluateInternal(input, CO, &V);
  }

  Interpreter::CompilationResult
  Interpreter::echo(const std::string& input, StoredValueRef* V /* = 0 */) {
    CompilationOptions CO;
    CO.DeclarationExtraction = 0;
    CO.ValuePrinting = CompilationOptions::VPEnabled;
    CO.ResultEvaluation = 0;

    return EvaluateInternal(input, CO, V);
  }

  Interpreter::CompilationResult
  Interpreter::execute(const std::string& input) {
    CompilationOptions CO;
    CO.DeclarationExtraction = 0;
    CO.ValuePrinting = 0;
    CO.ResultEvaluation = 0;
    CO.DynamicScoping = 0;
    CO.Debug = isPrintingAST();
    CO.IRDebug = isPrintingIR();

    // Disable warnings which doesn't make sense when using the prompt
    // This gets reset with the clang::Diagnostics().Reset()
    // TODO: Here might be useful to issue unused variable diagnostic,
    // because we don't do declaration extraction and the decl won't be visible
    // anymore.
    ignoreFakeDiagnostics();

    // Wrap the expression
    std::string WrapperName;
    std::string Wrapper = input;
    WrapInput(Wrapper, WrapperName);
    
    const Transaction* lastT = m_IncrParser->Compile(Wrapper, CO);
    assert(lastT->getState() == Transaction::kCommitted && "Must be committed");
    if (lastT->getIssuedDiags() == Transaction::kNone)
      if (RunFunction(lastT->getWrapperFD()) < kExeFirstError)
        return Interpreter::kSuccess;

    return Interpreter::kFailure;
  }

  Interpreter::CompilationResult Interpreter::emitAllDecls(Transaction* T) {
    assert(getCodeGenerator() && "No CodeGenerator?");
    m_IncrParser->markWholeTransactionAsUsed(T);
    m_IncrParser->codeGenTransaction(T);

    // The static initializers might run anything and can thus cause more
    // decls that need to end up in a transaction. But this one is done
    // with CodeGen...
    T->setState(Transaction::kCommitted);
    if (runStaticInitializersOnce(*T))
      return Interpreter::kSuccess;

    return Interpreter::kFailure;
  }

  bool Interpreter::ShouldWrapInput(const std::string& input) {
    llvm::OwningPtr<llvm::MemoryBuffer> buf;
    buf.reset(llvm::MemoryBuffer::getMemBuffer(input, "Cling Preparse Buf"));
    Lexer WrapLexer(SourceLocation(), getSema().getLangOpts(), input.c_str(), 
                    input.c_str(), input.c_str() + input.size());
    Token Tok;
    WrapLexer.Lex(Tok);

    const tok::TokenKind kind = Tok.getKind();

    if (kind == tok::raw_identifier && !Tok.needsCleaning()) {
      StringRef keyword(Tok.getRawIdentifierData(), Tok.getLength());
      if (keyword.equals("using"))
        return false;
      if (keyword.equals("extern"))
        return false;
      if (keyword.equals("namespace"))
        return false;
      if (keyword.equals("template"))
        return false;
    }
    else if (kind == tok::hash) {
      WrapLexer.Lex(Tok);
      if (Tok.is(tok::raw_identifier) && !Tok.needsCleaning()) {
        StringRef keyword(Tok.getRawIdentifierData(), Tok.getLength());
        if (keyword.equals("include"))
          return false;
      }
    }

    return true;
  }

  void Interpreter::WrapInput(std::string& input, std::string& fname) {
    fname = createUniqueWrapper();
    input.insert(0, "void " + fname + "() {\n ");
    input.append("\n;\n}");
  }

  Interpreter::ExecutionResult
  Interpreter::RunFunction(const FunctionDecl* FD, StoredValueRef* res /*=0*/) {
    if (getCI()->getDiagnostics().hasErrorOccurred())
      return kExeCompilationError;

    if (!m_IncrParser->hasCodeGenerator()) {
      return kExeNoCodeGen;
    }

    if (!FD)
      return kExeUnkownFunction;

    std::string mangledNameIfNeeded;
    maybeMangleDeclName(FD, mangledNameIfNeeded);
    ExecutionContext::ExecutionResult ExeRes =
       m_ExecutionContext->executeFunction(mangledNameIfNeeded.c_str(),
                                           getCI()->getASTContext(),
                                           FD->getResultType(), res);
    if (res && res->isValid())
      res->get().setLLVMType(getLLVMType(res->get().getClangType()));
    return ConvertExecutionResult(ExeRes);
  }

  void Interpreter::createUniqueName(std::string& out) {
    out += utils::Synthesize::UniquePrefix;
    llvm::raw_string_ostream(out) << m_UniqueCounter++;
  }

  bool Interpreter::isUniqueName(llvm::StringRef name) {
    return name.startswith(utils::Synthesize::UniquePrefix);
  }

  llvm::StringRef Interpreter::createUniqueWrapper() {
    const size_t size 
      = sizeof(utils::Synthesize::UniquePrefix) + sizeof(m_UniqueCounter);
    llvm::SmallString<size> out(utils::Synthesize::UniquePrefix);
    llvm::raw_svector_ostream(out) << m_UniqueCounter++;

    return (getCI()->getASTContext().Idents.getOwn(out)).getName();
  }

  bool Interpreter::isUniqueWrapper(llvm::StringRef name) {
    return name.startswith(utils::Synthesize::UniquePrefix);
  }

  Interpreter::CompilationResult
  Interpreter::DeclareInternal(const std::string& input, 
                               const CompilationOptions& CO,
                               Transaction** T /* = 0 */) const {
    // Disable warnings which doesn't make sense when using the prompt
    // This gets reset with the clang::Diagnostics().Reset()
    ignoreFakeDiagnostics();

    Transaction* lastT = m_IncrParser->Compile(input, CO);
    if (lastT->getIssuedDiags() != Transaction::kErrors) {
      if (T)
        *T = lastT;
      return Interpreter::kSuccess;
    }

    return Interpreter::kFailure;
  }

  Interpreter::CompilationResult
  Interpreter::EvaluateInternal(const std::string& input, 
                                const CompilationOptions& CO,
                                StoredValueRef* V, /* = 0 */
                                Transaction** T /* = 0 */) {
    // Disable warnings which doesn't make sense when using the prompt
    // This gets reset with the clang::Diagnostics().Reset()
    ignoreFakeDiagnostics();

    // Wrap the expression
    std::string WrapperName;
    std::string Wrapper = input;
    WrapInput(Wrapper, WrapperName);

    Transaction* lastT = m_IncrParser->Compile(Wrapper, CO);
    assert((!V || lastT->size()) && "No decls created!?");
    assert((lastT->getState() == Transaction::kCommitted
           || lastT->getState() == Transaction::kRolledBack) 
           && "Not committed?");
    assert(lastT->getWrapperFD() && "Must have wrapper!");
    if (lastT->getIssuedDiags() != Transaction::kErrors)
      if (RunFunction(lastT->getWrapperFD(), V) < kExeFirstError)
        return Interpreter::kSuccess;
    if (V)
      *V = StoredValueRef::invalidValue();

    return Interpreter::kFailure;
  }

  Interpreter::CompilationResult
  Interpreter::loadFile(const std::string& filename,
                        bool allowSharedLib /*=true*/) {
    if (allowSharedLib) {
      bool tryCode;
      if (getDynamicLibraryManager()->loadLibrary(filename, false, &tryCode)
          == DynamicLibraryManager::kLoadLibSuccess)
        return kSuccess;
      if (!tryCode)
        return kFailure;
    }

    std::string code;
    code += "#include \"" + filename + "\"";
    CompilationResult res = declare(code);
    return res;
  }

  void Interpreter::installLazyFunctionCreator(void* (*fp)(const std::string&)) {
    m_ExecutionContext->installLazyFunctionCreator(fp);
  }

  void Interpreter::suppressLazyFunctionCreatorDiags(bool suppressed/*=true*/) {
    m_ExecutionContext->suppressLazyFunctionCreatorDiags(suppressed);
  }

  StoredValueRef Interpreter::Evaluate(const char* expr, DeclContext* DC,
                                       bool ValuePrinterReq) {
    Sema& TheSema = getCI()->getSema();
    // The evaluation should happen on the global scope, because of the wrapper
    // that is created. 
    //
    // We can't PushDeclContext, because we don't have scope.
    Sema::ContextRAII pushDC(TheSema, 
                             TheSema.getASTContext().getTranslationUnitDecl());

    StoredValueRef Result;
    getCallbacks()->SetIsRuntime(true);
    if (ValuePrinterReq)
      echo(expr, &Result);
    else 
      evaluate(expr, Result);
    getCallbacks()->SetIsRuntime(false);

    return Result;
  }

  void Interpreter::setCallbacks(InterpreterCallbacks* C) {
    // We need it to enable LookupObject callback.
    m_Callbacks.reset(C);

    // FIXME: We should add a multiplexer in the ASTContext, too.
    llvm::OwningPtr<ExternalASTSource> astContextExternalSource;
    astContextExternalSource.reset(getSema().getExternalSource());
    clang::ASTContext& Ctx = getSema().getASTContext();
    // FIXME: This is a gross hack. We must make multiplexer in the astcontext,
    // or a derived class that extends what we need.
    Ctx.ExternalSource.take(); // FIXME: make sure we delete it.
    Ctx.setExternalSource(astContextExternalSource);
  }

  //FIXME: Get rid of that.
  clang::ASTDeserializationListener*
  Interpreter::getASTDeserializationListener() const {
    if (!m_Callbacks)
      return 0;
    return m_Callbacks->getInterpreterDeserializationListener();
  }


  const Transaction* Interpreter::getFirstTransaction() const {
    return m_IncrParser->getFirstTransaction();
  }

  void Interpreter::enableDynamicLookup(bool value /*=true*/) {
    m_DynamicLookupEnabled = value;

    if (isDynamicLookupEnabled()) {
     if (loadModuleForHeader("cling/Interpreter/DynamicLookupRuntimeUniverse.h")
         != kSuccess)
      declare("#include \"cling/Interpreter/DynamicLookupRuntimeUniverse.h\"");
    }
  }

  Interpreter::ExecutionResult
  Interpreter::runStaticInitializersOnce(const Transaction& T) const {
    assert(m_IncrParser->hasCodeGenerator() && "Running on what?");
    assert(T.getState() == Transaction::kCommitted && "Must be committed");
    // Forward to ExecutionContext; should not be called by
    // anyone except for IncrementalParser.
    llvm::Module* module = m_IncrParser->getCodeGenerator()->GetModule();
    ExecutionContext::ExecutionResult ExeRes
       = m_ExecutionContext->runStaticInitializersOnce(module);

    // Reset the module builder to clean up global initializers, c'tors, d'tors
    getCodeGenerator()->HandleTranslationUnit(getCI()->getASTContext());

    return ConvertExecutionResult(ExeRes);
  }

  void Interpreter::runStaticDestructorsOnce() {
    m_ExecutionContext->runStaticDestructorsOnce(getModule());
  }

  void Interpreter::maybeMangleDeclName(const clang::NamedDecl* D,
                                        std::string& mangledName) const {
    ///Get the mangled name of a NamedDecl.
    ///
    ///D - mangle this decl's name
    ///mangledName - put the mangled name in here
    if (!m_MangleCtx) {
      m_MangleCtx.reset(getCI()->getASTContext().createMangleContext());
    }
    if (m_MangleCtx->shouldMangleDeclName(D)) {
      llvm::raw_string_ostream RawStr(mangledName);
      switch(D->getKind()) {
      case Decl::CXXConstructor:
        //Ctor_Complete,          // Complete object ctor
        //Ctor_Base,              // Base object ctor
        //Ctor_CompleteAllocating // Complete object allocating ctor (unused)
        m_MangleCtx->mangleCXXCtor(cast<CXXConstructorDecl>(D), 
                                   Ctor_Complete, RawStr);
        break;

      case Decl::CXXDestructor:
        //Dtor_Deleting, // Deleting dtor
        //Dtor_Complete, // Complete object dtor
        //Dtor_Base      // Base object dtor
        m_MangleCtx->mangleCXXDtor(cast<CXXDestructorDecl>(D),
                                   Dtor_Complete, RawStr);
        break;

      default :
        m_MangleCtx->mangleName(D, RawStr);
        break;
      }
      RawStr.flush();
    } else {
      mangledName = D->getNameAsString();
    }
  }

  void Interpreter::ignoreFakeDiagnostics() const {
    DiagnosticsEngine& Diag = getCI()->getDiagnostics();
    // Disable warnings which doesn't make sense when using the prompt
    // This gets reset with the clang::Diagnostics().Reset()
    Diag.setDiagnosticMapping(clang::diag::warn_unused_expr,
                              clang::diag::MAP_IGNORE, SourceLocation());
    Diag.setDiagnosticMapping(clang::diag::warn_unused_call,
                              clang::diag::MAP_IGNORE, SourceLocation());
    Diag.setDiagnosticMapping(clang::diag::warn_unused_comparison,
                              clang::diag::MAP_IGNORE, SourceLocation());
    Diag.setDiagnosticMapping(clang::diag::ext_return_has_expr,
                              clang::diag::MAP_IGNORE, SourceLocation());
    // Very very ugly. TODO: Revisit and extract out as interpreter arg
    Diag.setDiagnosticMapping(clang::diag::ext_auto_type_specifier,
                              clang::diag::MAP_IGNORE, SourceLocation());
  }

  bool Interpreter::addSymbol(const char* symbolName,  void* symbolAddress) {
    // Forward to ExecutionContext;
    if (!symbolName || !symbolAddress )
      return false;

    return m_ExecutionContext->addSymbol(symbolName, symbolAddress);
  }

  void* Interpreter::getAddressOfGlobal(const clang::NamedDecl* D,
                                        bool* fromJIT /*=0*/) const {
    // Return a symbol's address, and whether it was jitted.
    std::string mangledName;
    maybeMangleDeclName(D, mangledName);
    return getAddressOfGlobal(mangledName.c_str(), fromJIT);
  }

  void* Interpreter::getAddressOfGlobal(const char* SymName,
                                        bool* fromJIT /*=0*/) const {
    // Return a symbol's address, and whether it was jitted.
    llvm::Module* module = m_IncrParser->getCodeGenerator()->GetModule();
    return m_ExecutionContext->getAddressOfGlobal(module, SymName, fromJIT);
  }

  const llvm::Type* Interpreter::getLLVMType(QualType QT) {
    if (!m_IncrParser->hasCodeGenerator())
      return 0;

    // Note: The first thing this routine does is getCanonicalType(), so we
    //       do not need to do that first.
    return getCodeGenerator()->ConvertType(QT);
  }

} // namespace cling
