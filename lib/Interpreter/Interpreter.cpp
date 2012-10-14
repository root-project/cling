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
#include "cling/Interpreter/CompilationOptions.h"
#include "cling/Interpreter/InterpreterCallbacks.h"
#include "cling/Interpreter/LookupHelper.h"
#include "cling/Interpreter/StoredValueRef.h"
#include "cling/Interpreter/Transaction.h"
#include "cling/Utils/AST.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Mangle.h"
#include "clang/AST/DeclarationName.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/Utils.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Parse/Parser.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Overload.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaInternal.h"
#include "clang/Sema/TemplateDeduction.h"

#include "llvm/Linker.h"
#include "llvm/LLVMContext.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

#include <iostream>
#include <sstream>
#include <vector>

using namespace clang;

namespace {
static bool tryLinker(const std::string& filename,
                      const cling::InvocationOptions& Opts,
                      llvm::Module* module) {
  assert(module && "Module must exist for linking!");
  llvm::Linker L("cling", module, llvm::Linker::QuietWarnings
                 | llvm::Linker::QuietErrors);
  for (std::vector<llvm::sys::Path>::const_iterator I
         = Opts.LibSearchPath.begin(), E = Opts.LibSearchPath.end(); I != E;
       ++I) {
    L.addPath(*I);
  }
  L.addSystemPaths();
  bool Native = true;
  if (L.LinkInLibrary(filename, Native)) {
    // that didn't work, try bitcode:
    llvm::sys::Path FilePath(filename);
    std::string Magic;
    if (!FilePath.getMagicNumber(Magic, 64)) {
      // filename doesn't exist...
      L.releaseModule();
      return false;
    }
    if (llvm::sys::IdentifyFileType(Magic.c_str(), 64)
        == llvm::sys::Bitcode_FileType) {
      // We are promised a bitcode file, complain if it fails
      L.setFlags(0);
      if (L.LinkInFile(llvm::sys::Path(filename), Native)) {
        L.releaseModule();
        return false;
      }
    } else {
      // Nothing the linker can handle
      L.releaseModule();
      return false;
    }
  } else if (Native) {
    // native shared library, load it!
    llvm::sys::Path SoFile = L.FindLib(filename);
    assert(!SoFile.isEmpty() && "The shared lib exists but can't find it!");
    std::string errMsg;
    bool hasError = llvm::sys::DynamicLibrary
      ::LoadLibraryPermanently(SoFile.str().c_str(), &errMsg);
    if (hasError) {
      llvm::errs() << "Could not load shared library!\n"
                   << "\n"
                   << errMsg.c_str();
      L.releaseModule();
      return false;
    }
  }
  L.releaseModule();
  return true;
}

static bool canWrapForCall(const std::string& input_line) {
   // Whether input_line can be wrapped into a function.
   // "1" can, "#include <vector>" can't.
   if (input_line.length() > 1 && input_line[0] == '#') return false;
   if (input_line.compare(0, strlen("extern "), "extern ") == 0) return false;
   if (input_line.compare(0, strlen("using "), "using ") == 0) return false;
   return true;
}

} // unnamed namespace

namespace cling {

  // "Declared" to the JIT in RuntimeUniverse.h
  namespace runtime {
    namespace internal {
      int local_cxa_atexit(void (*func) (void*), void* arg,
                           void* dso, Interpreter* interp) {
        return interp->CXAAtExit(func, arg, dso);
      }
      struct __trigger__cxa_atexit {
        ~__trigger__cxa_atexit();
      };
      __trigger__cxa_atexit::~__trigger__cxa_atexit() {}
    }
  }

  // This function isn't referenced outside its translation unit, but it
  // can't use the "static" keyword because its address is used for
  // GetMainExecutable (since some platforms don't support taking the
  // address of main, and some platforms can't implement GetMainExecutable
  // without being given the address of a function in the main executable).
  llvm::sys::Path GetExecutablePath(const char *Argv0) {
    // This just needs to be some symbol in the binary; C++ doesn't
    // allow taking the address of ::main however.
    void *MainAddr = (void*) (intptr_t) GetExecutablePath;
    return llvm::sys::Path::GetMainExecutable(Argv0, MainAddr);
  }

  void Interpreter::unload() {
    m_IncrParser->unloadTransaction(0);
  }

  Interpreter::Interpreter(int argc, const char* const *argv,
                           const char* llvmdir /*= 0*/) :
    m_UniqueCounter(0), m_PrintAST(false), m_DynamicLookupEnabled(false) {

    m_LLVMContext.reset(new llvm::LLVMContext);
    std::vector<unsigned> LeftoverArgsIdx;
    m_Opts = InvocationOptions::CreateFromArgs(argc, argv, LeftoverArgsIdx);
    std::vector<const char*> LeftoverArgs;

    for (size_t I = 0, N = LeftoverArgsIdx.size(); I < N; ++I) {
      LeftoverArgs.push_back(argv[LeftoverArgsIdx[I]]);
    }

    m_IncrParser.reset(new IncrementalParser(this, LeftoverArgs.size(),
                                             &LeftoverArgs[0],
                                             llvmdir));
    m_LookupHelper.reset(new LookupHelper(m_IncrParser->getParser()));

    m_ExecutionContext.reset(new ExecutionContext());

    // Add path to interpreter's include files
    // Try to find the headers in the src folder first
#ifdef CLING_SRCDIR_INCL
    llvm::sys::Path SrcP(CLING_SRCDIR_INCL);
    if (SrcP.canRead())
      AddIncludePath(SrcP.str());
#endif

    llvm::sys::Path P = GetExecutablePath(argv[0]);
    if (!P.isEmpty()) {
      P.eraseComponent();  // Remove /cling from foo/bin/clang
      P.eraseComponent();  // Remove /bin   from foo/bin
      // Get foo/include
      P.appendComponent("include");
      if (P.canRead())
        AddIncludePath(P.str());
      else {
#ifdef CLING_INSTDIR_INCL
        llvm::sys::Path InstP(CLING_INSTDIR_INCL);
        if (InstP.canRead())
          AddIncludePath(InstP.str());
#endif
      }
    }

    m_ExecutionContext->addSymbol("local_cxa_atexit",
                  (void*)(intptr_t)&cling::runtime::internal::local_cxa_atexit);

    // Enable incremental processing, which prevents the preprocessor destroying
    // the lexer on EOF token.
    getSema().getPreprocessor().enableIncrementalProcessing();

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

      // Set up the gCling variable
      std::stringstream initializer;
      initializer << "gCling=(cling::Interpreter*)" << (uintptr_t)this << ";";
      evaluate(initializer.str());
    }
    else {
      declare("#include \"cling/Interpreter/CValuePrinter.h\"");
    }

    handleFrontendOptions();
  }

  Interpreter::~Interpreter() {
    for (size_t I = 0, N = m_AtExitFuncs.size(); I < N; ++I) {
      const CXAAtExitElement& AEE = m_AtExitFuncs[N - I - 1];
      (*AEE.m_Func)(AEE.m_Arg);
    }
  }

  const char* Interpreter::getVersion() const {
    return "$Id$";
  }

  void Interpreter::handleFrontendOptions() {
    if (m_Opts.ShowVersion) {
      llvm::outs() << getVersion() << '\n';
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
    const bool IsUserSupplied = false;
    const bool IsFramework = false;
    const bool IsSysRootRelative = true;
    headerOpts.AddPath(incpath, frontend::Angled, IsUserSupplied, IsFramework,
                       IsSysRootRelative);

    Preprocessor& PP = CI->getPreprocessor();
    ApplyHeaderSearchOptions(PP.getHeaderSearchInfo(), headerOpts,
                                    PP.getLangOpts(),
                                    PP.getTargetInfo().getTriple());
  }

  // Copied from clang/lib/Frontend/CompilerInvocation.cpp
  void Interpreter::DumpIncludePath() {
    const HeaderSearchOptions Opts(getCI()->getHeaderSearchOpts());
    std::vector<std::string> Res;
    if (Opts.Sysroot != "/") {
      Res.push_back("-isysroot");
      Res.push_back(Opts.Sysroot);
    }

    /// User specified include entries.
    for (unsigned i = 0, e = Opts.UserEntries.size(); i != e; ++i) {
      const HeaderSearchOptions::Entry &E = Opts.UserEntries[i];
      if (E.IsFramework && (E.Group != frontend::Angled || !E.IsUserSupplied))
        llvm::report_fatal_error("Invalid option set!");
      if (E.IsUserSupplied) {
        switch (E.Group) {
        case frontend::After:
          Res.push_back("-idirafter");
          break;

        case frontend::Quoted:
          Res.push_back("-iquote");
          break;

        case frontend::System:
          Res.push_back("-isystem");
          break;

        case frontend::IndexHeaderMap:
          Res.push_back("-index-header-map");
          Res.push_back(E.IsFramework? "-F" : "-I");
          break;

        case frontend::CSystem:
          Res.push_back("-c-isystem");
          break;

        case frontend::CXXSystem:
          Res.push_back("-cxx-isystem");
          break;

        case frontend::ObjCSystem:
          Res.push_back("-objc-isystem");
          break;

        case frontend::ObjCXXSystem:
          Res.push_back("-objcxx-isystem");
          break;

        case frontend::Angled:
          Res.push_back(E.IsFramework ? "-F" : "-I");
          break;
        }
      } else {
        if (E.Group != frontend::Angled && E.Group != frontend::System)
          llvm::report_fatal_error("Invalid option set!");
        Res.push_back(E.Group == frontend::Angled ? "-iwithprefixbefore" :
                      "-iwithprefix");
      }
      Res.push_back(E.Path);
    }

    if (!Opts.ResourceDir.empty()) {
      Res.push_back("-resource-dir");
      Res.push_back(Opts.ResourceDir);
    }
    if (!Opts.ModuleCachePath.empty()) {
      Res.push_back("-fmodule-cache-path");
      Res.push_back(Opts.ModuleCachePath);
    }
    if (!Opts.UseStandardSystemIncludes)
      Res.push_back("-nostdinc");
    if (!Opts.UseStandardCXXIncludes)
      Res.push_back("-nostdinc++");
    if (Opts.UseLibcxx)
      Res.push_back("-stdlib=libc++");
    if (Opts.Verbose)
      Res.push_back("-v");

    // print'em all
    for (unsigned i = 0; i < Res.size(); ++i) {
      llvm::outs() << Res[i] <<"\n";
    }
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
                       const Decl** D /* = 0 */) {
    CompilationOptions CO;
    CO.DeclarationExtraction = 1;
    CO.ValuePrinting = CompilationOptions::VPAuto;
    CO.DynamicScoping = isDynamicLookupEnabled();
    CO.Debug = isPrintingAST();

    if (!canWrapForCall(input))
      return declare(input, D);

    if (EvaluateInternal(input, CO, V) == Interpreter::kFailure) {
      if (D)
        *D = 0;
      return Interpreter::kFailure;
    }

    if (D)
      *D = m_IncrParser->getLastTransaction()->getFirstDecl().getSingleDecl();

    return Interpreter::kSuccess;
  }

  Interpreter::CompilationResult
  Interpreter::parse(const std::string& input) {
    CompilationOptions CO;
    CO.CodeGeneration = 0;
    CO.DeclarationExtraction = 0;
    CO.ValuePrinting = 0;
    CO.DynamicScoping = isDynamicLookupEnabled();
    CO.Debug = isPrintingAST();

    return DeclareInternal(input, CO);
  }

  Interpreter::CompilationResult
  Interpreter::declare(const std::string& input, const Decl** D /* = 0 */) {
    CompilationOptions CO;
    CO.DeclarationExtraction = 0;
    CO.ValuePrinting = 0;
    CO.DynamicScoping = isDynamicLookupEnabled();
    CO.Debug = isPrintingAST();

    return DeclareInternal(input, CO, D);
  }

  Interpreter::CompilationResult
  Interpreter::evaluate(const std::string& input, StoredValueRef* V /* = 0 */) {
    // Here we might want to enforce further restrictions like: Only one
    // ExprStmt can be evaluated and etc. Such enforcement cannot happen in the
    // worker, because it is used from various places, where there is no such
    // rule
    CompilationOptions CO;
    CO.DeclarationExtraction = 0;
    CO.ValuePrinting = 0;

    return EvaluateInternal(input, CO, V);
  }

  Interpreter::CompilationResult
  Interpreter::echo(const std::string& input, StoredValueRef* V /* = 0 */) {
    CompilationOptions CO;
    CO.DeclarationExtraction = 0;
    CO.ValuePrinting = CompilationOptions::VPEnabled;

    return EvaluateInternal(input, CO, V);
  }

  void Interpreter::WrapInput(std::string& input, std::string& fname) {
    fname = createUniqueWrapper();
    input.insert(0, "void " + fname + "() {\n ");
    input.append("\n;\n}");
  }

  bool Interpreter::RunFunction(llvm::StringRef fname,
                                clang::QualType retType,
                                StoredValueRef* res) {
    if (getCI()->getDiagnostics().hasErrorOccurred())
      return false;

    if (!m_IncrParser->hasCodeGenerator()) {
      return true;
    }

    std::string mangledNameIfNeeded;

    Sema& S = getCI()->getSema();
    FunctionDecl* FD 
      = cast_or_null<FunctionDecl>(utils::Lookup::Named(&S, fname.str().c_str()));
    
    if (FD) {
      mangleName(FD, mangledNameIfNeeded);
      m_ExecutionContext->executeFunction(mangledNameIfNeeded.c_str(),
                                          getCI()->getASTContext(),
                                          retType, res);
      return true;
    }

    return false;
  }

  void Interpreter::createUniqueName(std::string& out) {
    out = "Un1Qu3";
    llvm::raw_string_ostream(out) << m_UniqueCounter++;
  }

  bool Interpreter::isUniqueName(llvm::StringRef name) {
    return name.startswith("Un1Qu3");
  }

  llvm::StringRef Interpreter::createUniqueWrapper() {
    const size_t size = sizeof("__cling_Un1Qu3") + sizeof(m_UniqueCounter);
    llvm::SmallString<size> out("__cling_Un1Qu3");
    llvm::raw_svector_ostream(out) << m_UniqueCounter++;

    return (getCI()->getASTContext().Idents.getOwn(out)).getName();
  }

  bool Interpreter::isUniqueWrapper(llvm::StringRef name) {
    return name.startswith("__cling_Un1Qu3");
  }

  Interpreter::CompilationResult
  Interpreter::DeclareInternal(const std::string& input, 
                               const CompilationOptions& CO,
                               const clang::Decl** D /* = 0 */) {

    if (m_IncrParser->Compile(input, CO) != IncrementalParser::kFailed) {
      if (D)
        *D = m_IncrParser->getLastTransaction()->getFirstDecl().getSingleDecl();
      return Interpreter::kSuccess;
    }

    return Interpreter::kFailure;
  }

  Interpreter::CompilationResult
  Interpreter::EvaluateInternal(const std::string& input, 
                                const CompilationOptions& CO,
                                StoredValueRef* V /* = 0 */) {

    Sema& TheSema = getCI()->getSema();

    DiagnosticsEngine& Diag = getCI()->getDiagnostics();
    // Disable warnings which doesn't make sense when using the prompt
    // This gets reset with the clang::Diagnostics().Reset()
    Diag.setDiagnosticMapping(clang::diag::warn_unused_expr,
                              clang::diag::MAP_IGNORE, SourceLocation());
    Diag.setDiagnosticMapping(clang::diag::warn_unused_call,
                              clang::diag::MAP_IGNORE, SourceLocation());
    Diag.setDiagnosticMapping(clang::diag::warn_unused_comparison,
                              clang::diag::MAP_IGNORE, SourceLocation());

    // Wrap the expression
    std::string WrapperName;
    std::string Wrapper = input;
    WrapInput(Wrapper, WrapperName);
    QualType RetTy = getCI()->getASTContext().VoidTy;

    if (V) {
      const Transaction* CurT = m_IncrParser->Parse(Wrapper);
      assert(CurT->size() && "No decls created by Parse!");

      // Find the wrapper function declaration.
      //
      // Note: The parse may have created a whole set of decls if a template
      //       instantiation happened.  Our wrapper function should be the
      //       last decl in the set.
      //
      FunctionDecl* TopLevelFD 
        = dyn_cast<FunctionDecl>(CurT->getLastDecl().getSingleDecl());
      assert(TopLevelFD && "No Decls Parsed?");
      DeclContext* CurContext = TheSema.CurContext;
      TheSema.CurContext = TopLevelFD;
      ASTContext& Context(getCI()->getASTContext());
      // We have to be able to mark the expression for printout. There are three
      // scenarios:
      // 0: Expression printing disabled - don't do anything just disable the
      //    consumer
      //    is our marker, even if there wasn't missing ';'.
      // 1: Expression printing enabled - make sure we don't have NullStmt,
      //    which is used as a marker to suppress the print out.
      // 2: Expression printing auto - do nothing - rely on the omitted ';' to
      //    not produce the suppress marker.
      if (CompoundStmt* CS = dyn_cast<CompoundStmt>(TopLevelFD->getBody())) {
        // Collect all Stmts, contained in the CompoundStmt
        llvm::SmallVector<Stmt *, 4> Stmts;
        for (CompoundStmt::body_iterator iStmt = CS->body_begin(),
               eStmt = CS->body_end(); iStmt != eStmt; ++iStmt)
          Stmts.push_back(*iStmt);

        size_t indexOfLastExpr = Stmts.size();
        while(indexOfLastExpr--) {
          // find the trailing expression statement (skip e.g. null statements)
          if (Expr* E = dyn_cast_or_null<Expr>(Stmts[indexOfLastExpr])) {
            RetTy = E->getType();
            if (!RetTy->isVoidType()) {
              // Change the void function's return type
              FunctionProtoType::ExtProtoInfo EPI;
              QualType FuncTy = Context.getFunctionType(RetTy,/* ArgArray = */0,
                                                        /* NumArgs = */0, EPI);
              TopLevelFD->setType(FuncTy);
              // Strip the parenthesis if any
              if (ParenExpr* PE = dyn_cast<ParenExpr>(E))
                E = PE->getSubExpr();

              // Change it with return stmt
              Stmts[indexOfLastExpr]
                = TheSema.ActOnReturnStmt(SourceLocation(), E).take();
            }
            // even if void: we found an expression
            break;
          }
        }

        // case 1:
        if (CO.ValuePrinting == CompilationOptions::VPEnabled)
          if (indexOfLastExpr < Stmts.size() - 1 &&
              isa<NullStmt>(Stmts[indexOfLastExpr + 1]))
            Stmts.erase(Stmts.begin() + indexOfLastExpr);
        // Stmts.insert(Stmts.begin() + indexOfLastExpr + 1,
        //              TheSema.ActOnNullStmt(SourceLocation()).take());

        // Update the CompoundStmt body
        CS->setStmts(TheSema.getASTContext(), Stmts.data(), Stmts.size());

      }

      TheSema.CurContext = CurContext;

      m_IncrParser->commitCurrentTransaction();
    }
    else
      m_IncrParser->Compile(Wrapper, CO);

    // get the result
    if (!V) {
       if (RunFunction(WrapperName, RetTy)) {
        return Interpreter::kSuccess;
      }
    } else if (RunFunction(WrapperName, RetTy, V)) {
      return Interpreter::kSuccess;
    } else {
       *V = StoredValueRef::invalidValue();
    }

    return Interpreter::kFailure;
  }

  bool Interpreter::loadFile(const std::string& filename,
                             bool allowSharedLib /*=true*/) {
    if (allowSharedLib) {
      llvm::Module* module = m_IncrParser->getCodeGenerator()->GetModule();
      if (module) {
        if (tryLinker(filename, getOptions(), module))
          return true;
        if (filename.compare(0, 3, "lib") == 0) {
          // starts with "lib", try without (the llvm::Linker forces
          // a "lib" in front, which makes it liblib...
          if (tryLinker(filename.substr(3, std::string::npos),
                        getOptions(), module))
            return true;
        }
      }
    }

    std::string code;
    code += "#include \"" + filename + "\"";
    return declare(code) == Interpreter::kSuccess;
  }

  void Interpreter::installLazyFunctionCreator(void* (*fp)(const std::string&)) {
    m_ExecutionContext->installLazyFunctionCreator(fp);
  }

  StoredValueRef Interpreter::Evaluate(const char* expr, DeclContext* DC,
                                       bool ValuePrinterReq) {
    Sema& TheSema = getCI()->getSema();
    if (!DC)
      DC = TheSema.getASTContext().getTranslationUnitDecl();

    // Set up the declaration context
    DeclContext* CurContext;

    CurContext = TheSema.CurContext;
    TheSema.CurContext = DC;

    StoredValueRef Result;
    if (TheSema.getExternalSource()) {
      (ValuePrinterReq) ? echo(expr, &Result) : evaluate(expr, &Result);
    }
    else
      (ValuePrinterReq) ? echo(expr, &Result) : evaluate(expr, &Result);

    TheSema.CurContext = CurContext;

    return Result;
  }

  void Interpreter::setCallbacks(InterpreterCallbacks* C) {
    // We need it to enable LookupObject callback.
    m_Callbacks.reset(C);
  }

  void Interpreter::enableDynamicLookup(bool value /*=true*/) {
    m_DynamicLookupEnabled = value;

    if (isDynamicLookupEnabled()) {
      declare("#include \"cling/Interpreter/DynamicLookupRuntimeUniverse.h\"");
    }
    else
      setCallbacks(0);
  }

  void Interpreter::runStaticInitializersOnce() const {
    // Forward to ExecutionContext; should not be called by
    // anyone except for IncrementalParser.
    llvm::Module* module = m_IncrParser->getCodeGenerator()->GetModule();
    m_ExecutionContext->runStaticInitializersOnce(module);
  }

  int Interpreter::CXAAtExit(void (*func) (void*), void* arg, void* dso) {
    // Register a CXAAtExit function
    Decl* LastTLD 
      = m_IncrParser->getLastTransaction()->getLastDecl().getSingleDecl();
    m_AtExitFuncs.push_back(CXAAtExitElement(func, arg, dso, LastTLD));
    return 0; // happiness
  }

  void Interpreter::mangleName(const clang::NamedDecl* D,
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
      m_MangleCtx->mangleName(D, RawStr);
      RawStr.flush();
    } else {
      mangledName = D->getNameAsString();
    }
  }

  bool Interpreter::addSymbol(const char* symbolName,  void* symbolAddress) {
    // Forward to ExecutionContext;
    if (!symbolName || !symbolAddress )
      return false;

    return m_ExecutionContext->addSymbol(symbolName,  symbolAddress);
  }


  void* Interpreter::getAddressOfGlobal(const clang::NamedDecl* D,
                                        bool* fromJIT /*=0*/) const {
    // Return a symbol's address, and whether it was jitted.
    std::string mangledName;
    mangleName(D, mangledName);
    llvm::Module* module = m_IncrParser->getCodeGenerator()->GetModule();
    return m_ExecutionContext->getAddressOfGlobal(module,
                                                  mangledName.c_str(),
                                                  fromJIT);
  }

} // namespace cling
