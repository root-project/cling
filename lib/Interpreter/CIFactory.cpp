//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Axel Naumann <axel@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "ClingUtils.h"
#include "DeclCollector.h"
#include "cling-compiledata.h"

#include "cling/Interpreter/CIFactory.h"
#include "cling/Interpreter/InvocationOptions.h"
#include "cling/Utils/Paths.h"
#include "cling/Utils/Platform.h"

#include "clang/AST/ASTContext.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/Version.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Job.h"
#include "clang/Driver/Tool.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Frontend/VerifyDiagnosticConsumer.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "clang/Serialization/ASTReader.h"

#include "llvm/Config/llvm-config.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetOptions.h"

#include <cstdio>
#include <ctime>
#include <memory>

using namespace clang;
using namespace cling;

namespace {
  // This function isn't referenced outside its translation unit, but it
  // can't use the "static" keyword because its address is used for
  // GetMainExecutable (since some platforms don't support taking the
  // address of main, and some platforms can't implement GetMainExecutable
  // without being given the address of a function in the main executable).
  std::string GetExecutablePath(const char *Argv0) {
    // This just needs to be some symbol in the binary; C++ doesn't
    // allow taking the address of ::main however.
    void *MainAddr = (void*) intptr_t(GetExecutablePath);
    return llvm::sys::fs::getMainExecutable(Argv0, MainAddr);
  }

  class AdditionalArgList {
    typedef std::vector< std::pair<const char*,std::string> > container_t;
    container_t m_Saved;

  public:
    
    void addArgument(const char* arg, std::string value = std::string()) {
      m_Saved.push_back(std::make_pair(arg,std::move(value)));
    }
    container_t::const_iterator begin() const { return m_Saved.begin(); }
    container_t::const_iterator end() const { return m_Saved.end(); }
    bool empty() const { return m_Saved.empty(); }
  };

#ifndef _MSC_VER

  static void ReadCompilerIncludePaths(const char* Compiler,
                                       llvm::SmallVectorImpl<char>& Buf,
                                       AdditionalArgList& Args,
                                       bool Verbose) {
    std::string CppInclQuery("LC_ALL=C ");
    CppInclQuery.append(Compiler);
    CppInclQuery.append(" -xc++ -E -v /dev/null 2>&1 >/dev/null "
                        "| awk '/^#include </,/^End of search"
                        "/{if (!/^#include </ && !/^End of search/){ print }}' "
                        "| GREP_OPTIONS= grep -E \"(c|g)\\+\\+\"");

    if (Verbose)
      llvm::errs() << "Looking for C++ headers with:\n  " << CppInclQuery << "\n";

    if (FILE *PF = ::popen(CppInclQuery.c_str(), "r")) {
      Buf.resize(Buf.capacity_in_bytes());
      while (fgets(&Buf[0], Buf.capacity_in_bytes(), PF) && Buf[0]) {
        llvm::StringRef Path(&Buf[0]);
        // Skip leading and trailing whitespace
        Path = Path.trim();
        if (!Path.empty()) {
          if (!llvm::sys::fs::is_directory(Path)) {
            if (Verbose)
              cling::utils::LogNonExistantDirectory(Path);
          }
          else
            Args.addArgument("-I", Path.str());
        }
      }
      ::pclose(PF);
    } else {
      llvm::errs() << "popen failed";
      // Don't be overly verbose, we already printed the command
      if (!Verbose)
        llvm::errs() << " for '" << CppInclQuery << "'\n";
    }

    // Return the query in Buf on failure
    if (Args.empty()) {
      Buf.resize(0);
      Buf.insert(Buf.begin(), CppInclQuery.begin(), CppInclQuery.end());
    } else if (Verbose) {
      llvm::errs() << "Found:\n";
      for (const auto& Arg : Args)
        llvm::errs() << "  " << Arg.second << "\n";
    }
  }

  static bool AddCxxPaths(llvm::StringRef PathStr, AdditionalArgList& Args,
                          bool Verbose) {
    if (Verbose)
      llvm::errs() << "Looking for C++ headers in \"" << PathStr << "\"\n";

    llvm::SmallVector<llvm::StringRef, 6> Paths;
    if (!utils::SplitPaths(PathStr, Paths, utils::kFailNonExistant,
                           platform::kEnvDelim, Verbose))
      return false;

    if (Verbose) {
      llvm::errs() << "Found:\n";
      for (llvm::StringRef Path : Paths)
        llvm::errs() << " " << Path << "\n";
    }

    for (llvm::StringRef Path : Paths)
      Args.addArgument("-I", Path.str());

    return true;
  }

#endif
  
  ///\brief Adds standard library -I used by whatever compiler is found in PATH.
  static void AddHostArguments(llvm::StringRef clingBin,
                               std::vector<const char*>& args,
                               const char* llvmdir, const CompilerOptions& opts) {
    static AdditionalArgList sArguments;
    if (sArguments.empty()) {
      const bool Verbose = opts.Verbose;
#ifdef _MSC_VER
      // Honor %INCLUDE%. It should know essential search paths with vcvarsall.bat.
      if (const char* Includes = getenv("INCLUDE")) {
        SmallVector<StringRef, 8> Dirs;
        utils::SplitPaths(Includes, Dirs, utils::kAllowNonExistant,
                          platform::kEnvDelim, Verbose);
        for (const llvm::StringRef& Path : Dirs)
          sArguments.addArgument("-I", Path.str());
      }

      // When built with access to the proper Windows APIs, try to actually find
      // the correct include paths first. Init for UnivSDK.empty check below.
      std::string VSDir, WinSDK,
                  UnivSDK(opts.NoBuiltinInc ? "" : CLING_UCRT_VERSION);
      if (platform::GetVisualStudioDirs(VSDir,
                                        opts.NoBuiltinInc ? nullptr : &WinSDK,
                                        opts.NoBuiltinInc ? nullptr : &UnivSDK,
                                        Verbose)) {
        if (!opts.NoCXXInc) {
          const std::string VSIncl = VSDir + "\\VC\\include";
          if (Verbose)
            llvm::errs() << "Adding VisualStudio SDK: '" << VSIncl << "'\n";
          sArguments.addArgument("-I", std::move(VSIncl));
        }
        if (!opts.NoBuiltinInc) {
          if (!WinSDK.empty()) {
            WinSDK.append("\\include");
            if (Verbose)
              llvm::errs() << "Adding Windows SDK: '" << WinSDK << "'\n";
            sArguments.addArgument("-I", std::move(WinSDK));
          } else {
            VSDir.append("\\VC\\PlatformSDK\\Include");
            if (Verbose)
              llvm::errs() << "Adding Platform SDK: '" << VSDir << "'\n";
            sArguments.addArgument("-I", std::move(VSDir));
          }
        }
      }

#if LLVM_MSC_PREREQ(1900)
      if (!UnivSDK.empty()) {
        if (Verbose)
          llvm::errs() << "Adding UniversalCRT SDK: '" << UnivSDK << "'\n";
        sArguments.addArgument("-I", std::move(UnivSDK));
      }
#endif

#else // _MSC_VER

      // Skip LLVM_CXX execution if -nostdinc++ was provided.
      if (!opts.NoCXXInc) {
        // Need sArguments.empty as a check condition later
        assert(sArguments.empty() && "Arguments not empty");

        SmallString<2048> buffer;

  #ifdef _LIBCPP_VERSION
        // Try to use a version of clang that is located next to cling
        // in case cling was built with a new/custom libc++
        std::string clang = llvm::sys::path::parent_path(clingBin);
        buffer.assign(clang);
        llvm::sys::path::append(buffer, "clang");
        clang.assign(&buffer[0], buffer.size());

        if (llvm::sys::fs::is_regular_file(clang)) {
          if (!opts.StdLib) {
  #if defined(_LIBCPP_VERSION)
            clang.append(" -stdlib=libc++");
  #elif defined(__GLIBCXX__)
            clang.append(" -stdlib=libstdc++");
  #endif
          }
          ReadCompilerIncludePaths(clang.c_str(), buffer, sArguments, Verbose);
        }
  #endif // _LIBCPP_VERSION

  // first try the include directory cling was built with
  #ifdef CLING_CXX_INCL
        if (sArguments.empty())
          AddCxxPaths(CLING_CXX_INCL, sArguments, Verbose);
  #endif
  // Then try the absolute path i.e.: '/usr/bin/g++'
  #ifdef CLING_CXX_PATH
        if (sArguments.empty())
          ReadCompilerIncludePaths(CLING_CXX_PATH, buffer, sArguments, Verbose);
  #endif
  // Finally try the relative path 'g++'
  #ifdef CLING_CXX_RLTV
        if (sArguments.empty())
          ReadCompilerIncludePaths(CLING_CXX_RLTV, buffer, sArguments, Verbose);
  #endif

        if (sArguments.empty()) {
          // buffer is a copy of the query string that failed
          llvm::errs() << "ERROR in cling::CIFactory::createCI(): cannot extract"
                          " standard library include paths!\n";

  #if defined(CLING_CXX_PATH) || defined(CLING_CXX_RLTV)
          // Only when ReadCompilerIncludePaths called do we have the command
          // Verbose has already printed the command
          if (!Verbose)
            llvm::errs() << "Invoking:\n  " << buffer.c_str() << "\n";

          llvm::errs() << "Results was:\n";
          const int ExitCode = system(buffer.c_str());
          llvm::errs() << "With exit code " << ExitCode << "\n";
  #elif !defined(CLING_CXX_INCL)
          // Technically a valid configuration that just wants to use libClangs
          // internal header detection, but for now give a hint about why.
          llvm::errs() << "CLING_CXX_INCL, CLING_CXX_PATH, and CLING_CXX_RLTV"
                          " are undefined, there was probably an error during"
                          " configuration.\n";
  #endif
        } else
          sArguments.addArgument("-nostdinc++");
      }

  #if defined(__APPLE__)

      if (!opts.NoBuiltinInc && !opts.SysRoot) {
        std::string sysRoot;
        if (platform::GetISysRoot(sysRoot, Verbose)) {
          if (Verbose)
            llvm::errs() << "Using SDK \"" << sysRoot << "\"\n";
          sArguments.addArgument("-isysroot", std::move(sysRoot));
        }
      }

    #if defined(__GLIBCXX__)
      // Avoid '__float128 is not supported on this target' errors
      if (!opts.StdVersion)
        sArguments.addArgument("-std=c++11");
    #endif //__GLIBCXX__
  #endif // __APPLE__

#endif // _MSC_VER

      if (!opts.ResourceDir && !opts.NoBuiltinInc) {
        std::string resourcePath;
        if (!llvmdir) {
          // FIXME: The first arg really does need to be argv[0] on FreeBSD.
          //
          // Note: The second arg is not used for Apple, FreeBSD, Linux,
          //       or cygwin, and can only be used on systems which support
          //       the use of dladdr().
          //
          // Note: On linux and cygwin this uses /proc/self/exe to find the path
          // Note: On Apple it uses _NSGetExecutablePath().
          // Note: On FreeBSD it uses getprogpath().
          // Note: Otherwise it uses dladdr().
          //
          resourcePath
            = CompilerInvocation::GetResourcesPath("cling",
                                            (void*)intptr_t(GetExecutablePath));
        } else {
          llvm::SmallString<512> tmp(llvmdir);
          llvm::sys::path::append(tmp, "lib", "clang", CLANG_VERSION_STRING);
          resourcePath.assign(&tmp[0], tmp.size());
        }

        // FIXME: Handle cases, where the cling is part of a library/framework.
        // There we can't rely on the find executable logic.
        if (!llvm::sys::fs::is_directory(resourcePath)) {
          llvm::errs()
            << "ERROR in cling::CIFactory::createCI():\n  resource directory "
            << resourcePath << " not found!\n";
          resourcePath = "";
        } else {
          sArguments.addArgument("-resource-dir", std::move(resourcePath));
        }
      }
    }

    for (auto& arg : sArguments) {
      args.push_back(arg.first);
      args.push_back(arg.second.c_str());
    }
  }

  static void SetClingCustomLangOpts(LangOptions& Opts) {
    Opts.EmitAllDecls = 0; // Otherwise if PCH attached will codegen all decls.
#ifdef _MSC_VER
    Opts.Exceptions = 0;
    if (Opts.CPlusPlus) {
      Opts.CXXExceptions = 0;
    }
#else
    Opts.Exceptions = 1;
    if (Opts.CPlusPlus) {
      Opts.CXXExceptions = 1;
    }
#endif // _MSC_VER
    Opts.Deprecated = 1;
    //Opts.Modules = 1;

    // See test/CodeUnloading/PCH/VTables.cpp which implicitly compares clang
    // to cling lang options. They should be the same, we should not have to
    // give extra lang options to their invocations on any platform.
    // Except -fexceptions -fcxx-exceptions.

    Opts.Deprecated = 1;
    Opts.GNUKeywords = 0;
    Opts.Trigraphs = 1; // o no??! but clang has it on by default...

#ifdef __APPLE__
    Opts.Blocks = 1;
    Opts.MathErrno = 0;
#endif

    // C++11 is turned on if cling is built with C++11: it's an interpreter;
    // cross-language compilation doesn't make sense.
    // Extracted from Boost/config/compiler.
    // SunProCC has no C++11.
    // VisualC's support is not obvious to extract from Boost...

    // The value of __cplusplus in GCC < 5.0 (e.g. 4.9.3) when
    // either -std=c++1y or -std=c++14 is specified is 201300L, which fails
    // the test for C++14 or more (201402L) as previously specified.
    // I would claim that the check should be relaxed to:

#if __cplusplus > 201103L
    if (Opts.CPlusPlus) Opts.CPlusPlus14 = 1;
#endif
#if __cplusplus >= 201103L
    if (Opts.CPlusPlus) Opts.CPlusPlus11 = 1;
#endif

#ifdef _REENTRANT
    Opts.POSIXThreads = 1;
#endif
  }

  static void SetClingTargetLangOpts(LangOptions& Opts,
                                     const TargetInfo& Target) {
    if (Target.getTriple().getOS() == llvm::Triple::Win32) {
      Opts.MicrosoftExt = 1;
#ifdef _MSC_VER
      Opts.MSCompatibilityVersion = (_MSC_VER * 100000);
#endif
      // Should fix http://llvm.org/bugs/show_bug.cgi?id=10528
      Opts.DelayedTemplateParsing = 1;
    } else {
      Opts.MicrosoftExt = 0;
    }
  }

  // This must be a copy of clang::getClangToolFullVersion(). Luckily
  // we'll notice quickly if it ever changes! :-)
  static std::string CopyOfClanggetClangToolFullVersion(StringRef ToolName) {
    std::string buf;
    llvm::raw_string_ostream OS(buf);
#ifdef CLANG_VENDOR
    OS << CLANG_VENDOR;
#endif
    OS << ToolName << " version " CLANG_VERSION_STRING " "
       << getClangFullRepositoryVersion();

    // If vendor supplied, include the base LLVM version as well.
#ifdef CLANG_VENDOR
    OS << " (based on LLVM " << PACKAGE_VERSION << ")";
#endif

    return OS.str();
  }

  ///\brief Check the compile-time clang version vs the run-time clang version,
  /// a mismatch could cause havoc. Reports if clang versions differ.
  static void CheckClangCompatibility() {
    if (clang::getClangToolFullVersion("cling")
        != CopyOfClanggetClangToolFullVersion("cling"))
      llvm::errs()
        << "Warning in cling::CIFactory::createCI():\n  "
        "Using incompatible clang library! "
        "Please use the one provided by cling!\n";
    return;
  }

  /// \brief Retrieves the clang CC1 specific flags out of the compilation's
  /// jobs. Returns NULL on error.
  static const llvm::opt::ArgStringList
  *GetCC1Arguments(clang::DiagnosticsEngine *Diagnostics,
                   clang::driver::Compilation *Compilation) {
    // We expect to get back exactly one Command job, if we didn't something
    // failed. Extract that job from the Compilation.
    const clang::driver::JobList &Jobs = Compilation->getJobs();
    if (!Jobs.size() || !isa<clang::driver::Command>(*Jobs.begin())) {
      // diagnose this...
      return NULL;
    }

    // The one job we find should be to invoke clang again.
    const clang::driver::Command *Cmd
      = cast<clang::driver::Command>(&(*Jobs.begin()));
    if (llvm::StringRef(Cmd->getCreator().getName()) != "clang") {
      // diagnose this...
      return NULL;
    }

    return &Cmd->getArguments();
  }

  /// Set cling's preprocessor defines to match the cling binary.
  static void SetPreprocessorFromBinary(PreprocessorOptions& PPOpts) {
#ifdef _MSC_VER
    PPOpts.addMacroDef("_HAS_EXCEPTIONS=0");
#ifdef _DEBUG
    PPOpts.addMacroDef("_DEBUG=1");
#elif defined(NDEBUG)
    PPOpts.addMacroDef("NDEBUG=1");
#else // well, what else?
    PPOpts.addMacroDef("NDEBUG=1");
#endif
#endif

    // Since cling, uses clang instead, macros always sees __CLANG__ defined
    // In addition, clang also defined __GNUC__, we add the following two macros
    // to allow scripts, and more important, dictionary generation to know which
    // of the two is the underlying compiler.

#ifdef __clang__
    PPOpts.addMacroDef("__CLING__clang__=" ClingStringify(__clang__));
#elif defined(__GNUC__)
    PPOpts.addMacroDef("__CLING__GNUC__=" ClingStringify(__GNUC__));
#endif

// https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_dual_abi.html
#ifdef _GLIBCXX_USE_CXX11_ABI
    PPOpts.addMacroDef("_GLIBCXX_USE_CXX11_ABI="
                       ClingStringify(_GLIBCXX_USE_CXX11_ABI));
#endif
  }

  /// Set target-specific preprocessor defines.
  static void SetPreprocessorFromTarget(PreprocessorOptions& PPOpts,
                                        const llvm::Triple& TTriple) {
    if (TTriple.getEnvironment() == llvm::Triple::Cygnus) {
      // clang "forgets" the basic arch part needed by winnt.h:
      if (TTriple.getArch() == llvm::Triple::x86) {
        PPOpts.addMacroDef("_X86_=1");
      } else if (TTriple.getArch() == llvm::Triple::x86_64) {
        PPOpts.addMacroDef("__x86_64=1");
      } else {
        llvm::errs() << "Warning in cling::CIFactory::createCI():\n"
          "unhandled target architecture "
        << TTriple.getArchName() << '\n';
      }
    }
  }

  template <class CONTAINER>
  static void insertBehind(CONTAINER& To, const CONTAINER& From) {
    To.insert(To.end(), From.begin(), From.end());
  }

  static void AddRuntimeIncludePaths(llvm::StringRef ClingBin,
                                     clang::HeaderSearchOptions& HOpts) {
    if (HOpts.Verbose)
      llvm::errs() << "Adding runtime include paths:\n";
    // Add configuration paths to interpreter's include files.
#ifdef CLING_INCLUDE_PATHS
    if (HOpts.Verbose)
      llvm::errs() << "  \"" CLING_INCLUDE_PATHS "\"\n";
    utils::AddIncludePaths(CLING_INCLUDE_PATHS, HOpts);
#endif
    llvm::SmallString<512> P(ClingBin);
    if (!P.empty()) {
      // Remove /cling from foo/bin/clang
      llvm::StringRef ExeIncl = llvm::sys::path::parent_path(P);
      // Remove /bin   from foo/bin
      ExeIncl = llvm::sys::path::parent_path(ExeIncl);
      P.resize(ExeIncl.size());
      // Get foo/include
      llvm::sys::path::append(P, "include");
      if (llvm::sys::fs::is_directory(P.str()))
        utils::AddIncludePaths(P.str(), HOpts, nullptr);
    }
  }

  static CompilerInstance*
  createCIImpl(std::unique_ptr<llvm::MemoryBuffer> Buffer,
               const CompilerOptions& COpts, const char* LLVMDir,
               bool OnlyLex) {
    // Follow clang -v convention of printing version on first line
    if (COpts.Verbose)
      llvm::errs() << "cling version " << ClingStringify(CLING_VERSION) << '\n';

    // Create an instance builder, passing the LLVMDir and arguments.
    //

    CheckClangCompatibility();

    //  Initialize the llvm library.
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmParser();
    llvm::InitializeNativeTargetAsmPrinter();

    const size_t argc = COpts.Remaining.size();
    const char* const* argv = &COpts.Remaining[0];
    std::vector<const char*> argvCompile(argv, argv+1);
    argvCompile.reserve(argc+5);

    if (!COpts.Language) {
      // We do C++ by default; append right after argv[0] if no "-x" given
      argvCompile.push_back("-x");
      argvCompile.push_back( "c++");
    }
    // argv[0] already inserted, get the rest
    argvCompile.insert(argvCompile.end(), argv+1, argv + argc);

    // Add host specific includes, -resource-dir if necessary, and -isysroot
    std::string ClingBin = GetExecutablePath(argv[0]);
    AddHostArguments(ClingBin, argvCompile, LLVMDir, COpts);

    // Be explicit about the stdlib on OS X
    // Would be nice on Linux but will warn 'argument unused during compilation'
    // when -nostdinc++ is passed
#ifdef __APPLE__
      if (!COpts.StdLib) {
  #ifdef _LIBCPP_VERSION
        argvCompile.push_back("-stdlib=libc++");
  #elif defined(__GLIBCXX__)
        argvCompile.push_back("-stdlib=libstdc++");
  #endif
      }
#endif

    argvCompile.push_back("-c");
    argvCompile.push_back("-");

    clang::CompilerInvocation*
      Invocation = new clang::CompilerInvocation;
    // The compiler invocation is the owner of the diagnostic options.
    // Everything else points to them.
    DiagnosticOptions& DiagOpts = Invocation->getDiagnosticOpts();
    TextDiagnosticPrinter* DiagnosticPrinter
      = new TextDiagnosticPrinter(llvm::errs(), &DiagOpts);
    llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs> DiagIDs(new DiagnosticIDs());
    llvm::IntrusiveRefCntPtr<DiagnosticsEngine>
      Diags(new DiagnosticsEngine(DiagIDs, &DiagOpts,
                                  DiagnosticPrinter, /*Owns it*/ true));
    clang::driver::Driver Driver(argv[0], llvm::sys::getDefaultTargetTriple(),
                                 *Diags);
    //Driver.setWarnMissingInput(false);
    Driver.setCheckInputsExist(false); // think foo.C(12)
    llvm::ArrayRef<const char*>RF(&(argvCompile[0]), argvCompile.size());
    std::unique_ptr<clang::driver::Compilation>
      Compilation(Driver.BuildCompilation(RF));
    const clang::driver::ArgStringList* CC1Args
      = GetCC1Arguments(Diags.get(), Compilation.get());
    if (CC1Args == NULL) {
      delete Invocation;
      return 0;
    }

    clang::CompilerInvocation::CreateFromArgs(*Invocation, CC1Args->data() + 1,
                                              CC1Args->data() + CC1Args->size(),
                                              *Diags);
    // We appreciate the error message about an unknown flag (or do we? if not
    // we should switch to a different DiagEngine for parsing the flags).
    // But in general we'll happily go on.
    Diags->Reset();

    // Create and setup a compiler instance.
    std::unique_ptr<CompilerInstance> CI(new CompilerInstance());
    CI->createFileManager();

    llvm::StringRef PCHFileName
      = Invocation->getPreprocessorOpts().ImplicitPCHInclude;
    if (!PCHFileName.empty()) {
      // Load target options etc from PCH.
      struct PCHListener: public ASTReaderListener {
        CompilerInvocation& m_Invocation;

        PCHListener(CompilerInvocation& I): m_Invocation(I) {}

        bool ReadLanguageOptions(const LangOptions &LangOpts,
                                 bool /*Complain*/,
                                 bool /*AllowCompatibleDifferences*/) override {
          *m_Invocation.getLangOpts() = LangOpts;
          return true;
        }
        bool ReadTargetOptions(const TargetOptions &TargetOpts,
                               bool /*Complain*/,
                               bool /*AllowCompatibleDifferences*/) override {
          m_Invocation.getTargetOpts() = TargetOpts;
          return true;
        }
        bool ReadPreprocessorOptions(const PreprocessorOptions &PPOpts,
                                     bool /*Complain*/,
                                std::string &/*SuggestedPredefines*/) override {
          // Import selected options, e.g. don't overwrite ImplicitPCHInclude.
          PreprocessorOptions& myPP = m_Invocation.getPreprocessorOpts();
          insertBehind(myPP.Macros, PPOpts.Macros);
          insertBehind(myPP.Includes, PPOpts.Includes);
          insertBehind(myPP.MacroIncludes, PPOpts.MacroIncludes);
          return true;
        }
      };
      PCHListener listener(*Invocation);
      ASTReader::readASTFileControlBlock(PCHFileName,
                                         CI->getFileManager(),
                                         CI->getPCHContainerReader(),
                                         false /*FindModuleFileExtensions*/,
                                         listener);
    }

    Invocation->getFrontendOpts().DisableFree = true;
    // Copied from CompilerInstance::createDiagnostics:
    // Chain in -verify checker, if requested.
    if (DiagOpts.VerifyDiagnostics)
      Diags->setClient(new clang::VerifyDiagnosticConsumer(*Diags));
    // Configure our handling of diagnostics.
    ProcessWarningOptions(*Diags, DiagOpts);

    CI->setInvocation(Invocation);
    CI->setDiagnostics(Diags.get());

    if (PCHFileName.empty()) {
      // Set the language options, which cling needs
      SetClingCustomLangOpts(CI->getLangOpts());
    }

    PreprocessorOptions& PPOpts = CI->getInvocation().getPreprocessorOpts();
    SetPreprocessorFromBinary(PPOpts);

    PPOpts.addMacroDef("__CLING__");
    if (CI->getLangOpts().CPlusPlus11 == 1) {
      // http://llvm.org/bugs/show_bug.cgi?id=13530
      PPOpts.addMacroDef("__CLING__CXX11");
    }

    if (CI->getDiagnostics().hasErrorOccurred())
      return 0;

    CI->setTarget(TargetInfo::CreateTargetInfo(CI->getDiagnostics(),
                                               Invocation->TargetOpts));
    if (!CI->hasTarget())
      return 0;

    CI->getTarget().adjust(CI->getLangOpts());

    if (PCHFileName.empty())
      SetClingTargetLangOpts(CI->getLangOpts(), CI->getTarget());

    SetPreprocessorFromTarget(PPOpts, CI->getTarget().getTriple());

    // Set up source managers
    SourceManager* SM = new SourceManager(CI->getDiagnostics(),
                                          CI->getFileManager(),
                                          /*UserFilesAreVolatile*/ true);
    CI->setSourceManager(SM); // FIXME: SM leaks.

    // As main file we want
    // * a virtual file that is claiming to be huge
    // * with an empty memory buffer attached (to bring the content)
    FileManager& FM = SM->getFileManager();

    // When asking for the input file below (which does not have a directory
    // name), clang will call $PWD "." which is terrible if we ever change
    // directories (see ROOT-7114). By asking for $PWD (and not ".") it will
    // be registered as $PWD instead, which is stable even after chdirs.
    FM.getDirectory(platform::GetCwd());

    // Build the virtual file, Give it a name that's likely not to ever
    // be #included (so we won't get a clash in clangs cache).
    const char* Filename = "<<< cling interactive line includer >>>";
    const FileEntry* FE = FM.getVirtualFile(Filename, 1U << 15U, time(0));

    // Tell ASTReader to create a FileID even if this file does not exist:
    SM->setFileIsTransient(FE);
    FileID MainFileID = SM->createFileID(FE, SourceLocation(), SrcMgr::C_User);
    SM->setMainFileID(MainFileID);
    const SrcMgr::SLocEntry& MainFileSLocE = SM->getSLocEntry(MainFileID);
    const SrcMgr::ContentCache* MainFileCC
      = MainFileSLocE.getFile().getContentCache();
    if (!Buffer)
      Buffer = llvm::MemoryBuffer::getMemBuffer("/*CLING DEFAULT MEMBUF*/;\n");
    const_cast<SrcMgr::ContentCache*>(MainFileCC)->setBuffer(std::move(Buffer));

    // Set up the preprocessor
    CI->createPreprocessor(TU_Complete);
    Preprocessor& PP = CI->getPreprocessor();
    PP.getBuiltinInfo().initializeBuiltins(PP.getIdentifierTable(),
                                           PP.getLangOpts());

    // Set up the ASTContext
    CI->createASTContext();

    if (OnlyLex) {
      class IgnoreConsumer: public clang::ASTConsumer {
      };
      std::unique_ptr<clang::ASTConsumer> ignoreConsumer(new IgnoreConsumer());
      CI->setASTConsumer(std::move(ignoreConsumer));
    } else {
      std::unique_ptr<cling::DeclCollector>
        stateCollector(new cling::DeclCollector());

      // Add the callback keeping track of the macro definitions
      PP.addPPCallbacks(stateCollector->MakePPAdapter());
      CI->setASTConsumer(std::move(stateCollector));
    }

    // Set up Sema
    CodeCompleteConsumer* CCC = 0;
    CI->createSema(TU_Complete, CCC);

    // Set CodeGen options
    // want debug info
    //CI->getCodeGenOpts().setDebugInfo(clang::CodeGenOptions::FullDebugInfo);
    // CI->getCodeGenOpts().EmitDeclMetadata = 1; // For unloading, for later
    CI->getCodeGenOpts().CXXCtorDtorAliases = 0; // aliasing the complete
                                                 // ctor to the base ctor causes
                                                 // the JIT to crash
    CI->getCodeGenOpts().VerifyModule = 0; // takes too long

    if (!OnlyLex) {
      // -nobuiltininc
      clang::HeaderSearchOptions& HOpts = CI->getHeaderSearchOpts();
      if (CI->getHeaderSearchOpts().UseBuiltinIncludes)
        AddRuntimeIncludePaths(ClingBin, HOpts);

      // Write a marker to know the rest of the output is from clang
      if (COpts.Verbose)
        llvm::errs() << "Setting up system headers with clang:\n";

      // ### FIXME:
      // Want to update LLVM to 3.9 realease and better testing first, but
      // ApplyHeaderSearchOptions shouldn't even be called here:
      //   1. It's already been called via CI->createPreprocessor(TU_Complete)
      //   2. It could corrupt clang's directory cache
      // HeaderSearchOptions.::AddSearchPath is a better alternative

      clang::ApplyHeaderSearchOptions(PP.getHeaderSearchInfo(), HOpts,
                                      PP.getLangOpts(),
                                      PP.getTargetInfo().getTriple());
    }

    return CI.release(); // Passes over the ownership to the caller.
  }

} // unnamed namespace

namespace cling {

CompilerInstance* CIFactory::createCI(llvm::StringRef Code,
                                      const InvocationOptions& Opts,
                                      const char* LLVMDir) {
  return createCIImpl(llvm::MemoryBuffer::getMemBuffer(Code),
                      Opts.CompilerOpts, LLVMDir, false /*OnlyLex*/);
}

CompilerInstance* CIFactory::createCI(MemBufPtr_t Buffer, int argc,
                                      const char* const *argv,
                                      const char* LLVMDir, bool OnlyLex) {
  return createCIImpl(std::move(Buffer), CompilerOptions(argc, argv),
                      LLVMDir, OnlyLex);
}

} // namespace cling

