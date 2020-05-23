//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Axel Naumann <axel@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "ClingUtils.h"
#include <cling-compiledata.h>

#include "cling/Interpreter/CIFactory.h"
#include "cling/Interpreter/InvocationOptions.h"
#include "cling/Utils/Output.h"
#include "cling/Utils/Paths.h"
#include "cling/Utils/Platform.h"

#include "clang/AST/ASTContext.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/Version.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Job.h"
#include "clang/Driver/Tool.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Frontend/MultiplexConsumer.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Frontend/VerifyDiagnosticConsumer.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "clang/Serialization/ASTReader.h"
#include "clang/Serialization/ASTWriter.h"
#include "clang/Serialization/SerializationDiagnostic.h"

#include "llvm/Config/llvm-config.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetOptions.h"

#include <cstdio>
#include <ctime>
#include <memory>

using namespace clang;
using namespace cling;

namespace {
  static constexpr unsigned CxxStdCompiledWith() {
    // The value of __cplusplus in GCC < 5.0 (e.g. 4.9.3) when
    // either -std=c++1y or -std=c++14 is specified is 201300L, which fails
    // the test for C++14 or more (201402L) as previously specified.
    // I would claim that the check should be relaxed to:
#if __cplusplus > 201402L
    return 17;
#elif __cplusplus > 201103L || (defined(LLVM_ON_WIN32) && _MSC_VER >= 1900)
    return 14;
#elif __cplusplus >= 201103L
    return 11;
#else
#error "Unknown __cplusplus version"
#endif
  }

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

    CppInclQuery.append(" -xc++ -E -v /dev/null 2>&1 |"
                        " sed -n -e '/^.include/,${' -e '/^ \\/.*++/p' -e '}'");

    if (Verbose)
      cling::log() << "Looking for C++ headers with:\n  " << CppInclQuery << "\n";

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
            Args.addArgument("-cxx-isystem", Path.str());
        }
      }
      ::pclose(PF);
    } else {
      ::perror("popen failure");
      // Don't be overly verbose, we already printed the command
      if (!Verbose)
        cling::errs() << " for '" << CppInclQuery << "'\n";
    }

    // Return the query in Buf on failure
    if (Args.empty()) {
      Buf.resize(0);
      Buf.insert(Buf.begin(), CppInclQuery.begin(), CppInclQuery.end());
    } else if (Verbose) {
      cling::log() << "Found:\n";
      for (const auto& Arg : Args)
        cling::log() << "  " << Arg.second << "\n";
    }
  }

  static bool AddCxxPaths(llvm::StringRef PathStr, AdditionalArgList& Args,
                          bool Verbose) {
    if (Verbose)
      cling::log() << "Looking for C++ headers in \"" << PathStr << "\"\n";

    llvm::SmallVector<llvm::StringRef, 6> Paths;
    if (!utils::SplitPaths(PathStr, Paths, utils::kFailNonExistant,
                           platform::kEnvDelim, Verbose))
      return false;

    if (Verbose) {
      cling::log() << "Found:\n";
      for (llvm::StringRef Path : Paths)
        cling::log() << " " << Path << "\n";
    }

    for (llvm::StringRef Path : Paths)
      Args.addArgument("-cxx-isystem", Path.str());

    return true;
  }

#endif

  static std::string getResourceDir(const char* llvmdir) {
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
      return CompilerInvocation::GetResourcesPath(
          "cling", (void*)intptr_t(GetExecutablePath));
    } else {
      std::string resourcePath;
      llvm::SmallString<512> tmp(llvmdir);
      llvm::sys::path::append(tmp, "lib", "clang", CLANG_VERSION_STRING);
      resourcePath.assign(&tmp[0], tmp.size());
      return resourcePath;
    }
  }

  ///\brief Adds standard library -I used by whatever compiler is found in PATH.
  static void AddHostArguments(llvm::StringRef clingBin,
                               std::vector<const char*>& args,
                               const char* llvmdir, const CompilerOptions& opts) {
    static AdditionalArgList sArguments;
    if (sArguments.empty()) {
      const bool Verbose = opts.Verbose;
#ifdef _MSC_VER
      // When built with access to the proper Windows APIs, try to actually find
      // the correct include paths first. Init for UnivSDK.empty check below.
      std::string VSDir, WinSDK,
                  UnivSDK(opts.NoBuiltinInc ? "" : CLING_UCRT_VERSION);
      if (platform::GetVisualStudioDirs(VSDir,
                                        opts.NoBuiltinInc ? nullptr : &WinSDK,
                                        opts.NoBuiltinInc ? nullptr : &UnivSDK,
                                        Verbose)) {
        if (!opts.NoCXXInc) {
          // The Visual Studio 2017 path is very different than the previous
          // versions (see also GetVisualStudioDirs() in PlatformWin.cpp)
          const std::string VSIncl = VSDir + "\\include";
          if (Verbose)
            cling::log() << "Adding VisualStudio SDK: '" << VSIncl << "'\n";
          sArguments.addArgument("-I", std::move(VSIncl));
        }
        if (!opts.NoBuiltinInc) {
          if (!WinSDK.empty()) {
            WinSDK.append("\\include");
            if (Verbose)
              cling::log() << "Adding Windows SDK: '" << WinSDK << "'\n";
            sArguments.addArgument("-I", std::move(WinSDK));
          } else {
            // Since Visual Studio 2017, this is not valid anymore...
            VSDir.append("\\VC\\PlatformSDK\\Include");
            if (Verbose)
              cling::log() << "Adding Platform SDK: '" << VSDir << "'\n";
            sArguments.addArgument("-I", std::move(VSDir));
          }
        }
      }

#if LLVM_MSC_PREREQ(1900)
      if (!UnivSDK.empty()) {
        if (Verbose)
          cling::log() << "Adding UniversalCRT SDK: '" << UnivSDK << "'\n";
        sArguments.addArgument("-I", std::move(UnivSDK));
      }
#endif

      // Windows headers use '__declspec(dllexport) __cdecl' for most funcs
      // causing a lot of warnings for different redeclarations (eg. coming from
      // the test suite).
      // Do not warn about such cases.
      sArguments.addArgument("-Wno-dll-attribute-on-redeclaration");
      sArguments.addArgument("-Wno-inconsistent-dllimport");

      // Assume Windows.h might be included, and don't spew a ton of warnings
      sArguments.addArgument("-Wno-ignored-attributes");
      sArguments.addArgument("-Wno-nonportable-include-path");
      sArguments.addArgument("-Wno-microsoft-enum-value");
      sArguments.addArgument("-Wno-expansion-to-defined");

      // silent many warnings (mostly during ROOT compilation)
      sArguments.addArgument("-Wno-constant-conversion");
      sArguments.addArgument("-Wno-unknown-escape-sequence");
      sArguments.addArgument("-Wno-microsoft-unqualified-friend");
      sArguments.addArgument("-Wno-deprecated-declarations");

      //sArguments.addArgument("-fno-threadsafe-statics");

      //sArguments.addArgument("-Wno-dllimport-static-field-def");
      //sArguments.addArgument("-Wno-microsoft-template");

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

  // First try the relative path 'g++'
  #ifdef CLING_CXX_RLTV
        if (sArguments.empty())
          ReadCompilerIncludePaths(CLING_CXX_RLTV, buffer, sArguments, Verbose);
  #endif
  // Then try the include directory cling was built with
  #ifdef CLING_CXX_INCL
        if (sArguments.empty())
          AddCxxPaths(CLING_CXX_INCL, sArguments, Verbose);
  #endif
  // Finally try the absolute path i.e.: '/usr/bin/g++'
  #ifdef CLING_CXX_PATH
        if (sArguments.empty())
          ReadCompilerIncludePaths(CLING_CXX_PATH, buffer, sArguments, Verbose);
  #endif

        if (sArguments.empty()) {
          // buffer is a copy of the query string that failed
          cling::errs() << "ERROR in cling::CIFactory::createCI(): cannot extract"
                          " standard library include paths!\n";

  #if defined(CLING_CXX_PATH) || defined(CLING_CXX_RLTV)
          // Only when ReadCompilerIncludePaths called do we have the command
          // Verbose has already printed the command
          if (!Verbose)
            cling::errs() << "Invoking:\n  " << buffer.c_str() << "\n";

          cling::errs() << "Results was:\n";
          const int ExitCode = system(buffer.c_str());
          cling::errs() << "With exit code " << ExitCode << "\n";
  #elif !defined(CLING_CXX_INCL)
          // Technically a valid configuration that just wants to use libClangs
          // internal header detection, but for now give a hint about why.
          cling::errs() << "CLING_CXX_INCL, CLING_CXX_PATH, and CLING_CXX_RLTV"
                          " are undefined, there was probably an error during"
                          " configuration.\n";
  #endif
        } else
          sArguments.addArgument("-nostdinc++");
      }

  #ifdef CLING_OSX_SYSROOT
    sArguments.addArgument("-isysroot", CLING_OSX_SYSROOT);
  #endif

#endif // _MSC_VER

      if (!opts.ResourceDir && !opts.NoBuiltinInc) {
        std::string resourcePath = getResourceDir(llvmdir);

        // FIXME: Handle cases, where the cling is part of a library/framework.
        // There we can't rely on the find executable logic.
        if (!llvm::sys::fs::is_directory(resourcePath)) {
          cling::errs()
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

  static void SetClingCustomLangOpts(LangOptions& Opts,
                                     const CompilerOptions& CompilerOpts) {
    Opts.EmitAllDecls = 0; // Otherwise if PCH attached will codegen all decls.
#ifdef _MSC_VER
#ifdef _DEBUG
    // FIXME: This requires bufferoverflowu.lib, but adding:
    // #pragma comment(lib, "bufferoverflowu.lib") still gives errors!
    // Opts.setStackProtector(clang::LangOptions::SSPStrong);
#endif // _DEBUG
#ifdef _CPPRTTI
    Opts.RTTIData = 1;
#else
    Opts.RTTIData = 0;
#endif // _CPPRTTI
    Opts.Trigraphs = 0;
    Opts.setDefaultCallingConv(clang::LangOptions::DCC_CDecl);
#else // !_MSC_VER
    Opts.Trigraphs = 1;
//    Opts.RTTIData = 1;
//    Opts.DefaultCallingConventions = 1;
//    Opts.StackProtector = 0;
#endif // _MSC_VER

    Opts.Exceptions = 1;
    if (Opts.CPlusPlus) {
      Opts.CXXExceptions = 1;
    }

    //Opts.Modules = 1;

    // See test/CodeUnloading/PCH/VTables.cpp which implicitly compares clang
    // to cling lang options. They should be the same, we should not have to
    // give extra lang options to their invocations on any platform.
    // Except -fexceptions -fcxx-exceptions.

    Opts.Deprecated = 1;

#ifdef __APPLE__
    Opts.Blocks = 1;
    Opts.MathErrno = 0;
#endif

#ifdef _REENTRANT
    Opts.POSIXThreads = 1;
#endif
#ifdef __FAST_MATH__
    Opts.FastMath = 1;
#endif

    if (CompilerOpts.DefaultLanguage(&Opts)) {
#ifdef __STRICT_ANSI__
      Opts.GNUMode = 0;
#else
      Opts.GNUMode = 1;
#endif
      Opts.GNUKeywords = 0;
    }
  }

  static void SetClingTargetLangOpts(LangOptions& Opts,
                                     const TargetInfo& Target,
                                     const CompilerOptions& CompilerOpts) {
    if (Target.getTriple().getOS() == llvm::Triple::Win32) {
      Opts.MicrosoftExt = 1;
#ifdef _MSC_VER
      Opts.MSCompatibilityVersion = (_MSC_VER * 100000);
      Opts.MSVCCompat = 1;
      Opts.ThreadsafeStatics = 0; // FIXME: this is removing the thread guard around static init!
#endif
      // Should fix http://llvm.org/bugs/show_bug.cgi?id=10528
      Opts.DelayedTemplateParsing = 1;
    } else {
      Opts.MicrosoftExt = 0;
    }

    if (CompilerOpts.DefaultLanguage(&Opts)) {
#if _GLIBCXX_USE_FLOAT128
      // We are compiling with libstdc++ with __float128 enabled.
      if (!Target.hasFloat128Type()) {
        // clang currently supports native __float128 only on few targets, and
        // this target does not have it. The most visible consequence of this is
        // a specialization
        //    __is_floating_point_helper<__float128>
        // in include/c++/6.3.0/type_traits:344 that clang then rejects. The
        // specialization is protected by !if _GLIBCXX_USE_FLOAT128 (which is
        // unconditionally set in c++config.h) and #if !__STRICT_ANSI__. Tweak
        // the latter by disabling GNUMode.
        // the nvptx backend doesn't support 128 bit float
        // a error message is not necessary
        if(!CompilerOpts.CUDADevice) {
          cling::errs()
            << "Disabling gnu++: "
               "clang has no __float128 support on this target!\n";
        }
        Opts.GNUMode = 0;
      }
#endif //_GLIBCXX_USE_FLOAT128
    }
  }

  // This must be a copy of clang::getClangToolFullVersion(). Luckily
  // we'll notice quickly if it ever changes! :-)
  static std::string CopyOfClanggetClangToolFullVersion(StringRef ToolName) {
    cling::stdstrstream OS;
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
      cling::errs()
        << "Warning in cling::CIFactory::createCI():\n  "
        "Using incompatible clang library! "
        "Please use the one provided by cling!\n";
    return;
  }

  /// \brief Retrieves the clang CC1 specific flags out of the compilation's
  /// jobs. Returns NULL on error.
  static const llvm::opt::ArgStringList*
  GetCC1Arguments(clang::driver::Compilation *Compilation,
                  clang::DiagnosticsEngine* = nullptr) {
    // We expect to get back exactly one Command job, if we didn't something
    // failed. Extract that job from the Compilation.
    const clang::driver::JobList &Jobs = Compilation->getJobs();
    if (!Jobs.size() || !isa<clang::driver::Command>(*Jobs.begin())) {
      // diagnose this better...
      cling::errs() << "No Command jobs were built.\n";
      return nullptr;
    }

    // The one job we find should be to invoke clang again.
    const clang::driver::Command *Cmd
      = cast<clang::driver::Command>(&(*Jobs.begin()));
    if (llvm::StringRef(Cmd->getCreator().getName()) != "clang") {
      // diagnose this better...
      cling::errs() << "Clang wasn't the first job.\n";
      return nullptr;
    }

    return &Cmd->getArguments();
  }

  /// \brief Splits the given environment variable by the path separator.
  /// Can be used to extract the paths from LD_LIBRARY_PATH.
  static SmallVector<StringRef, 4> getPathsFromEnv(const char* EnvVar) {
    if (!EnvVar) return {};
    SmallVector<StringRef, 4> Paths;
    StringRef(EnvVar).split(Paths, ':', -1, false);
    return Paths;
  }

  /// \brief Adds all the paths to the prebuilt module paths of the given
  /// HeaderSearchOptions.
  static void addPrebuiltModulePaths(clang::HeaderSearchOptions& Opts,
                                     const SmallVectorImpl<StringRef>& Paths) {
    for (StringRef ModulePath : Paths)
      Opts.AddPrebuiltModulePath(ModulePath);
  }

  static std::string getIncludePathForHeader(const clang::HeaderSearch& HS,
                                             llvm::StringRef header) {
    for (auto Dir = HS.search_dir_begin(), E = HS.search_dir_end();
         Dir != E; ++Dir) {
      llvm::SmallString<512> headerPath(Dir->getName());
      llvm::sys::path::append(headerPath, header);
      if (llvm::sys::fs::exists(headerPath.str()))
        return Dir->getName().str();
    }
    return {};
  }

  static void collectModuleMaps(clang::CompilerInstance& CI,
                           llvm::SmallVectorImpl<std::string> &ModuleMapFiles) {
    assert(CI.getLangOpts().Modules && "Using overlay without -fmodules");

    const clang::HeaderSearch& HS = CI.getPreprocessor().getHeaderSearchInfo();
    clang::HeaderSearchOptions& HSOpts = CI.getHeaderSearchOpts();

    // We can't use "assert.h" because it is defined in the resource dir, too.
#ifdef LLVM_ON_WIN32
    llvm::SmallString<256> vcIncLoc(getIncludePathForHeader(HS, "vcruntime.h"));
    llvm::SmallString<256> servIncLoc(getIncludePathForHeader(HS, "windows.h"));
#endif
    llvm::SmallString<128> cIncLoc(getIncludePathForHeader(HS, "time.h"));

    llvm::SmallString<256> stdIncLoc(getIncludePathForHeader(HS, "cassert"));
    llvm::SmallString<256> boostIncLoc(getIncludePathForHeader(HS, "boost/version.hpp"));
    llvm::SmallString<256> tinyxml2IncLoc(getIncludePathForHeader(HS, "tinyxml2.h"));
    llvm::SmallString<256> cudaIncLoc(getIncludePathForHeader(HS, "cuda.h"));
    llvm::SmallString<256> clingIncLoc(getIncludePathForHeader(HS,
                                        "cling/Interpreter/RuntimeUniverse.h"));

    // Re-add cling as the modulemap are in cling/*modulemap
    llvm::sys::path::append(clingIncLoc, "cling");

    // Construct a column of modulemap overlay file if needed.
    auto maybeAppendOverlayEntry
       = [&HSOpts, &ModuleMapFiles](llvm::StringRef SystemDir,
                                    const std::string& Filename,
                                    const std::string& Location,
                                    std::string& overlay) -> void {

      assert(llvm::sys::fs::exists(SystemDir) && "Must exist!");

      std::string modulemapFilename = "module.modulemap";
      llvm::SmallString<512> systemLoc(SystemDir);
      llvm::sys::path::append(systemLoc, modulemapFilename);
      // Check if we need to mount a custom modulemap. We may have it, for
      // instance when we are on osx or using libc++.
      if (llvm::sys::fs::exists(systemLoc.str())) {
        if (HSOpts.Verbose)
          cling::log() << "Loading '" << systemLoc.str() << "'\n";

        // If the library had its own modulemap file, use it. This should handle
        // the case where we use libc++ on Unix.
        if (!HSOpts.ImplicitModuleMaps)
           ModuleMapFiles.push_back(systemLoc.str().str());

        return;
      }

      llvm::SmallString<512> originalLoc(Location);
      assert(llvm::sys::fs::exists(originalLoc.str()) && "Must exist!");
      llvm::sys::path::append(originalLoc, Filename);
      assert(llvm::sys::fs::exists(originalLoc.str()));

      if (HSOpts.Verbose)
        cling::log() << "'" << systemLoc << "' does not exist. Mounting '"
                     << originalLoc.str() << "' as '" << systemLoc << "'\n";

      if (!HSOpts.ImplicitModuleMaps) {
         modulemapFilename = Filename;
         llvm::sys::path::remove_filename(systemLoc);
         llvm::sys::path::append(systemLoc, modulemapFilename);
      }

      if (!overlay.empty())
        overlay += ",\n";

      overlay += "{ 'name': '" + SystemDir.str() + "', 'type': 'directory',\n";
      overlay += "'contents': [\n   { 'name': '" + modulemapFilename + "', ";
      overlay += "'type': 'file',\n  'external-contents': '";
      overlay += originalLoc.str().str() + "'\n";
      overlay += "}\n ]\n }";

      if (HSOpts.ImplicitModuleMaps)
         return;

      ModuleMapFiles.push_back(systemLoc.str().str());
    };

    if (!HSOpts.ImplicitModuleMaps) {
      // Register the modulemap files.
      llvm::SmallString<512> resourceDirLoc(HSOpts.ResourceDir);
      llvm::sys::path::append(resourceDirLoc, "include", "module.modulemap");
      ModuleMapFiles.push_back(resourceDirLoc.str().str());
      llvm::SmallString<512> clingModuleMap(clingIncLoc);
      llvm::sys::path::append(clingModuleMap, "module.modulemap");
      ModuleMapFiles.push_back(clingModuleMap.str().str());
#ifdef __APPLE__
      llvm::SmallString<512> libcModuleMap(cIncLoc);
      llvm::sys::path::append(libcModuleMap, "module.modulemap");
      ModuleMapFiles.push_back(libcModuleMap.str().str());
      llvm::SmallString<512> stdModuleMap(stdIncLoc);
      llvm::sys::path::append(stdModuleMap, "module.modulemap");
      ModuleMapFiles.push_back(stdModuleMap.str().str());
#endif // __APPLE__
    }

    std::string MOverlay;
#ifdef LLVM_ON_WIN32
    maybeAppendOverlayEntry(vcIncLoc.str(), "vcruntime.modulemap",
                            clingIncLoc.str(), MOverlay);
    maybeAppendOverlayEntry(servIncLoc.str(), "services_msvc.modulemap",
                            clingIncLoc.str(), MOverlay);
    maybeAppendOverlayEntry(cIncLoc.str(), "libc_msvc.modulemap",
                            clingIncLoc.str(), MOverlay);
    maybeAppendOverlayEntry(stdIncLoc.str(), "std_msvc.modulemap",
                            clingIncLoc.str(), MOverlay);
#else
    maybeAppendOverlayEntry(cIncLoc.str(), "libc.modulemap", clingIncLoc.str(),
                            MOverlay);
    maybeAppendOverlayEntry(stdIncLoc.str(), "std.modulemap", clingIncLoc.str(),
                            MOverlay);
#endif // LLVM_ON_WIN32

    if (!tinyxml2IncLoc.empty())
      maybeAppendOverlayEntry(tinyxml2IncLoc.str(), "tinyxml2.modulemap",
                              clingIncLoc.str(), MOverlay);
    if (!cudaIncLoc.empty())
      maybeAppendOverlayEntry(cudaIncLoc.str(), "cuda.modulemap",
                              clingIncLoc.str(), MOverlay);
    if (!boostIncLoc.empty())
      maybeAppendOverlayEntry(boostIncLoc.str(), "boost.modulemap",
                              clingIncLoc.str(), MOverlay);

    if (/*needsOverlay*/!MOverlay.empty()) {
      // Virtual modulemap overlay file
      MOverlay.insert(0, "{\n 'version': 0,\n 'roots': [\n");

      MOverlay += "]\n }\n ]\n }\n";

      const std::string VfsOverlayFileName = "modulemap.overlay.yaml";
      if (HSOpts.Verbose)
        cling::log() << VfsOverlayFileName << "\n" << MOverlay;

      // Set up the virtual modulemap overlay file
      std::unique_ptr<llvm::MemoryBuffer> Buffer =
        llvm::MemoryBuffer::getMemBuffer(MOverlay);

      IntrusiveRefCntPtr<clang::vfs::FileSystem> FS =
        vfs::getVFSFromYAML(std::move(Buffer), nullptr, VfsOverlayFileName);
      if (!FS.get())
        llvm::errs() << "Error in modulemap.overlay!\n";

      // Load virtual modulemap overlay file
      CI.getInvocation().addOverlay(FS);
    }
  }

  static void setupCxxModules(clang::CompilerInstance& CI) {
    assert(CI.getLangOpts().Modules);
    clang::HeaderSearchOptions& HSOpts = CI.getHeaderSearchOpts();
    // Register prebuilt module paths where we will lookup module files.
    addPrebuiltModulePaths(HSOpts,
                           getPathsFromEnv(getenv("CLING_PREBUILT_MODULE_PATH")));

    // Register all modulemaps necessary for cling to run. If we have specified
    // -fno-implicit-module-maps then we have to add them explicitly to the list
    // of modulemap files to load.
    llvm::SmallVector<std::string, 4> ModuleMaps;

    collectModuleMaps(CI, ModuleMaps);

    assert(HSOpts.ImplicitModuleMaps == ModuleMaps.empty() &&
           "We must have register the modulemaps by hand!");
    // Prepend the modulemap files we attached so that they will be loaded.
    if (!HSOpts.ImplicitModuleMaps) {
      FrontendOptions& FrontendOpts = CI.getInvocation().getFrontendOpts();
      FrontendOpts.ModuleMapFiles.insert(FrontendOpts.ModuleMapFiles.begin(),
                                         ModuleMaps.begin(), ModuleMaps.end());
    }
  }

#if defined(_MSC_VER) || defined(NDEBUG)
static void stringifyPreprocSetting(PreprocessorOptions& PPOpts,
                                    const std::string &Name, int Val) {
  smallstream Strm;
  Strm << Name << "=" << Val;
  if (std::find(PPOpts.Macros.begin(), PPOpts.Macros.end(),
                std::make_pair(Name, true))
      == PPOpts.Macros.end()
      && std::find(PPOpts.Macros.begin(), PPOpts.Macros.end(),
                   std::make_pair(Name, false))
      == PPOpts.Macros.end())
    PPOpts.addMacroDef(Strm.str());
}

#define STRINGIFY_PREPROC_SETTING(PP, name) \
  stringifyPreprocSetting(PP, #name, name)
#endif

  /// Set cling's preprocessor defines to match the cling binary.
  static void SetPreprocessorFromBinary(PreprocessorOptions& PPOpts) {
#ifdef _MSC_VER
// FIXME: Stay consistent with the _HAS_EXCEPTIONS flag settings in SetClingCustomLangOpts
//    STRINGIFY_PREPROC_SETTING(PPOpts, _HAS_EXCEPTIONS);
#ifdef _DEBUG
    STRINGIFY_PREPROC_SETTING(PPOpts, _DEBUG);
#endif
#endif

#ifdef NDEBUG
    STRINGIFY_PREPROC_SETTING(PPOpts, NDEBUG);
#endif
    // Since cling, uses clang instead, macros always sees __CLANG__ defined
    // In addition, clang also defined __GNUC__, we add the following two macros
    // to allow scripts, and more important, dictionary generation to know which
    // of the two is the underlying compiler.

#ifdef __clang__
    PPOpts.addMacroDef("__CLING__clang__=" ClingStringify(__clang__));
#elif defined(__GNUC__)
    PPOpts.addMacroDef("__CLING__GNUC__=" ClingStringify(__GNUC__));
    PPOpts.addMacroDef("__CLING__GNUC_MINOR__=" ClingStringify(__GNUC_MINOR__));
#elif defined(_MSC_VER)
    PPOpts.addMacroDef("__CLING__MSVC__=" ClingStringify(_MSC_VER));
#endif

// https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_dual_abi.html
#ifdef _GLIBCXX_USE_CXX11_ABI
    PPOpts.addMacroDef("_GLIBCXX_USE_CXX11_ABI="
                       ClingStringify(_GLIBCXX_USE_CXX11_ABI));
#endif

#if defined(LLVM_ON_WIN32)
    PPOpts.addMacroDef("CLING_EXPORT=__declspec(dllimport)");
    // prevent compilation error G47C585C4: STL1000: Unexpected compiler
    // version, expected Clang 6 or newer.
    PPOpts.addMacroDef("_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH");
#else
    PPOpts.addMacroDef("CLING_EXPORT=");
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
        cling::errs() << "Warning in cling::CIFactory::createCI():\n"
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
      cling::log() << "Adding runtime include paths:\n";
    // Add configuration paths to interpreter's include files.
#ifdef CLING_INCLUDE_PATHS
    if (HOpts.Verbose)
      cling::log() << "  \"" CLING_INCLUDE_PATHS "\"\n";
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
      if (llvm::sys::fs::is_directory(P.str())) {
        utils::AddIncludePaths(P.str(), HOpts, nullptr);
        llvm::sys::path::append(P, "clang");
        if (!llvm::sys::fs::is_directory(P.str())) {
          // LLVM is not installed. Try resolving clang from its usual location.
          llvm::SmallString<512> PParent = llvm::sys::path::parent_path(P);
          P = PParent;
          llvm::sys::path::append(P, "..", "tools", "clang", "include");
          if (llvm::sys::fs::is_directory(P.str()))
            utils::AddIncludePaths(P.str(), HOpts, nullptr);
        }
      }
    }
  }

  static llvm::IntrusiveRefCntPtr<DiagnosticsEngine>
  SetupDiagnostics(DiagnosticOptions& DiagOpts) {
    // The compiler invocation is the owner of the diagnostic options.
    // Everything else points to them.
    llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs> DiagIDs(new DiagnosticIDs());

    std::unique_ptr<TextDiagnosticPrinter>
      DiagnosticPrinter(new TextDiagnosticPrinter(cling::errs(), &DiagOpts));

    llvm::IntrusiveRefCntPtr<DiagnosticsEngine>
      Diags(new DiagnosticsEngine(DiagIDs, &DiagOpts,
                                  DiagnosticPrinter.get(), /*Owns it*/ true));
    DiagnosticPrinter.release();

    return Diags;
  }

  static bool
  SetupCompiler(CompilerInstance* CI, const CompilerOptions& CompilerOpts,
                bool Lang = true, bool Targ = true) {
    LangOptions& LangOpts = CI->getLangOpts();
    // Set the language options, which cling needs.
    // This may have already been done via a precompiled header
    if (Lang)
      SetClingCustomLangOpts(LangOpts, CompilerOpts);

    PreprocessorOptions& PPOpts = CI->getInvocation().getPreprocessorOpts();
    SetPreprocessorFromBinary(PPOpts);

    // Sanity check that clang delivered the language standard requested
    if (CompilerOpts.DefaultLanguage(&LangOpts)) {
      switch (CxxStdCompiledWith()) {
        case 17: assert(LangOpts.CPlusPlus1z && "Language version mismatch");
          // fall-through!
        case 14: assert(LangOpts.CPlusPlus14 && "Language version mismatch");
          // fall-through!
        case 11: assert(LangOpts.CPlusPlus11 && "Language version mismatch");
          break;
        default: assert(false && "You have an unhandled C++ standard!");
      }
    }

    PPOpts.addMacroDef("__CLING__");
    if (LangOpts.CPlusPlus11 == 1)
      PPOpts.addMacroDef("__CLING__CXX11");
    if (LangOpts.CPlusPlus14 == 1)
      PPOpts.addMacroDef("__CLING__CXX14");

    if (CI->getDiagnostics().hasErrorOccurred()) {
      cling::errs() << "Compiler error too early in initialization.\n";
      return false;
    }

    CI->setTarget(TargetInfo::CreateTargetInfo(CI->getDiagnostics(),
                                               CI->getInvocation().TargetOpts));
    if (!CI->hasTarget()) {
      cling::errs() << "Could not determine compiler target.\n";
      return false;
    }

    CI->getTarget().adjust(LangOpts);

    // This may have already been done via a precompiled header
    if (Targ)
      SetClingTargetLangOpts(LangOpts, CI->getTarget(), CompilerOpts);

    SetPreprocessorFromTarget(PPOpts, CI->getTarget().getTriple());
    return true;
  }

  class ActionScan {
    std::set<const clang::driver::Action*> m_Visited;
    llvm::SmallVector<clang::driver::Action::ActionClass, 2> m_Kinds;

    bool find (const clang::driver::Action* A) {
      if (A && !m_Visited.count(A)) {
        if (std::find(m_Kinds.begin(), m_Kinds.end(), A->getKind()) !=
            m_Kinds.end())
          return true;

        m_Visited.insert(A);
        return find(*A->input_begin());
      }
      return false;
    }

  public:
    ActionScan(clang::driver::Action::ActionClass a, int b = -1) {
      m_Kinds.push_back(a);
      if (b != -1)
        m_Kinds.push_back(clang::driver::Action::ActionClass(b));
    }

    bool find (clang::driver::Compilation* C) {
      for (clang::driver::Action* A : C->getActions()) {
        if (find(A))
          return true;
      }
      return false;
    }
  };

  static void HandleProgramActions(CompilerInstance &CI) {
    const clang::FrontendOptions& FrontendOpts = CI.getFrontendOpts();
    if (FrontendOpts.ProgramAction == clang::frontend::ModuleFileInfo) {
      // Copied from FrontendActions.cpp
      // FIXME: Remove when we switch to the new driver.

      class DumpModuleInfoListener : public ASTReaderListener {
        llvm::raw_ostream &Out;

      public:
        DumpModuleInfoListener(llvm::raw_ostream &Out) : Out(Out) { }

#define DUMP_BOOLEAN(Value, Text)                                       \
        Out.indent(4) << Text << ": " << (Value? "Yes" : "No") << "\n"

        bool ReadFullVersionInformation(StringRef FullVersion) override {
          Out.indent(2)
            << "Generated by "
            << (FullVersion == getClangFullRepositoryVersion()? "this"
                                                              : "a different")
            << " Clang: " << FullVersion << "\n";
          return ASTReaderListener::ReadFullVersionInformation(FullVersion);
        }

        void ReadModuleName(StringRef ModuleName) override {
          Out.indent(2) << "Module name: " << ModuleName << "\n";
        }
        void ReadModuleMapFile(StringRef ModuleMapPath) override {
          Out.indent(2) << "Module map file: " << ModuleMapPath << "\n";
        }

        bool ReadLanguageOptions(const LangOptions &LangOpts, bool Complain,
                                 bool AllowCompatibleDifferences) override {
          Out.indent(2) << "Language options:\n";
#define LANGOPT(Name, Bits, Default, Description)                       \
          DUMP_BOOLEAN(LangOpts.Name, Description);
#define ENUM_LANGOPT(Name, Type, Bits, Default, Description)            \
          Out.indent(4) << Description << ": "                          \
                    << static_cast<unsigned>(LangOpts.get##Name()) << "\n";
#define VALUE_LANGOPT(Name, Bits, Default, Description) \
          Out.indent(4) << Description << ": " << LangOpts.Name << "\n";
#define BENIGN_LANGOPT(Name, Bits, Default, Description)
#define BENIGN_ENUM_LANGOPT(Name, Type, Bits, Default, Description)
#include "clang/Basic/LangOptions.def"

          if (!LangOpts.ModuleFeatures.empty()) {
            Out.indent(4) << "Module features:\n";
            for (StringRef Feature : LangOpts.ModuleFeatures)
              Out.indent(6) << Feature << "\n";
          }

          return false;
        }

        bool ReadTargetOptions(const TargetOptions &TargetOpts, bool Complain,
                               bool AllowCompatibleDifferences) override {
          Out.indent(2) << "Target options:\n";
          Out.indent(4) << "  Triple: " << TargetOpts.Triple << "\n";
          Out.indent(4) << "  CPU: " << TargetOpts.CPU << "\n";
          Out.indent(4) << "  ABI: " << TargetOpts.ABI << "\n";

          if (!TargetOpts.FeaturesAsWritten.empty()) {
            Out.indent(4) << "Target features:\n";
            for (unsigned I = 0, N = TargetOpts.FeaturesAsWritten.size();
                 I != N; ++I) {
              Out.indent(6) << TargetOpts.FeaturesAsWritten[I] << "\n";
            }
          }

          return false;
        }

        bool ReadDiagnosticOptions(IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts,
                                   bool Complain) override {
          Out.indent(2) << "Diagnostic options:\n";
#define DIAGOPT(Name, Bits, Default) DUMP_BOOLEAN(DiagOpts->Name, #Name);
#define ENUM_DIAGOPT(Name, Type, Bits, Default)                         \
          Out.indent(4) << #Name << ": " << DiagOpts->get##Name() << "\n";
#define VALUE_DIAGOPT(Name, Bits, Default)                              \
      Out.indent(4) << #Name << ": " << DiagOpts->Name << "\n";
#include "clang/Basic/DiagnosticOptions.def"

          Out.indent(4) << "Diagnostic flags:\n";
          for (const std::string &Warning : DiagOpts->Warnings)
            Out.indent(6) << "-W" << Warning << "\n";
          for (const std::string &Remark : DiagOpts->Remarks)
            Out.indent(6) << "-R" << Remark << "\n";

          return false;
        }

        bool ReadHeaderSearchOptions(const HeaderSearchOptions &HSOpts,
                                     StringRef SpecificModuleCachePath,
                                     bool Complain) override {
          Out.indent(2) << "Header search options:\n";
          Out.indent(4) << "System root [-isysroot=]: '"
                        << HSOpts.Sysroot << "'\n";
          Out.indent(4) << "Resource dir [ -resource-dir=]: '"
                        << HSOpts.ResourceDir << "'\n";
          Out.indent(4) << "Module Cache: '" << SpecificModuleCachePath
                        << "'\n";
          DUMP_BOOLEAN(HSOpts.UseBuiltinIncludes,
                       "Use builtin include directories [-nobuiltininc]");
          DUMP_BOOLEAN(HSOpts.UseStandardSystemIncludes,
                       "Use standard system include directories [-nostdinc]");
          DUMP_BOOLEAN(HSOpts.UseStandardCXXIncludes,
                       "Use standard C++ include directories [-nostdinc++]");
          DUMP_BOOLEAN(HSOpts.UseLibcxx,
                       "Use libc++ (rather than libstdc++) [-stdlib=]");
          return false;
        }

        bool ReadPreprocessorOptions(const PreprocessorOptions &PPOpts,
                                     bool Complain,
                                    std::string &SuggestedPredefines) override {
          Out.indent(2) << "Preprocessor options:\n";
          DUMP_BOOLEAN(PPOpts.UsePredefines,
                       "Uses compiler/target-specific predefines [-undef]");
          DUMP_BOOLEAN(PPOpts.DetailedRecord,
                       "Uses detailed preprocessing record (for indexing)");

          if (!PPOpts.Macros.empty()) {
            Out.indent(4) << "Predefined macros:\n";
          }

          for (std::vector<std::pair<std::string, bool/*isUndef*/> >::const_iterator
                 I = PPOpts.Macros.begin(), IEnd = PPOpts.Macros.end();
               I != IEnd; ++I) {
            Out.indent(6);
            if (I->second)
              Out << "-U";
            else
              Out << "-D";
            Out << I->first << "\n";
          }
          return false;
        }

        /// Indicates that a particular module file extension has been read.
        void readModuleFileExtension(
                         const ModuleFileExtensionMetadata &Metadata) override {
          Out.indent(2) << "Module file extension '"
                        << Metadata.BlockName << "' " << Metadata.MajorVersion
                        << "." << Metadata.MinorVersion;
          if (!Metadata.UserInfo.empty()) {
            Out << ": ";
            Out.write_escaped(Metadata.UserInfo);
          }

          Out << "\n";
        }

        /// Tells the \c ASTReaderListener that we want to receive the
        /// input files of the AST file via \c visitInputFile.
        bool needsInputFileVisitation() override { return true; }

        /// Tells the \c ASTReaderListener that we want to receive the
        /// input files of the AST file via \c visitInputFile.
        bool needsSystemInputFileVisitation() override { return true; }

        /// Indicates that the AST file contains particular input file.
        ///
        /// \returns true to continue receiving the next input file, false to stop.
        bool visitInputFile(StringRef Filename, bool isSystem,
                            bool isOverridden, bool isExplicitModule) override {

          Out.indent(2) << "Input file: " << Filename;

          if (isSystem || isOverridden || isExplicitModule) {
            Out << " [";
            if (isSystem) {
              Out << "System";
              if (isOverridden || isExplicitModule)
                Out << ", ";
            }
            if (isOverridden) {
              Out << "Overridden";
              if (isExplicitModule)
                Out << ", ";
            }
            if (isExplicitModule)
              Out << "ExplicitModule";

            Out << "]";
          }

          Out << "\n";

          return true;
        }
#undef DUMP_BOOLEAN
      };

      std::unique_ptr<llvm::raw_fd_ostream> OutFile;
      StringRef OutputFileName = FrontendOpts.OutputFile;
      if (!OutputFileName.empty() && OutputFileName != "-") {
        std::error_code EC;
        OutFile.reset(new llvm::raw_fd_ostream(OutputFileName.str(), EC,
                                               llvm::sys::fs::F_Text));
      }
      llvm::raw_ostream &Out = OutFile.get()? *OutFile.get() : llvm::outs();
      StringRef CurInput = FrontendOpts.Inputs[0].getFile();
      Out << "Information for module file '" << CurInput << "':\n";
      auto &FileMgr = CI.getFileManager();
      auto Buffer = FileMgr.getBufferForFile(CurInput);
      StringRef Magic = (*Buffer)->getMemBufferRef().getBuffer();
      bool IsRaw = (Magic.size() >= 4 && Magic[0] == 'C' && Magic[1] == 'P' &&
                    Magic[2] == 'C' && Magic[3] == 'H');
      Out << "  Module format: " << (IsRaw ? "raw" : "obj") << "\n";
      Preprocessor &PP = CI.getPreprocessor();
      DumpModuleInfoListener Listener(Out);
      HeaderSearchOptions &HSOpts =
        PP.getHeaderSearchInfo().getHeaderSearchOpts();
      ASTReader::readASTFileControlBlock(CurInput, FileMgr,
                                         CI.getPCHContainerReader(),
                                         /*FindModuleFileExtensions=*/true,
                                         Listener,
                                         HSOpts.ModulesValidateDiagnosticOptions);
    }
  }

  static CompilerInstance*
  createCIImpl(std::unique_ptr<llvm::MemoryBuffer> Buffer,
               const CompilerOptions& COpts,
               const char* LLVMDir,
               std::unique_ptr<clang::ASTConsumer> customConsumer,
               const CIFactory::ModuleFileExtensions& moduleExtensions,
               bool OnlyLex, bool HasInput = false) {
    // Follow clang -v convention of printing version on first line
    if (COpts.Verbose)
      cling::log() << "cling version " << ClingStringify(CLING_VERSION) << '\n';

    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllAsmParsers();
    llvm::InitializeAllAsmPrinters();
    llvm::InitializeAllTargetMCs();

    // Create an instance builder, passing the LLVMDir and arguments.
    //

    CheckClangCompatibility();

    const size_t argc = COpts.Remaining.size();
    const char* const* argv = &COpts.Remaining[0];
    std::vector<const char*> argvCompile(argv, argv+1);
    argvCompile.reserve(argc+5);

    // Variables for storing the memory of the C-string arguments.
    // FIXME: We shouldn't use C-strings in the first place, but just use
    // std::string for clang arguments.
    std::string overlayArg;
    std::string cacheArg;

    // If user has enabled C++ modules we add some special module flags to the
    // compiler invocation.
    if (COpts.CxxModules) {
      // Enables modules in clang.
      argvCompile.push_back("-fmodules");
      argvCompile.push_back("-fcxx-modules");
      // We want to use modules in local-submodule-visibility mode. This mode
      // will probably be the future default mode for C++ modules in clang, so
      // we want to start using right now.
      // Maybe we have to remove this flag in the future when clang makes this
      // mode the default and removes this internal flag.
      argvCompile.push_back("-Xclang");
      argvCompile.push_back("-fmodules-local-submodule-visibility");
      // If we got a cache path, then we are supposed to place any modules
      // we have to build in this directory.
      if (!COpts.CachePath.empty()) {
        cacheArg = std::string("-fmodules-cache-path=") + COpts.CachePath;
        argvCompile.push_back(cacheArg.c_str());
      }
      // Disable the module hash. This gives us a flat file layout in the
      // modules cache directory. In clang this is used to prevent modules from
      // different compiler invocations to not collide, but we only have one
      // compiler invocation in cling, so we don't need this.
      argvCompile.push_back("-Xclang");
      argvCompile.push_back("-fdisable-module-hash");
      // Disable the warning when we import a module from extern C. Some headers
      // from the STL are doing this and we can't really do anything about this.
      argvCompile.push_back("-Wno-module-import-in-extern-c");
      // Disable the warning when we import a module in a function body. This
      // is a ROOT-specific issue tracked by ROOT-9088.
      // FIXME: Remove after merging ROOT's PR1306.
      argvCompile.push_back("-Wno-modules-import-nested-redundant");
      // FIXME: We get an error "'cling/module.modulemap' from the precompiled
      //  header has been overridden". This comes from a bug that rootcling
      // introduces by adding a lot of garbage in the PCH/PCM files because it
      // essentially serializes its current state of the AST. That usually
      // includes a few memory buffers which override their own contents.
      // We know how to remove this: just implement a callback in clang
      // which calls back the interpreter when a module file is built. This is
      // a lot of work as it needs fixing rootcling. See RE-0003.
      argvCompile.push_back("-Xclang");
      argvCompile.push_back("-fno-validate-pch");
    }

    if (!COpts.Language) {
      // We do C++ by default; append right after argv[0] if no "-x" given
      argvCompile.push_back("-x");
      argvCompile.push_back( "c++");
    }

    if (COpts.DefaultLanguage()) {
      // By default, set the standard to what cling was compiled with.
      // clang::driver::Compilation will do various things while initializing
      // and by enforcing the std version now cling is telling clang what to
      // do, rather than after clang has dedcuded a default.
      switch (CxxStdCompiledWith()) {
        case 17: argvCompile.emplace_back("-std=c++1z"); break;
        case 14: argvCompile.emplace_back("-std=c++14"); break;
        case 11: argvCompile.emplace_back("-std=c++11"); break;
        default: llvm_unreachable("Unrecognized C++ version");
      }
    }

    // This argument starts the cling instance with the x86 target. Otherwise,
    // the first job in the joblist starts the cling instance with the nvptx
    // target.
    if(COpts.CUDAHost)
      argvCompile.push_back("--cuda-host-only");

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

    if (!COpts.HasOutput || !HasInput) {
      argvCompile.push_back("-c");
      argvCompile.push_back("-");
    }

    auto InvocationPtr = std::make_shared<clang::CompilerInvocation>();

    // The compiler invocation is the owner of the diagnostic options.
    // Everything else points to them.
    DiagnosticOptions& DiagOpts = InvocationPtr->getDiagnosticOpts();
    llvm::IntrusiveRefCntPtr<DiagnosticsEngine> Diags =
        SetupDiagnostics(DiagOpts);
    if (!Diags) {
      cling::errs() << "Could not setup diagnostic engine.\n";
      return nullptr;
    }

    llvm::Triple TheTriple(llvm::sys::getProcessTriple());
#ifdef LLVM_ON_WIN32
    // COFF format currently needs a few changes in LLVM to function properly.
    TheTriple.setObjectFormat(llvm::Triple::COFF);
#endif
    clang::driver::Driver Drvr(argv[0], TheTriple.getTriple(), *Diags);
    //Drvr.setWarnMissingInput(false);
    Drvr.setCheckInputsExist(false); // think foo.C(12)
    llvm::ArrayRef<const char*>RF(&(argvCompile[0]), argvCompile.size());
    std::unique_ptr<clang::driver::Compilation>
      Compilation(Drvr.BuildCompilation(RF));
    if (!Compilation) {
      cling::errs() << "Couldn't create clang::driver::Compilation.\n";
      return nullptr;
    }

    const driver::ArgStringList* CC1Args = GetCC1Arguments(Compilation.get());
    if (!CC1Args) {
      cling::errs() << "Could not get cc1 arguments.\n";
      return nullptr;
    }

    clang::CompilerInvocation::CreateFromArgs(*InvocationPtr, CC1Args->data() + 1,
                                              CC1Args->data() + CC1Args->size(),
                                              *Diags);
    // We appreciate the error message about an unknown flag (or do we? if not
    // we should switch to a different DiagEngine for parsing the flags).
    // But in general we'll happily go on.
    Diags->Reset();

    // Create and setup a compiler instance.
    std::unique_ptr<CompilerInstance> CI(new CompilerInstance());
    CI->setInvocation(InvocationPtr);
    CI->setDiagnostics(Diags.get()); // Diags is ref-counted
    if (!OnlyLex)
      CI->getDiagnosticOpts().ShowColors = cling::utils::ColorizeOutput();

    // Copied from CompilerInstance::createDiagnostics:
    // Chain in -verify checker, if requested.
    if (DiagOpts.VerifyDiagnostics)
      Diags->setClient(new clang::VerifyDiagnosticConsumer(*Diags));
    // Configure our handling of diagnostics.
    ProcessWarningOptions(*Diags, DiagOpts);

    if (COpts.HasOutput && !OnlyLex) {
      ActionScan scan(clang::driver::Action::PrecompileJobClass,
                      clang::driver::Action::PreprocessJobClass);
      if (!scan.find(Compilation.get())) {
        cling::errs() << "Only precompiled header or preprocessor "
                        "output is supported.\n";
        return nullptr;
      }
      if (!SetupCompiler(CI.get(), COpts))
        return nullptr;

      ProcessWarningOptions(*Diags, DiagOpts);
      return CI.release();
    }

    CI->createFileManager();
    clang::CompilerInvocation& Invocation = CI->getInvocation();
    std::string& PCHFile = Invocation.getPreprocessorOpts().ImplicitPCHInclude;
    bool InitLang = true, InitTarget = true;
    if (!PCHFile.empty()) {
      if (cling::utils::LookForFile(argvCompile, PCHFile,
              &CI->getFileManager(),
              COpts.Verbose ? "Precompiled header" : nullptr)) {
        // Load target options etc from PCH.
        struct PCHListener: public ASTReaderListener {
          CompilerInvocation& m_Invocation;
          bool m_ReadLang, m_ReadTarget;

          PCHListener(CompilerInvocation& I) :
            m_Invocation(I), m_ReadLang(false), m_ReadTarget(false) {}

          bool ReadLanguageOptions(const LangOptions &LangOpts,
                                   bool /*Complain*/,
                                   bool /*AllowCompatibleDifferences*/) override {
            *m_Invocation.getLangOpts() = LangOpts;
            m_ReadLang = true;
            return false;
          }
          bool ReadTargetOptions(const TargetOptions &TargetOpts,
                                 bool /*Complain*/,
                                 bool /*AllowCompatibleDifferences*/) override {
            m_Invocation.getTargetOpts() = TargetOpts;
            m_ReadTarget = true;
            return false;
          }
          bool ReadPreprocessorOptions(const PreprocessorOptions &PPOpts,
                                       bool /*Complain*/,
                                  std::string &/*SuggestedPredefines*/) override {
            // Import selected options, e.g. don't overwrite ImplicitPCHInclude.
            PreprocessorOptions& myPP = m_Invocation.getPreprocessorOpts();
            insertBehind(myPP.Macros, PPOpts.Macros);
            insertBehind(myPP.Includes, PPOpts.Includes);
            insertBehind(myPP.MacroIncludes, PPOpts.MacroIncludes);
            return false;
          }
        };
        PCHListener listener(Invocation);
        if (ASTReader::readASTFileControlBlock(PCHFile,
                                               CI->getFileManager(),
                                               CI->getPCHContainerReader(),
                                               false /*FindModuleFileExt*/,
                                               listener,
                                         /*ValidateDiagnosticOptions=*/false)) {
          // When running interactively pass on the info that the PCH
          // has failed so that IncrmentalParser::Initialize won't try again.
          if (!HasInput && llvm::sys::Process::StandardInIsUserInput()) {
            const unsigned ID = Diags->getCustomDiagID(
                                       clang::DiagnosticsEngine::Level::Error,
                                       "Problems loading PCH: '%0'.");
            
            Diags->Report(ID) << PCHFile;
            // If this was the only error, then don't let it stop anything
            if (Diags->getClient()->getNumErrors() == 1)
              Diags->Reset(true);
            // Clear the include so no one else uses it.
            std::string().swap(PCHFile);
          }
        }
        // All we care about is if Language and Target options were successful.
        InitLang = !listener.m_ReadLang;
        InitTarget = !listener.m_ReadTarget;
      }
    }

    FrontendOptions& FrontendOpts = Invocation.getFrontendOpts();

    // Register the externally constructed extensions.
    assert(FrontendOpts.ModuleFileExtensions.empty() && "Extensions exist!");
    for (auto& E : moduleExtensions)
      FrontendOpts.ModuleFileExtensions.push_back(E);

    FrontendOpts.DisableFree = true;

    // Set up compiler language and target
    if (!SetupCompiler(CI.get(), COpts, InitLang, InitTarget))
      return nullptr;

    // Set up source managers
    SourceManager* SM = new SourceManager(CI->getDiagnostics(),
                                          CI->getFileManager(),
                                          /*UserFilesAreVolatile*/ true);
    CI->setSourceManager(SM); // CI now owns SM

    if (FrontendOpts.ShowTimers)
      CI->createFrontendTimer();

    if (FrontendOpts.ModulesEmbedAllFiles)
       CI->getSourceManager().setAllFilesAreTransient(true);

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
    // be #included (so we won't get a clash in clang's cache).
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

    // Create TargetInfo for the other side of CUDA and OpenMP compilation.
    if ((CI->getLangOpts().CUDA || CI->getLangOpts().OpenMPIsDevice) &&
        !CI->getFrontendOpts().AuxTriple.empty()) {
          auto TO = std::make_shared<TargetOptions>();
          TO->Triple = CI->getFrontendOpts().AuxTriple;
          TO->HostTriple = CI->getTarget().getTriple().str();
          CI->setAuxTarget(TargetInfo::CreateTargetInfo(CI->getDiagnostics(), TO));
    }

    // Set up the preprocessor
    CI->createPreprocessor(TU_Complete);

    // With modules, we now start adding prebuilt module paths to the CI.
    // Modules from those paths are treated like they are never out of date
    // and we don't update them on demand.
    // This mostly helps ROOT where we can't just recompile any out of date
    // modules because we would miss the annotations that rootcling creates.
    if (COpts.CxxModules) {
      setupCxxModules(*CI);
    }

    Preprocessor& PP = CI->getPreprocessor();

    PP.getBuiltinInfo().initializeBuiltins(PP.getIdentifierTable(),
                                           PP.getLangOpts());

    // Set up the ASTContext
    CI->createASTContext();

    std::vector<std::unique_ptr<ASTConsumer>> Consumers;

    if (!OnlyLex) {
      assert(customConsumer && "Need to specify a custom consumer"
                               " when not in OnlyLex mode");
      Consumers.push_back(std::move(customConsumer));
    }

    // With C++ modules, we now attach the consumers that will handle the
    // generation of the PCM file itself in case we want to generate
    // a C++ module with the current interpreter instance.
    if (COpts.CxxModules && !COpts.ModuleName.empty()) {
      // Code below from the (private) code in the GenerateModuleAction class.
      llvm::SmallVector<char, 256> Output;
      llvm::sys::path::append(Output, COpts.CachePath,
                              COpts.ModuleName + ".pcm");
      StringRef ModuleOutputFile = StringRef(Output.data(), Output.size());

      std::unique_ptr<raw_pwrite_stream> OS =
          CI->createOutputFile(ModuleOutputFile, /*Binary=*/true,
                               /*RemoveFileOnSignal=*/false, "",
                               /*Extension=*/"", /*useTemporary=*/true,
                               /*CreateMissingDirectories=*/true);
      assert(OS);

      std::string Sysroot;

      auto Buffer = std::make_shared<PCHBuffer>();

      Consumers.push_back(llvm::make_unique<PCHGenerator>(
          CI->getPreprocessor(), ModuleOutputFile, Sysroot, Buffer,
          CI->getFrontendOpts().ModuleFileExtensions,
          /*AllowASTWithErrors=*/false,
          /*IncludeTimestamps=*/
          +CI->getFrontendOpts().BuildingImplicitModule));
      Consumers.push_back(
          CI->getPCHContainerWriter().CreatePCHContainerGenerator(
              *CI, "", ModuleOutputFile, std::move(OS), Buffer));

      // Set the current module name for clang. With that clang doesn't start
      // to build the current module on demand when we include a header
      // from the current module.
      CI->getLangOpts().CurrentModule = COpts.ModuleName;
      CI->getLangOpts().setCompilingModule(LangOptions::CMK_ModuleMap);

      // Push the current module to the build stack so that clang knows when
      // we have a cyclic dependency.
      SM->pushModuleBuildStack(COpts.ModuleName,
                               FullSourceLoc(SourceLocation(), *SM));
    }

    std::unique_ptr<clang::MultiplexConsumer> multiConsumer(
        new clang::MultiplexConsumer(std::move(Consumers)));
    CI->setASTConsumer(std::move(multiConsumer));

    // Set up Sema
    CodeCompleteConsumer* CCC = 0;
    // Make sure we inform Sema we compile a Module.
    CI->createSema(COpts.ModuleName.empty() ? TU_Complete : TU_Module, CCC);

    // Set CodeGen options.
    CodeGenOptions& CGOpts = CI->getCodeGenOpts();
#ifdef _MSC_VER
    CGOpts.MSVolatile = 1;
    CGOpts.RelaxedAliasing = 1;
    CGOpts.EmitCodeView = 1;
    CGOpts.CXXCtorDtorAliases = 1;
#endif
    // Reduce amount of emitted symbols by optimizing more.
    // FIXME: We have a bug when we switch to -O2, for some cases it takes
    // several minutes to optimize, while the same code compiled by clang -O2
    // takes only a few seconds.
    CGOpts.OptimizationLevel = 0;
    // Taken from a -O2 run of clang:
    CGOpts.DiscardValueNames = 1;
    CGOpts.OmitLeafFramePointer = 1;
    CGOpts.UnrollLoops = 1;
    CGOpts.VectorizeLoop = 1;
    CGOpts.VectorizeSLP = 1;
    CGOpts.DisableO0ImplyOptNone = 1; // Enable dynamic opt level switching.

    CGOpts.setInlining((CGOpts.OptimizationLevel == 0)
                       ? CodeGenOptions::OnlyAlwaysInlining
                       : CodeGenOptions::NormalInlining);

    // CGOpts.setDebugInfo(clang::CodeGenOptions::FullDebugInfo);
    // CGOpts.EmitDeclMetadata = 1; // For unloading, for later
    // aliasing the complete ctor to the base ctor causes the JIT to crash
    CGOpts.CXXCtorDtorAliases = 0;
    CGOpts.VerifyModule = 0; // takes too long

    if (!OnlyLex) {
      // -nobuiltininc
      clang::HeaderSearchOptions& HOpts = CI->getHeaderSearchOpts();
      if (CI->getHeaderSearchOpts().UseBuiltinIncludes)
        AddRuntimeIncludePaths(ClingBin, HOpts);

      // Write a marker to know the rest of the output is from clang
      if (COpts.Verbose)
        cling::log() << "Setting up system headers with clang:\n";

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

    // Tell the diagnostic client that we are entering file parsing mode as the
    // handling of modulemap files may issue diagnostics.
    // FIXME: Consider moving in SetupDiagnostics.
    DiagnosticConsumer& DClient = CI->getDiagnosticClient();
    DClient.BeginSourceFile(CI->getLangOpts(), &PP);

    for (const auto& Filename : FrontendOpts.ModuleMapFiles) {
      if (auto* File = FM.getFile(Filename))
        PP.getHeaderSearchInfo().loadModuleMapFile(File, /*IsSystem*/ false);
      else
        CI->getDiagnostics().Report(diag::err_module_map_not_found) << Filename;
    }

    HandleProgramActions(*CI);

    return CI.release(); // Passes over the ownership to the caller.
  }

} // unnamed namespace

namespace cling {

CompilerInstance*
CIFactory::createCI(llvm::StringRef Code, const InvocationOptions& Opts,
                    const char* LLVMDir,
                    std::unique_ptr<clang::ASTConsumer> consumer,
                    const ModuleFileExtensions& moduleExtensions) {
  return createCIImpl(llvm::MemoryBuffer::getMemBuffer(Code), Opts.CompilerOpts,
                      LLVMDir, std::move(consumer), moduleExtensions,
                      false /*OnlyLex*/,
                      !Opts.IsInteractive());
}

CompilerInstance* CIFactory::createCI(
    MemBufPtr_t Buffer, int argc, const char* const* argv, const char* LLVMDir,
    std::unique_ptr<clang::ASTConsumer> consumer,
    const ModuleFileExtensions& moduleExtensions, bool OnlyLex /*false*/) {
  return createCIImpl(std::move(Buffer), CompilerOptions(argc, argv),  LLVMDir,
                      std::move(consumer), moduleExtensions, OnlyLex);
}

} // namespace cling

