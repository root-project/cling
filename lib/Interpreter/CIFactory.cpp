//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Axel Naumann <axel@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "cling/Interpreter/CIFactory.h"

#include "DeclCollector.h"
#include "cling-compiledata.h"

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

#include "llvm/Config/config.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"

#include <ctime>
#include <cstdio>

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

// FIXME: This code has been taken (copied from) llvm/tools/clang/lib/Driver/WindowsToolChain.cpp
// and should probably go to some platform utils place.
// the code for VS 11.0 and 12.0 common tools (vs110comntools and vs120comntools) 
// has been implemented (added) in getVisualStudioDir()
#ifdef _MSC_VER

/// \brief Read registry string.
/// This also supports a means to look for high-versioned keys by use
/// of a $VERSION placeholder in the key path.
/// $VERSION in the key path is a placeholder for the version number,
/// causing the highest value path to be searched for and used.
/// I.e. "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\VisualStudio\\$VERSION".
/// There can be additional characters in the component.  Only the numberic
/// characters are compared.
static bool getSystemRegistryString(const char *keyPath, const char *valueName,
                                    char *value, size_t maxLength) {
  HKEY hRootKey = NULL;
  HKEY hKey = NULL;
  const char* subKey = NULL;
  DWORD valueType;
  DWORD valueSize = maxLength - 1;
  long lResult;
  bool returnValue = false;

  if (strncmp(keyPath, "HKEY_CLASSES_ROOT\\", 18) == 0) {
    hRootKey = HKEY_CLASSES_ROOT;
    subKey = keyPath + 18;
  } else if (strncmp(keyPath, "HKEY_USERS\\", 11) == 0) {
    hRootKey = HKEY_USERS;
    subKey = keyPath + 11;
  } else if (strncmp(keyPath, "HKEY_LOCAL_MACHINE\\", 19) == 0) {
    hRootKey = HKEY_LOCAL_MACHINE;
    subKey = keyPath + 19;
  } else if (strncmp(keyPath, "HKEY_CURRENT_USER\\", 18) == 0) {
    hRootKey = HKEY_CURRENT_USER;
    subKey = keyPath + 18;
  } else {
    return false;
  }

  const char *placeHolder = strstr(subKey, "$VERSION");
  char bestName[256];
  bestName[0] = '\0';
  // If we have a $VERSION placeholder, do the highest-version search.
  if (placeHolder) {
    const char *keyEnd = placeHolder - 1;
    const char *nextKey = placeHolder;
    // Find end of previous key.
    while ((keyEnd > subKey) && (*keyEnd != '\\'))
      keyEnd--;
    // Find end of key containing $VERSION.
    while (*nextKey && (*nextKey != '\\'))
      nextKey++;
    size_t partialKeyLength = keyEnd - subKey;
    char partialKey[256];
    if (partialKeyLength > sizeof(partialKey))
      partialKeyLength = sizeof(partialKey);
    strncpy(partialKey, subKey, partialKeyLength);
    partialKey[partialKeyLength] = '\0';
    HKEY hTopKey = NULL;
    lResult = RegOpenKeyEx(hRootKey, partialKey, 0, KEY_READ | KEY_WOW64_32KEY,
                           &hTopKey);
    if (lResult == ERROR_SUCCESS) {
      char keyName[256];
      int bestIndex = -1;
      double bestValue = 0.0;
      DWORD index, size = sizeof(keyName) - 1;
      for (index = 0; RegEnumKeyEx(hTopKey, index, keyName, &size, NULL,
          NULL, NULL, NULL) == ERROR_SUCCESS; index++) {
        const char *sp = keyName;
        while (*sp && !isDigit(*sp))
          sp++;
        if (!*sp)
          continue;
        const char *ep = sp + 1;
        while (*ep && (isDigit(*ep) || (*ep == '.')))
          ep++;
        char numBuf[32];
        strncpy(numBuf, sp, sizeof(numBuf) - 1);
        numBuf[sizeof(numBuf) - 1] = '\0';
        double dvalue = strtod(numBuf, NULL);
        if (dvalue > bestValue) {
          // Test that InstallDir is indeed there before keeping this index.
          // Open the chosen key path remainder.
          strcpy(bestName, keyName);
          // Append rest of key.
          strncat(bestName, nextKey, sizeof(bestName) - 1);
          bestName[sizeof(bestName) - 1] = '\0';
          lResult = RegOpenKeyEx(hTopKey, bestName, 0,
                                 KEY_READ | KEY_WOW64_32KEY, &hKey);
          if (lResult == ERROR_SUCCESS) {
            lResult = RegQueryValueEx(hKey, valueName, NULL, &valueType,
              (LPBYTE)value, &valueSize);
            if (lResult == ERROR_SUCCESS) {
              bestIndex = (int)index;
              bestValue = dvalue;
              returnValue = true;
            }
            RegCloseKey(hKey);
          }
        }
        size = sizeof(keyName) - 1;
      }
      RegCloseKey(hTopKey);
    }
  } else {
    lResult = RegOpenKeyEx(hRootKey, subKey, 0, KEY_READ | KEY_WOW64_32KEY,
                           &hKey);
    if (lResult == ERROR_SUCCESS) {
      lResult = RegQueryValueEx(hKey, valueName, NULL, &valueType,
        (LPBYTE)value, &valueSize);
      if (lResult == ERROR_SUCCESS)
        returnValue = true;
      RegCloseKey(hKey);
    }
  }
  return returnValue;
}

/// \brief Get Windows SDK installation directory.
static bool getWindowsSDKDir(std::string &path) {
  char windowsSDKInstallDir[256];
  // Try the Windows registry.
  bool hasSDKDir = getSystemRegistryString(
   "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Microsoft SDKs\\Windows\\$VERSION",
                                           "InstallationFolder",
                                           windowsSDKInstallDir,
                                           sizeof(windowsSDKInstallDir) - 1);
    // If we have both vc80 and vc90, pick version we were compiled with.
  if (hasSDKDir && windowsSDKInstallDir[0]) {
    path = windowsSDKInstallDir;
    return true;
  }
  return false;
}

  // Get Visual Studio installation directory.
static bool getVisualStudioDir(std::string &path) {
  // First check the environment variables that vsvars32.bat sets.
  const char* vcinstalldir = getenv("VCINSTALLDIR");
  if (vcinstalldir) {
    char *p = const_cast<char *>(strstr(vcinstalldir, "\\VC"));
    if (p)
      *p = '\0';
    path = vcinstalldir;
    return true;
  }
  int VSVersion = (_MSC_VER / 100) - 6;
  std::stringstream keyName;
  keyName << "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\VisualStudio\\" << VSVersion << ".0";
  char vsIDEInstallDir[256];
  char vsExpressIDEInstallDir[256];
  // Then try the windows registry.
  bool hasVCDir = getSystemRegistryString(keyName.str().c_str(),
    "InstallDir", vsIDEInstallDir, sizeof(vsIDEInstallDir) - 1);
  keyName.str(std::string());
  keyName << "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\VCExpress\\" << VSVersion << ".0";
  bool hasVCExpressDir = getSystemRegistryString(keyName.str().c_str(),
    "InstallDir", vsExpressIDEInstallDir, sizeof(vsExpressIDEInstallDir) - 1);
    // If we have both vc80 and vc90, pick version we were compiled with.
  if (hasVCDir && vsIDEInstallDir[0]) {
    char *p = (char*)strstr(vsIDEInstallDir, "\\Common7\\IDE");
    if (p)
      *p = '\0';
    path = vsIDEInstallDir;
    return true;
  }

  if (hasVCExpressDir && vsExpressIDEInstallDir[0]) {
    char *p = (char*)strstr(vsExpressIDEInstallDir, "\\Common7\\IDE");
    if (p)
      *p = '\0';
    path = vsExpressIDEInstallDir;
    return true;
  }

  // Try the environment.
  const char *vs120comntools = getenv("VS120COMNTOOLS");
  const char *vs110comntools = getenv("VS110COMNTOOLS");
  const char *vs100comntools = getenv("VS100COMNTOOLS");
  const char *vs90comntools = getenv("VS90COMNTOOLS");
  const char *vs80comntools = getenv("VS80COMNTOOLS");
  const char *vscomntools = NULL;

  // Try to find the version that we were compiled with
  if(false) {}
  #if (_MSC_VER >= 1800)  // VC120
  else if(vs120comntools) {
    vscomntools = vs120comntools;
  }
  #elif (_MSC_VER >= 1700)  // VC110
  else if(vs110comntools) {
    vscomntools = vs110comntools;
  }
  #elif (_MSC_VER >= 1600)  // VC100
  else if(vs100comntools) {
    vscomntools = vs100comntools;
  }
  #elif (_MSC_VER == 1500) // VC80
  else if(vs90comntools) {
    vscomntools = vs90comntools;
  }
  #elif (_MSC_VER == 1400) // VC80
  else if(vs80comntools) {
    vscomntools = vs80comntools;
  }
  #endif
  // Otherwise find any version we can
  else if (vs120comntools)
    vscomntools = vs120comntools;
  else if (vs110comntools)
    vscomntools = vs110comntools;
  else if (vs100comntools)
    vscomntools = vs100comntools;
  else if (vs90comntools)
    vscomntools = vs90comntools;
  else if (vs80comntools)
    vscomntools = vs80comntools;

  if (vscomntools && *vscomntools) {
    const char *p = strstr(vscomntools, "\\Common7\\Tools");
    path = p ? std::string(vscomntools, p) : vscomntools;
    return true;
  }
  return false;
}

#endif // _MSC_VER

namespace cling {
  //
  //  Dummy function so we can use dladdr to find the executable path.
  //
  void locate_cling_executable()
  {
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
      = cast<clang::driver::Command>(*Jobs.begin());
    if (llvm::StringRef(Cmd->getCreator().getName()) != "clang") {
      // diagnose this...
      return NULL;
    }

    return &Cmd->getArguments();
  }

  CompilerInstance* CIFactory::createCI(llvm::StringRef code,
                                        int argc,
                                        const char* const *argv,
                                        const char* llvmdir) {
    return createCI(llvm::MemoryBuffer::getMemBuffer(code), argc, argv,
                    llvmdir, new DeclCollector());
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

  ///\brief Check the compile-time C++ ABI version vs the run-time ABI version,
  /// a mismatch could cause havoc. Reports if ABI versions differ.
  static void CheckABICompatibility() {
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
    static const char* runABIQuery
      = "echo '#include <vector>' | " LLVM_CXX " -xc++ -dM -E - "
      "| grep 'define " CLING_CXXABIS "' | awk '{print $3}'";
    bool ABIReadError = true;
    if (FILE *pf = ::popen(runABIQuery, "r")) {
      char buf[2048];
      if (fgets(buf, 2048, pf)) {
        size_t lenbuf = strlen(buf);
        if (lenbuf) {
          ABIReadError = false;
          buf[lenbuf - 1] = 0;   // remove trailing \n
          int runABIVersion = ::atoi(buf);
          if (runABIVersion != CLING_CXXABIV) {
            llvm::errs()
              << "Warning in cling::CIFactory::createCI():\n  "
              "C++ ABI mismatch, compiled with "
              CLING_CXXABIS " v" << CLING_CXXABIV
              << " running with v" << runABIVersion << "\n";
          }
        }
      }
      ::pclose(pf);
    }
    if (ABIReadError) {
      llvm::errs()
        << "Warning in cling::CIFactory::createCI():\n  "
        "Possible C++ standard library mismatch, compiled with "
        CLING_CXXABIS " v" << CLING_CXXABIV
        << " but extraction of runtime standard library version failed.\n"
        "Invoking:\n"
        "    " << runABIQuery << "\n"
        "results in\n";
      int ExitCode = system(runABIQuery);
      llvm::errs() << "with exit code " << ExitCode << "\n";
    }
#endif
#undef CLING_CXXABIV
#undef CLING_CXXABIS
  }

  ///\brief Adds standard library -I used by whatever compiler is found in PATH.
  static void AddHostCXXIncludes(std::vector<const char*>& args) {
    static bool IncludesSet = false;
    static std::vector<std::string> HostCXXI;
    if (!IncludesSet) {
      IncludesSet = true;
#ifdef _MSC_VER
      // Honor %INCLUDE%. It should know essential search paths with vcvarsall.bat.
      if (const char *cl_include_dir = getenv("INCLUDE")) {
        SmallVector<StringRef, 8> Dirs;
        StringRef(cl_include_dir).split(Dirs, ";");
        for (SmallVectorImpl<StringRef>::iterator I = Dirs.begin(), E = Dirs.end();
             I != E; ++I) {
          StringRef d = *I;
          if (d.size() == 0)
            continue;
          HostCXXI.push_back("-I");
          HostCXXI.push_back(d);
        }
      }
      std::string VSDir;
      std::string WindowsSDKDir;

      // When built with access to the proper Windows APIs, try to actually find
      // the correct include paths first.
      if (getVisualStudioDir(VSDir)) {
        HostCXXI.push_back("-I");
        HostCXXI.push_back(VSDir + "\\VC\\include");
        if (getWindowsSDKDir(WindowsSDKDir)) {
          HostCXXI.push_back("-I");
          HostCXXI.push_back(WindowsSDKDir + "\\include");
        }
        else {
          HostCXXI.push_back("-I");
          HostCXXI.push_back(VSDir + "\\VC\\PlatformSDK\\Include");
        }
      }
#else // _MSC_VER
      static const char *CppInclQuery =
        "echo | LC_ALL=C " LLVM_CXX " -xc++ -E -v - 2>&1 >/dev/null "
        "| awk '/^#include </,/^End of search"
        "/{if (!/^#include </ && !/^End of search/){ print }}' "
        "| grep -E \"(c|g)\\+\\+\"";
      if (FILE *pf = ::popen(CppInclQuery, "r")) {

        HostCXXI.push_back("-nostdinc++");
        char buf[2048];
        while (fgets(buf, sizeof(buf), pf) && buf[0]) {
          size_t lenbuf = strlen(buf);
          buf[lenbuf - 1] = 0;   // remove trailing \n
          // Skip leading whitespace:
          const char* start = buf;
          while (start < buf + lenbuf && *start == ' ')
            ++start;
          if (*start) {
            HostCXXI.push_back("-I");
            HostCXXI.push_back(start);
          }
        }
        ::pclose(pf);
      }
      // HostCXXI contains at least -nostdinc++, -I
      if (HostCXXI.size() < 3) {
        llvm::errs() << "ERROR in cling::CIFactory::createCI(): cannot extract "
          "standard library include paths!\n"
          "Invoking:\n"
          "    " << CppInclQuery << "\n"
          "results in\n";
        int ExitCode = system(CppInclQuery);
        llvm::errs() << "with exit code " << ExitCode << "\n";
      }
#endif // _MSC_VER
      CheckABICompatibility();
    }

    for (std::vector<std::string>::const_iterator
           I = HostCXXI.begin(), E = HostCXXI.end(); I != E; ++I)
      args.push_back(I->c_str());
  }

  CompilerInstance* CIFactory::createCI(llvm::MemoryBuffer* buffer,
                                        int argc,
                                        const char* const *argv,
                                        const char* llvmdir,
                                        DeclCollector* stateCollector) {
    // Create an instance builder, passing the llvmdir and arguments.
    //

    CheckClangCompatibility();

    //  Initialize the llvm library.
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::SmallString<512> resource_path;
    if (llvmdir) {
      resource_path = llvmdir;
      llvm::sys::path::append(resource_path,"lib", "clang", CLANG_VERSION_STRING);
    } else {
      // FIXME: The first arg really does need to be argv[0] on FreeBSD.
      //
      // Note: The second arg is not used for Apple, FreeBSD, Linux,
      //       or cygwin, and can only be used on systems which support
      //       the use of dladdr().
      //
      // Note: On linux and cygwin this uses /proc/self/exe to find the path.
      //
      // Note: On Apple it uses _NSGetExecutablePath().
      //
      // Note: On FreeBSD it uses getprogpath().
      //
      // Note: Otherwise it uses dladdr().
      //
      resource_path
        = CompilerInvocation::GetResourcesPath("cling",
                                       (void*)(intptr_t) locate_cling_executable
                                               );
    }
    if (!llvm::sys::fs::is_directory(resource_path.str())) {
      llvm::errs()
        << "ERROR in cling::CIFactory::createCI():\n  resource directory "
        << resource_path.str() << " not found!\n";
      resource_path = "";
    }

    std::vector<const char*> argvCompile(argv, argv + argc);
    // We do C++ by default; append right after argv[0] name
    // Only insert it if there is no other "-x":
    bool hasMinusX = false;
    for (const char* const* iarg = argv; iarg < argv + argc;
         ++iarg) {
      if (!hasMinusX)
        hasMinusX = !strcmp(*iarg, "-x");
      if (hasMinusX)
        break;
    }
    if (!hasMinusX) {
      argvCompile.insert(argvCompile.begin() + 1,"-x");
      argvCompile.insert(argvCompile.begin() + 2, "c++");
    }
    argvCompile.insert(argvCompile.begin() + 3,"-c");
    argvCompile.insert(argvCompile.begin() + 4,"-");
    AddHostCXXIncludes(argvCompile);

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
                                 "cling.out",
                                 *Diags);
    //Driver.setWarnMissingInput(false);
    Driver.setCheckInputsExist(false); // think foo.C(12)
    llvm::ArrayRef<const char*>RF(&(argvCompile[0]), argvCompile.size());
    llvm::OwningPtr<clang::driver::Compilation>
      Compilation(Driver.BuildCompilation(RF));
    const clang::driver::ArgStringList* CC1Args
      = GetCC1Arguments(Diags.getPtr(), Compilation.get());
    if (CC1Args == NULL) {
      return 0;
    }
    clang::CompilerInvocation::CreateFromArgs(*Invocation, CC1Args->data() + 1,
                                              CC1Args->data() + CC1Args->size(),
                                              *Diags);
    Invocation->getFrontendOpts().DisableFree = true;
    // Copied from CompilerInstance::createDiagnostics:
    // Chain in -verify checker, if requested.
    if (DiagOpts.VerifyDiagnostics)
      Diags->setClient(new clang::VerifyDiagnosticConsumer(*Diags));
    // Configure our handling of diagnostics.
    ProcessWarningOptions(*Diags, DiagOpts);

    if (Invocation->getHeaderSearchOpts().UseBuiltinIncludes &&
        !resource_path.empty()) {
      // Update ResourceDir
      // header search opts' entry for resource_path/include isn't
      // updated by providing a new resource path; update it manually.
      clang::HeaderSearchOptions& Opts = Invocation->getHeaderSearchOpts();
      llvm::SmallString<512> oldResInc(Opts.ResourceDir);
      llvm::sys::path::append(oldResInc, "include");
      llvm::SmallString<512> newResInc(resource_path);
      llvm::sys::path::append(newResInc, "include");
      bool foundOldResInc = false;
      for (unsigned i = 0, e = Opts.UserEntries.size();
           !foundOldResInc && i != e; ++i) {
        HeaderSearchOptions::Entry &E = Opts.UserEntries[i];
        if (!E.IsFramework && E.Group == clang::frontend::System
            && E.IgnoreSysRoot && oldResInc.str() == E.Path) {
          E.Path = newResInc.c_str();
          foundOldResInc = true;
        }
      }

      Opts.ResourceDir = resource_path.str();
    }

    // Create and setup a compiler instance.
    llvm::OwningPtr<CompilerInstance> CI(new CompilerInstance());
    CI->setInvocation(Invocation);
    CI->setDiagnostics(Diags.getPtr());
    {
      //
      //  Buffer the error messages while we process
      //  the compiler options.
      //

      // Set the language options, which cling needs
      SetClingCustomLangOpts(CI->getLangOpts());

      CI->getInvocation().getPreprocessorOpts().addMacroDef("__CLING__");
      if (CI->getLangOpts().CPlusPlus11 == 1) {
        // http://llvm.org/bugs/show_bug.cgi?id=13530
        CI->getInvocation().getPreprocessorOpts()
          .addMacroDef("__CLING__CXX11");
      }

      if (CI->getDiagnostics().hasErrorOccurred())
        return 0;
    }
    CI->setTarget(TargetInfo::CreateTargetInfo(CI->getDiagnostics(),
                                               &Invocation->getTargetOpts()));
    if (!CI->hasTarget()) {
      return 0;
    }
    CI->getTarget().setForcedLangOptions(CI->getLangOpts());
    SetClingTargetLangOpts(CI->getLangOpts(), CI->getTarget());
    if (CI->getTarget().getTriple().getOS() == llvm::Triple::Cygwin) {
      // clang "forgets" the basic arch part needed by winnt.h:
      if (CI->getTarget().getTriple().getArch() == llvm::Triple::x86) {
        CI->getInvocation().getPreprocessorOpts().addMacroDef("_X86_=1");
      } else if (CI->getTarget().getTriple().getArch()
                 == llvm::Triple::x86_64) {
        CI->getInvocation().getPreprocessorOpts().addMacroDef("__x86_64=1");
      } else {
        llvm::errs()
          << "Warning in cling::CIFactory::createCI():\n  "
          "unhandled target architecture "
          << CI->getTarget().getTriple().getArchName() << '\n';
      }
    }

    // Set up source and file managers
    CI->createFileManager();
    SourceManager* SM = new SourceManager(CI->getDiagnostics(),
                                          CI->getFileManager(),
                                          /*UserFilesAreVolatile*/ true);
    CI->setSourceManager(SM); // FIXME: SM leaks.

    // As main file we want
    // * a virtual file that is claiming to be huge
    // * with an empty memory buffer attached (to bring the content)
    FileManager& FM = SM->getFileManager();
    // Build the virtual file
    const char* Filename = "InteractiveInputLineIncluder.h";
    const std::string& CGOptsMainFileName
      = CI->getInvocation().getCodeGenOpts().MainFileName;
    if (!CGOptsMainFileName.empty())
      Filename = CGOptsMainFileName.c_str();
    const FileEntry* FE
      = FM.getVirtualFile(Filename, 1U << 15U, time(0));
    FileID MainFileID = SM->createMainFileID(FE, SrcMgr::C_User);
    const SrcMgr::SLocEntry& MainFileSLocE = SM->getSLocEntry(MainFileID);
    const SrcMgr::ContentCache* MainFileCC
      = MainFileSLocE.getFile().getContentCache();
    if (!buffer)
      buffer = llvm::MemoryBuffer::getMemBuffer("/*CLING DEFAULT MEMBUF*/\n");
    const_cast<SrcMgr::ContentCache*>(MainFileCC)->setBuffer(buffer);

    // Set up the preprocessor
    CI->createPreprocessor();
    Preprocessor& PP = CI->getPreprocessor();
    PP.getBuiltinInfo().InitializeBuiltins(PP.getIdentifierTable(),
                                           PP.getLangOpts());

    // Set up the ASTContext
    ASTContext *Ctx = new ASTContext(CI->getLangOpts(),
                                     PP.getSourceManager(), &CI->getTarget(),
                                     PP.getIdentifierTable(),
                                     PP.getSelectorTable(), PP.getBuiltinInfo(),
                                     /*size_reserve*/0, /*DelayInit*/false);
    CI->setASTContext(Ctx);

    //FIXME: This is bad workaround for use-cases which need only a CI to parse
    //things, like pragmas. In that case we cannot controll the interpreter's
    // state collector thus detach from whatever possible.
    if (stateCollector) {
      // Add the callback keeping track of the macro definitions
      PP.addPPCallbacks(stateCollector);
    }
    else
      stateCollector = new DeclCollector();
    // Set up the ASTConsumers
    CI->setASTConsumer(stateCollector);

    // Set up Sema
    CodeCompleteConsumer* CCC = 0;
    CI->createSema(TU_Complete, CCC);

    CI->takeASTConsumer(); // Transfer to ownership to the PP only

    // Set CodeGen options
    // CI->getCodeGenOpts().DebugInfo = 1; // want debug info
    // CI->getCodeGenOpts().EmitDeclMetadata = 1; // For unloading, for later
    CI->getCodeGenOpts().OptimizationLevel = 0; // see pure SSA, that comes out
    CI->getCodeGenOpts().CXXCtorDtorAliases = 0; // aliasing the complete
                                                 // ctor to the base ctor causes
                                                 // the JIT to crash
    CI->getCodeGenOpts().VerifyModule = 0; // takes too long

    return CI.take(); // Passes over the ownership to the caller.
  }

  void CIFactory::SetClingCustomLangOpts(LangOptions& Opts) {
    Opts.EmitAllDecls = 0; // Otherwise if PCH attached will codegen all decls.
    Opts.Exceptions = 1;
    if (Opts.CPlusPlus) {
      Opts.CXXExceptions = 1;
    }
    Opts.Deprecated = 1;
    //Opts.Modules = 1;

    // C++11 is turned on if cling is built with C++11: it's an interperter;
    // cross-language compilation doesn't make sense.
    // Extracted from Boost/config/compiler.
    // SunProCC has no C++11.
    // VisualC's support is not obvious to extract from Boost...
#if /*GCC*/ (defined(__GNUC__) && defined(__GXX_EXPERIMENTAL_CXX0X__))   \
  || /*clang*/ (defined(__has_feature) && __has_feature(cxx_decltype))   \
  || /*ICC*/ ((!(defined(_WIN32) || defined(_WIN64)) && defined(__STDC_HOSTED__) && defined(__INTEL_COMPILER) && (__STDC_HOSTED__ && (__INTEL_COMPILER <= 1200))) || defined(__GXX_EXPERIMENTAL_CPP0X__))
    if (Opts.CPlusPlus)
      Opts.CPlusPlus11 = 1;
#endif

  }

  void CIFactory::SetClingTargetLangOpts(LangOptions& Opts,
                                         const TargetInfo& Target) {
    if (Target.getTriple().getOS() == llvm::Triple::Win32) {
      Opts.MicrosoftExt = 1;
      Opts.MSCVersion = 1300;
      // Should fix http://llvm.org/bugs/show_bug.cgi?id=10528
      Opts.DelayedTemplateParsing = 1;
    } else {
      Opts.MicrosoftExt = 0;
    }
  }
} // end namespace
