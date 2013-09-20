//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Axel Naumann <axel@cern.ch>
//------------------------------------------------------------------------------

#include "cling/Interpreter/CIFactory.h"

#include "DeclCollector.h"

#include "clang/AST/ASTContext.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/Version.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Job.h"
#include "clang/Driver/Tool.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Lex/Preprocessor.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"

#include <ctime>

using namespace clang;

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
                    llvmdir);
  }

  CompilerInstance* CIFactory::createCI(llvm::MemoryBuffer* buffer,
                                        int argc,
                                        const char* const *argv,
                                        const char* llvmdir) {
    // Create an instance builder, passing the llvmdir and arguments.
    //
    //  Initialize the llvm library.
    llvm::InitializeNativeTarget();
    llvm::InitializeAllAsmPrinters();
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

    //______________________________________
    DiagnosticOptions* DefaultDiagnosticOptions = new DiagnosticOptions();
    DefaultDiagnosticOptions->ShowColors = llvm::sys::Process::StandardErrHasColors() ? 1 : 0;
    TextDiagnosticPrinter* DiagnosticPrinter
      = new TextDiagnosticPrinter(llvm::errs(), DefaultDiagnosticOptions);
    llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs> DiagIDs(new DiagnosticIDs());
    DiagnosticsEngine* Diagnostics
      = new DiagnosticsEngine(DiagIDs, DefaultDiagnosticOptions,
                              DiagnosticPrinter, /*Owns it*/ true); // LEAKS!

    std::vector<const char*> argvCompile(argv, argv + argc);
    // We do C++ by default; append right after argv[0] name
    // Only insert it if there is no other "-x":
    bool haveMinusX = false;
    for (const char* const* iarg = argv; !haveMinusX && iarg < argv + argc;
         ++iarg) {
      haveMinusX = !strcmp(*iarg, "-x");
    }
    if (!haveMinusX) {
      argvCompile.insert(argvCompile.begin() + 1,"-x");
      argvCompile.insert(argvCompile.begin() + 2, "c++");
    }
    argvCompile.push_back("-c");
    argvCompile.push_back("-");

    clang::driver::Driver Driver(argv[0], llvm::sys::getDefaultTargetTriple(),
                                 "cling.out",
                                 *Diagnostics);
    //Driver.setWarnMissingInput(false);
    Driver.setCheckInputsExist(false); // think foo.C(12)
    llvm::ArrayRef<const char*>RF(&(argvCompile[0]), argvCompile.size());
    llvm::OwningPtr<clang::driver::Compilation>
      Compilation(Driver.BuildCompilation(RF));
    const clang::driver::ArgStringList* CC1Args
      = GetCC1Arguments(Diagnostics, Compilation.get());
    if (CC1Args == NULL) {
      return 0;
    }
    clang::CompilerInvocation*
      Invocation = new clang::CompilerInvocation;
    clang::CompilerInvocation::CreateFromArgs(*Invocation, CC1Args->data() + 1,
                                              CC1Args->data() + CC1Args->size(),
                                              *Diagnostics);
    Invocation->getFrontendOpts().DisableFree = true;

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
          E.Path = newResInc.data();
          foundOldResInc = true;
        }
      }

      Opts.ResourceDir = resource_path.str();
    }

    // Create and setup a compiler instance.
    CompilerInstance* CI = new CompilerInstance();
    CI->setInvocation(Invocation);

    CI->createDiagnostics(DiagnosticPrinter, /*ShouldOwnClient=*/ false);
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

      if (CI->getDiagnostics().hasErrorOccurred()) {
        delete CI;
        CI = 0;
        return 0;
      }
    }
    CI->setTarget(TargetInfo::CreateTargetInfo(CI->getDiagnostics(),
                                               &Invocation->getTargetOpts()));
    if (!CI->hasTarget()) {
      delete CI;
      CI = 0;
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
        llvm::errs() << "Warning: unhandled target architecture "
                     << CI->getTarget().getTriple().getArchName() << '\n';
      }
    }

    // Set up source and file managers
    CI->createFileManager();
    SourceManager* SM = new SourceManager(CI->getDiagnostics(),
                                          CI->getFileManager(),
                                          /*UserFilesAreVolatile*/ true); 
    CI->setSourceManager(SM); // FIXME: SM leaks.

    // Set up the memory buffer
    if (buffer)
      CI->getSourceManager().createMainFileIDForMemBuffer(buffer);
    else {
      // As main file we want
      // * a virtual file that is claiming to be huge 
      // * with an empty memory buffer attached (to bring the content)
      SourceManager& SM = CI->getSourceManager();
      FileManager& FM = SM.getFileManager();
      // Build the virtual file
      const char* Filename = "InteractiveInputLineIncluder.h";
      const std::string& CGOptsMainFileName
        = CI->getInvocation().getCodeGenOpts().MainFileName;
      if (!CGOptsMainFileName.empty())
        Filename = CGOptsMainFileName.c_str();
      const FileEntry* FE
        = FM.getVirtualFile(Filename, 1U << 15U, time(0));
      FileID MainFileID = SM.createMainFileID(FE, SrcMgr::C_User);
      const SrcMgr::SLocEntry& MainFileSLocE = SM.getSLocEntry(MainFileID);
      const SrcMgr::ContentCache* MainFileCC
        = MainFileSLocE.getFile().getContentCache();
      llvm::MemoryBuffer* MainFileMB
        = llvm::MemoryBuffer::getMemBuffer("/*CLING MAIN FILE*/\n");
      const_cast<SrcMgr::ContentCache*>(MainFileCC)->setBuffer(MainFileMB);
    }

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

    // Set up the ASTConsumers
    CI->setASTConsumer(new DeclCollector());

    // Set up Sema
    CodeCompleteConsumer* CCC = 0;
    CI->createSema(TU_Complete, CCC);

    // Set CodeGen options
    // CI->getCodeGenOpts().DebugInfo = 1; // want debug info
    // CI->getCodeGenOpts().EmitDeclMetadata = 1; // For unloading, for later
    CI->getCodeGenOpts().OptimizationLevel = 0; // see pure SSA, that comes out
    CI->getCodeGenOpts().CXXCtorDtorAliases = 0; // aliasing the complete
                                                 // ctor to the base ctor causes
                                                 // the JIT to crash
    // When asserts are on, TURN ON not compare the VerifyModule
    assert(CI->getCodeGenOpts().VerifyModule = 1);

    return CI;
  }

  void CIFactory::SetClingCustomLangOpts(LangOptions& Opts) {
    Opts.EmitAllDecls = 0; // Otherwise if PCH attached will codegen all decls.
    Opts.Exceptions = 1;
    Opts.CXXExceptions = 1;
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
