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
#include "clang/Driver/ArgList.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Job.h"
#include "clang/Driver/Tool.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Lex/Preprocessor.h"

#include "llvm/LLVMContext.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

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
  static const clang::driver::ArgStringList
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
    llvm::sys::Path resource_path;
    if (llvmdir) {
      resource_path = llvmdir;
      resource_path.appendComponent("lib");
      resource_path.appendComponent("clang");
      resource_path.appendComponent(CLANG_VERSION_STRING);
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
    if (!resource_path.canRead()) {
      llvm::errs()
        << "ERROR in cling::CIFactory::createCI():\n  resource directory "
        << resource_path.str() << " not found!\n";
      resource_path = "";
    }

    //______________________________________
    DiagnosticOptions DefaultDiagnosticOptions;
    DefaultDiagnosticOptions.ShowColors = 1;
    TextDiagnosticPrinter* DiagnosticPrinter
      = new TextDiagnosticPrinter(llvm::errs(), DefaultDiagnosticOptions);
    llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs> DiagIDs(new DiagnosticIDs());
    DiagnosticsEngine* Diagnostics
      = new DiagnosticsEngine(DiagIDs, DiagnosticPrinter,
                              /*Owns it*/ true); // LEAKS!

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

    bool IsProduction = false;
    assert(IsProduction = true && "set IsProduction if asserts are on.");
    clang::driver::Driver Driver(argv[0], llvm::sys::getDefaultTargetTriple(),
                                 "cling.out",
                                 IsProduction,
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
      llvm::sys::Path oldResInc(Opts.ResourceDir);
      oldResInc.appendComponent("include");
      llvm::sys::Path newResInc(resource_path);
      newResInc.appendComponent("include");
      bool foundOldResInc = false;
      for (unsigned i = 0, e = Opts.UserEntries.size();
           !foundOldResInc && i != e; ++i) {
        HeaderSearchOptions::Entry &E = Opts.UserEntries[i];
        if (!E.IsUserSupplied && !E.IsFramework
            && E.Group == clang::frontend::System && E.IgnoreSysRoot
            && E.IsInternal && !E.ImplicitExternC
            && oldResInc.str() == E.Path) {
          E.Path = newResInc.str();
          foundOldResInc = true;
        }
      }

      Opts.ResourceDir = resource_path.str();
    }

    // Create and setup a compiler instance.
    CompilerInstance* CI = new CompilerInstance();
    CI->setInvocation(Invocation);

    CI->createDiagnostics(CC1Args->size(), CC1Args->data() + 1);
    {
      //
      //  Buffer the error messages while we process
      //  the compiler options.
      //

      // Set the language options, which cling needs
      SetClingCustomLangOpts(CI->getLangOpts());

      CI->getInvocation().getPreprocessorOpts().addMacroDef("__CLING__");
      if (CI->getDiagnostics().hasErrorOccurred()) {
        delete CI;
        CI = 0;
        return 0;
      }
    }
    CI->setTarget(TargetInfo::CreateTargetInfo(CI->getDiagnostics(),
                                               Invocation->getTargetOpts()));
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
    CI->createSourceManager(CI->getFileManager());

    // Set up the memory buffer
    if (buffer)
      CI->getSourceManager().createMainFileIDForMemBuffer(buffer);

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
    Opts.EmitAllDecls = 1;
    Opts.Exceptions = 1;
    Opts.CXXExceptions = 1;
    Opts.Deprecated = 1;
    //Opts.Modules = 1;
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
