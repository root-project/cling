//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Axel Naumann <axel@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "cling/Interpreter/InvocationOptions.h"
#include "cling/Interpreter/ClingOptions.h"
#include "cling/Utils/Output.h"

#include "clang/Basic/LangOptions.h"
#include "clang/Driver/Options.h"

#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/Option.h"
#include "llvm/Option/OptTable.h"

#include <memory>

using namespace clang;
using namespace clang::driver;

using namespace llvm;
using namespace llvm::opt;

using namespace cling;
using namespace cling::driver::clingoptions;

namespace {

// MSVC C++ backend currently does not support -nostdinc++. Translate it to
// -nostdinc so users scripts are insulated from mundane implementation details.
#if defined(LLVM_ON_WIN32) && !defined(_LIBCPP_VERSION)
#define CLING_TRANSLATE_NOSTDINCxx
// Likely to be string-pooled, but make sure it's valid after func exit.
static const char kNoStdInc[] = "-nostdinc";
#endif

#define PREFIX(NAME, VALUE) const char *const NAME[] = VALUE;
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM, \
               HELPTEXT, METAVAR, VALUES)
#include "cling/Interpreter/ClingOptions.inc"
#undef OPTION
#undef PREFIX

  static const OptTable::Info ClingInfoTable[] = {
#define PREFIX(NAME, VALUE)
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM, \
               HELPTEXT, METAVAR, VALUES)   \
  { PREFIX, NAME, HELPTEXT, METAVAR, OPT_##ID, Option::KIND##Class, PARAM, \
    FLAGS, OPT_##GROUP, OPT_##ALIAS, ALIASARGS, VALUES },
#include "cling/Interpreter/ClingOptions.inc"
#undef OPTION
#undef PREFIX
  };

  class ClingOptTable : public OptTable {
  public:
    ClingOptTable()
      : OptTable(ClingInfoTable) {}
  };

  static OptTable* CreateClingOptTable() {
    return new ClingOptTable();
  }

  static void ParseStartupOpts(cling::InvocationOptions& Opts,
                               InputArgList& Args) {
    Opts.ErrorOut = Args.hasArg(OPT__errorout);
    Opts.NoLogo = Args.hasArg(OPT__nologo);
    Opts.ShowVersion = Args.hasArg(OPT_version);
    Opts.Help = Args.hasArg(OPT_help);
    Opts.NoRuntime = Args.hasArg(OPT_noruntime);
    if (Arg* MetaStringArg = Args.getLastArg(OPT__metastr, OPT__metastr_EQ)) {
      Opts.MetaString = MetaStringArg->getValue();
      if (Opts.MetaString.empty()) {
        cling::errs() << "ERROR: meta string must be non-empty! Defaulting to '.'.\n";
        Opts.MetaString = ".";
      }
    }
  }

  static void Extend(std::vector<std::string>& A, std::vector<std::string> B) {
    A.reserve(A.size()+B.size());
    for (std::string& Val: B)
      A.push_back(std::move(Val));
  }

  static void ParseLinkerOpts(cling::InvocationOptions& Opts,
                              InputArgList& Args /* , Diags */) {
    Extend(Opts.LibsToLoad, Args.getAllArgValues(OPT_l));
    Extend(Opts.LibSearchPath, Args.getAllArgValues(OPT_L));
  }
}

CompilerOptions::CompilerOptions(int argc, const char* const* argv)
    : Language(false), ResourceDir(false), SysRoot(false), NoBuiltinInc(false),
      NoCXXInc(false), StdVersion(false), StdLib(false), HasOutput(false),
      Verbose(false), CxxModules(false), CUDAHost(false), CUDADevice(false) {
  if (argc && argv) {
    // Preserve what's already in Remaining, the user might want to push args
    // to clang while still using main's argc, argv
    // insert should/usually does call reserve, but its not part of the standard
    Remaining.reserve(Remaining.size() + argc);
    Remaining.insert(Remaining.end(), argv, argv+argc);
    Parse(argc, argv);
  }
}

void CompilerOptions::Parse(int argc, const char* const argv[],
                            std::vector<std::string>* Inputs) {
  unsigned MissingArgIndex, MissingArgCount;
  std::unique_ptr<OptTable> OptsC1(createDriverOptTable());
  ArrayRef<const char *> ArgStrings(argv+1, argv + argc);

  InputArgList Args(OptsC1->ParseArgs(ArgStrings, MissingArgIndex,
                    MissingArgCount, 0,
                    options::NoDriverOption | options::CLOption));

  for (const Arg* arg : Args) {
    switch (arg->getOption().getID()) {
      // case options::OPT_d_Flag:
      case options::OPT_E:
      case options::OPT_o: HasOutput = true; break;
      case options::OPT_x:
        Language = true;
        CUDAHost =
            (CUDADevice) ? 0 : llvm::StringRef(arg->getValue()) == "cuda";
        break;
      case options::OPT_resource_dir: ResourceDir = true; break;
      case options::OPT_isysroot: SysRoot = true; break;
      case options::OPT_std_EQ: StdVersion = true; break;
      case options::OPT_stdlib_EQ: StdLib = true; break;
      // case options::OPT_nostdlib:
      case options::OPT_nobuiltininc: NoBuiltinInc = true; break;
      // case options::OPT_nostdinc:
      case options::OPT_nostdincxx: NoCXXInc = true; break;
      case options::OPT_v: Verbose = true; break;
      case options::OPT_fmodules: CxxModules = true; break;
      case options::OPT_fmodule_name_EQ: LLVM_FALLTHROUGH;
      case options::OPT_fmodule_name: ModuleName = arg->getValue(); break;
      case options::OPT_fmodules_cache_path: CachePath = arg->getValue(); break;
      case options::OPT_cuda_path_EQ: CUDAPath = arg->getValue(); break;
      case options::OPT_cuda_gpu_arch_EQ: CUDAGpuArch = arg->getValue(); break;
      case options::OPT_Xcuda_fatbinary:
        CUDAFatbinaryArgs.push_back(arg->getValue());
        break;
      case options::OPT_cuda_device_only:
        Language = true;
        CUDADevice = true;
        CUDAHost = false;
        break;

      default:
        if (Inputs && arg->getOption().getKind() == Option::InputClass)
          Inputs->push_back(arg->getValue());
        break;
    }
  }
}

bool CompilerOptions::DefaultLanguage(const LangOptions* LangOpts) const {
  // When StdVersion is set (-std=c++11, -std=gnu++11, etc.) then definitely
  // don't setup the defaults, as they may interfere with what the user is doing
  if (StdVersion)
    return false;

  // Also don't set up the defaults when language is explicitly set; unless
  // the language was set to generate a PCH, in which case definitely do.
  if (Language)
    return HasOutput || (LangOpts && LangOpts->CompilingPCH) || CUDAHost ||
           CUDADevice;

  return true;
}

InvocationOptions::InvocationOptions(int argc, const char* const* argv) :
  MetaString("."), ErrorOut(false), NoLogo(false), ShowVersion(false),
  Help(false), NoRuntime(false) {

  ArrayRef<const char *> ArgStrings(argv, argv + argc);
  unsigned MissingArgIndex, MissingArgCount;
  std::unique_ptr<OptTable> Opts(CreateClingOptTable());

  InputArgList Args(Opts->ParseArgs(ArgStrings, MissingArgIndex,
                    MissingArgCount, 0,
                    options::NoDriverOption | options::CLOption));

  // Forward unknown arguments.
  for (const Arg* arg : Args) {
    switch (arg->getOption().getKind()) {
      case Option::FlagClass:
        // pass -v to clang as well
        if (arg->getOption().getID() != OPT_v)
          break;
        /* Falls through. */
      case Option::UnknownClass:
      case Option::InputClass:
        // prune "-" we need to control where it appears when invoking clang
        if (!arg->getSpelling().equals("-")) {
          if (const char* Arg = argv[arg->getIndex()]) {
#ifdef CLING_TRANSLATE_NOSTDINCxx
            if (!::strcmp(Arg, "-nostdinc++"))
              Arg = kNoStdInc;
#endif
            CompilerOpts.Remaining.push_back(Arg);
          }
        }
      default:
        break;
    }
  }

  // Get Input list and any compiler specific flags we're interested in
  CompilerOpts.Parse(argc, argv, &Inputs);

  ParseStartupOpts(*this, Args);
  ParseLinkerOpts(*this, Args);
}

void InvocationOptions::PrintHelp() {
  std::unique_ptr<OptTable> Opts(CreateClingOptTable());

  Opts->PrintHelp(cling::outs(), "cling",
                  "cling: LLVM/clang C++ Interpreter: http://cern.ch/cling");

  cling::outs() << "\n\n";

  std::unique_ptr<OptTable> OptsC1(createDriverOptTable());
  OptsC1->PrintHelp(cling::outs(), "clang -cc1",
                    "LLVM 'Clang' Compiler: http://clang.llvm.org");
}
