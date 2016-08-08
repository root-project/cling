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

#define PREFIX(NAME, VALUE) const char *const NAME[] = VALUE;
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM, \
               HELPTEXT, METAVAR)
#include "cling/Interpreter/ClingOptions.inc"
#undef OPTION
#undef PREFIX

  static const OptTable::Info ClingInfoTable[] = {
#define PREFIX(NAME, VALUE)
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM, \
               HELPTEXT, METAVAR)   \
  { PREFIX, NAME, HELPTEXT, METAVAR, OPT_##ID, Option::KIND##Class, PARAM, \
    FLAGS, OPT_##GROUP, OPT_##ALIAS, ALIASARGS },
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

CompilerOptions::CompilerOptions(int argc, const char* const* argv) :
  Language(false), ResourceDir(false), SysRoot(false), NoBuiltinInc(false),
  NoCXXInc(false), StdVersion(false), StdLib(false), HasOutput(false),
  Verbose(false) {
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
      case options::OPT_x: Language = true; break;
      case options::OPT_resource_dir: ResourceDir = true; break;
      case options::OPT_isysroot: SysRoot = true; break;
      case options::OPT_std_EQ: StdVersion = true; break;
      case options::OPT_stdlib_EQ: StdLib = true; break;
      // case options::OPT_nostdlib:
      case options::OPT_nobuiltininc: NoBuiltinInc = true; break;
      // case options::OPT_nostdinc:
      case options::OPT_nostdincxx: NoCXXInc = true; break;
      case options::OPT_v: Verbose = true; break;

      default:
        if (Inputs && arg->getOption().getKind() == Option::InputClass)
          Inputs->push_back(arg->getValue());
        break;
    }
  }
}

InvocationOptions::InvocationOptions(int argc, const char* const* argv) :
  MetaString("."), ErrorOut(false), NoLogo(false), ShowVersion(false),
  Help(false) {

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
      case Option::UnknownClass:
      case Option::InputClass:
        // prune "-" we need to control where it appears when invoking clang
        if (!arg->getSpelling().equals("-"))
          CompilerOpts.Remaining.push_back(argv[arg->getIndex()]);
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
