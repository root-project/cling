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

#include "clang/Driver/Options.h"

#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/Option.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>

using namespace clang;
using namespace cling::driver::clingoptions;

using namespace llvm;
using namespace llvm::opt;

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
      : OptTable(ClingInfoTable,
                 sizeof(ClingInfoTable) / sizeof(ClingInfoTable[0])) {}
  };

  static OptTable* CreateClingOptTable() {
    return new ClingOptTable();
  }

  static void ParseStartupOpts(cling::InvocationOptions& Opts,
                               InputArgList& Args /* , Diags */) {
    Opts.ErrorOut = Args.hasArg(OPT__errorout);
    Opts.NoLogo = Args.hasArg(OPT__nologo);
    Opts.ShowVersion = Args.hasArg(OPT_version);
    Opts.Verbose = Args.hasArg(OPT_v);
    Opts.Help = Args.hasArg(OPT_help);
    if (Args.hasArg(OPT__metastr, OPT__metastr_EQ)) {
      Arg* MetaStringArg = Args.getLastArg(OPT__metastr, OPT__metastr_EQ);
      Opts.MetaString = MetaStringArg->getValue();
      if (Opts.MetaString.length() == 0) {
        llvm::errs() << "ERROR: meta string must be non-empty! Defaulting to '.'.\n";
        Opts.MetaString = ".";
      }
    }
  }

  static void ParseLinkerOpts(cling::InvocationOptions& Opts,
                              InputArgList& Args /* , Diags */) {
    Opts.LibsToLoad = Args.getAllArgValues(OPT_l);
    std::vector<std::string> LibPaths = Args.getAllArgValues(OPT_L);
    for (size_t i = 0; i < LibPaths.size(); ++i)
      Opts.LibSearchPath.push_back(LibPaths[i]);
  }

  static void ParseInputs(cling::InvocationOptions& Opts,
                          int argc, const char* const argv[]) {
    if (argc <= 1) { return; }
    unsigned MissingArgIndex, MissingArgCount;
    std::unique_ptr<OptTable> OptsC1(clang::driver::createDriverOptTable());
    Opts.Inputs.clear();
    // see Driver::getIncludeExcludeOptionFlagMasks()
    unsigned ExcludeOptionFlagMasks
      = clang::driver::options::NoDriverOption | clang::driver::options::CLOption;
    std::unique_ptr<InputArgList> Args(
        OptsC1->ParseArgs(argv+1, argv + argc, MissingArgIndex, MissingArgCount,
                          0, ExcludeOptionFlagMasks));
    for (ArgList::const_iterator it = Args->begin(),
           ie = Args->end(); it != ie; ++it) {
      if ( (*it)->getOption().getKind() == Option::InputClass ) {
          Opts.Inputs.push_back(std::string( (*it)->getValue() ) );
      }
    }
  }
}

cling::InvocationOptions
cling::InvocationOptions::CreateFromArgs(int argc, const char* const argv[],
                                         std::vector<unsigned>& leftoverArgs
                                         /* , Diagnostic &Diags */) {
  InvocationOptions ClingOpts;
  std::unique_ptr<OptTable> Opts(CreateClingOptTable());
  unsigned MissingArgIndex, MissingArgCount;
  // see Driver::getIncludeExcludeOptionFlagMasks()
  unsigned ExcludeOptionFlagMasks
    = clang::driver::options::NoDriverOption | clang::driver::options::CLOption;
  std::unique_ptr<InputArgList> Args(
    Opts->ParseArgs(argv, argv + argc, MissingArgIndex, MissingArgCount,
                    0, ExcludeOptionFlagMasks));

  //if (MissingArgCount)
  //  Diags.Report(diag::err_drv_missing_argument)
  //    << Args->getArgString(MissingArgIndex) << MissingArgCount;

  // Forward unknown arguments.
  for (ArgList::const_iterator it = Args->begin(),
         ie = Args->end(); it != ie; ++it) {
    if ((*it)->getOption().getKind() == Option::UnknownClass
        ||(*it)->getOption().getKind() == Option::InputClass) {
      leftoverArgs.push_back((*it)->getIndex());
    }
  }
  ParseStartupOpts(ClingOpts, *Args /* , Diags */);
  ParseLinkerOpts(ClingOpts, *Args /* , Diags */);
  ParseInputs(ClingOpts, argc, argv);
  return ClingOpts;
}

void cling::InvocationOptions::PrintHelp() {
  std::unique_ptr<OptTable> Opts(CreateClingOptTable());

  Opts->PrintHelp(llvm::outs(), "cling",
                  "cling: LLVM/clang C++ Interpreter: http://cern.ch/cling");

  llvm::outs() << "\n\n";

  std::unique_ptr<OptTable> OptsC1(clang::driver::createDriverOptTable());
  OptsC1->PrintHelp(llvm::outs(), "clang -cc1",
                    "LLVM 'Clang' Compiler: http://clang.llvm.org");

}
