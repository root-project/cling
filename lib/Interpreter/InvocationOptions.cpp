//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Axel Naumann <axel@cern.ch>
//------------------------------------------------------------------------------

#include "cling/Interpreter/InvocationOptions.h"
#include "cling/Interpreter/ClingOptions.h"

#include "clang/Driver/Options.h"

#include "llvm/ADT/OwningPtr.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/Option.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace clang::driver;
using namespace cling::driver::clingoptions;

using namespace llvm;
using namespace llvm::opt;

namespace {

#define PREFIX(NAME, VALUE) const char *const NAME[] = VALUE;
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, FLAGS, PARAM, \
               HELPTEXT, METAVAR)
#include "cling/Interpreter/ClingOptions.inc"
#undef OPTION
#undef PREFIX

  static const OptTable::Info ClingInfoTable[] = {
#define PREFIX(NAME, VALUE)
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, FLAGS, PARAM, \
               HELPTEXT, METAVAR)   \
  { PREFIX, NAME, HELPTEXT, METAVAR, OPT_##ID, Option::KIND##Class, PARAM, \
    FLAGS, OPT_##GROUP, OPT_##ALIAS },
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

}

cling::InvocationOptions
cling::InvocationOptions::CreateFromArgs(int argc, const char* const argv[],
                                         std::vector<unsigned>& leftoverArgs
                                         /* , Diagnostic &Diags */) {
  InvocationOptions ClingOpts;
  llvm::OwningPtr<OptTable> Opts(CreateClingOptTable());
  unsigned MissingArgIndex, MissingArgCount;
  llvm::OwningPtr<InputArgList> Args(
    Opts->ParseArgs(argv, argv + argc, MissingArgIndex, MissingArgCount));

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
  return ClingOpts;
}

void cling::InvocationOptions::PrintHelp() {
  llvm::OwningPtr<OptTable> Opts(CreateClingOptTable());

  // We need stream that doesn't close its file descriptor, thus we are not
  // using llvm::outs. Keeping file descriptor open we will be able to use
  // the results in pipes (Savannah #99234).
  Opts->PrintHelp(llvm::errs(), "cling",
                  "cling: LLVM/clang C++ Interpreter: http://cern.ch/cling");

  llvm::OwningPtr<OptTable> OptsC1(createDriverOptTable());
  OptsC1->PrintHelp(llvm::errs(), "clang -cc1",
                    "LLVM 'Clang' Compiler: http://clang.llvm.org");

}
