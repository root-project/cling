//===--- ClingOptions.td - Options for cling -----------------------------------===//
//
// CLING - the C++ LLVM-based InterpreterG :)
//
//===----------------------------------------------------------------------===//
//
//  This file defines the options accepted by cling.
//
//===----------------------------------------------------------------------===//

// Include the common option parsing interfaces.
include "llvm/Option/OptParser.td"

#ifndef NDEBUG
def _debugFlags_EQ : Joined<["--"], "debug-only=">;
def _debugFlags : Flag<["--"], "debug-only">;
#endif
def _errorout : Flag<["--"], "errorout">, HelpText<"Do not recover from input errors">;
// Re-implement to forward to our help
def help : Flag<["-", "--"], "help">, HelpText<"Print this help text">;
def L : JoinedOrSeparate<["-"], "L">, HelpText<"Add directory to library search path">, MetaVarName<"<directory>">;
def l : JoinedOrSeparate<["-"], "l">, HelpText<"Load a library before prompt">, MetaVarName<"<library>">;
def _metastr_EQ : Joined<["--"], "metastr=">, HelpText<"Set the meta command tag, default '.'">;
def _metastr : Separate<["--"], "metastr">, HelpText<"Set the meta command tag, default '.'">;
def _nologo : Flag<["--"], "nologo">, HelpText<"Do not show startup-banner">;
def noruntime : Flag<["-", "--"], "noruntime">, HelpText<"Disable runtime support (no null checking, no value printing)">;
def _ptrcheck : Flag<["--"], "ptrcheck">, HelpText<"Enable injection of pointer validity checks">;
def version : Flag<["-", "--"], "version">, HelpText<"Print the compiler version">;
def v : Flag<["-"], "v">, HelpText<"Enable verbose output">;
