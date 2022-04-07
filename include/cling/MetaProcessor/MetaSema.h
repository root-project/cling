//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_META_SEMA_H
#define CLING_META_SEMA_H

#include "cling/MetaProcessor/MetaProcessor.h"
#include "cling/Interpreter/Transaction.h"

#include "clang/Basic/FileManager.h" // for DenseMap<FileEntry*>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"

namespace llvm {
  class StringRef;
  class raw_ostream;
}

namespace cling {
  class Interpreter;
  class MetaProcessor;
  class Value;

  ///\brief Semantic analysis for our home-grown language. All implementation
  /// details of the commands should go here.
  class MetaSema {
  private:
    Interpreter& m_Interpreter;
    MetaProcessor& m_MetaProcessor;
    bool m_IsQuitRequested;

    llvm::DenseMap<const clang::FileEntry*, const Transaction*> m_FEToTransaction;
    llvm::DenseMap<const Transaction*, const clang::FileEntry*> m_TransactionToFE;
  public:
    enum SwitchMode {
      kOff = 0,
      kOn = 1,
      kToggle = 2
    };
    enum ActionResult {
      AR_Failure = 0,
      AR_Success = 1
    };

    MetaSema(Interpreter& interp, MetaProcessor& meta);

    const Interpreter& getInterpreter() const { return m_Interpreter; }
    bool isQuitRequested() const { return m_IsQuitRequested; }

    ///\brief L command includes the given file or loads the given library. If
    /// \c file is empty print the list of library paths.
    ///
    ///\param[in] file - The file/library to be loaded.
    ///\param[out] transaction - Transaction containing the loaded file.
    ///
    ActionResult actOnLCommand(llvm::StringRef file,
                               Transaction** transaction = 0);

    ///\brief O command sets the optimization level.
    ///
    ///\param[in] optLevel - The optimization level to set.
    ///
    ActionResult actOnOCommand(int optLevel);

    ///\brief O command prints the current optimization level.
    ///
    void actOnOCommand();

    ///\brief T command prepares the tag files for giving semantic hints.
    ///
    ///\param[in] inputFile - The source file of the map.
    ///\param[in] outputFile - The forward declaration file.
    ///
    ActionResult actOnTCommand(llvm::StringRef inputFile,
                               llvm::StringRef outputFile);

    ///\brief < Redirect command.
    ///
    ///\param[in] file - The file where the output is redirected
    ///\param[in] stream - The optional stream to redirect.
    ///\param[in] append - Write or append to the file.
    ///
    ActionResult actOnRedirectCommand(llvm::StringRef file,
                                      MetaProcessor::RedirectionScope stream,
                                      bool append);

    ///\brief Actions that need to be performed on occurance of a comment.
    ///
    /// That is useful when the comments are meaningful for the interpreter. For
    /// example when we run in -verify mode.
    ///
    ///\param[in] comment - The comment to act on.
    ///
    void actOnComment(llvm::StringRef comment) const;

    ///\brief Actions to be performed on a given file. Loads the given file and
    /// calls a function with the name of the file.
    ///
    /// If the function needs arguments they are specified after the filename in
    /// parenthesis.
    ///
    ///\param[in] file - The filename to load.
    ///\param[in] args - The optional list of arguments.
    ///\param[out] result - If not NULL, will hold the value of the last
    ///                     expression.
    ///
    ActionResult actOnxCommand(llvm::StringRef file, llvm::StringRef args,
                               Value* result);

    ///\brief Actions to be performed on quit.
    ///
    void actOnqCommand();

    ///\brief Actions to be performed on request to cancel continuation.
    ///
    void actOnAtCommand();

    ///\brief Unloads the last N inputs lines.
    ///
    ///\param[in] N - The inputs to unload.
    ///
    ActionResult actOnUndoCommand(unsigned N = 1);

    ///\brief Actions to be performed on unload command.
    ///
    ///\param[in] file - The file to unload.
    ///
    ActionResult actOnUCommand(llvm::StringRef file);

    ///\brief Actions to be performed on add include path. It registers new
    /// folder where header files can be searched. If \c path is empty print the
    /// list of include paths.
    ///
    ///\param[in] path - The path to add to header search.
    ///
    void actOnICommand(llvm::StringRef path) const;

    ///\brief Changes the input mode to raw input. In that mode we act more like
    /// a compiler by bypassing many of cling's features.
    ///
    ///\param[in] mode - either on/off or toggle.
    ///
    void actOnrawInputCommand(SwitchMode mode = kToggle) const;

    ///\brief Generates debug info for the JIT.
    ///
    ///\param[in] mode - either on/off or toggle.
    ///
    void actOndebugCommand(llvm::Optional<int> mode) const;

    ///\brief Prints out the the Debug information of the state changes.
    ///
    ///\param[in] mode - either on/off or toggle.
    ///
    void actOnprintDebugCommand(SwitchMode mode = kToggle) const;

    ///\brief Store the interpreter's state.
    ///
    ///\param[in] name - Name of the files where the state will be stored
    ///
    void actOnstoreStateCommand(llvm::StringRef name) const;

    ///\brief Compare the interpreter's state with the one previously stored
    ///
    ///\param[in] name - Name of the files where the previous state was stored
    ///
    void actOncompareStateCommand(llvm::StringRef name) const;

    ///\brief Show stats for various internal data structures.
    ///
    ///\param[in] name - Name of the structure.
    ///\param[in] filter - Optional predicate for filtering displayed stats.
    ///
    void actOnstatsCommand(llvm::StringRef name,
                           llvm::StringRef filter = llvm::StringRef()) const;

    ///\brief Switches on/off the experimental dynamic extensions (dynamic
    /// scopes) and late binding.
    ///
    ///\param[in] mode - either on/off or toggle.
    ///
    void actOndynamicExtensionsCommand(SwitchMode mode = kToggle) const;

    ///\brief Prints out the help message with the description of the meta
    /// commands.
    ///
    void actOnhelpCommand() const;

    ///\brief Prints out some file statistics.
    ///
    void actOnfileExCommand() const;

    ///\brief Prints out some CINT-like file statistics.
    ///
    void actOnfilesCommand() const;

    ///\brief Prints out class CINT-like style.
    ///
    ///\param[in] className - the specific class to be printed.
    ///\param[in] verbose - set to true for more verbosity
    ///\note if empty class name, all classes are printed
    ///
    void actOnClassCommand(llvm::StringRef className, bool verbose) const;

    ///\brief Prints out namespace names.
    ///
    void actOnNamespaceCommand() const;

    ///\brief Prints out information about global variables.
    ///
    ///\param[in] varName - The name of the global variable
    //                      if empty prints them all.
    ///
    void actOngCommand(llvm::StringRef varName) const;

    ///\brief Prints out information about typedefs.
    ///
    ///\param[in] typedefName - The name of typedef if empty prints them all.
    ///
    void actOnTypedefCommand(llvm::StringRef typedefName) const;

    ///\brief '.! cmd [args]' syntax.
    ///
    ///\param[in] commandLine - shell command + optional list of parameters.
    ///\param[out] result - if not NULL will hold shell exit code at return.
    ///
    ActionResult actOnShellCommand(llvm::StringRef commandLine,
                                   Value* result) const;

    ///\brief Register the file as an upload point for the current
    ///  Transaction: when unloading that file, all transactions after
    ///  the current one will be reverted.
    ///
    ///\param [in] T - The unload point - any later Transaction will be
    /// reverted when filename is unloaded.
    ///\param [in] filename - The name of the file to be used as unload
    ///  point.
    void registerUnloadPoint(const Transaction* T, llvm::StringRef filename);

  };

} // end namespace cling

#endif // CLING_META_PARSER_H
