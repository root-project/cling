//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_META_SEMA_H
#define CLING_META_SEMA_H

namespace llvm {
  class StringRef;
  namespace sys {
    class Path;
  }
}

namespace cling {
  class Interpreter;
  class MetaProcessor;

  ///\brief Semantic analysis for our home-grown language. All implementation 
  /// details of the commands should go here.
  class MetaSema {
  private:
    Interpreter& m_Interpreter;
    MetaProcessor& m_MetaProcessor;
  public:
    enum SwitchMode {
      kOff = 0,
      kOn = 1,
      kToggle = 2
    };
  public:
    MetaSema(Interpreter& interp, MetaProcessor& meta) 
      : m_Interpreter(interp), m_MetaProcessor(meta) {}
    const Interpreter& getInterpreter() { return m_Interpreter; }
    
    ///\brief L command includes the given file or loads the given library.
    ///
    ///\param[in] file - The file/library to be loaded.
    ///
    void actOnLCommand(llvm::sys::Path file) const;

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
    ///
    void actOnxCommand(llvm::sys::Path file, llvm::StringRef args) const;

    ///\brief Actions to be performed on quit.
    ///
    void actOnqCommand() const;

    ///\brief Actions to be performed on unload command. For now it tries to 
    /// unload the last transaction.
    ///
    void actOnUCommand() const;

    ///\brief Actions to be performed on add include path. It registers new 
    /// folder where header files can be searched.
    ///
    ///\param[in] path - The path to add to header search.
    ///
    void actOnICommand(llvm::sys::Path path) const;

    ///\brief Changes the input mode to raw input. In that mode we act more like
    /// a compiler by bypassing many of cling's features.
    ///
    ///\param[in] mode - either on/off or toggle.
    ///
    void actOnrawInputCommand(SwitchMode mode = kToggle) const;

    ///\brief Prints out the the AST representation of the input.
    ///
    ///\param[in] mode - either on/off or toggle.
    ///
    void actOnprintASTCommand(SwitchMode mode = kToggle) const;

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
    ///
    void actOnclassCommand(llvm::StringRef className) const;

    ///\brief Prints out class CINT-like style more verbosely.
    ///
    void actOnClassCommand() const;

    ///\brief Prints out information about global variables.
    ///
    ///\param[in] varName - The name of the global variable 
    //                      if empty prints them all.
    ///
    void actOngCommand(llvm::StringRef varName) const;
  };

} // end namespace cling

#endif // CLING_META_PARSER_H
