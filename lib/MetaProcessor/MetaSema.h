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

    void actOnLCommand(llvm::sys::Path path) const;
    void actOnComment(llvm::StringRef comment) const;
    void actOnxCommand(llvm::sys::Path path, llvm::StringRef args) const;
    void actOnqCommand() const;
    void actOnUCommand() const;
    void actOnICommand(llvm::sys::Path path) const;
    void actOnrawInputCommand(SwitchMode mode = kToggle) const;
    void actOnprintASTCommand(SwitchMode mode = kToggle) const;
    void actOndynamicExtensionsCommand(SwitchMode mode = kToggle) const;
    void actOnhelpCommand() const;
    void actOnfileExCommand() const;
    void actOnfilesCommand() const;
    void actOnclassCommand(llvm::StringRef className) const;
    void actOnClassCommand() const;
  };

} // end namespace cling

#endif // CLING_META_PARSER_H
