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
    void ActOnLCommand(llvm::sys::Path path) const;
    void ActOnComment(llvm::StringRef comment) const;
    void ActOnxCommand(llvm::sys::Path path, llvm::StringRef args) const;
    void ActOnqCommand() const;
    void ActOnUCommand() const;
    void ActOnICommand(llvm::sys::Path path) const;
    void ActOnrawInputCommand(SwitchMode mode = kToggle) const;
    void ActOnprintASTCommand(SwitchMode mode = kToggle) const;
    void ActOndynamicExtensionsCommand(SwitchMode mode = kToggle) const;
    void ActOnhelpCommand() const;
    void ActOnfileExCommand() const;
    void ActOnfilesCommand() const;
    void ActOnclassCommand(llvm::StringRef className) const;
    void ActOnClassCommand() const;
  };

} // end namespace cling

#endif // CLING_META_PARSER_H
