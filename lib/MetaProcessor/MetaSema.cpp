//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vvasilev@cern.ch>
//------------------------------------------------------------------------------

#include "MetaSema.h"

#include "Display.h"

#include "cling/Interpreter/Interpreter.h"
#include "cling/MetaProcessor/MetaProcessor.h"

#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/CompilerInstance.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h" // for llvm::outs() FIXME

#include <cstdlib>

namespace cling {

  void MetaSema::actOnLCommand(llvm::sys::Path file) const {
    m_Interpreter.loadFile(file.str());
    // TODO: extra checks. Eg if the path is readable, if the file exists...
  }

  void MetaSema::actOnComment(llvm::StringRef comment) const {
    // Some of the comments are meaningful for the cling::Interpreter
    m_Interpreter.declare(comment);
  }

  void MetaSema::actOnxCommand(llvm::sys::Path file, llvm::StringRef args)
  {
    // Fall back to the meta processor for now.
    m_LastResultedValue = StoredValueRef::invalidValue();
    m_MetaProcessor.executeFile(file.str(), args.str(), &m_LastResultedValue);
    //m_Interpreter.loadFile(path.str());
    // TODO: extra checks. Eg if the path is readable, if the file exists...
  }

  void MetaSema::actOnqCommand() {
    m_IsQuitRequested = true;
  }

  void MetaSema::actOnUCommand() const {
     m_Interpreter.unload();
  }

  void MetaSema::actOnICommand(llvm::sys::Path path) const {
    if (path.isEmpty())
      m_Interpreter.DumpIncludePath();
    else
      m_Interpreter.AddIncludePath(path.str());
  }

  void MetaSema::actOnrawInputCommand(SwitchMode mode/* = kToggle*/) const {
    if (mode == kToggle) {
      bool flag = !m_Interpreter.isRawInputEnabled();
      m_Interpreter.enableRawInput(flag);
      // FIXME:
      llvm::outs() << (flag ? "U" :"Not u") << "sing raw input\n";
    }
    else
      m_Interpreter.enableRawInput(mode);
  }

  void MetaSema::actOnprintASTCommand(SwitchMode mode/* = kToggle*/) const {
    if (mode == kToggle) {
      bool flag = !m_Interpreter.isPrintingAST();
      m_Interpreter.enablePrintAST(flag);
      // FIXME:
      llvm::outs() << (flag ? "P" : "Not p") << "rinting AST\n";
    }
    else
      m_Interpreter.enablePrintAST(mode);
  }

  void MetaSema::actOndynamicExtensionsCommand(SwitchMode mode/* = kToggle*/) 
    const {
    if (mode == kToggle) {
      bool flag = !m_Interpreter.isDynamicLookupEnabled();
      m_Interpreter.enableDynamicLookup(flag);
      // FIXME:
      llvm::outs() << (flag ? "U" : "Not u") << "sing dynamic extensions\n";
    }
    else
      m_Interpreter.enableDynamicLookup(mode);
  }

  void MetaSema::actOnhelpCommand() const {
    std::string& metaString = m_Interpreter.getOptions().MetaString;
    llvm::outs() << "Cling meta commands usage\n";
    llvm::outs() << "Syntax: .Command [arg0 arg1 ... argN]\n";
    llvm::outs() << "\n";
    llvm::outs() << metaString << "q\t\t\t\t- Exit the program\n";
    llvm::outs() << metaString << "L <filename>\t\t\t - Load file or library\n";
    llvm::outs() << metaString << "(x|X) <filename>[args]\t\t- Same as .L and runs a ";
    llvm::outs() << "function with signature ";
    llvm::outs() << "\t\t\t\tret_type filename(args)\n";
    llvm::outs() << metaString << "I [path]\t\t\t- Shows the include path. If a path is ";
    llvm::outs() << "given - \n\t\t\t\tadds the path to the include paths\n";
    llvm::outs() << metaString << "@ \t\t\t\t- Cancels and ignores the multiline input\n";
    llvm::outs() << metaString << "rawInput [0|1]\t\t\t- Toggle wrapping and printing ";
    llvm::outs() << "the execution\n\t\t\t\tresults of the input\n";
    llvm::outs() << metaString << "dynamicExtensions [0|1]\t- Toggles the use of the ";
    llvm::outs() << "dynamic scopes and the \t\t\t\tlate binding\n";
    llvm::outs() << metaString << "printAST [0|1]\t\t\t- Toggles the printing of input's ";
    llvm::outs() << "corresponding \t\t\t\tAST nodes\n";
    llvm::outs() << metaString << "help\t\t\t\t- Shows this information\n";
  }

  void MetaSema::actOnfileExCommand() const {
    const clang::SourceManager& SM = m_Interpreter.getCI()->getSourceManager();
    SM.getFileManager().PrintStats();

    llvm::outs() << "\n***\n\n";

    for (clang::SourceManager::fileinfo_iterator I = SM.fileinfo_begin(),
           E = SM.fileinfo_end(); I != E; ++I) {
      llvm::outs() << (*I).first->getName();
      llvm::outs() << "\n";
    }

  }

  void MetaSema::actOnfilesCommand() const {
    typedef llvm::SmallVectorImpl<Interpreter::LoadedFileInfo*> LoadedFiles_t;
    const LoadedFiles_t& LoadedFiles = m_Interpreter.getLoadedFiles();
    for (LoadedFiles_t::const_iterator I = LoadedFiles.begin(),
           E = LoadedFiles.end(); I != E; ++I) {
      char cType[] = { 'S', 'D', 'B' };
      llvm::outs() << '[' << cType[(*I)->getType()] << "] " 
                   << (*I)->getName() << '\n';
    }
  }

  void MetaSema::actOnclassCommand(llvm::StringRef className) const {
    if (!className.empty()) 
      DisplayClass(llvm::outs(), &m_Interpreter, className.str().c_str(), true);
    else
      DisplayClasses(llvm::outs(), &m_Interpreter, false);
  }

  void MetaSema::actOnClassCommand() const {
    DisplayClasses(llvm::outs(), &m_Interpreter, true);
  }

  void MetaSema::actOngCommand(llvm::StringRef varName) const {
    if (varName.empty())
      DisplayGlobals(llvm::outs(), &m_Interpreter);
    else
      DisplayGlobal(llvm::outs(), &m_Interpreter, varName.str().c_str());
  }

  void MetaSema::actOnTypedefCommand(llvm::StringRef typedefName) const {
    if (typedefName.empty())
      DisplayTypedefs(llvm::outs(), &m_Interpreter);
    else
      DisplayTypedef(llvm::outs(), &m_Interpreter, typedefName.str().c_str());
  }
  
  void MetaSema::actOnShellCommand(llvm::StringRef commandLine) const {
    llvm::StringRef trimmed(commandLine.trim(" \t\n\v\f\r "));
    if (!trimmed.empty())
       (void)std::system(trimmed.str().c_str());
  }

} // end namespace cling
