//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vvasilev@cern.ch>
//------------------------------------------------------------------------------

#include "MetaSema.h"

#include "Display.h"

#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/StoredValueRef.h"
#include "cling/MetaProcessor/MetaProcessor.h"

#include "clang/AST/ASTContext.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Serialization/ASTReader.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/GenericValue.h"

#include "clang/Lex/Preprocessor.h"
#include "clang/Basic/SourceManager.h"

#include <cstdlib>

namespace cling {

  MetaSema::MetaSema(Interpreter& interp, MetaProcessor& meta) 
    : m_Interpreter(interp), m_MetaProcessor(meta), m_IsQuitRequested(false),
      m_Outs(m_MetaProcessor.getOuts()){ }

  MetaSema::ActionResult MetaSema::actOnLCommand(llvm::StringRef file) const {
    // TODO: extra checks. Eg if the path is readable, if the file exists...
    if (m_Interpreter.loadFile(file.str()) == Interpreter::kSuccess)
      return AR_Success;
    return AR_Failure;
  }

  void MetaSema::actOnComment(llvm::StringRef comment) const {
    // Some of the comments are meaningful for the cling::Interpreter
    m_Interpreter.declare(comment);
  }

  MetaSema::ActionResult MetaSema::actOnxCommand(llvm::StringRef file, 
                                                 llvm::StringRef args, 
                                                 StoredValueRef* result) {
    // Fall back to the meta processor for now.
    Interpreter::CompilationResult compRes = Interpreter::kFailure;
    m_MetaProcessor.executeFile(file.str(), args.str(), compRes, result);
    ActionResult actionResult = AR_Failure;
    if (compRes == Interpreter::kSuccess)
       actionResult = AR_Success;
    return actionResult;

    //m_Interpreter.loadFile(path.str());
    // TODO: extra checks. Eg if the path is readable, if the file exists...
  }

  void MetaSema::actOnqCommand() {
    m_IsQuitRequested = true;
  }

  MetaSema::ActionResult MetaSema::actOnUCommand() const {
     // FIXME: unload, once implemented, must return success / failure
     m_Interpreter.unload();
     return AR_Success;
  }

  void MetaSema::actOnICommand(llvm::StringRef path) const {
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
      m_Outs << (flag ? "U" :"Not u") << "sing raw input\n";
    }
    else
      m_Interpreter.enableRawInput(mode);
  }

  void MetaSema::actOnprintASTCommand(SwitchMode mode/* = kToggle*/) const {
    if (mode == kToggle) {
      bool flag = !m_Interpreter.isPrintingAST();
      m_Interpreter.enablePrintAST(flag);
      // FIXME:
      m_Outs << (flag ? "P" : "Not p") << "rinting AST\n";
    }
    else
      m_Interpreter.enablePrintAST(mode);
  }

  void MetaSema::actOnprintIRCommand(SwitchMode mode/* = kToggle*/) const {
    if (mode == kToggle) {
      bool flag = !m_Interpreter.isPrintingIR();
      m_Interpreter.enablePrintIR(flag);
      // FIXME:
      m_Outs << (flag ? "P" : "Not p") << "rinting IR\n";
    }
    else
      m_Interpreter.enablePrintIR(mode);
  }

  void MetaSema::actOnstoreStateCommand(llvm::StringRef name) const {
    m_Interpreter.storeInterpreterState(name);
  }

  void MetaSema::actOncompareStateCommand(llvm::StringRef name) const {
    m_Interpreter.compareInterpreterState(name);
  }

  void MetaSema::actOndynamicExtensionsCommand(SwitchMode mode/* = kToggle*/) 
    const {
    if (mode == kToggle) {
      bool flag = !m_Interpreter.isDynamicLookupEnabled();
      m_Interpreter.enableDynamicLookup(flag);
      // FIXME:
      m_Outs << (flag ? "U" : "Not u") << "sing dynamic extensions\n";
    }
    else
      m_Interpreter.enableDynamicLookup(mode);
  }

  void MetaSema::actOnhelpCommand() const {
    std::string& metaString = m_Interpreter.getOptions().MetaString;
    m_Outs << "Cling meta commands usage\n";
    m_Outs << "Syntax: .Command [arg0 arg1 ... argN]\n";
    m_Outs << "\n";
    m_Outs << metaString << "q\t\t\t\t- Exit the program\n";
    m_Outs << metaString << "L <filename>\t\t\t - Load file or library\n";
    m_Outs << metaString << "(x|X) <filename>[args]\t\t- Same as .L and runs a ";
    m_Outs << "function with signature ";
    m_Outs << "\t\t\t\tret_type filename(args)\n";
    m_Outs << metaString << "I [path]\t\t\t- Shows the include path. If a path is ";
    m_Outs << "given - \n\t\t\t\tadds the path to the include paths\n";
    m_Outs << metaString << "@ \t\t\t\t- Cancels and ignores the multiline input\n";
    m_Outs << metaString << "rawInput [0|1]\t\t\t- Toggle wrapping and printing ";
    m_Outs << "the execution\n\t\t\t\tresults of the input\n";
    m_Outs << metaString << "dynamicExtensions [0|1]\t- Toggles the use of the ";
    m_Outs << "dynamic scopes and the \t\t\t\tlate binding\n";
    m_Outs << metaString << "printAST [0|1]\t\t\t- Toggles the printing of input's ";
    m_Outs << "corresponding \t\t\t\tAST nodes\n";
    m_Outs << metaString << "help\t\t\t\t- Shows this information\n";
  }

  void MetaSema::actOnfileExCommand() const {
    const clang::SourceManager& SM = m_Interpreter.getCI()->getSourceManager();
    SM.getFileManager().PrintStats();

    m_Outs << "\n***\n\n";

    for (clang::SourceManager::fileinfo_iterator I = SM.fileinfo_begin(),
           E = SM.fileinfo_end(); I != E; ++I) {
      m_Outs << (*I).first->getName();
      m_Outs << "\n";
    }
    /* Only available in clang's trunk:
    clang::ASTReader* Reader = m_Interpreter.getCI()->getModuleManager();
    const clang::serialization::ModuleManager& ModMan
      = Reader->getModuleManager();
    for (clang::serialization::ModuleManager::ModuleConstIterator I
           = ModMan.begin(), E = ModMan.end(); I != E; ++I) {
      typedef
        std::vector<llvm::PointerIntPair<const clang::FileEntry*, 1, bool> >
        InputFiles_t;
      const InputFiles_t& InputFiles = (*I)->InputFilesLoaded;
      for (InputFiles_t::const_iterator IFI = InputFiles.begin(),
             IFE = InputFiles.end(); IFI != IFE; ++IFI) {
        m_Outs << IFI->getPointer()->getName();
        m_Outs << "\n";
      }
    }
    */
  }

  void MetaSema::actOnfilesCommand() const { 
    m_Interpreter.printIncludedFiles(m_Outs);
  }

  void MetaSema::actOnclassCommand(llvm::StringRef className) const {
    if (!className.empty()) 
      DisplayClass(m_Outs, &m_Interpreter, className.str().c_str(), true);
    else
      DisplayClasses(m_Outs, &m_Interpreter, false);
  }

  void MetaSema::actOnClassCommand() const {
    DisplayClasses(m_Outs, &m_Interpreter, true);
  }

  void MetaSema::actOngCommand(llvm::StringRef varName) const {
    if (varName.empty())
      DisplayGlobals(m_Outs, &m_Interpreter);
    else
      DisplayGlobal(m_Outs, &m_Interpreter, varName.str().c_str());
  }

  void MetaSema::actOnTypedefCommand(llvm::StringRef typedefName) const {
    if (typedefName.empty())
      DisplayTypedefs(m_Outs, &m_Interpreter);
    else
      DisplayTypedef(m_Outs, &m_Interpreter, typedefName.str().c_str());
  }
  
  MetaSema::ActionResult
  MetaSema::actOnShellCommand(llvm::StringRef commandLine,
                              StoredValueRef* result) const {
    llvm::StringRef trimmed(commandLine.trim(" \t\n\v\f\r "));
    if (!trimmed.empty()) {
      int ret = std::system(trimmed.str().c_str());

      // Build the result
      clang::ASTContext& Ctx = m_Interpreter.getCI()->getASTContext();
      llvm::GenericValue retGV;
      retGV.IntVal = llvm::APInt(sizeof(int) * 8, ret, true /*isSigned*/);
      Value V(retGV, Ctx.IntTy);
      if (result)
        *result = StoredValueRef::bitwiseCopy(Ctx, V);

      return (ret == 0) ? AR_Success : AR_Failure;
    }
    if (result)
      *result = StoredValueRef();
    // nothing to run - should this be success or failure?
    return AR_Failure;
  }

} // end namespace cling
