//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vvasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "cling/Interpreter/DynamicLibraryManager.h"
#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/Transaction.h"
#include "cling/Interpreter/Value.h"

#include "cling/MetaProcessor/Display.h"
#include "cling/MetaProcessor/MetaProcessor.h"
#include "cling/MetaProcessor/MetaSema.h"

#include "clang/AST/ASTContext.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Sema/Sema.h"
#include "clang/Serialization/ASTReader.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/Support/Casting.h"


#include "clang/Lex/Preprocessor.h"
#include "clang/Basic/SourceManager.h"

#include <cstdlib>
#include <iostream>

namespace cling {

  MetaSema::MetaSema(Interpreter& interp, MetaProcessor& meta)
    : m_Interpreter(interp), m_MetaProcessor(meta), m_IsQuitRequested(false) { }

  MetaSema::ActionResult MetaSema::actOnLCommand(llvm::StringRef file,
                                             Transaction** transaction /*= 0*/){
    ActionResult result = actOnUCommand(file);
    if (result != AR_Success)
      return result;

    // In case of libraries we get .L lib.so, which might automatically pull in
    // decls (from header files). Thus we want to take the restore point before
    // loading of the file and revert exclusively if needed.
    const Transaction* unloadPoint = m_Interpreter.getLastTransaction();
     // fprintf(stderr,"DEBUG: Load for %s unloadPoint is %p\n",file.str().c_str(),unloadPoint);
    // TODO: extra checks. Eg if the path is readable, if the file exists...
    std::string canFile = m_Interpreter.lookupFileOrLibrary(file);
    if (canFile.empty())
      canFile = file;
    if (m_Interpreter.loadFile(canFile, true /*allowSharedLib*/, transaction)
        == Interpreter::kSuccess) {
      registerUnloadPoint(unloadPoint, canFile);
      return AR_Success;
    }
    return AR_Failure;
  }

  MetaSema::ActionResult MetaSema::actOnOCommand(int optLevel) {
    if (optLevel >= 0 && optLevel < 4) {
      m_Interpreter.setDefaultOptLevel(optLevel);
      return AR_Success;
    }
    m_MetaProcessor.getOuts()
      << "Refusing to set invalid cling optimization level "
      << optLevel << '\n';
    return AR_Failure;
  }

  void MetaSema::actOnOCommand() {
    m_MetaProcessor.getOuts() << "Current cling optimization level: "
                              << m_Interpreter.getDefaultOptLevel() << '\n';
  }

  MetaSema::ActionResult MetaSema::actOnTCommand(llvm::StringRef inputFile,
                                                 llvm::StringRef outputFile) {
    m_Interpreter.GenerateAutoLoadingMap(inputFile, outputFile);
    return AR_Success;
  }

  MetaSema::ActionResult MetaSema::actOnRedirectCommand(llvm::StringRef file,
                         MetaProcessor::RedirectionScope stream,
                         bool append) {

    m_MetaProcessor.setStdStream(file, stream, append);
    return AR_Success;
  }

  void MetaSema::actOnComment(llvm::StringRef comment) const {
    // Some of the comments are meaningful for the cling::Interpreter
    m_Interpreter.declare(comment);
  }

  namespace {
    /// Replace non-identifier chars by '_'
    std::string normalizeDotXFuncName(const std::string& FuncName) {
      std::string ret = FuncName;
      // Prepend '_' if name starts with a digit.
      if (ret[0] >= '0' && ret[0] <= '9')
        ret.insert(ret.begin(), '_');
      for (char& c: ret) {
        // Instead of "escaping" all non-C++-id chars, only escape those that
        // are fairly certainly file names, to keep helpful error messages for
        // broken quoting or parsing. Example:
        // "Cannot find '_func_1___'" is much less helpful than
        // "Cannot find '/func(1)*&'"
        // I.e. find a compromise between helpful diagnostics and common file
        // name (stem) ingredients.
        if (c == '+' || c == '-' || c == '=' || c == '.' || c == ' '
            || c == '@')
          c = '_';
      }
      return ret;
    }
  }

  MetaSema::ActionResult MetaSema::actOnxCommand(llvm::StringRef file,
                                                 llvm::StringRef args,
                                                 Value* result) {

    // Check if there is a function named after the file.
    assert(!args.empty() && "Arguments must be provided (at least \"()\"");
    cling::Transaction* T = 0;
    MetaSema::ActionResult actionResult = actOnLCommand(file, &T);
    // T can be nullptr if there is no code (but comments)
    if (actionResult == AR_Success && T) {
      std::string expression;
      std::string FuncName = llvm::sys::path::stem(file);
      if (!FuncName.empty()) {
        FuncName = normalizeDotXFuncName(FuncName);
        if (T->containsNamedDecl(FuncName)) {
          expression = FuncName + args.str();
          // Give the user some context in case we have a problem invoking
          expression += " /* '.x' tries to invoke a function with the same name as the macro */";

          // Above transaction might have set a different OptLevel; use that.
          int prevOptLevel = m_Interpreter.getDefaultOptLevel();
          m_Interpreter.setDefaultOptLevel(T->getCompilationOpts().OptLevel);
          if (m_Interpreter.echo(expression, result) != Interpreter::kSuccess)
            actionResult = AR_Failure;
          m_Interpreter.setDefaultOptLevel(prevOptLevel);
        }
      } else
        FuncName = file; // Not great, but pass the diagnostics below something

      if (expression.empty()) {
        using namespace clang;
        static const char msg[] = "Failed to call `%0%1` to execute the macro.\n"
            "Add this function or rename the macro. Falling back to `.L`.";

        DiagnosticsEngine& Diags = m_Interpreter.getDiagnostics();
        unsigned diagID = Diags.getCustomDiagID(DiagnosticsEngine::Level::Warning, msg);
        //FIXME: Figure out how to pass in proper source locations, which we can
        // use with -verify.
        Diags.Report(SourceLocation(), diagID) << FuncName << args.str();
        return AR_Success;
      }
    }
    return actionResult;
  }

  void MetaSema::actOnqCommand() {
    m_IsQuitRequested = true;
  }

  void MetaSema::actOnAtCommand() {
    m_MetaProcessor.cancelContinuation();
  }

  MetaSema::ActionResult MetaSema::actOnUndoCommand(unsigned N/*=1*/) {
    m_Interpreter.unload(N);
    return AR_Success;
  }

  MetaSema::ActionResult MetaSema::actOnUCommand(llvm::StringRef file) {
    // FIXME: unload, once implemented, must return success / failure
    // Lookup the file
    clang::SourceManager& SM = m_Interpreter.getSema().getSourceManager();
    clang::FileManager& FM = SM.getFileManager();

    //Get the canonical path, taking into account interp and system search paths
    std::string canonicalFile = m_Interpreter.lookupFileOrLibrary(file);
    const clang::FileEntry* Entry
      = FM.getFile(canonicalFile, /*OpenFile*/false, /*CacheFailure*/false);
    if (Entry) {
      Watermarks::iterator Pos = m_Watermarks.find(Entry);
       //fprintf(stderr,"DEBUG: unload request for %s\n",file.str().c_str());

      if (Pos != m_Watermarks.end()) {
        const Transaction* unloadPoint = Pos->second;
        // Search for the transaction, i.e. verify that is has not already
        // been unloaded ; This can be removed once all transaction unload
        // properly information MetaSema that it has been unloaded.
        bool found = false;
        //for (auto t : m_Interpreter.m_IncrParser->getAllTransactions()) {
        for(const Transaction *t = m_Interpreter.getFirstTransaction();
            t != 0; t = t->getNext()) {
           //fprintf(stderr,"DEBUG: On unload check For %s unloadPoint is %p are t == %p\n",file.str().c_str(),unloadPoint, t);
          if (t == unloadPoint ) {
            found = true;
            break;
          }
        }
        if (!found) {
          m_MetaProcessor.getOuts() << "!!!ERROR: Transaction for file: " << file << " has already been unloaded\n";
        } else {
           //fprintf(stderr,"DEBUG: On Unload For %s unloadPoint is %p\n",file.str().c_str(),unloadPoint);
          while(m_Interpreter.getLastTransaction() != unloadPoint) {
             //fprintf(stderr,"DEBUG: unload transaction %p (searching for %p)\n",m_Interpreter.getLastTransaction(),unloadPoint);
            const clang::FileEntry* EntryUnloaded
              = m_ReverseWatermarks[m_Interpreter.getLastTransaction()];
            if (EntryUnloaded) {
              Watermarks::iterator PosUnloaded
                = m_Watermarks.find(EntryUnloaded);
              if (PosUnloaded != m_Watermarks.end()) {
                m_Watermarks.erase(PosUnloaded);
              }
            }
            m_Interpreter.unload(/*numberOfTransactions*/1);
          }
        }
        DynamicLibraryManager* DLM = m_Interpreter.getDynamicLibraryManager();
        if (DLM->isLibraryLoaded(canonicalFile))
          DLM->unloadLibrary(canonicalFile);
        m_Watermarks.erase(Pos);
      }
    }
    return AR_Success;
  }

  void MetaSema::actOnICommand(llvm::StringRef path) const {
    if (path.empty())
      m_Interpreter.DumpIncludePath();
    else
      m_Interpreter.AddIncludePath(path.str());
  }

  void MetaSema::actOnrawInputCommand(SwitchMode mode/* = kToggle*/) const {
    if (mode == kToggle) {
      bool flag = !m_Interpreter.isRawInputEnabled();
      m_Interpreter.enableRawInput(flag);
      // FIXME:
      m_MetaProcessor.getOuts() << (flag ? "U" :"Not u") << "sing raw input\n";
    }
    else
      m_Interpreter.enableRawInput(mode);
  }

  void MetaSema::actOndebugCommand(llvm::Optional<int> mode) const {
    clang::CodeGenOptions& CGO = m_Interpreter.getCI()->getCodeGenOpts();
    if (!mode) {
      bool flag = CGO.getDebugInfo() == clang::codegenoptions::NoDebugInfo;
      if (flag)
        CGO.setDebugInfo(clang::codegenoptions::LimitedDebugInfo);
      else
        CGO.setDebugInfo(clang::codegenoptions::NoDebugInfo);
      // FIXME:
      m_MetaProcessor.getOuts() << (flag ? "G" : "Not g")
                                << "enerating debug symbols\n";
    }
    else {
      static const int NumDebInfos = 5;
      clang::codegenoptions::DebugInfoKind DebInfos[NumDebInfos] = {
        clang::codegenoptions::NoDebugInfo,
        clang::codegenoptions::LocTrackingOnly,
        clang::codegenoptions::DebugLineTablesOnly,
        clang::codegenoptions::LimitedDebugInfo,
        clang::codegenoptions::FullDebugInfo
      };
      if (*mode >= NumDebInfos)
        mode = NumDebInfos - 1;
      else if (*mode < 0)
        mode = 0;
      CGO.setDebugInfo(DebInfos[*mode]);
      if (!*mode) {
        m_MetaProcessor.getOuts() << "Not generating debug symbols\n";
      } else {
        m_MetaProcessor.getOuts() << "Generating debug symbols level "
                                  << *mode << '\n';
      }
    }
  }

  void MetaSema::actOnprintDebugCommand(SwitchMode mode/* = kToggle*/) const {
    if (mode == kToggle) {
      bool flag = !m_Interpreter.isPrintingDebug();
      m_Interpreter.enablePrintDebug(flag);
      // FIXME:
      m_MetaProcessor.getOuts() << (flag ? "P" : "Not p") << "rinting Debug\n";
    }
    else
      m_Interpreter.enablePrintDebug(mode);
  }

  void MetaSema::actOnstoreStateCommand(llvm::StringRef name) const {
    m_Interpreter.storeInterpreterState(name);
  }

  void MetaSema::actOncompareStateCommand(llvm::StringRef name) const {
    m_Interpreter.compareInterpreterState(name);
  }

  void MetaSema::actOnstatsCommand(llvm::StringRef name,
                                   llvm::StringRef args) const {
    m_Interpreter.dump(name, args);
  }

  void MetaSema::actOndynamicExtensionsCommand(SwitchMode mode/* = kToggle*/)
    const {
    if (mode == kToggle) {
      bool flag = !m_Interpreter.isDynamicLookupEnabled();
      m_Interpreter.enableDynamicLookup(flag);
      // FIXME:
      m_MetaProcessor.getOuts()
        << (flag ? "U" : "Not u") << "sing dynamic extensions\n";
    }
    else
      m_Interpreter.enableDynamicLookup(mode);
  }

  void MetaSema::actOnhelpCommand() const {
    std::string& metaString = m_Interpreter.getOptions().MetaString;
    llvm::raw_ostream& outs = m_MetaProcessor.getOuts();
    outs << "\n Cling (C/C++ interpreter) meta commands usage\n"
      " All commands must be preceded by a '" << metaString << "', except\n"
      " for the evaluation statement { }\n"
      " ==============================================================================\n"
      " Syntax: " << metaString << "Command [arg0 arg1 ... argN]\n"
      "\n"
      "   " << metaString << "L <filename>\t\t- Load the given file or library\n\n"

      "   " << metaString << "(x|X) <filename>[args]\t- Same as .L and runs a function with"
                             "\n\t\t\t\t  signature: ret_type filename(args)\n"
      "\n"
      "   " << metaString << "> <filename>\t\t- Redirect command to a given file\n"
        "      '>' or '1>'\t\t- Redirects the stdout stream only\n"
        "      '2>'\t\t\t- Redirects the stderr stream only\n"
        "      '&>' (or '2>&1')\t\t- Redirects both stdout and stderr\n"
        "      '>>'\t\t\t- Appends to the given file\n"
      "\n"
      "   " << metaString << "undo [n]\t\t\t- Unloads the last 'n' inputs lines\n"
      "\n"
      "   " << metaString << "U <filename>\t\t- Unloads the given file\n"
      "\n"
      "   " << metaString << "I [path]\t\t\t- Shows the include path. If a path is given -"
                             "\n\t\t\t\t  adds the path to the include paths\n"
      "\n"
      "   " << metaString << "O <level>\t\t\t- Sets the optimization level (0-3)"
                             "\n\t\t\t\t  (not yet implemented)\n"
      "\n"
      "   " << metaString << "class <name>\t\t- Prints out class <name> in a CINT-like style\n"
      "\n"
      "   " << metaString << "files \t\t\t- Prints out some CINT-like file statistics\n"
      "\n"
      "   " << metaString << "fileEx \t\t\t- Prints out some file statistics\n"
      "\n"
      "   " << metaString << "g \t\t\t\t- Prints out information about global variable"
                             "\n\t\t\t\t  'name' - if no name is given, print them all\n"
      "\n"
      "   " << metaString << "@ \t\t\t\t- Cancels and ignores the multiline input\n"
      "\n"
      "   " << metaString << "rawInput [0|1]\t\t- Toggle wrapping and printing the"
                             "\n\t\t\t\t  execution results of the input\n"
      "\n"
      "   " << metaString << "dynamicExtensions [0|1]\t- Toggles the use of the dynamic scopes and the"
                             "\n\t\t\t\t  late binding\n"
      "\n"
      "   " << metaString << "printDebug [0|1]\t\t- Toggles the printing of input's corresponding"
                             "\n\t\t\t\t  state changes\n"
      "\n"
      "   " << metaString << "storeState <filename>\t- Store the interpreter's state to a given file\n"
      "\n"
      "   " << metaString << "compareState <filename>\t- Compare the interpreter's state with the one"
                             "\n\t\t\t\t  saved in a given file\n"
      "\n"
      "   " << metaString << "stats [name]\t\t- Show stats for internal data structures\n"
                             "\t\t\t\t  'ast'  abstract syntax tree stats\n"
                             "\t\t\t\t  'asttree [filter]'  abstract syntax tree layout\n"
                             "\t\t\t\t  'decl' dump ast declarations\n"
                             "\t\t\t\t  'undo' show undo stack\n"
      "\n"
      "   " << metaString << "help\t\t\t- Shows this information\n"
      "\n"
      "   " << metaString << "q\t\t\t\t- Exit the program\n"
      "\n";
  }

  void MetaSema::actOnfileExCommand() const {
    const clang::SourceManager& SM = m_Interpreter.getCI()->getSourceManager();
    SM.getFileManager().PrintStats();

    m_MetaProcessor.getOuts() << "\n***\n\n";

    for (clang::SourceManager::fileinfo_iterator I = SM.fileinfo_begin(),
           E = SM.fileinfo_end(); I != E; ++I) {
      m_MetaProcessor.getOuts() << (*I).first->getName();
      m_MetaProcessor.getOuts() << "\n";
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
        m_MetaProcessor.getOuts() << IFI->getPointer()->getName();
        m_MetaProcessor.getOuts() << "\n";
      }
    }
    */
  }

  void MetaSema::actOnfilesCommand() const {
    m_Interpreter.printIncludedFiles(m_MetaProcessor.getOuts());
  }

  void MetaSema::actOnclassCommand(llvm::StringRef className) const {
    if (!className.empty())
      DisplayClass(m_MetaProcessor.getOuts(),
                   &m_Interpreter, className.str().c_str(), true);
    else
      DisplayClasses(m_MetaProcessor.getOuts(), &m_Interpreter, false);
  }

  void MetaSema::actOnClassCommand() const {
    DisplayClasses(m_MetaProcessor.getOuts(), &m_Interpreter, true);
  }

  void MetaSema::actOnNamespaceCommand() const {
    DisplayNamespaces(m_MetaProcessor.getOuts(), &m_Interpreter);
  }

  void MetaSema::actOngCommand(llvm::StringRef varName) const {
    if (varName.empty())
      DisplayGlobals(m_MetaProcessor.getOuts(), &m_Interpreter);
    else
      DisplayGlobal(m_MetaProcessor.getOuts(),
                    &m_Interpreter, varName.str().c_str());
  }

  void MetaSema::actOnTypedefCommand(llvm::StringRef typedefName) const {
    if (typedefName.empty())
      DisplayTypedefs(m_MetaProcessor.getOuts(), &m_Interpreter);
    else
      DisplayTypedef(m_MetaProcessor.getOuts(),
                     &m_Interpreter, typedefName.str().c_str());
  }

  MetaSema::ActionResult
  MetaSema::actOnShellCommand(llvm::StringRef commandLine,
                              Value* result) const {
    llvm::StringRef trimmed(commandLine.trim(" \t\n\v\f\r "));
    if (!trimmed.empty()) {
      int ret = std::system(trimmed.str().c_str());

      // Build the result
      clang::ASTContext& Ctx = m_Interpreter.getCI()->getASTContext();
      if (result) {
        *result = Value(Ctx.IntTy, m_Interpreter);
        result->getAs<long long>() = ret;
      }

      return (ret == 0) ? AR_Success : AR_Failure;
    }
    if (result)
      *result = Value();
    // nothing to run - should this be success or failure?
    return AR_Failure;
  }

  void MetaSema::registerUnloadPoint(const Transaction* unloadPoint,
                                     llvm::StringRef filename) {
    std::string canFile = m_Interpreter.lookupFileOrLibrary(filename);
    if (canFile.empty())
      canFile = filename;
    clang::SourceManager& SM = m_Interpreter.getSema().getSourceManager();
    clang::FileManager& FM = SM.getFileManager();
    const clang::FileEntry* Entry
      = FM.getFile(canFile, /*OpenFile*/false, /*CacheFailure*/false);
    if (Entry && !m_Watermarks[Entry]) { // register as a watermark
      m_Watermarks[Entry] = unloadPoint;
      m_ReverseWatermarks[unloadPoint] = Entry;
      //fprintf(stderr,"DEBUG: Load for %s recorded unloadPoint %p\n",file.str().c_str(),unloadPoint);
    }
  }
} // end namespace cling
