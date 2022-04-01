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
#include <cctype>
#include <algorithm>

namespace {
  ///\brief Make a valid C++ identifier, replacing illegal characters in `S'
  /// by '_'. It does not take into account valid Unicode ranges.
  ///\param[in] S - std::string& that (may) contain chars that cannot be part
  ///               of an identifier
  ///
  static std::string makeValidCXXIdentifier(const std::string& S) {
    std::string ret(S);
    if (std::isdigit(ret[0]))
      ret.insert(ret.begin(), '_');
    std::replace_if(ret.begin(), ret.end(),
                    [] (char c) { return (c != '_')
                                    && !std::isdigit(c) && !std::isalpha(c); },
                    '_');
    return ret;
  }
}

namespace cling {
  MetaSema::MetaSema(Interpreter& interp, MetaProcessor& meta)
    : m_Interpreter(interp), m_MetaProcessor(meta), m_IsQuitRequested(false) { }

  MetaSema::ActionResult MetaSema::actOnLCommand(llvm::StringRef file,
                                            Transaction** transaction /*= 0*/) {
    if (file.empty()) {
      m_Interpreter.DumpDynamicLibraryInfo();
      return AR_Success;
    }

    if (actOnUCommand(file) != AR_Success)
      return AR_Failure;

    // In case of libraries we get .L lib.so, which might automatically pull in
    // decls (from header files). Thus we want to take the restore point before
    // loading of the file and revert exclusively if needed.
    const Transaction* unloadPoint = m_Interpreter.getLastTransaction();
    // fprintf(stderr,"DEBUG: Load for %s unloadPoint is %p\n", file.str().c_str(), unloadPoint);

    std::string pathname(m_Interpreter.lookupFileOrLibrary(file));
    if (pathname.empty())
      pathname = file;
    if (m_Interpreter.loadFile(pathname, /*allowSharedLib=*/true, transaction)
        == Interpreter::kSuccess) {
      registerUnloadPoint(unloadPoint, pathname);
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
      << "Refusing to set invalid cling optimization level " << optLevel << '\n';
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
                         MetaProcessor::RedirectionScope stream, bool append) {
    m_MetaProcessor.setStdStream(file, stream, append);
    return AR_Success;
  }

  void MetaSema::actOnComment(llvm::StringRef comment) const {
    // Some of the comments are meaningful for the cling::Interpreter
    m_Interpreter.declare(comment);
  }

  MetaSema::ActionResult MetaSema::actOnxCommand(llvm::StringRef file,
                                                 llvm::StringRef args,
                                                 Value* result) {
    assert(!args.empty() && "Arguments must be provided (at least \"()\"");

    enum CallResult { CR_NoSuchDecl, CR_Failure, CR_Success };
    auto tryCallFunction = [this] (cling::Transaction* T, std::string func,
                                   llvm::StringRef args, Value* ret) {
      if (!T->containsNamedDecl(func))
        return CR_NoSuchDecl;
      std::string S;
      llvm::raw_string_ostream OS(S);
      OS << func << args << " /* .x tries to invoke function `" << func << "` */";

      // Transaction `T' might have a different OptLevel; use that.
      struct OptLevelRAII {
        OptLevelRAII(Interpreter& I, int L)
          : m_Interp(I), m_OptLevel(I.getDefaultOptLevel()) { I.setDefaultOptLevel(L); }
        ~OptLevelRAII() { m_Interp.setDefaultOptLevel(m_OptLevel); }
        Interpreter& m_Interp;
        int m_OptLevel;
      } RAII(m_Interpreter, T->getCompilationOpts().OptLevel);
      return (m_Interpreter.echo(OS.str(), ret) == Interpreter::kSuccess)
              ? CR_Success : CR_Failure;
    };
    
    cling::Transaction* T = nullptr;
    if (actOnLCommand(file, &T) != AR_Success || !T)
      return AR_Failure;

    // First, try function named after `file`; add any alternatives below.
    const std::string tryCallThese[] = {
      makeValidCXXIdentifier(llvm::sys::path::stem(file)),
      // FIXME: this provides an entry point that is independent from the macro
      // filename (and still works if file is renamed); should we enable this?
      //"__main__",
    };
    bool noAlternativeFound = 0;
    for (auto &func : tryCallThese) {
      CallResult CR = tryCallFunction(T, func, args, result);
      if (CR == CR_Success)
        return AR_Success;
      noAlternativeFound |= (CR == CR_NoSuchDecl);
    }

    if (noAlternativeFound) {
      static constexpr char msg[] = "Failed to call `%0%1` to execute the macro.\n"
            "Add this function or rename the macro. Falling back to `.L`.";
      clang::DiagnosticsEngine& Diags = m_Interpreter.getDiagnostics();
      unsigned diagID = Diags.getCustomDiagID(clang::DiagnosticsEngine::Level::Warning, msg);
      //FIXME: Figure out how to pass in proper source locations, which we can
      // use with -verify.
      Diags.Report(clang::SourceLocation(), diagID) << tryCallThese[0] << args.str();
      return AR_Success;  //FIXME: should this be AR_Failure?
    }
    return AR_Failure;
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
    //FIXME: search for the transaction, i.e. verify that it has not already
    // been unloaded, e.g. through `.undo X'.
    auto interpreterHasTransaction = [] (const Interpreter& Interp,
                                         const Transaction* T) {
      for (const Transaction* I = Interp.getFirstTransaction();
           I != 0; I = I->getNext())
        if (I == T)
          return true;
      return false;
    };
    clang::FileManager& FM = m_Interpreter.getSema().getSourceManager().getFileManager();

    std::string pathname(m_Interpreter.lookupFileOrLibrary(file));
    const auto FE = FM.getFile(pathname, /*OpenFile=*/false,
                               /*CacheFailure=*/false);
    auto TI = m_FEToTransaction.find(FE);
    if (!FE || TI == m_FEToTransaction.end())
      return AR_Success;

    const Transaction* unloadPoint = (*TI).second;
    if (interpreterHasTransaction(m_Interpreter, unloadPoint)) {
      // Revert all the transactions that came after `unloadPoint'.
      while (m_Interpreter.getLastTransaction() != unloadPoint) {
        if (const auto ThisFE = m_TransactionToFE[m_Interpreter.getLastTransaction()]) {
          auto I = m_FEToTransaction.find(ThisFE);
          if (I != m_FEToTransaction.end())
            m_FEToTransaction.erase(I);
        }
        m_Interpreter.unload(/*numberOfTransactions=*/1);
      }

      DynamicLibraryManager* DLM = m_Interpreter.getDynamicLibraryManager();
      if (DLM->isLibraryLoaded(pathname))
        DLM->unloadLibrary(pathname);
    } else {
      m_MetaProcessor.getOuts() << "!!!ERROR: Transaction for file: " << file
                                << " has already been unloaded\n";
    }
    m_FEToTransaction.erase(TI);
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
      m_MetaProcessor.getOuts() << (flag ? "U" :"Not u") << "sing raw input\n";
    }
    else
      m_Interpreter.enableRawInput(mode);
  }

  void MetaSema::actOndebugCommand(llvm::Optional<int> mode) const {
    constexpr clang::codegenoptions::DebugInfoKind DebugInfo[] = {
      clang::codegenoptions::NoDebugInfo,
      clang::codegenoptions::LocTrackingOnly,
      clang::codegenoptions::DebugLineTablesOnly,
      clang::codegenoptions::LimitedDebugInfo,
      clang::codegenoptions::FullDebugInfo
    };
    constexpr int N = (int)std::extent<decltype(DebugInfo)>::value;

    clang::CodeGenOptions& CGO = m_Interpreter.getCI()->getCodeGenOpts();
    if (!mode) {
      bool flag = (CGO.getDebugInfo() == clang::codegenoptions::NoDebugInfo);
      if (flag)
        CGO.setDebugInfo(clang::codegenoptions::LimitedDebugInfo);
      else
        CGO.setDebugInfo(clang::codegenoptions::NoDebugInfo);
      m_MetaProcessor.getOuts() << (flag ? "G" : "Not g") << "enerating debug symbols\n";
    } else {
      mode = (*mode < 0) ? 0 : ((*mode >= N) ? N - 1 : *mode);
      CGO.setDebugInfo(DebugInfo[*mode]);
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

  void MetaSema::actOndynamicExtensionsCommand(SwitchMode mode/* = kToggle*/) const {
    if (mode == kToggle) {
      bool flag = !m_Interpreter.isDynamicLookupEnabled();
      m_Interpreter.enableDynamicLookup(flag);
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
      "   " << metaString << "L <filename>\t\t- Load the given file or library\n"
      "\n"
      "   " << metaString << "(x|X) <filename>[(args)]\t- Same as .L and runs a function with"
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
      "   " << metaString << "(I|include) [path]\t\t- Shows all include paths. If a path is given,"
                             "\n\t\t\t\t  adds the path to the include paths.\n"
      "\n"
      "   " << metaString << "O <level>\t\t\t- Sets the optimization level (0-3)"
                             "\n\t\t\t\t  If no level is given, prints the current setting.\n"
      "\n"
      "   " << metaString << "class <name>\t\t- Prints out class <name> in a CINT-like style (one-level).\n"
                             "\t\t\t\t  If no name is given, prints out list of all classes.\n"
      "\n"
      "   " << metaString << "Class <name>\t\t\t- Prints out class <name> in a CINT-like style (all-levels).\n"
                             "\t\t\t\t  If no name is given, prints out list of all classes.\n"
      "\n"
      "   " << metaString << "namespace\t\t\t- Prints list of all known namespaces\n"
      "\n"
      "   " << metaString << "typedef <name>\t\t- Prints out typedef <name> in a CINT-like style\n"
                             "\t\t\t\t  If no name is given, prints out list of all typedefs.\n"
      "\n"
      "   " << metaString << "files\t\t\t- Prints names of all included (parsed) files\n"
      "\n"
      "   " << metaString << "fileEx\t\t\t- Prints out included (parsed) file statistics\n"
                             "\t\t\t\t  as well as a list of their names\n"
      "\n"
      "   " << metaString << "g <var>\t\t\t\t- Prints out information about global variable"
                             "\n\t\t\t\t  'var' - if no name is given, print them all\n"
      "\n"
      "   " << metaString << "@ \t\t\t\t- Cancels and ignores the multiline input\n"
      "\n"
      "   " << metaString << "rawInput [0|1]\t\t- Toggle wrapping and printing the"
                             "\n\t\t\t\t  execution results of the input\n"
      "\n"
      "   " << metaString << "dynamicExtensions [0|1]\t- Toggles the use of the dynamic scopes"
                             "\n\t\t\t\t  and the late binding\n"
      "\n"
      "   " << metaString << "debug <level>\t\t- Generates debug symbols (level is optional, 0 to disable)\n"
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
      "   " << metaString << "T <filePath> <comment>\t- Generate autoload map\n"
      "\n"
      "   " << metaString << "trace <repr> <id>\t\t- Dump trace of requested respresentation\n"
                             "\t\t\t\t  (see " << metaString << "stats arguments for <repr>)\n"
      "\n"
      "   " << metaString << "help\t\t\t- Shows this information (also " << metaString << "?)\n"
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
#if 0
    // Only available in clang's trunk:
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
#endif
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
    llvm::StringRef trimmed(commandLine.trim(" \t\n\v\f\r"));
    if (!trimmed.empty()) {
      int exitStatus = std::system(trimmed.str().c_str());

      // Build the result
      clang::ASTContext& Ctx = m_Interpreter.getCI()->getASTContext();
      if (result) {
        *result = Value(Ctx.IntTy, m_Interpreter);
        result->getAs<long long>() = exitStatus;
      }
      return (exitStatus == 0) ? AR_Success : AR_Failure;
    }
    if (result)
      *result = Value();
    return AR_Failure; //FIXME: should this be success or failure?
  }

  void MetaSema::registerUnloadPoint(const Transaction* unloadPoint,
                                     llvm::StringRef filename) {
    std::string pathname(m_Interpreter.lookupFileOrLibrary(filename));
    if (pathname.empty())
      pathname = filename;

    clang::FileManager& FM = m_Interpreter.getSema().getSourceManager().getFileManager();
    const clang::FileEntry* FE = FM.getFile(pathname, /*OpenFile=*/false,
                                            /*CacheFailure=*/false);
    if (FE && !m_FEToTransaction[FE]) {
      m_FEToTransaction[FE] = unloadPoint;
      m_TransactionToFE[unloadPoint] = FE;
    }
  }
} // end namespace cling
