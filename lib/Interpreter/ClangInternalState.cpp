//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "cling/Interpreter/ClangInternalState.h"
#include "cling/Utils/Output.h"
#include "cling/Utils/Platform.h"

#include "clang/AST/ASTContext.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/CodeGen/ModuleBuilder.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Signals.h"

#include <cstdio>
#include <sstream>
#include <string>
#include <time.h>

using namespace clang;

namespace cling {

  ClangInternalState::ClangInternalState(const ASTContext& AC,
                                         const Preprocessor& PP,
                                         const llvm::Module* M,
                                         CodeGenerator* CG,
                                         const std::string& name)
    : m_ASTContext(AC), m_Preprocessor(PP), m_CodeGen(CG), m_Module(M),
#if defined(LLVM_ON_WIN32)
      m_DiffCommand("diff.exe -u --text "),
#else
      m_DiffCommand("diff -u --text "),
#endif
      m_Name(name), m_DiffPair(nullptr) {
    store();
  }

  ClangInternalState::~ClangInternalState() {
    // cleanup the temporary files:
    remove(m_LookupTablesFile.c_str());
    remove(m_IncludedFilesFile.c_str());
    remove(m_ASTFile.c_str());
    remove(m_LLVMModuleFile.c_str());
    remove(m_MacrosFile.c_str());
  }

  void ClangInternalState::store() {
    // Cannot use the stack (private copy ctor)
    std::unique_ptr<llvm::raw_fd_ostream> m_LookupTablesOS;
    std::unique_ptr<llvm::raw_fd_ostream> m_IncludedFilesOS;
    std::unique_ptr<llvm::raw_fd_ostream> m_ASTOS;
    std::unique_ptr<llvm::raw_fd_ostream> m_LLVMModuleOS;
    std::unique_ptr<llvm::raw_fd_ostream> m_MacrosOS;

    m_LookupTablesOS.reset(createOutputFile("lookup",
                                            &m_LookupTablesFile));
    m_IncludedFilesOS.reset(createOutputFile("included",
                                             &m_IncludedFilesFile));
    m_ASTOS.reset(createOutputFile("ast", &m_ASTFile));
    m_LLVMModuleOS.reset(createOutputFile("module", &m_LLVMModuleFile));
    m_MacrosOS.reset(createOutputFile("macros", &m_MacrosFile));

    printLookupTables(*m_LookupTablesOS.get(), m_ASTContext);
    printIncludedFiles(*m_IncludedFilesOS.get(),
                       m_ASTContext.getSourceManager());
    printAST(*m_ASTOS.get(), m_ASTContext);
    if (m_Module)
      printLLVMModule(*m_LLVMModuleOS.get(), *m_Module, *m_CodeGen);
    printMacroDefinitions(*m_MacrosOS.get(), m_Preprocessor);
  }
  namespace {
    std::string getCurrentTimeAsString() {
      time_t rawtime;
      struct tm * timeinfo;
      char buffer [80];

      time (&rawtime);
      timeinfo = localtime (&rawtime);

      strftime (buffer, 80, "%I_%M_%S", timeinfo);
      return buffer;
    }
  }

  // Copied with modifications from CompilerInstance.cpp
  llvm::raw_fd_ostream*
  ClangInternalState::createOutputFile(llvm::StringRef OutFile,
                                       std::string *TempPathName/*=0*/,
                                       bool RemoveFileOnSignal/*=true*/) {
    std::unique_ptr<llvm::raw_fd_ostream> OS;
    std::string OSFile;
    llvm::SmallString<256> OutputPath;
    llvm::sys::path::system_temp_directory(/*erasedOnReboot*/false, OutputPath);

    // Only create the temporary if the parent directory exists (or create
    // missing directories is true) and we can actually write to OutPath,
    // otherwise we want to fail early.
    llvm::SmallString<256> TempPath(OutputPath);
    llvm::sys::fs::make_absolute(TempPath);
    assert(llvm::sys::fs::is_directory(TempPath.str()) && "Must be a folder.");
    // Create a temporary file.
    llvm::sys::path::append(TempPath, "cling-" + OutFile);
    TempPath += "-" + getCurrentTimeAsString();
    TempPath += "-%%%%%%%%";
    int fd;
    if (llvm::sys::fs::createUniqueFile(TempPath.str(), fd, TempPath)
        != std::errc::no_such_file_or_directory) {
      OS.reset(new llvm::raw_fd_ostream(fd, /*shouldClose=*/true));
      OSFile = TempPath.str();
    }

    // Make sure the out stream file gets removed if we crash.
    if (RemoveFileOnSignal)
      llvm::sys::RemoveFileOnSignal(OSFile);

    if (TempPathName)
      *TempPathName = OSFile;

    return OS.release();
  }

  void ClangInternalState::compare(const std::string& name, bool verbose) {
    assert(name == m_Name && "Different names!?");
    m_DiffPair.reset(new ClangInternalState(m_ASTContext, m_Preprocessor,
                                            m_Module, m_CodeGen, name));
    std::string differences = "";
    // Ignore the builtins
    llvm::SmallVector<llvm::StringRef, 1024> builtinNames;
    const clang::Builtin::Context& BuiltinCtx = m_ASTContext.BuiltinInfo;
    for (auto i = clang::Builtin::NotBuiltin+1;
         i != clang::Builtin::FirstTSBuiltin; ++i) {
      llvm::StringRef Name(BuiltinCtx.getName(i));
      if (Name.startswith("__builtin"))
        builtinNames.emplace_back(Name);
    }

    for (auto&& BuiltinInfo: m_ASTContext.getTargetInfo().getTargetBuiltins()) {
      llvm::StringRef Name(BuiltinInfo.Name);
      if (!Name.startswith("__builtin"))
        builtinNames.emplace_back(Name);
#ifndef NDEBUG
      else // Make sure it's already in the list
        assert(std::find(builtinNames.begin(), builtinNames.end(),
                         Name) == builtinNames.end() && "Not in list!");
#endif
    }

    builtinNames.push_back(".*__builtin.*");

    differentContent(m_LookupTablesFile, m_DiffPair->m_LookupTablesFile,
                     "lookup tables", verbose, &builtinNames);

    differentContent(m_IncludedFilesFile, m_DiffPair->m_IncludedFilesFile,
                     "included files", verbose);

    differentContent(m_ASTFile, m_DiffPair->m_ASTFile, "AST", verbose);

    if (m_Module) {
      assert(m_CodeGen && "Must have CodeGen set");
      // We want to skip the intrinsics
      builtinNames.clear();
      for (const auto& Func : m_Module->getFunctionList()) {
        if (Func.isIntrinsic())
          builtinNames.emplace_back(Func.getName());
      }
      differentContent(m_LLVMModuleFile, m_DiffPair->m_LLVMModuleFile,
                       "llvm Module", verbose, &builtinNames);
    }

    differentContent(m_MacrosFile, m_DiffPair->m_MacrosFile,
                     "Macro Definitions", verbose);
  }

  bool ClangInternalState::differentContent(const std::string& file1,
                                            const std::string& file2,
                                            const char* type,
                                            bool verbose,
            const llvm::SmallVectorImpl<llvm::StringRef>* ignores/*=0*/) const {

    std::string diffCall = m_DiffCommand;
    if (ignores) {
      for (const llvm::StringRef& ignore : *ignores) {
        diffCall += " --ignore-matching-lines=\".*";
        diffCall += ignore;
        diffCall += ".*\"";
      }
    }
    diffCall += " ";
    diffCall += file1;
    diffCall += " ";
    diffCall += file2;

    llvm::SmallString<1024> Difs;
    platform::Popen(diffCall, Difs);

    if (verbose)
      cling::log() << diffCall << "\n";

    if (Difs.empty())
      return false;

    if (type) {
      cling::log() << "Differences in the " << type << ":\n";
      cling::log() << Difs << "\n";
    }
    return true;
  }

  class DumpLookupTables : public RecursiveASTVisitor<DumpLookupTables> {
  private:
    llvm::raw_ostream& m_OS;
  public:
    DumpLookupTables(llvm::raw_ostream& OS) : m_OS(OS) { }
    bool VisitDecl(Decl* D) {
      if (DeclContext* DC = dyn_cast<DeclContext>(D))
        VisitDeclContext(DC);
      return true;
    }

    bool VisitDeclContext(DeclContext* DC) {
      // If the lookup is pending for building, force its creation.
      if (DC == DC->getPrimaryContext() && !DC->getLookupPtr())
        DC->buildLookup();
      DC->dumpLookups(m_OS);
      return true;
    }
  };

  void ClangInternalState::printLookupTables(llvm::raw_ostream& Out,
                                             const ASTContext& C) {
    DumpLookupTables dumper(Out);
    dumper.TraverseDecl(C.getTranslationUnitDecl());
  }

  void ClangInternalState::printIncludedFiles(llvm::raw_ostream& Out,
                                              const SourceManager& SM) {
    Out << "Legend: [p] parsed; [P] parsed and open; [r] from AST file\n\n";
    for (clang::SourceManager::fileinfo_iterator I = SM.fileinfo_begin(),
           E = SM.fileinfo_end(); I != E; ++I) {
      const clang::FileEntry *FE = I->first;
      // Our error recovery purges the cache of the FileEntry, but keeps
      // the FileEntry's pointer so that if it was used by smb (like the
      // SourceManager) it wouldn't be dangling. In that case we shouldn't
      // print the FileName, because semantically it is not there.
      if (!I->second)
        continue;
      std::string fileName(FE->getName());
      if (!(fileName.compare(0, 5, "/usr/") == 0 &&
            fileName.find("/bits/") != std::string::npos) &&
          fileName.compare("-")) {
        if (I->second->getRawBuffer()) {
          // There is content - a memory buffer or a file.
          // We know it's a file because we started off the FileEntry.
          if (FE->isOpen())
            Out << "[P] ";
          else
            Out << "[p] ";
        } else
          Out << "[r] ";
        Out << fileName << '\n';
      }
    }
  }

  void ClangInternalState::printAST(llvm::raw_ostream& Out, const ASTContext& C) {
    TranslationUnitDecl* TU = C.getTranslationUnitDecl();
    unsigned Indentation = 0;
    bool PrintInstantiation = false;
    std::string ErrMsg;
    clang::PrintingPolicy policy = C.getPrintingPolicy();
    TU->print(Out, policy, Indentation, PrintInstantiation);
    // TODO: For future when we relpace the bump allocation with slab.
    //
    //Out << "Allocated memory: " << C.getAllocatedMemory();
    //Out << "Side table allocated memory: " << C.getSideTableAllocatedMemory();
    Out.flush();
  }

  void ClangInternalState::printLLVMModule(llvm::raw_ostream& Out,
                                           const llvm::Module& M,
                                           CodeGenerator& CG) {
    M.print(Out, /*AssemblyAnnotationWriter*/ 0);
    CG.print(Out);
  }

  void ClangInternalState::printMacroDefinitions(llvm::raw_ostream& Out,
                                                const clang::Preprocessor& PP) {
    stdstrstream contentsOS;
    PP.printMacros(contentsOS);
    Out << "Ordered Alphabetically:\n";
    std::vector<std::string> elems;
    {
      // Split the string into lines.
      char delim = '\n';
      std::stringstream ss(contentsOS.str());
      std::string item;
      while (std::getline(ss, item, delim)) {
        elems.push_back(item);
      }
      // Sort them alphabetically
      std::sort(elems.begin(), elems.end());
    }
    for(std::vector<std::string>::iterator I = elems.begin(),
          E = elems.end(); I != E; ++I)
      Out << *I << '\n';
    Out.flush();
  }
} // end namespace cling
