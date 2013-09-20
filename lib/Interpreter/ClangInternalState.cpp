//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#include "cling/Interpreter/ClangInternalState.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/SourceManager.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Signals.h"

#include <cstdio>
#include <string>

using namespace clang;

namespace cling {
  ClangInternalState::ClangInternalState(ASTContext& C, const std::string& name)
    : m_ASTContext(C), m_DiffCommand("diff -u "), m_Name(name) {
    store();
  }

  ClangInternalState::~ClangInternalState() {
    // cleanup the temporary files:
    remove(m_LookupTablesFile.c_str());
    remove(m_IncludedFilesFile.c_str());
    remove(m_ASTFile.c_str());
  }

  void ClangInternalState::store() {
    m_LookupTablesOS.reset(createOutputFile("lookup", &m_LookupTablesFile));
    m_IncludedFilesOS.reset(createOutputFile("included", &m_IncludedFilesFile));
    m_ASTOS.reset(createOutputFile("ast", &m_ASTFile));
    
    printLookupTables(*m_LookupTablesOS.get(), m_ASTContext);
    printIncludedFiles(*m_IncludedFilesOS.get(), 
                       m_ASTContext.getSourceManager());
    printAST(*m_ASTOS.get(), m_ASTContext);
  }

  // Copied with modifications from CompilerInstance.cpp
  llvm::raw_fd_ostream* 
  ClangInternalState::createOutputFile(llvm::StringRef OutFile,
                                       std::string *TempPathName/*=0*/,
                                       bool RemoveFileOnSignal/*=true*/) {
    llvm::OwningPtr<llvm::raw_fd_ostream> OS;
    std::string OSFile;
    llvm::SmallString<256> OutputPath;
    llvm::sys::path::system_temp_directory(/*erasedOnReboot*/false, OutputPath);

    // Only create the temporary if the parent directory exists (or create
    // missing directories is true) and we can actually write to OutPath,
    // otherwise we want to fail early.
    llvm::SmallString<256> AbsPath(OutputPath);
    llvm::sys::fs::make_absolute(AbsPath);
    llvm::sys::Path OutPath(AbsPath);
    assert(!OutPath.isRegularFile() && "Must be a folder.");
    // Create a temporary file.
    llvm::SmallString<128> TempPath;
    TempPath = OutFile;
    TempPath += "-%%%%%%%%";
    int fd;
    if (llvm::sys::fs::unique_file(TempPath.str(), fd, TempPath,
                                   /*makeAbsolute=*/false, 0664)
        == llvm::errc::success) {
      OS.reset(new llvm::raw_fd_ostream(fd, /*shouldClose=*/true));
      OSFile = TempPath.str();
    }

    // Make sure the out stream file gets removed if we crash.
    if (RemoveFileOnSignal)
      llvm::sys::RemoveFileOnSignal(llvm::sys::Path(OSFile));
    
    if (TempPathName)
      *TempPathName = OSFile;
    
    return OS.take();
  }

  void ClangInternalState::compare(ClangInternalState& other) {
    std::string differences = "";
    if (differentContent(m_LookupTablesFile, other.m_LookupTablesFile, 
                         differences)) {
      llvm::errs() << "Differences in the lookup tablse\n";
      llvm::errs() << differences << "\n";
      differences = "";
    }

    if (differentContent(m_IncludedFilesFile, other.m_IncludedFilesFile, 
                         differences)) {
      llvm::errs() << "Differences in the included files\n";
      llvm::errs() << differences << "\n";
      differences = "";
    }

    if (differentContent(m_ASTFile, other.m_ASTFile, differences)) {
      llvm::errs() << "Differences in the AST \n";
      llvm::errs() << differences << "\n";
      differences = "";
    }
  }

  bool ClangInternalState::differentContent(const std::string& file1, 
                                            const std::string& file2, 
                                            std::string& differences) const {
    FILE* pipe = popen((m_DiffCommand + file1 + " " + file2).c_str() , "r");
    assert(pipe && "Error creating the pipe");
    assert(differences.empty() && "Must be empty");

    char buffer[128];
    while(!feof(pipe)) {
      if(fgets(buffer, 128, pipe) != NULL)
        differences += buffer;
    }
    pclose(pipe);
    return !differences.empty();
  }


  class DumpLookupTables : public RecursiveASTVisitor<DumpLookupTables> {
  private:
    //llvm::raw_ostream& m_OS;
  public:
    //DumpLookupTables(llvm::raw_ostream& OS) : m_OS(OS) { }
    DumpLookupTables(llvm::raw_ostream&) { }
    bool VisitDeclContext(DeclContext* DC) {
      //DC->dumpLookups(m_OS);
      return true;
    }
  };

  void ClangInternalState::printLookupTables(llvm::raw_ostream& Out, 
                                             ASTContext& C) {
    DumpLookupTables dumper(Out);
    dumper.TraverseDecl(C.getTranslationUnitDecl());
  }

  void ClangInternalState::printIncludedFiles(llvm::raw_ostream& Out, 
                                              SourceManager& SM) {
    for (clang::SourceManager::fileinfo_iterator I = SM.fileinfo_begin(),
           E = SM.fileinfo_end(); I != E; ++I) {
      const clang::SrcMgr::ContentCache &C = *I->second;
      const clang::FileEntry *FE = C.OrigEntry;
      std::string fileName(FE->getName());
      if (!(fileName.compare(0, 5, "/usr/") == 0 &&
            fileName.find("/bits/") != std::string::npos)) {
        Out << fileName << '\n';
      }
    }
  }

  void ClangInternalState::printAST(llvm::raw_ostream& Out, ASTContext& C) {
    TranslationUnitDecl* TU = C.getTranslationUnitDecl();
    unsigned Indentation = 0;
    bool PrintInstantiation = false;
    std::string ErrMsg;
    clang::PrintingPolicy policy = C.getPrintingPolicy();
    TU->print(Out, policy, Indentation, PrintInstantiation);
    Out.flush();
  }
} // end namespace cling
