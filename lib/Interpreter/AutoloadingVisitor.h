#ifndef CLING_AUTOLOADING_VISITOR_H
#define CLING_AUTOLOADING_VISITOR_H
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"

namespace cling {
  class NamespacePrinterRAII {
  public:
    NamespacePrinterRAII(std::string name) {
      llvm::outs()<< "namespace " <<name<<" {\n";
    }
    ~NamespacePrinterRAII() {
      llvm::outs()<<"\n}\n";
    }
  };

  class AutoloadingVisitor
    :public clang::RecursiveASTVisitor<AutoloadingVisitor> {
  public:
    AutoloadingVisitor(llvm::StringRef InFile,llvm::StringRef OutFile)
        :m_InFile(InFile),m_OutFile(OutFile){}
    bool VisitCXXRecordDecl(clang::CXXRecordDecl* Declaration);
    bool VisitFunctionDecl(clang::FunctionDecl* Declaration);
    bool VisitClassTemplateDecl(clang::ClassTemplateDecl* Declaration);

  private:
    llvm::StringRef m_InFile;
    llvm::StringRef m_OutFile;
  };
}//end namespace cling

#endif
