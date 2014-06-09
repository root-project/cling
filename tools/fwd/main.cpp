#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"

#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"


using namespace clang::tooling;
using namespace llvm;

// Apply a custom category to all command-line options so that they are the
// only ones displayed.
static cl::OptionCategory FwdCategory("fwd options");

// CommonOptionsParser declares HelpMessage with a description of the common
// command-line options related to the compilation database and input files.
// It's nice to have this help message in all tools.
static cl::extrahelp CommonHelp(CommonOptionsParser::HelpMessage);

// A help message for this specific tool can be added afterwards.
static cl::extrahelp MoreHelp("\nMore help text...");

class NamespacePrinterRAII {
public:
  NamespacePrinterRAII(std::string name) {
    llvm::outs()<< "namespace " <<name<<" {\n";
  }
  ~NamespacePrinterRAII() {
    llvm::outs()<<"\n}\n";
  }
};

class FindNamedClassVisitor
  :public clang::RecursiveASTVisitor<FindNamedClassVisitor> {
public:
  FindNamedClassVisitor(llvm::StringRef InFile):m_File(InFile){}
  bool VisitCXXRecordDecl(clang::CXXRecordDecl* Declaration) {

      if(Declaration->getName().startswith("_")
              || Declaration->getName().size()==0
              /*|| //TODO: Find a way to avoid templates here*/)
          return true;

      std::vector<NamespacePrinterRAII> scope;
      clang::DeclContext* c=Declaration->getEnclosingNamespaceContext();
      while(c->isNamespace()) {
        clang::NamespaceDecl* n=llvm::cast<clang::NamespaceDecl>(c);
        scope.emplace_back(n->getNameAsString());
        c=c->getParent();
      }

    llvm::outs() << "\n" << Declaration->getKindName()
                 << " __attribute__((annotate(\""
                 << m_File << "\"))) "
                 << Declaration->getName() << ";\n";
    return true;
  }
  bool VisitFunctionDecl(clang::FunctionDecl* Declaration) {

      if(Declaration->getName().startswith("_")
            || Declaration->getName().size()==0
            || Declaration->isCXXClassMember()
            || !Declaration->hasBody())
        return true;

      std::vector<NamespacePrinterRAII> scope;
      clang::DeclContext* c=Declaration->getEnclosingNamespaceContext();
      while(c->isNamespace()) {
        clang::NamespaceDecl* n=llvm::cast<clang::NamespaceDecl>(c);
        scope.emplace_back(n->getNameAsString());
        c=c->getParent();
      }

    llvm::outs() << "\n" << Declaration->getReturnType().getAsString()
                 << " " << Declaration->getName() << " () "
                 << "__attribute__((annotate(\""
                 << m_File << "\")));\n";

    //TODO: arg list, not sure if necessary

    return true;
  }

  bool VisitClassTemplateDecl(clang::ClassTemplateDecl* Declaration) {

   if(Declaration->getName().startswith("_")
          || Declaration->getName().size()==0)
     return true;

    std::vector<NamespacePrinterRAII> scope;
    clang::DeclContext* c=Declaration->getTemplatedDecl()->getEnclosingNamespaceContext();
    while(c->isNamespace()) {
      clang::NamespaceDecl* n=llvm::cast<clang::NamespaceDecl>(c);
      scope.emplace_back(n->getNameAsString());
      c=c->getParent();
    }

    llvm::outs()<<"template <";
    clang::TemplateParameterList* tl=Declaration->getTemplateParameters();
    for(auto it=tl->begin();it!=tl->end();++it) {
      if(llvm::isa<clang::NonTypeTemplateParmDecl>(*it)) {
        clang::NonTypeTemplateParmDecl* td=llvm::cast<clang::NonTypeTemplateParmDecl>(*it);
        llvm::outs()<<td->getType().getAsString();
      }
      else llvm::outs()<<"typename";
      llvm::outs()<<" "<<(*it)->getName();
      if((it+1)!=tl->end())
        llvm::outs()<<", ";
    }
    llvm::outs()<<"> class __attribute__((annotate(\""
                << m_File << "\"))) "
                << Declaration->getName() << ";\n";

    return true;
  }

private:
  llvm::StringRef m_File;
};

class FindNamedClassConsumer : public clang::ASTConsumer {
public:
  FindNamedClassConsumer(llvm::StringRef InFile):Visitor(InFile){}
  virtual void HandleTranslationUnit(clang::ASTContext &Context) {
    // Traversing the translation unit decl via a RecursiveASTVisitor
    // will visit all nodes in the AST.
    Visitor.TraverseDecl(Context.getTranslationUnitDecl());
  }
private:
  // A RecursiveASTVisitor implementation.
  FindNamedClassVisitor Visitor;
};


class FindNamedClassAction : public clang::ASTFrontendAction {
public:
  virtual clang::ASTConsumer *CreateASTConsumer(
    clang::CompilerInstance &Compiler, llvm::StringRef InFile) {
    return new FindNamedClassConsumer(InFile);
  }
};


int main(int argc, const char **argv) {
  CommonOptionsParser OptionsParser(argc, argv, FwdCategory);
  ClangTool Tool(OptionsParser.getCompilations(),
                 OptionsParser.getSourcePathList());

  return Tool.run(newFrontendActionFactory<FindNamedClassAction>());
}
