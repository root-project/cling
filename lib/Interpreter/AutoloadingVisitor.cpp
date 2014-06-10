#include "AutoloadingVisitor.h"
namespace cling {
  bool AutoloadingVisitor::VisitCXXRecordDecl(clang::CXXRecordDecl* Declaration) {
    if(Declaration->getName().startswith("_")
          || Declaration->getName().size() == 0
          /*|| //TODO: Find a way to avoid templates here*/)
      return true;

    std::vector<NamespacePrinterRAII> scope;
    clang::DeclContext* c=Declaration->getEnclosingNamespaceContext();
    while(c->isNamespace()) {
      clang::NamespaceDecl* n = llvm::cast<clang::NamespaceDecl>(c);
      scope.emplace_back(n->getNameAsString());
      c=c->getParent();
    }

    llvm::outs() << "\n" << Declaration->getKindName()
             << " __attribute__((annotate(\""
             << m_InFile << "\"))) "
             << Declaration->getName() << ";\n";
    return true;
  }
  bool AutoloadingVisitor::VisitFunctionDecl(clang::FunctionDecl* Declaration) {

    if(Declaration->getName().startswith("_")
        || Declaration->getName().size() == 0
        || Declaration->isCXXClassMember()
        || !Declaration->hasBody())
      return true;

    std::vector<NamespacePrinterRAII> scope;
    clang::DeclContext* c = Declaration->getEnclosingNamespaceContext();
    while(c->isNamespace()) {
      clang::NamespaceDecl* n = llvm::cast<clang::NamespaceDecl>(c);
      scope.emplace_back(n->getNameAsString());
      c=c->getParent();
    }

    llvm::outs() << "\n" << Declaration->getReturnType().getAsString()
             << " " << Declaration->getName() << " () "
             << "__attribute__((annotate(\""
             << m_InFile << "\")));\n";

    //TODO: arg list, not sure if necessary

    return true;
  }

  bool AutoloadingVisitor::VisitClassTemplateDecl
        (clang::ClassTemplateDecl* Declaration) {
   if(Declaration->getName().startswith("_")
      || Declaration->getName().size() == 0)
     return true;

    std::vector<NamespacePrinterRAII> scope;
    clang::DeclContext* c=
      Declaration->getTemplatedDecl()->getEnclosingNamespaceContext();
    while(c->isNamespace()) {
      clang::NamespaceDecl* n = llvm::cast<clang::NamespaceDecl>(c);
      scope.emplace_back(n->getNameAsString());
      c=c->getParent();
    }

    llvm::outs()<<"template <";
    clang::TemplateParameterList* tl=Declaration->getTemplateParameters();
    for(auto it=tl->begin();it!=tl->end();++it) {
      if(llvm::isa<clang::NonTypeTemplateParmDecl>(*it)) {
        clang::NonTypeTemplateParmDecl* td=llvm::cast<clang::NonTypeTemplateParmDecl>(*it);
        llvm::outs() << td->getType().getAsString();
      }
      else llvm::outs() << "typename";
      llvm::outs()<<" " << (*it)->getName();
      if((it+1) != tl->end())
        llvm::outs() << ", ";
    }
    llvm::outs()<<"> class __attribute__((annotate(\""
            << m_InFile << "\"))) "
            << Declaration->getName() << ";\n";

    return true;
  }
} // end namespace cling
