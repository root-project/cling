//TODO: Adapted from DeclPrinter, may need to be rewritten
#ifndef CLING_AUTOLOADING_VISITOR_H
#define CLING_AUTOLOADING_VISITOR_H
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/Basic/Module.h"
#include "llvm/Support/raw_ostream.h"

#include <set>

namespace cling {

  class ForwardDeclPrinter : public clang::DeclVisitor<ForwardDeclPrinter> {
    llvm::raw_ostream &Out;
    clang::PrintingPolicy Policy;
    unsigned Indentation;
    bool PrintInstantiation;

    llvm::raw_ostream& Indent() { return Indent(Indentation); }
    llvm::raw_ostream& Indent(unsigned Indentation);
    void ProcessDeclGroup(llvm::SmallVectorImpl<clang::Decl*>& Decls);

    void Print(clang::AccessSpecifier AS);

    std::set<std::string> ClassDeclNames;
    clang::SourceManager& m_SMgr;

  public:
    ForwardDeclPrinter(llvm::raw_ostream &Out, clang::SourceManager& smgr,
        const clang::PrintingPolicy &Policy =clang::PrintingPolicy(clang::LangOptions()),
        unsigned Indentation = 0, bool PrintInstantiation = false)
      : Out(Out), Policy(Policy), Indentation(Indentation),
        PrintInstantiation(PrintInstantiation),m_SMgr(smgr)
        { }

    void VisitDeclContext(clang::DeclContext *DC, bool Indent = true);

    void VisitTranslationUnitDecl(clang::TranslationUnitDecl *D);
    void VisitTypedefDecl(clang::TypedefDecl *D);
    void VisitTypeAliasDecl(clang::TypeAliasDecl *D);
    void VisitEnumDecl(clang::EnumDecl *D);
    void VisitRecordDecl(clang::RecordDecl *D);
    void VisitEnumConstantDecl(clang::EnumConstantDecl *D);
    void VisitEmptyDecl(clang::EmptyDecl *D);
    void VisitFunctionDecl(clang::FunctionDecl *D);
    void VisitFriendDecl(clang::FriendDecl *D);
    void VisitFieldDecl(clang::FieldDecl *D);
    void VisitVarDecl(clang::VarDecl *D);
    void VisitLabelDecl(clang::LabelDecl *D);
    void VisitParmVarDecl(clang::ParmVarDecl *D);
    void VisitFileScopeAsmDecl(clang::FileScopeAsmDecl *D);
    void VisitImportDecl(clang::ImportDecl *D);
    void VisitStaticAssertDecl(clang::StaticAssertDecl *D);
    void VisitNamespaceDecl(clang::NamespaceDecl *D);
    void VisitUsingDirectiveDecl(clang::UsingDirectiveDecl *D);
    void VisitNamespaceAliasDecl(clang::NamespaceAliasDecl *D);
    void VisitCXXRecordDecl(clang::CXXRecordDecl *D);
    void VisitLinkageSpecDecl(clang::LinkageSpecDecl *D);
    void VisitTemplateDecl(const clang::TemplateDecl *D);
    void VisitFunctionTemplateDecl(clang::FunctionTemplateDecl *D);
    void VisitClassTemplateDecl(clang::ClassTemplateDecl *D);
    void VisitClassTemplateSpecializationDecl(clang::ClassTemplateSpecializationDecl* D);

    void PrintTemplateParameters(const clang::TemplateParameterList *Params,
                               const clang::TemplateArgumentList *Args = 0);
    void prettyPrintAttributes(clang::Decl *D);
  };
}
#endif
