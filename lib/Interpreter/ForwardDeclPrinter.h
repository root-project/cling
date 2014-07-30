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
  class Transaction;

  class ForwardDeclPrinter : public clang::DeclVisitor<ForwardDeclPrinter> {
    llvm::raw_ostream &Out;
    clang::PrintingPolicy Policy; // intentional copy
    unsigned Indentation;
    bool PrintInstantiation;

    llvm::raw_ostream& Indent() { return Indent(Indentation); }
    llvm::raw_ostream& Indent(unsigned Indentation);
    void ProcessDeclGroup(llvm::SmallVectorImpl<clang::Decl*>& Decls);

    void Print(clang::AccessSpecifier AS);

    clang::SourceManager& m_SMgr;
    bool m_SkipFlag;
    //False by default, true if current item is not to be printed
  public:
    ForwardDeclPrinter(llvm::raw_ostream &Out, clang::SourceManager& smgr,
                       const Transaction& T,
                       unsigned Indentation = 0,
                       bool printMacros = false);

    ForwardDeclPrinter(llvm::raw_ostream &Out, clang::SourceManager& smgr,
                       const clang::PrintingPolicy& P,
                       unsigned Indentation = 0);

    clang::PrintingPolicy& getPolicy() { return Policy; }
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
    void prettyPrintAttributes(clang::Decl *D, std::string extra="");

    void printSemiColon(bool flag=true);
    //if flag is true , m_SkipFlag is obeyed and reset.

    bool hasNestedNameSpecifier(clang::QualType q);
    bool isOperator(clang::FunctionDecl* D);
    bool shouldSkipFunction(clang::FunctionDecl* D);
  };
}
#endif
