//TODO: Adapted from DeclPrinter, may need to be rewritten
#ifndef CLING_AUTOLOADING_VISITOR_H
#define CLING_AUTOLOADING_VISITOR_H

#include "clang/AST/DeclVisitor.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/Basic/Specifiers.h"
#include <set>

namespace clang {
  class ClassTemplateDecl;
  class ClassTemplateSpecializationDecl;
  class CXXRecordDecl;
  class Decl;
  class DeclContext;
  class EmptyDecl;
  class EnumDecl;
  class EnumConstantDecl;
  class FieldDecl;
  class FileScopeAsmDecl;
  class FriendDecl;
  class FunctionDecl;
  class FunctionTemplateDecl;
  class ImportDecl;
  class LabelDecl;
  class LinkageSpecDecl;
  class NamespaceDecl;
  class NamespaceAliasDecl;
  class ParmVarDecl;
  class QualType;
  class RecordDecl;
  class SourceManager;
  class StaticAssertDecl;
  class TemplateArgumentList;
  class TemplateDecl;
  class TemplateParameterList;
  class TranslationUnitDecl;
  class TypeAliasDecl;
  class TypedefDecl;
  class VarDecl;
  class UsingDirectiveDecl;
}

namespace llvm {
  class raw_ostream;
}

namespace cling {
  class Transaction;

  class ForwardDeclPrinter : public clang::DeclVisitor<ForwardDeclPrinter> {
  private:
    llvm::raw_ostream &m_Out;
    clang::PrintingPolicy m_Policy; // intentional copy
    unsigned m_Indentation;
    bool m_PrintInstantiation;

    clang::SourceManager& m_SMgr;
    bool m_SkipFlag;
    //False by default, true if current item is not to be printed

    std::set<llvm::StringRef> m_IncompatibleTypes;

  public:
    ForwardDeclPrinter(llvm::raw_ostream& Out, clang::SourceManager& SM,
                       const Transaction& T,
                       unsigned Indentation = 0,
                       bool printMacros = false);

    ForwardDeclPrinter(llvm::raw_ostream &Out, clang::SourceManager& SM,
                       const clang::PrintingPolicy& P,
                       unsigned Indentation = 0);

    void VisitDeclContext(clang::DeclContext *DC, bool shouldIndent = true);

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

    bool isIncompatibleType(clang::QualType q);
    bool isOperator(clang::FunctionDecl* D);
    bool shouldSkipFunction(clang::FunctionDecl* D);
  private:
    llvm::raw_ostream& Indent() { return Indent(m_Indentation); }
    llvm::raw_ostream& Indent(unsigned Indentation);

    void ProcessDeclGroup(llvm::SmallVectorImpl<clang::Decl*>& Decls);

    void Print(clang::AccessSpecifier AS);
  };
}
#endif
