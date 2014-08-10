//TODO: Adapted from DeclPrinter, may need to be rewritten
#ifndef CLING_AUTOLOADING_VISITOR_H
#define CLING_AUTOLOADING_VISITOR_H

#include "clang/AST/DeclVisitor.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/Basic/Specifiers.h"
#include <set>
#include <stack>

///\brief Generates forward declarations for a Decl or Transaction by implementing a DeclVisitor
///
///\Cases which do not work:
///\1. Nested name specifiers: Since there doesn't seem to be a way to forward declare B in the following example:
///    class A { class B{};};
///    We have chosen to skip all declarations using B.
///    This filters out many acceptable types, like when B is defined within a namespace.
///    The fix for this issue dhould go in isIncompatibleType, which currently just searches for "::" in the type name.
///
///

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
    clang::PrintingPolicy m_Policy; // intentional copy
    llvm::raw_ostream& m_Log;
    unsigned m_Indentation;
    bool m_PrintInstantiation;
    clang::SourceManager& m_SMgr;
    bool m_SkipFlag;
    //False by default, true if current item is not to be printed

    std::set<llvm::StringRef> m_IncompatibleNames;
    int m_SkipCounter;
    int m_TotalDecls;
  public:
    ForwardDeclPrinter(llvm::raw_ostream& OutS,
                       llvm::raw_ostream& LogS,
                       clang::SourceManager& SM,
                       const Transaction& T,
                       unsigned Indentation = 0,
                       bool printMacros = false);

    ForwardDeclPrinter(llvm::raw_ostream& OutS,
                       llvm::raw_ostream& LogS,
                       clang::SourceManager& SM,
                       const clang::PrintingPolicy& P,
                       unsigned Indentation = 0);

//    void VisitDeclContext(clang::DeclContext *DC, bool shouldIndent = true);

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
    void VisitUsingDecl(clang::UsingDecl* D);
    void VisitUsingShadowDecl(clang::UsingShadowDecl* D);
    void VisitNamespaceAliasDecl(clang::NamespaceAliasDecl *D);
    void VisitCXXRecordDecl(clang::CXXRecordDecl *D);
    void VisitLinkageSpecDecl(clang::LinkageSpecDecl *D);
    void VisitTemplateDecl(const clang::TemplateDecl *D);
    void VisitFunctionTemplateDecl(clang::FunctionTemplateDecl *D);
    void VisitClassTemplateDecl(clang::ClassTemplateDecl *D);
    void VisitClassTemplateSpecializationDecl(clang::ClassTemplateSpecializationDecl* D);
    void VisitTypeAliasTemplateDecl(clang::TypeAliasTemplateDecl* D);
    void printDeclType(clang::QualType T, llvm::StringRef DeclName, bool Pack = false);

    void PrintTemplateParameters(const clang::TemplateParameterList *Params,
                               const clang::TemplateArgumentList *Args = 0);
    void prettyPrintAttributes(clang::Decl *D, std::string extra = "");

    void printSemiColon(bool flag = true);
    //if flag is true , m_SkipFlag is obeyed and reset.

    bool isIncompatibleType(clang::QualType q, bool includeNNS = true);
    bool isOperator(clang::FunctionDecl* D);

    template<typename DeclT>
    bool shouldSkip(DeclT* D){return false;}

    bool shouldSkip(clang::FunctionDecl* D);
    bool shouldSkip(clang::CXXRecordDecl* D);
    bool shouldSkip(clang::TypedefDecl* D);
    bool shouldSkip(clang::VarDecl* D);
    bool shouldSkip(clang::EnumDecl* D);
    bool shouldSkip(clang::ClassTemplateSpecializationDecl* D);
    bool shouldSkip(clang::UsingDecl* D);
    bool shouldSkip(clang::UsingShadowDecl* D){return true;}
    bool shouldSkip(clang::UsingDirectiveDecl* D);
    bool shouldSkip(clang::ClassTemplateDecl* D);
    bool shouldSkip(clang::FunctionTemplateDecl* D);
    bool shouldSkip(clang::TypeAliasTemplateDecl* D);

    bool ContainsIncompatibleName(clang::TemplateParameterList* Params);

    void skipCurrentDecl(bool skip = true);

    void printStats();
  private:
    llvm::raw_ostream& Indent() { return Indent(m_Indentation); }
    llvm::raw_ostream& Indent(unsigned Indentation);

//    void ProcessDeclGroup(llvm::SmallVectorImpl<clang::Decl*>& Decls);

    void Print(clang::AccessSpecifier AS);

    llvm::raw_ostream& Out();
    llvm::raw_ostream& Log();

    std::stack<llvm::raw_ostream*> m_StreamStack;
  };
}
#endif
