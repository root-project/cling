//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Manasij Mukherjee  <manasij7479@gmail.com>
// author:  Vassil Vassilev <vvasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_AUTOLOADING_VISITOR_H
#define CLING_AUTOLOADING_VISITOR_H

#include "cling/Utils/Output.h"

#include "clang/AST/DeclVisitor.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/Basic/Specifiers.h"
#include "llvm/ADT/DenseMap.h"
#include <stack>
#include <set>

///\brief Generates forward declarations for a Decl or Transaction
///       by implementing a DeclVisitor
///
/// Important Points:
/// 1. Function arguments having an EnumConstant as a default value
///    are printed in the following way:
///    enum E {E_a, E_b};
///    void foo(E e = E_b){}
///    Generates:
///    enum E : unsigned int;
///    void foo(E e = E(1));
///    1 is the integral value of E_b.
///
/// 2. Decls, in general, are skipped when they depend on things
///    that were previously skipped.
///    The set of strings, m_IncompatibleNames facilitate this.
///    Examine the shouldSkip functions to see why specific types
///    are skipped.
///
/// 3. Log file:
///    The name of the file depends on the name of the file where
///    the forward declarations are written.
///    So, fwd.h produces a corresponding fwd.h.skipped, when
///    output logging is enabled.
///    The log messages are written in the shouldSkip functions to
///    simplify the design.
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
  class Sema;
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
    using IgnoreFilesFunc_t = bool (*)(const clang::PresumedLoc&);

    clang::PrintingPolicy m_Policy; // intentional copy
    llvm::raw_ostream& m_Log;
    unsigned m_Indentation;
    bool m_PrintInstantiation;
    clang::Preprocessor& m_PP;
    clang::SourceManager& m_SMgr;
    clang::ASTContext& m_Ctx;
    bool m_SkipFlag;
    //False by default, true if current item is not to be printed

    llvm::DenseMap<const clang::Decl*, bool> m_Visited; // fwd decl success
    std::stack<llvm::raw_ostream*> m_StreamStack;
    std::set<const char*> m_BuiltinNames;
    IgnoreFilesFunc_t m_IgnoreFile; // Call back to ignore some top level files.

  public:
    ForwardDeclPrinter(llvm::raw_ostream& OutS,
                       llvm::raw_ostream& LogS,
                       clang::Preprocessor& P,
                       clang::ASTContext& Ctx,
                       const Transaction& T,
                       unsigned Indentation = 0,
                       bool printMacros = false,
                       IgnoreFilesFunc_t ignoreFiles =
                          [](const clang::PresumedLoc&) { return false; } );

//    void VisitDeclContext(clang::DeclContext *DC, bool shouldIndent = true);

    void Visit(clang::Decl *D);
    void VisitTranslationUnitDecl(clang::TranslationUnitDecl *D);
    void VisitTypedefDecl(clang::TypedefDecl *D);
    void VisitTypeAliasDecl(clang::TypeAliasDecl *D);
    void VisitEnumDecl(clang::EnumDecl *D);
    void VisitRecordDecl(clang::RecordDecl *D);
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
    void VisitTagDecl(clang::TagDecl *D);
    void VisitLinkageSpecDecl(clang::LinkageSpecDecl *D);
    void VisitRedeclarableTemplateDecl(const clang::RedeclarableTemplateDecl *D);
    void VisitFunctionTemplateDecl(clang::FunctionTemplateDecl *D);
    void VisitClassTemplateDecl(clang::ClassTemplateDecl *D);
    void VisitClassTemplateSpecializationDecl(clang::ClassTemplateSpecializationDecl* D);
    void VisitTypeAliasTemplateDecl(clang::TypeAliasTemplateDecl* D);

    // Not coming from the RecursiveASTVisitor
    void Visit(clang::QualType QT);
    void Visit(const clang::Type* T);
    void VisitNestedNameSpecifier(const clang::NestedNameSpecifier* NNS);
    void VisitTemplateArgument(const clang::TemplateArgument& TA);
    void VisitTemplateName(const clang::TemplateName& TN);

    void printDeclType(llvm::raw_ostream& Stream, clang::QualType T,
                       llvm::StringRef DeclName, bool Pack = false);

    void PrintTemplateParameters(llvm::raw_ostream& Stream,
                                 clang::TemplateParameterList *Params,
                                 const clang::TemplateArgumentList *Args = 0);
    void prettyPrintAttributes(clang::Decl *D);

    bool isOperator(clang::FunctionDecl* D);
    bool hasDefaultArgument(clang::FunctionDecl* D);

    bool shouldSkip(clang::Decl* D) {
      switch (D->getKind()) {
#define DECL(TYPE, BASE) \
        case clang::Decl::TYPE: return shouldSkip((clang::TYPE##Decl*)D); break;
#define ABSTRACT_DECL(DECL)
#include "clang/AST/DeclNodes.inc"
#undef DECL
#undef ABSTRACT_DECL
      }
      return false;
    }

    std::string getNameIfPossible(clang::Decl* D) { return "<not named>"; }
    std::string getNameIfPossible(clang::NamedDecl* D) {
      return D->getNameAsString();
    }

    template <typename T> bool shouldSkip(T* D) {
      // Anything inside DCs except those below cannot be fwd declared.
      clang::Decl::Kind DCKind = D->getDeclContext()->getDeclKind();
      if (DCKind != clang::Decl::Namespace
          && DCKind != clang::Decl::TranslationUnit
          && DCKind != clang::Decl::LinkageSpec) {
        Log() << getNameIfPossible(D) <<" \n";
        skipDecl(D, "Incompatible DeclContext");
      } else {
        if (clang::NamedDecl* ND = clang::dyn_cast<clang::NamedDecl>(D)) {
          if (clang::IdentifierInfo* II = ND->getIdentifier()) {
            if (m_BuiltinNames.find(II->getNameStart()) != m_BuiltinNames.end()
                || !strncmp(II->getNameStart(), "__builtin_", 10))
              skipDecl(D, "builtin");
          }
        }
        if (!m_SkipFlag)
          if (shouldSkipImpl(D))
            skipDecl(D, "shouldSkip");
      }
      if (m_SkipFlag) {
        // Remember that we have tried to fwd declare this already.
        m_Visited.insert(std::pair<const clang::Decl*, bool>(
          getCanonicalOrNamespace(D), false));
      }
      return m_SkipFlag;
    }

    bool ContainsIncompatibleName(clang::TemplateParameterList* Params);

    void skipDecl(clang::Decl* D, const char* Reason);
    void resetSkip() { m_SkipFlag = false; }

    void printStats();
  private:
    llvm::raw_ostream& Indent() { return Indent(m_Indentation); }
    llvm::raw_ostream& Indent(unsigned Indentation);

//    void ProcessDeclGroup(llvm::SmallVectorImpl<clang::Decl*>& Decls);

    void Print(clang::AccessSpecifier AS);

    llvm::raw_ostream& Out() { return *m_StreamStack.top(); }
    llvm::raw_ostream& Log() { return m_Log; }

    bool shouldSkipImpl(clang::Decl*){return false;}

    bool shouldSkipImpl(clang::FunctionDecl* D);
    bool shouldSkipImpl(clang::FunctionTemplateDecl* D);
    bool shouldSkipImpl(clang::TagDecl* D);
    bool shouldSkipImpl(clang::VarDecl* D);
    bool shouldSkipImpl(clang::EnumDecl* D);
    bool shouldSkipImpl(clang::ClassTemplateSpecializationDecl* D);
    bool shouldSkipImpl(clang::UsingDirectiveDecl* D);
    bool shouldSkipImpl(clang::TypeAliasTemplateDecl* D);
    bool shouldSkipImpl(clang::EnumConstantDecl* D) { return false; };
    bool haveSkippedBefore(const clang::Decl* D) const {
      auto Found = m_Visited.find(getCanonicalOrNamespace(D));
      return (Found != m_Visited.end() && !Found->second);
    }
    const clang::Decl* getCanonicalOrNamespace(const clang::Decl* D) const {
      if (D->getKind() == clang::Decl::Namespace)
        return D;
      return D->getCanonicalDecl();
    }
    const clang::Decl* getCanonicalOrNamespace(const clang::NamespaceDecl* D) const {
      return D;
    }
    std::string PrintEnclosingDeclContexts(llvm::raw_ostream& Stream,
                                           const clang::DeclContext* DC);
    void PrintNamespaceOpen(llvm::raw_ostream& Stream,
                            const clang::NamespaceDecl* ND);
    void PrintLinkageOpen(llvm::raw_ostream& Stream,
                          const clang::LinkageSpecDecl* LSD);

    class StreamRAII {
      ForwardDeclPrinter& m_pr;
      clang::PrintingPolicy m_oldPol;
      largestream m_Stream;
      bool m_HavePopped;
    public:
      StreamRAII(ForwardDeclPrinter& pr, clang::PrintingPolicy* pol = 0):
        m_pr(pr), m_oldPol(pr.m_Policy), m_HavePopped(false) {
        m_pr.m_StreamStack.push(&static_cast<llvm::raw_ostream&>(m_Stream));
        if (pol)
          m_pr.m_Policy = *pol;
      }
      ~StreamRAII() {
        if (!m_HavePopped) {
          m_pr.m_StreamStack.pop();
          if (!m_pr.m_SkipFlag) {
            m_pr.Out() << m_Stream.str();
          }
        }
        m_pr.m_Policy = m_oldPol;
      }
      llvm::StringRef take(bool pop = false) {
        if (pop) {
          assert(!m_HavePopped && "No popping twice");
          m_HavePopped = true;
          m_pr.m_StreamStack.pop();
        }
        return m_Stream.str();
      }
    };
  };
}
#endif
