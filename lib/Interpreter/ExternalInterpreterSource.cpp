//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Elisavet Sakellari <elisavet.sakellari@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "ExternalInterpreterSource.h"
#include "cling/Interpreter/Interpreter.h"
#include "cling/Utils/Diagnostics.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTDiagnostic.h"
#include "clang/AST/ASTImporter.h"
#include "clang/AST/DeclContextInternals.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/Sema.h"

using namespace clang;

namespace {
  class ClingASTImporter : public ASTImporter {
  private:
    cling::ExternalInterpreterSource &m_Source;

  public:
    ClingASTImporter(ASTContext &ToContext, FileManager &ToFileManager,
                     ASTContext &FromContext, FileManager &FromFileManager,
                     bool MinimalImport,
                     cling::ExternalInterpreterSource& source):
      ASTImporter(ToContext, ToFileManager, FromContext, FromFileManager,
                  MinimalImport), m_Source(source) {}
    virtual ~ClingASTImporter() = default;

    Decl *Imported(Decl *From, Decl *To) override {
      ASTImporter::Imported(From, To);

      if (clang::TagDecl* toTagDecl = dyn_cast<TagDecl>(To)) {
        toTagDecl->setHasExternalLexicalStorage();
        toTagDecl->setMustBuildLookupTable();
        toTagDecl->setHasExternalVisibleStorage();
      }
      if (NamespaceDecl *toNamespaceDecl = dyn_cast<NamespaceDecl>(To)) {
        toNamespaceDecl->setHasExternalVisibleStorage();
      }
      if (ObjCContainerDecl *toContainerDecl = dyn_cast<ObjCContainerDecl>(To)) {
        toContainerDecl->setHasExternalLexicalStorage();
        toContainerDecl->setHasExternalVisibleStorage();
      }
      // Put the name of the Decl imported with the
      // DeclarationName coming from the parent, in  my map.
      if (NamedDecl *toNamedDecl = llvm::dyn_cast<NamedDecl>(To)) {
        NamedDecl *fromNamedDecl = llvm::dyn_cast<NamedDecl>(From);
        m_Source.addToImportedDecls(toNamedDecl->getDeclName(),
                                    fromNamedDecl->getDeclName());
      }
      if (DeclContext *toDeclContext = llvm::dyn_cast<DeclContext>(To)) {
        DeclContext *fromDeclContext = llvm::dyn_cast<DeclContext>(From);
        m_Source.addToImportedDeclContexts(toDeclContext, fromDeclContext);
      }
      return To;
    }
  };
}

namespace cling {

  ExternalInterpreterSource::ExternalInterpreterSource(
        const cling::Interpreter *parent, cling::Interpreter *child) :
        m_ParentInterpreter(parent), m_ChildInterpreter(child) {

    clang::DeclContext *parentTUDeclContext =
      m_ParentInterpreter->getCI()->getASTContext().getTranslationUnitDecl();

    clang::DeclContext *childTUDeclContext =
      m_ChildInterpreter->getCI()->getASTContext().getTranslationUnitDecl();

    // Also keep in the map of Decl Contexts the Translation Unit Decl Context
    m_ImportedDeclContexts.emplace(childTUDeclContext, parentTUDeclContext);

    FileManager &childFM = m_ChildInterpreter->getCI()->getFileManager();
    FileManager &parentFM = m_ParentInterpreter->getCI()->getFileManager();

    ASTContext &fromASTContext = m_ParentInterpreter->getCI()->getASTContext();
    ASTContext &toASTContext = m_ChildInterpreter->getCI()->getASTContext();
    ClingASTImporter* importer
      = new ClingASTImporter(toASTContext, childFM, fromASTContext, parentFM,
                             /*MinimalImport : ON*/ true, *this);
    m_Importer.reset(llvm::dyn_cast<ASTImporter>(importer));
  }

  ExternalInterpreterSource::~ExternalInterpreterSource() {}

  void ExternalInterpreterSource::ImportDecl(Decl *declToImport,
                                   DeclarationName &childDeclName,
                                   const DeclarationName &parentDeclName,
                                   const DeclContext *childCurrentDeclContext) {

    // Don't do the import if we have a Function Template or using decls. They
    // are not supported by clang.
    // FIXME: These are temporary checks and should be de-activated once clang
    // supports their import.
    if ((declToImport->isFunctionOrFunctionTemplate()
         && declToImport->isTemplateDecl()) || dyn_cast<UsingDecl>(declToImport)
         || dyn_cast<UsingDirectiveDecl>(declToImport)
         || dyn_cast<UsingShadowDecl>(declToImport)) {
#ifndef NDEBUG
      utils::DiagnosticsStore DS(
        m_Importer->getFromContext().getDiagnostics(), false, false, true);

      assert((m_Importer->Import(declToImport)==nullptr) && "Import worked!");
      assert(!DS.empty() &&
             DS[0].getID() == clang::diag::err_unsupported_ast_node &&
             "Import may be supported");
#endif
      return;
    }

    if (Decl *importedDecl = m_Importer->Import(declToImport)) {
      if (NamedDecl *importedNamedDecl = llvm::dyn_cast<NamedDecl>(importedDecl)) {
        SetExternalVisibleDeclsForName(childCurrentDeclContext,
                                       importedNamedDecl->getDeclName(),
                                       importedNamedDecl);
      }
      // Put the name of the Decl imported with the
      // DeclarationName coming from the parent, in  my map.
      m_ImportedDecls.emplace(childDeclName, parentDeclName);
    }
  }

  void ExternalInterpreterSource::ImportDeclContext(
                                  DeclContext *declContextToImport,
                                  DeclarationName &childDeclName,
                                  const DeclarationName &parentDeclName,
                                  const DeclContext *childCurrentDeclContext) {

    if (DeclContext *importedDeclContext =
                                m_Importer->ImportContext(declContextToImport)) {

      importedDeclContext->setHasExternalVisibleStorage(true);
      if (NamedDecl *importedNamedDecl = 
                            llvm::dyn_cast<NamedDecl>(importedDeclContext)) {
        SetExternalVisibleDeclsForName(childCurrentDeclContext,
                                       importedNamedDecl->getDeclName(),
                                       importedNamedDecl);
      }

      // Put the name of the DeclContext imported with the
      // DeclarationName coming from the parent, in  my map.
      m_ImportedDecls.emplace(childDeclName, parentDeclName);

      // And also put the declaration context I found from the parent Interpreter
      // in the map of the child Interpreter to have it for the future.
      m_ImportedDeclContexts.emplace(importedDeclContext, declContextToImport);
    }
  }

  bool ExternalInterpreterSource::Import(DeclContext::lookup_result lookup_result,
                                const DeclContext *childCurrentDeclContext,
                                DeclarationName &childDeclName,
                                const DeclarationName &parentDeclName) {



    for (NamedDecl* ND : lookup_result) {
      // Check if this Name we are looking for is
      // a DeclContext (for example a Namespace, function etc.).
      if (DeclContext *declContextToImport = llvm::dyn_cast<DeclContext>(ND)) {

        ImportDeclContext(declContextToImport, childDeclName,
                          parentDeclName, childCurrentDeclContext);

      }
      ImportDecl(ND, childDeclName, parentDeclName, childCurrentDeclContext);
    }
    return true;
  }

  ///\brief This is the one of the most important function of the class
  /// since from here initiates the lookup and import part of the missing
  /// Decl(s) (Contexts).
  ///
  bool ExternalInterpreterSource::FindExternalVisibleDeclsByName(
    const DeclContext *childCurrentDeclContext, DeclarationName childDeclName) {

    assert(childDeclName && "Child Decl name is empty");

    assert(childCurrentDeclContext->hasExternalVisibleStorage() &&
           "DeclContext has no visible decls in storage");

    // Search in the map of the stored Decl Contexts for this
    // Decl Context.
    auto IDeclContext = m_ImportedDeclContexts.find(childCurrentDeclContext);
    // If childCurrentDeclContext was found before and is already in the map,
    // then do the lookup using the stored pointer.
    if (IDeclContext == m_ImportedDeclContexts.end())
      return false;

    DeclContext *parentDC = IDeclContext->second;

    //Check if we have already found this declaration Name before
    DeclarationName parentDeclName;
    auto IDecl = m_ImportedDecls.find(childDeclName);
    if (IDecl != m_ImportedDecls.end()) {
      parentDeclName = IDecl->second;
    } else {
      // Get the identifier info from the parent interpreter
      // for this Name.
      const std::string name = childDeclName.getAsString();
      IdentifierTable &parentIdentifierTable =
                            m_ParentInterpreter->getCI()->getASTContext().Idents;
      IdentifierInfo &parentIdentifierInfo =
                            parentIdentifierTable.get(name);
      parentDeclName = DeclarationName(&parentIdentifierInfo);

      // Make sure then lookup name is right, this is an issue looking to import
      // a constructor, where lookup_result can hold the injected class name.
      // FIXME: Pre-filter childDeclName.getNameKind() and just drop import
      // of any unsupported types (i.e. CXXUsingDirective)
      const DeclarationName::NameKind ChildKind = childDeclName.getNameKind();
      if (parentDeclName.getNameKind() != ChildKind) {
        (void)parentDC->lookup(parentDeclName);
        const auto* DM = parentDC->getPrimaryContext()->getLookupPtr();
        assert(DM && "No lookup map");
        for (auto&& Entry : *DM) {
          const DeclarationName& DN = Entry.first;
          if (DN.getNameKind() == ChildKind) {
            if (DN.getAsString() == name) {
              parentDeclName = DN;
              break;
            }
          }
        }
      }
    }

    DeclContext::lookup_result lookup_result = parentDC->lookup(parentDeclName);

    // Check if we found this Name in the parent interpreter
    if (lookup_result.empty())
      return false;

    if (!Import(lookup_result, childCurrentDeclContext, childDeclName,
                parentDeclName))
      return false;

    // FIXME: The failure of this to work out of the box seems like a deeper
    // issue (in ASTImporter::ImportContext or
    // CXXRecordDecl::getVisibleConversionFunctions for example).

    // Constructing or importing a variable of type CXXRecordDecl.
    // Import the all constructors, conversion routines, and the destructor.

    const CXXRecordDecl* CXD = dyn_cast<CXXRecordDecl>(childCurrentDeclContext);
    if (!CXD && isa<VarDecl>(*lookup_result.begin())) {
      assert(lookup_result.size() == 1 && "More than one VarDecl?!");
      CXD = cast<VarDecl>(*lookup_result.begin())
                ->getType()
                ->getAsCXXRecordDecl();
      if (CXD)
        parentDC = cast<DeclContext>(const_cast<CXXRecordDecl*>(CXD));
    }

    if (!CXD)
      return true;

    ASTContext& AST = m_ChildInterpreter->getCI()->getASTContext();
    const auto CanonQT = CXD->getCanonicalDecl()
                             ->getTypeForDecl()
                             ->getCanonicalTypeUnqualified();
    const auto* DM = parentDC->getPrimaryContext()->getLookupPtr();
    assert(DM && "No lookup map");
    for (auto&& Entry : *DM) {
      const DeclarationName& ParentDN = Entry.first;
      const auto ParentKind = ParentDN.getNameKind();
      if (ParentKind < DeclarationName::CXXConstructorName ||
          ParentKind > DeclarationName::CXXConversionFunctionName)
        continue;

      if (m_ImportedDecls.find(childDeclName) == m_ImportedDecls.end())
        continue;

      DeclarationName ChildDN =
          AST.DeclarationNames.getCXXSpecialName(ParentDN.getNameKind(),
                                                 CanonQT);
      // FIXME: DeclContext::Import checks if the decl is a DeclContext.
      // Is that neccessary?
      DeclContext::lookup_result LR = parentDC->lookup(ParentDN);
      if (!LR.empty() &&
          !Import(LR, childCurrentDeclContext, ChildDN, ParentDN))
        return false;
    }
    return true;
  }

  ///\brief Make available to child all decls in parent's decl context
  /// that corresponds to child decl context.
  void ExternalInterpreterSource::completeVisibleDeclsMap(
                                const clang::DeclContext *childDeclContext) {
    assert (childDeclContext && "No child decl context!");

    if (!childDeclContext->hasExternalVisibleStorage())
      return;

    // Search in the map of the stored Decl Contexts for this
    // Decl Context.
    auto IDeclContext = m_ImportedDeclContexts.find(childDeclContext);
    // If childCurrentDeclContext was found before and is already in the map,
    // then do the lookup using the stored pointer.
    if (IDeclContext == m_ImportedDeclContexts.end()) return ;

    DeclContext *parentDeclContext = IDeclContext->second;

    // Filter the decls from the external source using the stem information
    // stored in Sema.
    StringRef filter =
      m_ChildInterpreter->getCI()->getPreprocessor().getCodeCompletionFilter();
    for (Decl* D : parentDeclContext->decls()) {
      if (NamedDecl* parentDecl = llvm::dyn_cast<NamedDecl>(D)) {
        DeclarationName childDeclName = parentDecl->getDeclName();
        if (auto II = childDeclName.getAsIdentifierInfo()) {
          StringRef name = II->getName();
          if (!name.empty() && name.startswith(filter))
            ImportDecl(parentDecl, childDeclName, childDeclName,
                       childDeclContext);
        }
      }
    }

    const_cast<DeclContext *>(childDeclContext)->
                                      setHasExternalVisibleStorage(false);
  }
} // end namespace cling
