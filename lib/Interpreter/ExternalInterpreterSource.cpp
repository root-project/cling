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

    void Imported(Decl *From, Decl *To) override {
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
    m_ImportedDeclContexts[childTUDeclContext] = parentTUDeclContext;

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
                                   DeclarationName &parentDeclName,
                                   const DeclContext *childCurrentDeclContext) {

    // Don't do the import if we have a Function Template or using decls. They
    // are not supported by clang.
    // FIXME: These are temporary checks and should be de-activated once clang
    // supports their import.
    if ((declToImport->isFunctionOrFunctionTemplate()
         && declToImport->isTemplateDecl()) || dyn_cast<UsingDecl>(declToImport)
         || dyn_cast<UsingShadowDecl>(declToImport)) {
#ifndef NDEBUG
      utils::DiagnosticsStore DS(
        m_Importer->getFromContext().getDiagnostics(), false, false, true);

      const Decl* To = llvm::cantFail(m_Importer->Import(declToImport));
      assert(To && "Import did not work!");
      assert((DS.empty() ||
              DS[0].getID() == clang::diag::err_unsupported_ast_node) &&
             "Import not supported!");
#endif
      return;
    }

    if (auto toOrErr = m_Importer->Import(declToImport)) {
      if (NamedDecl *importedNamedDecl = llvm::dyn_cast<NamedDecl>(*toOrErr)) {
        SetExternalVisibleDeclsForName(childCurrentDeclContext,
                                       importedNamedDecl->getDeclName(),
                                       importedNamedDecl);
      }
      // Put the name of the Decl imported with the
      // DeclarationName coming from the parent, in  my map.
      m_ImportedDecls[childDeclName] = parentDeclName;
    } else {
      logAllUnhandledErrors(toOrErr.takeError(), llvm::errs(),
                            "Error importing decl");
    }
  }

  void ExternalInterpreterSource::ImportDeclContext(
                                  DeclContext *declContextToImport,
                                  DeclarationName &childDeclName,
                                  DeclarationName &parentDeclName,
                                  const DeclContext *childCurrentDeclContext) {

    if (auto toOrErr = m_Importer->ImportContext(declContextToImport)) {

      DeclContext *importedDC = *toOrErr;
      importedDC->setHasExternalVisibleStorage(true);
      if (NamedDecl *importedND = llvm::dyn_cast<NamedDecl>(importedDC)) {
        SetExternalVisibleDeclsForName(childCurrentDeclContext,
                                       importedND->getDeclName(),
                                       importedND);
      }

      // Put the name of the DeclContext imported with the
      // DeclarationName coming from the parent, in  my map.
      m_ImportedDecls[childDeclName] = parentDeclName;

      // And also put the declaration context I found from the parent Interpreter
      // in the map of the child Interpreter to have it for the future.
      m_ImportedDeclContexts[importedDC] = declContextToImport;
    } else {
      logAllUnhandledErrors(toOrErr.takeError(), llvm::errs(),
                            "Error importing decl context");
    }
  }

  bool ExternalInterpreterSource::Import(DeclContext::lookup_result lookup_result,
                                const DeclContext *childCurrentDeclContext,
                                DeclarationName &childDeclName,
                                DeclarationName &parentDeclName) {



    for (DeclContext::lookup_iterator I = lookup_result.begin(),
          E = lookup_result.end(); I != E; ++I) {
      // Check if this Name we are looking for is
      // a DeclContext (for example a Namespace, function etc.).
      if (DeclContext *declContextToImport = llvm::dyn_cast<DeclContext>(*I)) {

        ImportDeclContext(declContextToImport, childDeclName,
                          parentDeclName, childCurrentDeclContext);

      }
      ImportDecl(*I, childDeclName, parentDeclName, childCurrentDeclContext);
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

    //Check if we have already found this declaration Name before
    DeclarationName parentDeclName;
    std::map<clang::DeclarationName,
             clang::DeclarationName>::iterator IDecl =
                                            m_ImportedDecls.find(childDeclName);
    if (IDecl != m_ImportedDecls.end()) {
      parentDeclName = IDecl->second;
    } else {
      // Get the identifier info from the parent interpreter
      // for this Name.
      std::string name = childDeclName.getAsString();
      IdentifierTable &parentIdentifierTable =
                            m_ParentInterpreter->getCI()->getASTContext().Idents;
      IdentifierInfo &parentIdentifierInfo =
                            parentIdentifierTable.get(name);
      parentDeclName = DeclarationName(&parentIdentifierInfo);
    }

    // Search in the map of the stored Decl Contexts for this
    // Decl Context.
    std::map<const clang::DeclContext *, clang::DeclContext *>::iterator
          IDeclContext = m_ImportedDeclContexts.find(childCurrentDeclContext);
    // If childCurrentDeclContext was found before and is already in the map,
    // then do the lookup using the stored pointer.
    if (IDeclContext == m_ImportedDeclContexts.end()) return false;

    DeclContext *parentDeclContext = IDeclContext->second;

    DeclContext::lookup_result lookup_result =
                                    parentDeclContext->lookup(parentDeclName);

    // Check if we found this Name in the parent interpreter
    if (!lookup_result.empty()) {
      if (Import(lookup_result,
                 childCurrentDeclContext, childDeclName, parentDeclName))
        return true;
    }

    return false;
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
    std::map<const clang::DeclContext *, clang::DeclContext *>::iterator
                  IDeclContext = m_ImportedDeclContexts.find(childDeclContext);
    // If childCurrentDeclContext was found before and is already in the map,
    // then do the lookup using the stored pointer.
    if (IDeclContext == m_ImportedDeclContexts.end()) return ;

    DeclContext *parentDeclContext = IDeclContext->second;

    // Filter the decls from the external source using the stem information
    // stored in Sema.
    StringRef filter =
      m_ChildInterpreter->getCI()->getPreprocessor().getCodeCompletionFilter();
    for (DeclContext::decl_iterator IDeclContext =
                                      parentDeclContext->decls_begin(),
                                    EDeclContext =
                                      parentDeclContext->decls_end();
                              IDeclContext != EDeclContext; ++IDeclContext) {
      if (NamedDecl* parentDecl = llvm::dyn_cast<NamedDecl>(*IDeclContext)) {
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
