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

#include "clang/AST/ASTContext.h"
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
                     bool MinimalImport, cling::ExternalInterpreterSource& source):
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
        m_Source.addToImportedDecls(toNamedDecl->getDeclName(),
                                    fromNamedDecl->getDeclName());
      }
      if (DeclContext *toDeclContext = llvm::dyn_cast<DeclContext>(To)) {
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
    m_ImportedDeclContexts[childTUDeclContext] = parentTUDeclContext;
  }

  ExternalInterpreterSource::~ExternalInterpreterSource() {}

  ClingASTImporter CreateClingASTImporter(ASTContext &toContext,
                                          ASTContext &fromContext,
                                          const Interpreter* child,
                                          const Interpreter* parent,
                                          ExternalInterpreterSource& source) {

    FileManager &childFM = child->getCI()->getFileManager();
    FileManager &parentFM = parent->getCI()->getFileManager();

    return ClingASTImporter(toContext, childFM, fromContext, parentFM,
                            /*MinimalImport : ON*/ true, source);
  }

  void ExternalInterpreterSource::ImportDecl(Decl *declToImport,
                                   ASTImporter &importer,
                                   DeclarationName &childDeclName,
                                   DeclarationName &parentDeclName,
                                   const DeclContext *childCurrentDeclContext) {

    // Don't do the import if we have a Function Template.
    // Not supported by clang.
    // FIXME: This is also a temporary check. Will be de-activated
    // once clang supports the import of function templates.
    if (declToImport->isFunctionOrFunctionTemplate() 
        && declToImport->isTemplateDecl())
      return;

    if (Decl *importedDecl = importer.Import(declToImport)) {
      if (NamedDecl *importedNamedDecl = llvm::dyn_cast<NamedDecl>(importedDecl)) {
        SetExternalVisibleDeclsForName(childCurrentDeclContext,
                                       importedNamedDecl->getDeclName(),
                                       importedNamedDecl);
      }
      // Put the name of the Decl imported with the
      // DeclarationName coming from the parent, in  my map.
      m_ImportedDecls[childDeclName] = parentDeclName;
    }
  }

  void ExternalInterpreterSource::ImportDeclContext(
                                  DeclContext *declContextToImport,
                                  ASTImporter &importer,
                                  DeclarationName &childDeclName,
                                  DeclarationName &parentDeclName,
                                  const DeclContext *childCurrentDeclContext) {

    if (DeclContext *importedDeclContext =
                                importer.ImportContext(declContextToImport)) {

      importedDeclContext->setHasExternalVisibleStorage(true);
      if (NamedDecl *importedNamedDecl = 
                            llvm::dyn_cast<NamedDecl>(importedDeclContext)) {
        SetExternalVisibleDeclsForName(childCurrentDeclContext,
                                       importedNamedDecl->getDeclName(),
                                       importedNamedDecl);
      }

      // Put the name of the DeclContext imported with the
      // DeclarationName coming from the parent, in  my map.
      m_ImportedDecls[childDeclName] = parentDeclName;

      // And also put the declaration context I found from the parent Interpreter
      // in the map of the child Interpreter to have it for the future.
      m_ImportedDeclContexts[importedDeclContext] = declContextToImport;
    }
  }

  bool ExternalInterpreterSource::Import(DeclContext::lookup_result lookup_result,
                                ASTContext &fromASTContext,
                                ASTContext &toASTContext,
                                const DeclContext *childCurrentDeclContext,
                                DeclarationName &childDeclName,
                                DeclarationName &parentDeclName) {

    // Cling's ASTImporter
    ClingASTImporter importer = CreateClingASTImporter(toASTContext,
                                                       fromASTContext,
                                                       m_ChildInterpreter,
                                                       m_ParentInterpreter,
                                                       *this);

    for (DeclContext::lookup_iterator I = lookup_result.begin(),
          E = lookup_result.end(); I != E; ++I) {
      // Check if this Name we are looking for is
      // a DeclContext (for example a Namespace, function etc.).
      if (DeclContext *declContextToImport = llvm::dyn_cast<DeclContext>(*I)) {

        ImportDeclContext(declContextToImport, importer, childDeclName,
                          parentDeclName, childCurrentDeclContext);

      }
      ImportDecl(*I, importer, childDeclName, parentDeclName,
                 childCurrentDeclContext);
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
      ASTContext &fromASTContext =
            Decl::castFromDeclContext(parentDeclContext)->getASTContext();
      ASTContext &toASTContext =
            Decl::castFromDeclContext(childCurrentDeclContext)->getASTContext();
      if (Import(lookup_result, fromASTContext, toASTContext,
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

    ASTContext &fromASTContext =
          Decl::castFromDeclContext(parentDeclContext)->getASTContext();
    ASTContext &toASTContext =
          Decl::castFromDeclContext(childDeclContext)->getASTContext();

    // Cling's ASTImporter
    ClingASTImporter importer = CreateClingASTImporter(toASTContext,
                                                       fromASTContext,
                                                       m_ChildInterpreter,
                                                       m_ParentInterpreter,
                                                       *this);

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
            ImportDecl(parentDecl, importer, childDeclName, childDeclName,
                       childDeclContext);
        }
      }
    }

    const_cast<DeclContext *>(childDeclContext)->
                                      setHasExternalVisibleStorage(false);
  }
} // end namespace cling
