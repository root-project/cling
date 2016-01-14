#include "ASTImportSource.h"
#include "cling/Interpreter/Interpreter.h"

using namespace clang;

namespace cling {

    ASTImportSource::ASTImportSource(cling::Interpreter *parent_interpreter,
    cling::Interpreter *child_interpreter) :
    m_parent_Interp(parent_interpreter), m_child_Interp(child_interpreter) {

      clang::DeclContext *parentTUDeclContext =
        clang::TranslationUnitDecl::castToDeclContext(
          m_parent_Interp->getCI()->getASTContext().getTranslationUnitDecl());

      clang::DeclContext *childTUDeclContext =
        clang::TranslationUnitDecl::castToDeclContext(
          m_child_Interp->getCI()->getASTContext().getTranslationUnitDecl());

      // Also keep in the map of Decl Contexts the Translation Unit Decl Context
      m_DeclContexts_map[childTUDeclContext] = parentTUDeclContext;
    }

    void ASTImportSource::ImportDecl(Decl *declToImport,
                                     ASTImporter &importer,
                                     DeclarationName &childDeclName,
                                     DeclarationName &parentDeclName,
                                     const DeclContext *childCurrentDeclContext) {

      // Don't do the import if we have a Function Template.
      // Not supported by clang.
      // FIXME: This is also a temporary check. Will be de-activated
      // once clang supports the import of function templates.
      if (declToImport->isFunctionOrFunctionTemplate() && declToImport->isTemplateDecl())
        return;

      if (Decl *importedDecl = importer.Import(declToImport)) {
        if (NamedDecl *importedNamedDecl = llvm::dyn_cast<NamedDecl>(importedDecl)) {
          std::vector < NamedDecl * > declVector{importedNamedDecl};
          llvm::ArrayRef < NamedDecl * > FoundDecls(declVector);
          SetExternalVisibleDeclsForName(childCurrentDeclContext,
                                         importedNamedDecl->getDeclName(),
                                         FoundDecls);
        }
        // Put the name of the Decl imported with the
        // DeclarationName coming from the parent, in  my map.
        m_DeclName_map[childDeclName] = parentDeclName;
      }
    }

    void ASTImportSource::ImportDeclContext(DeclContext *declContextToImport,
                                            ASTImporter &importer,
                                            DeclarationName &childDeclName,
                                            DeclarationName &parentDeclName,
                                            const DeclContext *childCurrentDeclContext) {

      if (DeclContext *importedDeclContext = importer.ImportContext(declContextToImport)) {

        importedDeclContext->setHasExternalVisibleStorage(true);

        if (NamedDecl *importedNamedDecl = llvm::dyn_cast<NamedDecl>(importedDeclContext)) {
          std::vector < NamedDecl * > declVector{importedNamedDecl};
          llvm::ArrayRef < NamedDecl * > FoundDecls(declVector);
          SetExternalVisibleDeclsForName(childCurrentDeclContext,
                                         importedNamedDecl->getDeclName(),
                                         FoundDecls);
        }
        // Put the name of the DeclContext imported with the
        // DeclarationName coming from the parent, in  my map.
        m_DeclName_map[childDeclName] = parentDeclName;

        // And also put the declaration context I found from the parent Interpreter
        // in the map of the child Interpreter to have it for the future.
        m_DeclContexts_map[importedDeclContext] = declContextToImport;
      }
    }

    bool ASTImportSource::Import(DeclContext::lookup_result lookup_result,
                                 ASTContext &from_ASTContext,
                                 ASTContext &to_ASTContext,
                                 const DeclContext *childCurrentDeclContext,
                                 DeclarationName &childDeclName,
                                 DeclarationName &parentDeclName) {

      // Prepare to import the Decl(Context)  we found in the
      // child interpreter by getting the file managers from
      // each interpreter.
      FileManager &child_FM = m_child_Interp->getCI()->getFileManager();
      FileManager &parent_FM = m_parent_Interp->getCI()->getFileManager();

      // Clang's ASTImporter
      ASTImporter importer(to_ASTContext, child_FM,
                           from_ASTContext, parent_FM,
                           /*MinimalImport : ON*/ true);

      for (DeclContext::lookup_iterator I = lookup_result.begin(),
             E = lookup_result.end();
             I != E; ++I) {
        // Check if this Name we are looking for is
        // a DeclContext (for example a Namespace, function etc.).
        if (DeclContext *declContextToImport = llvm::dyn_cast<DeclContext>(*I)) {

          ImportDeclContext(declContextToImport, importer, childDeclName,
                            parentDeclName, childCurrentDeclContext);

        } else if (Decl *declToImport = llvm::dyn_cast<Decl>(*I)) {

          // else it is a Decl
          ImportDecl(declToImport, importer, childDeclName,
                     parentDeclName, childCurrentDeclContext);
        }
      }
      return true;
    }

    ///\brief This is the most important function of the class ASTImportSource
    /// since from here initiates the lookup and import part of the missing
    /// Decl(s) (Contexts).
    ///
    bool ASTImportSource::FindExternalVisibleDeclsByName(
      const DeclContext *childCurrentDeclContext, DeclarationName childDeclName) {

      assert(childCurrentDeclContext->hasExternalVisibleStorage() &&
             "DeclContext has no visible decls in storage");

      //Check if we have already found this declaration Name before
      DeclarationName parentDeclName;
      std::map<clang::DeclarationName,
        clang::DeclarationName>::iterator II = m_DeclName_map.find(childDeclName);
      if (II != m_DeclName_map.end()) {
        parentDeclName = II->second;
      } else {
        // Get the identifier info from the parent interpreter
        // for this Name.
        llvm::StringRef name(childDeclName.getAsString());
        IdentifierTable &parentIdentifierTable =
          m_parent_Interp->getCI()->getASTContext().Idents;
        IdentifierInfo &parentIdentifierInfo = parentIdentifierTable.get(name);
        DeclarationName parentDeclNameTemp(&parentIdentifierInfo);
        parentDeclName = parentDeclNameTemp;
      }

      // Search in the map of the stored Decl Contexts for this
      // Decl Context.
      std::map<const clang::DeclContext *, clang::DeclContext *>::iterator I;
      if ((I = m_DeclContexts_map.find(childCurrentDeclContext))
           != m_DeclContexts_map.end()) {
        // If childCurrentDeclContext was found before and is already in the map,
        // then do the lookup using the stored pointer.
        DeclContext *parentDeclContext = I->second;

        Decl *fromDeclContext = Decl::castFromDeclContext(parentDeclContext);
        ASTContext &from_ASTContext = fromDeclContext->getASTContext();

        Decl *toDeclContext = Decl::castFromDeclContext(childCurrentDeclContext);
        ASTContext &to_ASTContext = toDeclContext->getASTContext();

        DeclContext::lookup_result lookup_result =
          parentDeclContext->lookup(parentDeclName);

        // Check if we found this Name in the parent interpreter
        if (!lookup_result.empty()) {
          // Do the import
          if (Import(lookup_result, from_ASTContext, to_ASTContext,
                     childCurrentDeclContext, childDeclName, parentDeclName))
            return true;
        }
      }
      return false;
    }
} // end namespace cling
