//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vvasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "cling/Interpreter/InterpreterCallbacks.h"

#include "cling/Interpreter/Interpreter.h"

#include "clang/AST/ASTContext.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/MultiplexConsumer.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/CodeCompleteConsumer.h"
#include "clang/Sema/MultiplexExternalSemaSource.h"
#include "clang/Sema/Sema.h"
#include "clang/Serialization/ASTDeserializationListener.h"
#include "clang/Serialization/ASTReader.h"

using namespace clang;

namespace cling {

  ///\brief Translates 'interesting' for the interpreter
  /// ASTDeserializationListener events into interpreter callback.
  ///
  class InterpreterPPCallbacks : public PPCallbacks {
  private:
    cling::InterpreterCallbacks* m_Callbacks;
  public:
    InterpreterPPCallbacks(InterpreterCallbacks* C) : m_Callbacks(C) { }
    ~InterpreterPPCallbacks() { }

    virtual void InclusionDirective(clang::SourceLocation HashLoc,
                                    const clang::Token &IncludeTok,
                                    llvm::StringRef FileName,
                                    bool IsAngled,
                                    clang::CharSourceRange FilenameRange,
                                    const clang::FileEntry *File,
                                    llvm::StringRef SearchPath,
                                    llvm::StringRef RelativePath,
                                    const clang::Module *Imported) {
      m_Callbacks->InclusionDirective(HashLoc, IncludeTok, FileName,
                                      IsAngled, FilenameRange, File,
                                      SearchPath, RelativePath, Imported);
    }

    virtual bool FileNotFound(llvm::StringRef FileName,
                              llvm::SmallVectorImpl<char>& RecoveryPath) {
      if (m_Callbacks)
        return m_Callbacks->FileNotFound(FileName, RecoveryPath);

      // Returning true would mean that the preprocessor should try to recover.
      return false;
    }
  };

  ///\brief Translates 'interesting' for the interpreter
  /// ASTDeserializationListener events into interpreter callback.
  ///
  class InterpreterDeserializationListener : public ASTDeserializationListener {
  private:
    cling::InterpreterCallbacks* m_Callbacks;
  public:
    InterpreterDeserializationListener(InterpreterCallbacks* C)
      : m_Callbacks(C) {}

    virtual void DeclRead(serialization::DeclID, const Decl *D) {
      if (m_Callbacks)
        m_Callbacks->DeclDeserialized(D);
    }

    virtual void TypeRead(serialization::TypeIdx, QualType T) {
      if (m_Callbacks)
        m_Callbacks->TypeDeserialized(T.getTypePtr());
    }
  };

  /// \brief Wraps an ASTDeserializationListener in an ASTConsumer so that
  /// it can be used with a MultiplexConsumer.
  class DeserializationListenerWrapper : public ASTConsumer {
    ASTDeserializationListener* m_Listener;

  public:
    DeserializationListenerWrapper(ASTDeserializationListener* Listener)
        : m_Listener(Listener) {}
    ASTDeserializationListener* GetASTDeserializationListener() override {
      return m_Listener;
    }
  };

  /// \brief wraps an ExternalASTSource in an ExternalSemaSource. No functional
  /// difference between the original source and this wrapper intended.
  class ExternalASTSourceWrapper : public ExternalSemaSource {
    ExternalASTSource* m_Source;

  public:
    ExternalASTSourceWrapper(ExternalASTSource* Source) : m_Source(Source) {
      assert(m_Source && "Can't wrap nullptr ExternalASTSource");
    }

    virtual Decl* GetExternalDecl(uint32_t ID) override {
      return m_Source->GetExternalDecl(ID);
    }

    virtual Selector GetExternalSelector(uint32_t ID) override {
      return m_Source->GetExternalSelector(ID);
    }

    virtual uint32_t GetNumExternalSelectors() override {
      return m_Source->GetNumExternalSelectors();
    }

    virtual Stmt* GetExternalDeclStmt(uint64_t Offset) override {
      return m_Source->GetExternalDeclStmt(Offset);
    }

    virtual CXXCtorInitializer**
    GetExternalCXXCtorInitializers(uint64_t Offset) override {
      return m_Source->GetExternalCXXCtorInitializers(Offset);
    }

    virtual CXXBaseSpecifier*
    GetExternalCXXBaseSpecifiers(uint64_t Offset) override {
      return m_Source->GetExternalCXXBaseSpecifiers(Offset);
    }

    virtual void updateOutOfDateIdentifier(IdentifierInfo& II) override {
      m_Source->updateOutOfDateIdentifier(II);
    }

    virtual bool FindExternalVisibleDeclsByName(const DeclContext* DC,
                                                DeclarationName Name) override {
      return m_Source->FindExternalVisibleDeclsByName(DC, Name);
    }

    virtual void completeVisibleDeclsMap(const DeclContext* DC) override {
      m_Source->completeVisibleDeclsMap(DC);
    }

    virtual Module* getModule(unsigned ID) override {
      return m_Source->getModule(ID);
    }

    virtual llvm::Optional<ASTSourceDescriptor>
    getSourceDescriptor(unsigned ID) override {
      return m_Source->getSourceDescriptor(ID);
    }

    virtual ExtKind hasExternalDefinitions(const Decl* D) override {
      return m_Source->hasExternalDefinitions(D);
    }

    virtual void
    FindExternalLexicalDecls(const DeclContext* DC,
                             llvm::function_ref<bool(Decl::Kind)> IsKindWeWant,
                             SmallVectorImpl<Decl*>& Result) override {
      m_Source->FindExternalLexicalDecls(DC, IsKindWeWant, Result);
    }

    virtual void FindFileRegionDecls(FileID File, unsigned Offset,
                                     unsigned Length,
                                     SmallVectorImpl<Decl*>& Decls) override {
      m_Source->FindFileRegionDecls(File, Offset, Length, Decls);
    }

    virtual void CompleteRedeclChain(const Decl* D) override {
      m_Source->CompleteRedeclChain(D);
    }

    virtual void CompleteType(TagDecl* Tag) override {
      m_Source->CompleteType(Tag);
    }

    virtual void CompleteType(ObjCInterfaceDecl* Class) override {
      m_Source->CompleteType(Class);
    }

    virtual void ReadComments() override { m_Source->ReadComments(); }

    virtual void StartedDeserializing() override {
      m_Source->StartedDeserializing();
    }

    virtual void FinishedDeserializing() override {
      m_Source->FinishedDeserializing();
    }

    virtual void StartTranslationUnit(ASTConsumer* Consumer) override {
      m_Source->StartTranslationUnit(Consumer);
    }

    virtual void PrintStats() override { m_Source->PrintStats(); }

    virtual bool layoutRecordType(
        const RecordDecl* Record, uint64_t& Size, uint64_t& Alignment,
        llvm::DenseMap<const FieldDecl*, uint64_t>& FieldOffsets,
        llvm::DenseMap<const CXXRecordDecl*, CharUnits>& BaseOffsets,
        llvm::DenseMap<const CXXRecordDecl*, CharUnits>& VirtualBaseOffsets)
        override {
      return m_Source->layoutRecordType(Record, Size, Alignment, FieldOffsets,
                                        BaseOffsets, VirtualBaseOffsets);
    }
  };

  ///\brief Translates 'interesting' for the interpreter ExternalSemaSource
  /// events into interpreter callbacks.
  ///
  class InterpreterExternalSemaSource : public clang::ExternalSemaSource {
  protected:
    ///\brief The interpreter callback which are subscribed for the events.
    ///
    /// Usually the callbacks is the owner of the class and the interpreter owns
    /// the callbacks so they can't be out of sync. Eg we notifying the wrong
    /// callback class.
    ///
    InterpreterCallbacks* m_Callbacks; // we don't own it.

    Sema* m_Sema; // we don't own // FIXME: once we remove ForgetSema delete.

  public:
    InterpreterExternalSemaSource(InterpreterCallbacks* C)
      : m_Callbacks(C), m_Sema(0) {}

    ~InterpreterExternalSemaSource() {
      // FIXME: Another gross hack due to the missing multiplexing AST external
      // source see Interpreter::setCallbacks.
      if (m_Sema) {
        ASTContext& C = m_Sema->getASTContext();
        // tell the owning ptr to not delete it, the callbacks will delete it.
        if (C.ExternalSource.get() == this)
          C.ExternalSource.resetWithoutRelease();
      }
    }

    virtual void InitializeSema(Sema& S) {
      m_Sema = &S;
    }

    virtual void ForgetSema() {
      m_Sema = 0;
    }

    InterpreterCallbacks* getCallbacks() const { return m_Callbacks; }

    /// \brief Provides last resort lookup for failed unqualified lookups.
    ///
    /// This gets translated into InterpreterCallback's call.
    ///
    ///\param[out] R The recovered symbol.
    ///\param[in] S The scope in which the lookup failed.
    ///
    ///\returns true if a suitable declaration is found.
    ///
    virtual bool LookupUnqualified(clang::LookupResult& R, clang::Scope* S) {
      if (m_Callbacks) {
        return m_Callbacks->LookupObject(R, S);
      }

      return false;
    }

    virtual bool FindExternalVisibleDeclsByName(const clang::DeclContext* DC,
                                                clang::DeclarationName Name) {
      if (m_Callbacks)
        return m_Callbacks->LookupObject(DC, Name);

      return false;
    }

    // Silence warning virtual function was hidden.
    using ExternalASTSource::CompleteType;
    virtual void CompleteType(TagDecl* Tag) {
      if (m_Callbacks)
        m_Callbacks->LookupObject(Tag);
    }

    void UpdateWithNewDeclsFwd(const DeclContext *DC, DeclarationName Name,
                               llvm::ArrayRef<NamedDecl*> Decls) {
      SetExternalVisibleDeclsForName(DC, Name, Decls);
    }
  };

  InterpreterCallbacks::InterpreterCallbacks(Interpreter* interp,
                             bool enableExternalSemaSourceCallbacks/* = false*/,
                        bool enableDeserializationListenerCallbacks/* = false*/,
                                             bool enablePPCallbacks/* = false*/)
    : m_Interpreter(interp), m_ExternalSemaSource(0), m_PPCallbacks(0),
      m_IsRuntime(false) {
    Sema& SemaRef = interp->getSema();
    ASTReader* Reader = m_Interpreter->getCI()->getModuleManager().get();
    ExternalSemaSource* externalSemaSrc = SemaRef.getExternalSource();
    if (enableExternalSemaSourceCallbacks)
      if (!externalSemaSrc || externalSemaSrc == Reader) {
        // If the ExternalSemaSource is the PCH reader we still need to insert
        // our listener.
        m_ExternalSemaSource = new InterpreterExternalSemaSource(this);
        m_ExternalSemaSource->InitializeSema(SemaRef);
        m_Interpreter->getSema().addExternalSource(m_ExternalSemaSource);

        // Overwrite the ExternalASTSource.
        clang::ASTContext& Ctx = SemaRef.getASTContext();
        auto ExistingSource = Ctx.getExternalSource();
        // If we already have source, we need to create a multiplexer with the
        // existing source.
        if (ExistingSource) {
          // Make sure the context is not deleting the existing source.
          // FIXME: We should delete this, but looking at the other TODO's in
          // the destructor we can't easily free our callbacks from here...
          Ctx.ExternalSource.resetWithoutRelease();

          // Wrap the existing source in a wrapper so that it becomes an
          // external sema source. This way we can use the existing multiplexer
          // for this.
          auto wrapper = new ExternalASTSourceWrapper(ExistingSource);

          // Wrap both the existing source and our source. We give our own
          // source preference to the existing one.
          IntrusiveRefCntPtr<ExternalASTSource> S;
          S = new MultiplexExternalSemaSource(*m_ExternalSemaSource, *wrapper);

          Ctx.setExternalSource(S);
        } else {
          // We don't have an existing source, so just set our own source.
          Ctx.setExternalSource(m_ExternalSemaSource);
        }
    }

    if (enableDeserializationListenerCallbacks && Reader) {
      // Create a new deserialization listener.
      m_DeserializationListener.
        reset(new InterpreterDeserializationListener(this));

      // Wrap the deserialization listener in an MultiplexConsumer and then
      // combine it with the existing Consumer.
      // FIXME: Maybe it's better to make MultiplexASTDeserializationListener
      // public instead. See also: https://reviews.llvm.org/D37475
      std::unique_ptr<DeserializationListenerWrapper> wrapper(
          new DeserializationListenerWrapper(m_DeserializationListener.get()));

      std::vector<std::unique_ptr<ASTConsumer>> Consumers;
      Consumers.push_back(std::move(wrapper));
      Consumers.push_back(m_Interpreter->getCI()->takeASTConsumer());

      std::unique_ptr<clang::MultiplexConsumer> multiConsumer(
          new clang::MultiplexConsumer(std::move(Consumers)));
      m_Interpreter->getCI()->setASTConsumer(std::move(multiConsumer));
    }

    if (enablePPCallbacks) {
      Preprocessor& PP = m_Interpreter->getCI()->getPreprocessor();
      m_PPCallbacks = new InterpreterPPCallbacks(this);
      PP.addPPCallbacks(std::unique_ptr<InterpreterPPCallbacks>(m_PPCallbacks));
    }
  }

  // pin the vtable here
  InterpreterCallbacks::~InterpreterCallbacks() {
    // FIXME: we have to remove the external source at destruction time. Needs
    // further tweaks of the patch in clang. This will be done later once the
    // patch is in clang's mainline.
  }

  void InterpreterCallbacks::SetIsRuntime(bool val) {
    m_IsRuntime = val;
  }

  ExternalSemaSource*
  InterpreterCallbacks::getInterpreterExternalSemaSource() const {
    return m_ExternalSemaSource;
  }

  ASTDeserializationListener*
  InterpreterCallbacks::getInterpreterDeserializationListener() const {
    return m_DeserializationListener.get();
  }

  bool InterpreterCallbacks::FileNotFound(llvm::StringRef FileName,
                                    llvm::SmallVectorImpl<char>& RecoveryPath) {
    // Default implementation is no op.
    return false;
  }

  bool InterpreterCallbacks::LookupObject(LookupResult&, Scope*) {
    // Default implementation is no op.
    return false;
  }

  bool InterpreterCallbacks::LookupObject(const DeclContext*, DeclarationName) {
    // Default implementation is no op.
    return false;
  }

  bool InterpreterCallbacks::LookupObject(TagDecl*) {
    // Default implementation is no op.
    return false;
  }

  void InterpreterCallbacks::UpdateWithNewDecls(const DeclContext *DC,
                                                DeclarationName Name,
                                             llvm::ArrayRef<NamedDecl*> Decls) {
    if (m_ExternalSemaSource)
      m_ExternalSemaSource->UpdateWithNewDeclsFwd(DC, Name, Decls);
  }
} // end namespace cling

// TODO: Make the build system in the testsuite aware how to build that class
// and extract it out there again.
#include "DynamicLookup.h"
#include "cling/Utils/AST.h"

#include "clang/AST/ASTContext.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Scope.h"
namespace cling {
namespace test {
  TestProxy* Tester = 0;

  extern "C" int printf(const char* fmt, ...);
  TestProxy::TestProxy(){}
  int TestProxy::Draw(){ return 12; }
  const char* TestProxy::getVersion(){ return "Interpreter.cpp"; }

  int TestProxy::Add10(int num) { return num + 10;}

  int TestProxy::Add(int a, int b) {
    return a + b;
  }

  void TestProxy::PrintString(std::string s) { printf("%s\n", s.c_str()); }

  bool TestProxy::PrintArray(int a[], size_t size) {
    for (unsigned i = 0; i < size; ++i)
      printf("%i", a[i]);

    printf("%s", "\n");

    return true;
  }

  void TestProxy::PrintArray(float a[][5], size_t size) {
    for (unsigned i = 0; i < size; ++i)
      for (unsigned j = 0; j < 5; ++j)
        printf("%i", (int)a[i][j]);

    printf("%s", "\n");
  }

  void TestProxy::PrintArray(int a[][4][5], size_t size) {
    for (unsigned i = 0; i < size; ++i)
      for (unsigned j = 0; j < 4; ++j)
        for (unsigned k = 0; k < 5; ++k)
          printf("%i", a[i][j][k]);

    printf("%s", "\n");
  }

  SymbolResolverCallback::SymbolResolverCallback(Interpreter* interp,
                                                 bool resolve)
    : InterpreterCallbacks(interp), m_Resolve(resolve), m_TesterDecl(0) {
    m_Interpreter->process("cling::test::Tester = new cling::test::TestProxy();");
  }

  SymbolResolverCallback::~SymbolResolverCallback() { }

  bool SymbolResolverCallback::LookupObject(LookupResult& R, Scope* S) {
    if (!ShouldResolveAtRuntime(R, S))
      return false;

    if (m_IsRuntime) {
      // We are currently parsing an EvaluateT() expression
      if (!m_Resolve)
        return false;

      // Only for demo resolve all unknown objects to cling::test::Tester
      if (!m_TesterDecl) {
        clang::Sema& SemaR = m_Interpreter->getSema();
        clang::NamespaceDecl* NSD = utils::Lookup::Namespace(&SemaR, "cling");
        NSD = utils::Lookup::Namespace(&SemaR, "test", NSD);
        m_TesterDecl = utils::Lookup::Named(&SemaR, "Tester", NSD);
      }
      assert (m_TesterDecl && "Tester not found!");
      R.addDecl(m_TesterDecl);
      return true; // Tell clang to continue.
    }

    // We are currently NOT parsing an EvaluateT() expression.
    // Escape the expression into an EvaluateT() expression.
    ASTContext& C = R.getSema().getASTContext();
    DeclContext* DC = 0;
    // For DeclContext-less scopes like if (dyn_expr) {}
    while (!DC) {
      DC = static_cast<DeclContext*>(S->getEntity());
      S = S->getParent();
    }

    // DynamicLookup only happens inside topmost functions:
    clang::DeclContext* TopmostDC = DC;
    while (!isa<TranslationUnitDecl>(TopmostDC->getParent())) {
      TopmostDC = TopmostDC->getParent();
    }
    FunctionDecl* TopmostFunc = dyn_cast<FunctionDecl>(TopmostDC);
    if (!TopmostFunc)
       return false;

    DeclarationName Name = R.getLookupName();
    IdentifierInfo* II = Name.getAsIdentifierInfo();
    SourceLocation Loc = R.getNameLoc();
    VarDecl* Res = VarDecl::Create(C, DC, Loc, Loc, II, C.DependentTy,
        /*TypeSourceInfo*/0, SC_None);

    // Annotate the decl to give a hint in cling. FIXME: Current implementation
    // is a gross hack, because TClingCallbacks shouldn't know about
    // EvaluateTSynthesizer at all!
    SourceRange invalidRange;
    TopmostFunc->addAttr(new (C) AnnotateAttr(invalidRange, C,
                                              "__ResolveAtRuntime", 0));
    R.addDecl(Res);
    DC->addDecl(Res);
    // Say that we can handle the situation. Clang should try to recover
    return true;
  }

  bool SymbolResolverCallback::ShouldResolveAtRuntime(LookupResult& R,
                                                      Scope* S) {
    if (R.getLookupKind() != Sema::LookupOrdinaryName)
      return false;

    if (R.isForRedeclaration())
      return false;

    if (!R.empty())
      return false;

    // FIXME: Figure out better way to handle:
    // C++ [basic.lookup.classref]p1:
    //   In a class member access expression (5.2.5), if the . or -> token is
    //   immediately followed by an identifier followed by a <, the
    //   identifier must be looked up to determine whether the < is the
    //   beginning of a template argument list (14.2) or a less-than operator.
    //   The identifier is first looked up in the class of the object
    //   expression. If the identifier is not found, it is then looked up in
    //   the context of the entire postfix-expression and shall name a class
    //   or function template.
    //
    // We want to ignore object(.|->)member<template>
    if (R.getSema().PP.LookAhead(0).getKind() == tok::less)
      // TODO: check for . or -> in the cached token stream
      return false;

    for (Scope* DepScope = S; DepScope; DepScope = DepScope->getParent()) {
      if (DeclContext* Ctx = static_cast<DeclContext*>(DepScope->getEntity())) {
        if (!Ctx->isDependentContext())
          // For now we support only the prompt.
          if (isa<FunctionDecl>(Ctx))
            return true;
      }
    }

    return false;
  }

} // end test
} // end cling
