#include "ForwardDeclPrinter.h"

#include "cling/Interpreter/DynamicLibraryManager.h"
#include "cling/Interpreter/Transaction.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Type.h"
#include "clang/Lex/Preprocessor.h"

#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

namespace cling {

  using namespace clang;


  ForwardDeclPrinter::ForwardDeclPrinter(llvm::raw_ostream& OutS,
                                         llvm::raw_ostream& LogS,
                                         SourceManager& SM,
                                         const Transaction& T,
                                         unsigned Indentation,
                                         bool printMacros)
    : m_Policy(clang::PrintingPolicy(clang::LangOptions())), m_Log(LogS),
      m_Indentation(Indentation), m_SMgr(SM), m_SkipFlag(false) {
    m_PrintInstantiation = false;
    m_Policy.SuppressTagKeyword = true;

    m_Policy.Bool = true; // Avoid printing _Bool instead of bool

    m_StreamStack.push(&OutS);

    m_SkipCounter = 0;
    m_TotalDecls = 0;

    // Suppress some unfixable warnings.
    // TODO: Find proper fix for these issues
    Out() << "#pragma clang diagnostic ignored \"-Wkeyword-compat\"" << "\n";
    Out() << "#pragma clang diagnostic ignored \"-Wignored-attributes\"" <<"\n";
    Out() << "#pragma clang diagnostic ignored \"-Wreturn-type-c-linkage\"" <<"\n";
    // Inject a special marker:
    Out() << "extern int __Cling_Autoloading_Map;\n";

    std::vector<std::string> macrodefs;
    if (printMacros) {
      for (auto mit = T.macros_begin(); mit != T.macros_end(); ++mit) {
        Transaction::MacroDirectiveInfo macro = *mit;
        if (macro.m_MD->getKind() == MacroDirective::MD_Define) {
          const MacroInfo* MI = macro.m_MD->getMacroInfo();
          if (MI ->getNumTokens() > 1)
            //FIXME: We can not display function like macros yet
            continue;
          Out() << "#define " << macro.m_II->getName() << ' ';
          for (unsigned i = 0, e = MI->getNumTokens(); i != e; ++i) {
            const Token &Tok = MI->getReplacementToken(i);
            Out() << Tok.getName() << ' ';
            macrodefs.push_back(macro.m_II->getName());
          }
          Out() << '\n';
        }
      }
    }

    for(auto dcit = T.decls_begin(); dcit != T.decls_end(); ++dcit) {
      const Transaction::DelayCallInfo& dci = *dcit;
      if (dci.m_DGR.isNull()) {
          break;
      }
      if (dci.m_Call == Transaction::kCCIHandleTopLevelDecl) {
        for (auto dit = dci.m_DGR.begin(); dit != dci.m_DGR.end(); ++dit) {
//          llvm::StringRef filename = m_SMgr.getFilename
//                            ((*dit)->getSourceRange().getBegin());
//#ifdef _POSIX_C_SOURCE
//          //Workaround for differnt expansion of macros to typedefs
//          if (filename.endswith("sys/types.h"))
//            continue;
//#endif
          //This may indicate a bug in cling.
          //This condition should ideally never be triggered
          //But is needed in case of generating fwd decls for
          // c++ <future> header.
          if (!(*dit)->getDeclContext()->isTranslationUnit())
            continue;

          Visit(*dit);
          printSemiColon();
        }

      }
    }
    if (printMacros) {
      for (auto m : macrodefs) {
        Out() << "#undef " << m << "\n";
      }
    }

  }

  ForwardDeclPrinter::ForwardDeclPrinter(llvm::raw_ostream& OutS,
                                         llvm::raw_ostream& LogS,
                                         clang::SourceManager& SM,
                                         const clang::PrintingPolicy& P,
                                         unsigned Indentation)
    : m_Policy(clang::PrintingPolicy(clang::LangOptions())), m_Log(LogS),
      m_Indentation(Indentation), m_SMgr(SM), m_SkipFlag(false) {
    m_PrintInstantiation = false;
    m_Policy.SuppressTagKeyword = true;
    m_StreamStack.push(&OutS);
  }

  llvm::raw_ostream& ForwardDeclPrinter::Out() {
    return *m_StreamStack.top();
  }
  llvm::raw_ostream& ForwardDeclPrinter::Log() {
    return m_Log;
  }


  void ForwardDeclPrinter::printDeclType(QualType T, StringRef DeclName, bool Pack) {
    // Normally, a PackExpansionType is written as T[3]... (for instance, as a
    // template argument), but if it is the type of a declaration, the ellipsis
    // is placed before the name being declared.
    if (auto *PET = T->getAs<PackExpansionType>()) {
      Pack = true;
      T = PET->getPattern();
    }
    T.print(Out(), m_Policy, (Pack ? "..." : "") + DeclName);
  }

  llvm::raw_ostream& ForwardDeclPrinter::Indent(unsigned Indentation) {
    for (unsigned i = 0; i != Indentation; ++i)
      Out() << "  ";
    return Out();
  }

  void ForwardDeclPrinter::prettyPrintAttributes(Decl *D, std::string extra) {
    if (D->getSourceRange().isInvalid())
      return;

    if (D->hasAttrs() && ! isa<FunctionDecl>(D)) {
      AttrVec &Attrs = D->getAttrs();
      for (AttrVec::const_iterator i=Attrs.begin(), e=Attrs.end(); i != e; ++i) {
        Attr *A = *i;
        if (!A->isImplicit() && !A->isInherited() && A->getKind() != attr::Kind::Final){
          A->printPretty(Out(), m_Policy);
        }
      }
    }

    SourceLocation spellingLoc = m_SMgr.getSpellingLoc(D->getLocStart());
    // Walk up the include chain.
    PresumedLoc PLoc = m_SMgr.getPresumedLoc(spellingLoc);
    llvm::SmallVector<PresumedLoc, 16> PLocs;
    while (true) {
      if (!m_SMgr.getPresumedLoc(PLoc.getIncludeLoc()).isValid())
        break;
      PLocs.push_back(PLoc);
      PLoc = m_SMgr.getPresumedLoc(PLoc.getIncludeLoc());
    }
    std::string file = PLocs[PLocs.size() -1].getFilename();
//    assert ( file.length() != 0 && "Filename Should not be blank");
    Out() << " __attribute__((annotate(\""
          << file;
    if (!extra.empty())
      Out() << " " << extra;
    Out() << "\"))) ";
  }


    //----------------------------------------------------------------------------
    // Common C declarations
    //----------------------------------------------------------------------------


  void ForwardDeclPrinter::VisitTranslationUnitDecl(TranslationUnitDecl *D) {
//    VisitDeclContext(D, false);
    for (auto it = D->decls_begin(); it != D->decls_end(); ++it) {
      Visit(*it);
      printSemiColon();
    }
  }

  void ForwardDeclPrinter::VisitTypedefDecl(TypedefDecl *D) {

    if (shouldSkip(D)) {
      skipCurrentDecl();
      return;
    }

    if (!m_Policy.SuppressSpecifiers) {
      Out() << "typedef ";

    if (D->isModulePrivate())
      Out() << "__module_private__ ";
    }
    QualType q = D->getTypeSourceInfo()->getType();
    q.removeLocalRestrict();
    q.print(Out(), m_Policy, D->getName());
    q.addRestrict();
    prettyPrintAttributes(D);
//      Indent() << ";\n";
    skipCurrentDecl(false);
  }

  void ForwardDeclPrinter::VisitTypeAliasDecl(TypeAliasDecl *D) {
      /*FIXME: Ugly Hack*/
//      if(!D->getLexicalDeclContext()->isNamespace()
//              && !D->getLexicalDeclContext()->isFileContext())
//          return;
    Out() << "using " << *D;
    prettyPrintAttributes(D);
    Out() << " = " << D->getTypeSourceInfo()->getType().getAsString(m_Policy);
//      Indent() << ";\n";
    skipCurrentDecl(false);
  }

  void ForwardDeclPrinter::VisitEnumDecl(EnumDecl *D) {
    if (shouldSkip(D)) {
      skipCurrentDecl();
      return;
    }

    if (!m_Policy.SuppressSpecifiers && D->isModulePrivate())
      Out() << "__module_private__ ";
    Out() << "enum ";
    prettyPrintAttributes(D,std::to_string(D->isFixed()));
    if (D->isScoped()) {
      if (D->isScopedUsingClassTag())
        Out() << "class ";
      else
        Out() << "struct ";
    }
    Out() << *D;

//      if (D->isFixed())
    Out() << " : " << D->getIntegerType().stream(m_Policy);

      if (D->isCompleteDefinition()) {
        for (auto eit = D->decls_begin(), end = D->decls_end() ;
                eit != end; ++ eit ){
          if (EnumConstantDecl* ecd = dyn_cast<EnumConstantDecl>(*eit)){
            VisitEnumConstantDecl(ecd);
          }
        }
      }
    skipCurrentDecl(false);
  }

  void ForwardDeclPrinter::VisitRecordDecl(RecordDecl *D) {
    if (!m_Policy.SuppressSpecifiers && D->isModulePrivate())
      Out() << "__module_private__ ";
    Out() << D->getKindName();
    prettyPrintAttributes(D);
    if (D->getIdentifier())
      Out() << ' ' << *D;

//    if (D->isCompleteDefinition()) {
//      Out << " {\n";
//      VisitDeclContext(D);
//      Indent() << "}";
//    }
    skipCurrentDecl(false);
  }

  void ForwardDeclPrinter::VisitEnumConstantDecl(EnumConstantDecl *D) {
    m_IncompatibleNames.insert(D->getName());
    //Because enums are forward declared.
  }

  void ForwardDeclPrinter::VisitFunctionDecl(FunctionDecl *D) {
    if (shouldSkip(D)) {
      if (D->getDeclName().isIdentifier())
        m_IncompatibleNames.insert(D->getName());
      skipCurrentDecl();
      return;
    }

    bool hasTrailingReturn = false;

    CXXConstructorDecl *CDecl = dyn_cast<CXXConstructorDecl>(D);
    CXXConversionDecl *ConversionDecl = dyn_cast<CXXConversionDecl>(D);

    if (!m_Policy.SuppressSpecifiers) {
      switch (D->getStorageClass()) {
      case SC_None: break;
      case SC_Extern: Out() << "extern "; break;
      case SC_Static: Out() << "static "; break;
      case SC_PrivateExtern: Out() << "__private_extern__ "; break;
      case SC_Auto: case SC_Register: case SC_OpenCLWorkGroupLocal:
        llvm_unreachable("invalid for functions");
      }

      if (D->isInlineSpecified())  Out() << "inline ";
      if (D->isVirtualAsWritten()) Out() << "virtual ";
      if (D->isModulePrivate())    Out() << "__module_private__ ";
      if (D->isConstexpr() && !D->isExplicitlyDefaulted())
        Out() << "constexpr ";
      if ((CDecl && CDecl->isExplicitSpecified()) ||
          (ConversionDecl && ConversionDecl->isExplicit()))
        Out() << "explicit ";
    }

    PrintingPolicy SubPolicy(m_Policy);
    SubPolicy.SuppressSpecifiers = false;
    std::string Proto = D->getNameInfo().getAsString();
    QualType Ty = D->getType();
    while (const ParenType *PT = dyn_cast<ParenType>(Ty)) {
      Proto = '(' + Proto + ')';
      Ty = PT->getInnerType();
    }

    if (const FunctionType *AFT = Ty->getAs<FunctionType>()) {
      const FunctionProtoType *FT = 0;
      if (D->hasWrittenPrototype())
        FT = dyn_cast<FunctionProtoType>(AFT);

      Proto += "(";
      if (FT) {
        llvm::raw_string_ostream POut(Proto);
        ForwardDeclPrinter ParamPrinter(POut, m_Log ,m_SMgr, SubPolicy, m_Indentation);
        for (unsigned i = 0, e = D->getNumParams(); i != e; ++i) {
          if (i) POut << ", ";
          ParamPrinter.VisitParmVarDecl(D->getParamDecl(i));
        }

        if (FT->isVariadic()) {
          if (D->getNumParams()) POut << ", ";
          POut << "...";
        }
      }
      else if (D->doesThisDeclarationHaveABody() && !D->hasPrototype()) {
        for (unsigned i = 0, e = D->getNumParams(); i != e; ++i) {
          if (i)
            Proto += ", ";
          Proto += D->getParamDecl(i)->getNameAsString();
        }
      }

      Proto += ")";

      if (FT) {
        if (FT->isConst())
          Proto += " const";
        if (FT->isVolatile())
          Proto += " volatile";
        if (FT->isRestrict())
          Proto += " __restrict";

        switch (FT->getRefQualifier()) {
        case RQ_None:
          break;
        case RQ_LValue:
          Proto += " &";
          break;
        case RQ_RValue:
          Proto += " &&";
          break;
        }
      }

      if (FT && FT->hasDynamicExceptionSpec()) {
        Proto += " throw(";
        if (FT->getExceptionSpecType() == EST_MSAny)
          Proto += "...";
        else
          for (unsigned I = 0, N = FT->getNumExceptions(); I != N; ++I) {
            if (I)
              Proto += ", ";

            Proto += FT->getExceptionType(I).getAsString(SubPolicy);
          }
        Proto += ")";
      } else if (FT && isNoexceptExceptionSpec(FT->getExceptionSpecType())) {
        Proto += " noexcept";
        if (FT->getExceptionSpecType() == EST_ComputedNoexcept) {
          Proto += "(";
          llvm::raw_string_ostream EOut(Proto);
          FT->getNoexceptExpr()->printPretty(EOut, 0, SubPolicy,
                                             m_Indentation);
          EOut.flush();
          //Proto += EOut.str()
          //Commented out to fix swap bug, no idea why this was here
          //Print was already being called earlier above
          Proto += ")";
        }
      }

      if (CDecl) {
        bool HasInitializerList = false;
        for (CXXConstructorDecl::init_const_iterator B = CDecl->init_begin(),
               E = CDecl->init_end();
             B != E; ++B) {
          CXXCtorInitializer *BMInitializer = (*B);
          if (BMInitializer->isInClassMemberInitializer())
            continue;

          if (!HasInitializerList) {
            Proto += " : ";
            Out() << Proto;
            Proto.clear();
            HasInitializerList = true;
          } else
            Out() << ", ";

          if (BMInitializer->isAnyMemberInitializer()) {
            FieldDecl *FD = BMInitializer->getAnyMember();
            Out() << *FD;
          } else {
            Out() << QualType(BMInitializer->getBaseClass(), 0).getAsString(m_Policy);
          }

          Out() << "(";
          if (!BMInitializer->getInit()) {
            // Nothing to print
          }
          else {
            Expr *Init = BMInitializer->getInit();
            if (ExprWithCleanups *Tmp = dyn_cast<ExprWithCleanups>(Init))
              Init = Tmp->getSubExpr();

            Init = Init->IgnoreParens();

            Expr *SimpleInit = 0;
            Expr **Args = 0;
            unsigned NumArgs = 0;
            if (ParenListExpr *ParenList = dyn_cast<ParenListExpr>(Init)) {
              Args = ParenList->getExprs();
              NumArgs = ParenList->getNumExprs();
            } else if (CXXConstructExpr *Construct
                       = dyn_cast<CXXConstructExpr>(Init)) {
              Args = Construct->getArgs();
              NumArgs = Construct->getNumArgs();
            } else
              SimpleInit = Init;

            if (SimpleInit)
              SimpleInit->printPretty(Out(), 0, m_Policy, m_Indentation);
            else {
              for (unsigned I = 0; I != NumArgs; ++I) {
                if (isa<CXXDefaultArgExpr>(Args[I]))
                  break;

                if (I)
                  Out() << ", ";
                Args[I]->printPretty(Out(), 0, m_Policy, m_Indentation);
              }
            }
          }
          Out() << ")";
          if (BMInitializer->isPackExpansion())
            Out() << "...";
        }
      }
      else if (!ConversionDecl && !isa<CXXDestructorDecl>(D)) {
        if (FT && FT->hasTrailingReturn()) {
          Out() << "auto " << Proto << " -> ";
          Proto.clear();
          hasTrailingReturn = true;
        }
        AFT->getReturnType().print(Out(), m_Policy, Proto);
        Proto.clear();
      }
      Out() << Proto;
    }
    else {
      Ty.print(Out(), m_Policy, Proto);
    }
    if (!hasTrailingReturn)
      prettyPrintAttributes(D);

    if (D->isPure())
      Out() << " = 0";
    else if (D->isDeletedAsWritten())
      Out() << " = delete";
    else if (D->isExplicitlyDefaulted())
      Out() << " = default";
    else if (D->doesThisDeclarationHaveABody() && !m_Policy.TerseOutput) {
      if (!D->hasPrototype() && D->getNumParams()) {
        // This is a K&R function definition, so we need to print the
        // parameters.
        Out() << '\n';
        ForwardDeclPrinter ParamPrinter(Out(), m_Log,m_SMgr, SubPolicy,
                                        m_Indentation);
        m_Indentation += m_Policy.Indentation;
        for (unsigned i = 0, e = D->getNumParams(); i != e; ++i) {
          Indent();
          ParamPrinter.VisitParmVarDecl(D->getParamDecl(i));
          Out() << ";\n";
        }
        m_Indentation -= m_Policy.Indentation;
      } else
        Out() << ' ';

      //    D->getBody()->printPretty(Out, 0, SubPolicy, Indentation);

    }
    skipCurrentDecl(false);
  }

  void ForwardDeclPrinter::VisitFriendDecl(FriendDecl *D) {
     skipCurrentDecl();
  }

  void ForwardDeclPrinter::VisitFieldDecl(FieldDecl *D) {
    if (!m_Policy.SuppressSpecifiers && D->isMutable())
      Out() << "mutable ";
    if (!m_Policy.SuppressSpecifiers && D->isModulePrivate())
      Out() << "__module_private__ ";
    Out() << D->getASTContext().getUnqualifiedObjCPointerType(D->getType()).
      stream(m_Policy, D->getName());

    if (D->isBitField()) {
      Out() << " : ";
      D->getBitWidth()->printPretty(Out(), 0, m_Policy, m_Indentation);
    }

    Expr *Init = D->getInClassInitializer();
    if (!m_Policy.SuppressInitializers && Init) {
      if (D->getInClassInitStyle() == ICIS_ListInit)
        Out() << " ";
      else
        Out() << " = ";
      Init->printPretty(Out(), 0, m_Policy, m_Indentation);
    }
    prettyPrintAttributes(D);
    skipCurrentDecl(false);
  }

  void ForwardDeclPrinter::VisitLabelDecl(LabelDecl *D) {
    Out() << *D << ":";
    skipCurrentDecl(false);
  }


  void ForwardDeclPrinter::VisitVarDecl(VarDecl *D) {
    if(shouldSkip(D)) {
      skipCurrentDecl();
      return;
    }

    if (D->isDefinedOutsideFunctionOrMethod() && D->getStorageClass() != SC_Extern)
      Out() << "extern ";

    m_Policy.Bool = true;
    //^This should not have been needed (already set in constructor)
    //But for some reason,without this _Bool is still printed in this path (eg: <iomanip>)


    if (!m_Policy.SuppressSpecifiers) {
      StorageClass SC = D->getStorageClass();
      if (SC != SC_None)
        Out() << VarDecl::getStorageClassSpecifierString(SC) << " ";

      switch (D->getTSCSpec()) {
      case TSCS_unspecified:
        break;
      case TSCS___thread:
        Out() << "__thread ";
        break;
      case TSCS__Thread_local:
        Out() << "_Thread_local ";
        break;
      case TSCS_thread_local:
        Out() << "thread_local ";
        break;
      }

      if (D->isModulePrivate())
        Out() << "__module_private__ ";
    }

    QualType T = D->getTypeSourceInfo()
      ? D->getTypeSourceInfo()->getType()
      : D->getASTContext().getUnqualifiedObjCPointerType(D->getType());

    //FIXME: It prints restrict as restrict
    //which is not valid C++
    //Should be __restrict
    //So, we ignore restrict here
    T.removeLocalRestrict();
//    T.print(Out(), m_Policy, D->getName());
    printDeclType(T,D->getName());
    //    llvm::outs()<<D->getName()<<"\n";
    T.addRestrict();

    Expr *Init = D->getInit();
    if (!m_Policy.SuppressInitializers && Init) {
      bool ImplicitInit = false;
      if (CXXConstructExpr *Construct =
          dyn_cast<CXXConstructExpr>(Init->IgnoreImplicit())) {
        if (D->getInitStyle() == VarDecl::CallInit &&
            !Construct->isListInitialization()) {
          ImplicitInit = Construct->getNumArgs() == 0 ||
            Construct->getArg(0)->isDefaultArgument();
        }
      }
      if (D->isDefinedOutsideFunctionOrMethod())
        prettyPrintAttributes(D);
      if (!ImplicitInit) {
        if ((D->getInitStyle() == VarDecl::CallInit)
            && !isa<ParenListExpr>(Init))
          Out() << "(";
        else if (D->getInitStyle() == VarDecl::CInit) {
          if (!D->isDefinedOutsideFunctionOrMethod())
            Out() << " = "; //Comment for skipping default function args
        }
        if (!D->isDefinedOutsideFunctionOrMethod()) {
          //Comment for skipping default function args
          bool isEnumConst = false;
          if (DeclRefExpr* dre = dyn_cast<DeclRefExpr>(Init)){
            if (EnumConstantDecl* decl = dyn_cast<EnumConstantDecl>(dre->getDecl())){
              printDeclType(D->getType(),"");
              // "" because we want only the type name, not the argument name.
              Out() << "(";
              decl->getInitVal().print(Out(),/*isSigned*/true);
              Out() << ")";
              isEnumConst = true;
            }
          }
          if (! isEnumConst)
            Init->printPretty(Out(), 0, m_Policy, m_Indentation);

        }
      if ((D->getInitStyle() == VarDecl::CallInit) && !isa<ParenListExpr>(Init))
        Out() << ")";
      }
    }

    skipCurrentDecl(false);
  }

  void ForwardDeclPrinter::VisitParmVarDecl(ParmVarDecl *D) {
    VisitVarDecl(D);
    skipCurrentDecl(false);
  }

  void ForwardDeclPrinter::VisitFileScopeAsmDecl(FileScopeAsmDecl *D) {
    Out() << "__asm (";
    D->getAsmString()->printPretty(Out(), 0, m_Policy, m_Indentation);
    Out() << ")";
    skipCurrentDecl(false);
  }

  void ForwardDeclPrinter::VisitImportDecl(ImportDecl *D) {
    Out() << "@import " << D->getImportedModule()->getFullModuleName()
          << ";\n";
    skipCurrentDecl(false);
  }

  void ForwardDeclPrinter::VisitStaticAssertDecl(StaticAssertDecl *D) {
    Out() << "static_assert(";
    D->getAssertExpr()->printPretty(Out(), 0, m_Policy, m_Indentation);
    Out() << ", ";
    D->getMessage()->printPretty(Out(), 0, m_Policy, m_Indentation);
    Out() << ")";
    skipCurrentDecl(false);
  }

  //----------------------------------------------------------------------------
  // C++ declarations
  //----------------------------------------------------------------------------
  void ForwardDeclPrinter::VisitNamespaceDecl(NamespaceDecl *D) {

//      VisitDeclContext(D);

    std::string output;
    llvm::raw_string_ostream stream(output);
    m_StreamStack.push(&stream);
    for (auto dit=D->decls_begin();dit!=D->decls_end();++dit) {
      Visit(*dit);
      printSemiColon();
    }
    m_StreamStack.pop();
    stream.flush();
    if ( output.length() == 0 ) {
      m_SkipFlag = true;
      m_IncompatibleNames.insert(D->getName());
      return;
    }
    if (D->isInline())
      Out() << "inline ";
    Out() << "namespace " << *D << " {\n" << output << "}\n";
    m_SkipFlag = true; //Don't print a semi after a namespace
  }

  void ForwardDeclPrinter::VisitUsingDirectiveDecl(UsingDirectiveDecl *D) {

    if (shouldSkip(D)) {
      skipCurrentDecl();
      return;
    }

    Out() << "using namespace ";
    if (D->getQualifier())
      D->getQualifier()->print(Out(), m_Policy);
    Out() << *D->getNominatedNamespaceAsWritten();
    skipCurrentDecl(false);
  }

  void ForwardDeclPrinter::VisitUsingDecl(UsingDecl *D) {
    if(shouldSkip(D)){
      skipCurrentDecl();
      return;
    }
    D->print(Out(),m_Policy);
    skipCurrentDecl(false);
  }
  void ForwardDeclPrinter::VisitUsingShadowDecl(UsingShadowDecl *D) {
    skipCurrentDecl();
  }

  void ForwardDeclPrinter::VisitTypeAliasTemplateDecl(TypeAliasTemplateDecl *D) {
    if (shouldSkip(D)){
      skipCurrentDecl();
    }
  }

  void ForwardDeclPrinter::VisitNamespaceAliasDecl(NamespaceAliasDecl *D) {
    Out() << "namespace " << *D << " = ";
    if (D->getQualifier())
      D->getQualifier()->print(Out(), m_Policy);
    Out() << *D->getAliasedNamespace();
    skipCurrentDecl(false);
  }

  void ForwardDeclPrinter::VisitEmptyDecl(EmptyDecl *D) {
//    prettyPrintAttributes(D);
      skipCurrentDecl();
  }

  void ForwardDeclPrinter::VisitCXXRecordDecl(CXXRecordDecl *D) {
    if (shouldSkip(D)) {
        skipCurrentDecl();
        return;
    }

    if (!m_Policy.SuppressSpecifiers && D->isModulePrivate())
      Out() << "__module_private__ ";
    Out() << D->getKindName();

//    if (D->isCompleteDefinition())
      prettyPrintAttributes(D);
    if (D->getIdentifier())
      Out() << ' ' << *D ;
    skipCurrentDecl(false);

  }

  void ForwardDeclPrinter::VisitLinkageSpecDecl(LinkageSpecDecl *D) {
    const char *l;
    if (D->getLanguage() == LinkageSpecDecl::lang_c)
      l = "C";
    else {
      assert(D->getLanguage() == LinkageSpecDecl::lang_cxx &&
             "unknown language in linkage specification");
      l = "C++";
    }

    Out() << "extern \"" << l << "\" ";
    if (D->hasBraces()) {
      Out() << "{\n";
//      VisitDeclContext(D); //To skip weird typedefs and struct definitions
      for (auto it = D->decls_begin(); it != D->decls_end(); ++it) {
        Visit(*it);
        printSemiColon();
      }
      Out() << "}";
    } else {
      Out() << "{\n"; // print braces anyway, as the decl may end up getting skipped
      Visit(*D->decls_begin());
      Out() << ";}\n";
    }
  }

  void ForwardDeclPrinter::PrintTemplateParameters(const TemplateParameterList *Params,
                                              const TemplateArgumentList *Args) {
    assert(Params);
    assert(!Args || Params->size() == Args->size());

    Out() << "template <";

    for (unsigned i = 0, e = Params->size(); i != e; ++i) {
      if (i != 0)
        Out() << ", ";

      const Decl *Param = Params->getParam(i);
      if (const TemplateTypeParmDecl *TTP =
          dyn_cast<TemplateTypeParmDecl>(Param)) {

        if (TTP->wasDeclaredWithTypename())
          Out() << "typename ";
        else
          Out() << "class ";

        if (TTP->isParameterPack())
          Out() << "...";

        Out() << *TTP;

        if (Args) {
          Out() << " = ";
          Args->get(i).print(m_Policy, Out());
        }
        else if (TTP->hasDefaultArgument() &&
                 !TTP->defaultArgumentWasInherited()) {
          Out() << " = ";
          Out() << TTP->getDefaultArgument().getAsString(m_Policy);
        }
      }
      else if (const NonTypeTemplateParmDecl *NTTP =
               dyn_cast<NonTypeTemplateParmDecl>(Param)) {
        StringRef Name;
        if (IdentifierInfo *II = NTTP->getIdentifier())
          Name = II->getName();
          printDeclType(NTTP->getType(), Name, NTTP->isParameterPack());

        if (Args) {
          Out() << " = ";
          Args->get(i).print(m_Policy, Out());
        }
        else if (NTTP->hasDefaultArgument() &&
                 !NTTP->defaultArgumentWasInherited()) {
          Out() << " = ";
          NTTP->getDefaultArgument()->printPretty(Out(), 0, m_Policy,
                                                  m_Indentation);
        }
      }
      else if (const TemplateTemplateParmDecl *TTPD =
               dyn_cast<TemplateTemplateParmDecl>(Param)) {
        VisitTemplateDecl(TTPD);
        // FIXME: print the default argument, if present.
      }
    }

    Out() << "> ";
  }

  void ForwardDeclPrinter::VisitTemplateDecl(const TemplateDecl *D) {

    PrintTemplateParameters(D->getTemplateParameters());

    if (const TemplateTemplateParmDecl *TTP =
          dyn_cast<TemplateTemplateParmDecl>(D)) {
      Out() << "class ";
    if (TTP->isParameterPack())
      Out() << "...";
    Out() << D->getName();
    }
    else {
      Visit(D->getTemplatedDecl());
    }
    skipCurrentDecl(false);
  }

  void ForwardDeclPrinter::VisitFunctionTemplateDecl(FunctionTemplateDecl *D) {
    if (shouldSkip(D)) {
      skipCurrentDecl();
      return;
    }

    if (m_PrintInstantiation) {
      TemplateParameterList *Params = D->getTemplateParameters();
      for (FunctionTemplateDecl::spec_iterator I = D->spec_begin(),
             E = D->spec_end(); I != E; ++I) {
        PrintTemplateParameters(Params, (*I)->getTemplateSpecializationArgs());
        Visit(*I);
      }
    }

    skipCurrentDecl(false);
    return VisitRedeclarableTemplateDecl(D);

  }

  void ForwardDeclPrinter::VisitClassTemplateDecl(ClassTemplateDecl *D) {
    if (shouldSkip(D) ) {
      skipCurrentDecl();
      return;
    }

    if (m_PrintInstantiation) {
      TemplateParameterList *Params = D->getTemplateParameters();
      for (ClassTemplateDecl::spec_iterator I = D->spec_begin(),
             E = D->spec_end(); I != E; ++I) {
        PrintTemplateParameters(Params, &(*I)->getTemplateArgs());
        Visit(*I);
        Out() << '\n';
      }
    }
    skipCurrentDecl(false);
    return VisitRedeclarableTemplateDecl(D);
  }

  void ForwardDeclPrinter::
  VisitClassTemplateSpecializationDecl(ClassTemplateSpecializationDecl* D) {

//    if (shouldSkip(D)) {
//      skipCurrentDecl();
//      return;
//    }

//    const TemplateArgumentList& iargs = D->getTemplateInstantiationArgs();

//    Out() << "template <> ";
//    VisitCXXRecordDecl(D->getCanonicalDecl());

//    Out() << "<";
//    for (uint i=0; i < iargs.size(); ++i){
//      if (iargs[i].getKind() == TemplateArgument::Pack)
//        continue;
//      if (i != 0 )
//        Out() << ", ";
//      iargs[i].print(m_Policy,Out());
//    }
//    Out() << ">";
//    skipCurrentDecl(false);

      skipCurrentDecl();
      //Above code doesn't work properly
      //Must find better and more general way to print specializations
  }

  void ForwardDeclPrinter::printSemiColon(bool flag) {
    if (flag) {
      if (!m_SkipFlag)
        Out() << ";\n";
      else
        m_SkipFlag = false;
    }
    else Out() << ";\n";
  }

  bool ForwardDeclPrinter::isIncompatibleType(QualType q, bool includeNNS) {
    //FIXME: This is a workaround and filters out many acceptable cases
    //Refer to Point#1
    QualType temp = q;
    while (temp.getTypePtr()->isAnyPointerType())//For 3 star programmers
      temp = temp.getTypePtr()->getPointeeType();

    while (temp.getTypePtr()->isReferenceType())//For move references
        temp = temp.getNonReferenceType();

    std::string str = QualType(temp.getTypePtr(),0).getAsString();
//    llvm::outs() << "Q:"<<str<<"\n";
    bool result =  m_IncompatibleNames.find(str) != m_IncompatibleNames.end();
    if (includeNNS)
      result = result || str.find("::") != std::string::npos;
    return result;
  }

  bool ForwardDeclPrinter::isOperator(FunctionDecl *D) {
    //TODO: Find a better check for this
    return D->getNameAsString().find("operator") == 0;
  }

  bool ForwardDeclPrinter::hasDefaultArgument(FunctionDecl *D) {
    auto N = D->getNumParams();
    for (uint i=0; i < N; ++i) {
      if (D->getParamDecl(i)->hasDefaultArg())
        return true;
    }
    return false;
  }

  bool ForwardDeclPrinter::shouldSkip(FunctionDecl *D) {
    bool param = false;
    //will be true if any of the params turn out to have incompatible types

    for (unsigned i = 0, e = D->getNumParams(); i != e; ++i) {
      const Type* type = D->getParamDecl(i)->getType().getTypePtr();
      while (type->isReferenceType())
        type = D->getParamDecl(i)->getType().getNonReferenceType().getTypePtr();
      while (type->isPointerType())
        type = type->getPointeeType().getTypePtr();

      if (const TemplateSpecializationType* tst
        = dyn_cast<TemplateSpecializationType>(type)){
          if (m_IncompatibleNames.find
                  (tst->getTemplateName().getAsTemplateDecl()->getName())
                  != m_IncompatibleNames.end()) {
//            Log() << "Function : Incompatible Type\n";
            return true;
          }
          for (uint i = 0; i < tst->getNumArgs(); ++i ) {
            const TemplateArgument& arg = tst->getArg(i);
            TemplateArgument::ArgKind kind = arg.getKind();
            if (kind == TemplateArgument::ArgKind::Type){
              if (m_IncompatibleNames.find(arg.getAsType().getAsString())
                      != m_IncompatibleNames.end()){
//                Log() << D->getName() << " Function : Incompatible Type\n";
                return true;
              }
            }
            if (kind == TemplateArgument::ArgKind::Expression) {
              Expr* expr = arg.getAsExpr();
              //TODO: Traverse this expr
            }
         }

      }
      if (isIncompatibleType(D->getParamDecl(i)->getType())){
//        Log() << "Function : Incompatible Type\n";
        param = true;
      }
    }

    if (D->getNameAsString().size() == 0
        || D->getNameAsString()[0] == '_'
        || D->getStorageClass() == SC_Static
        || D->isCXXClassMember()
        || isIncompatibleType(D->getReturnType())
        || param
        || isOperator(D)
        || D->isDeleted()
        || D->isDeletedAsWritten()){
        //FIXME: setDeletedAsWritten can be called from the
        //InclusionDiretctive callback.
        //Implement that if important functions are marked so.
        //Not important, as users do not need hints
        //about using Deleted functions
//      Log() <<"Function : Other\n";
      return true;
    }
    return false;
  }
  bool ForwardDeclPrinter::shouldSkip(CXXRecordDecl *D) {
    return D->getNameAsString().size() == 0;
  }

  bool ForwardDeclPrinter::shouldSkip(TypedefDecl *D) {
    if (const ElaboratedType* ET =
            dyn_cast<ElaboratedType>(D->getTypeSourceInfo()->getType().getTypePtr())) {
      if (isa<EnumType>(ET->getNamedType())) {
        m_IncompatibleNames.insert(D->getName());
        Log() << D->getName() << " Typedef : enum\n";
        return true;
      }
      if (isa<RecordType>(ET->getNamedType())) {
        m_IncompatibleNames.insert(D->getName());
        Log() << D->getName() << " Typedef : struct\n";
        return true;
      }
    } 

    if (const TemplateSpecializationType* tst
            = dyn_cast<TemplateSpecializationType>
            (D->getTypeSourceInfo()->getType().getTypePtr())){
      for (uint i = 0; i < tst->getNumArgs(); ++i ) {
        const TemplateArgument& arg = tst->getArg(i);
        if (arg.getKind() == TemplateArgument::ArgKind::Type)
          if (m_IncompatibleNames.find(arg.getAsType().getAsString())
                  != m_IncompatibleNames.end()){
            m_IncompatibleNames.insert(D->getName());
            Log() << D->getName() << " Typedef : Incompatible Type\n";
            return true;
          }
      }
    }
    if (isIncompatibleType(D->getTypeSourceInfo()->getType())) {
      m_IncompatibleNames.insert(D->getName());
      Log() << D->getName() << " Typedef : Incompatible Type\n";
      return true;
    }
    if (D->getUnderlyingType().getTypePtr()->isFunctionPointerType()) {
      const FunctionType* ft =
              D->getUnderlyingType().getTypePtr()
              ->getPointeeType().getTypePtr()->castAs<clang::FunctionType>();
      bool result = isIncompatibleType(ft->getReturnType(),false);
      if (const FunctionProtoType* fpt = dyn_cast<FunctionProtoType>(ft)){
        for (uint i = 0; i < fpt->getNumParams(); ++i){
          result = result || isIncompatibleType(fpt->getParamType(i),false);
          if (result){
            m_IncompatibleNames.insert(D->getName());
            Log() << D->getName() << " Typedef : Function Pointer\n";
            return true;
          }

        }
      }
      return false;
    }
    if (D->getUnderlyingType().getTypePtr()->isFunctionType()){
        const FunctionType* ft =D->getUnderlyingType().getTypePtr()
                                 ->castAs<clang::FunctionType>();
        bool result = isIncompatibleType(ft->getReturnType(),false);
        if (const FunctionProtoType* fpt = dyn_cast<FunctionProtoType>(ft)){
          for (uint i = 0; i < fpt->getNumParams(); ++i){
            result = result || isIncompatibleType(fpt->getParamType(i),false);
            if (result){
              m_IncompatibleNames.insert(D->getName());
              Log() << " Typedef : Function Pointer\n";
              return true;
            }
          }
        }
    }
    //TODO: Lot of logic overlap with above block, make an abstraction
    return false;
  }
  bool ForwardDeclPrinter::shouldSkip(VarDecl *D) {
    if (D->isDefinedOutsideFunctionOrMethod()){
      if (D->isCXXClassMember()){
        Log() << D->getName() <<" Var : Class Member\n";
        return true ;
      }
    }
    bool stc =  D->getStorageClass() == SC_Static;
    bool inctype = isIncompatibleType(D->getType());
    if (stc) Log() << D->getName() <<" Var : Static\n";
    if (inctype) Log() << D->getName() <<" Var : Incompatible Type\n";
    if (stc || inctype) {
      m_IncompatibleNames.insert(D->getName());
      return true;
    }
    return false;
  }
  bool ForwardDeclPrinter::shouldSkip(EnumDecl *D) {
    return D->getName().size() == 0;
  }
  void ForwardDeclPrinter::skipCurrentDecl(bool skip) {
    if (skip) {
      m_SkipFlag = true;
      m_SkipCounter++;
    }
    m_TotalDecls++;
  }
  bool ForwardDeclPrinter::shouldSkip(ClassTemplateSpecializationDecl *D) {
    if (llvm::isa<ClassTemplatePartialSpecializationDecl>(D)) {
        //TODO: How to print partial specializations?
        return true;

    }
    const TemplateArgumentList& iargs = D->getTemplateInstantiationArgs();
    for (uint i = 0; i < iargs.size(); ++i) {
      const TemplateArgument& arg = iargs[i];
      if (arg.getKind() == TemplateArgument::ArgKind::Type)
        if (m_IncompatibleNames.find(arg.getAsType().getAsString())
            !=m_IncompatibleNames.end())
          return true;
    }
    return false;
  }
  bool ForwardDeclPrinter::shouldSkip(UsingDirectiveDecl *D) {
    std::string str = D->getNominatedNamespace()->getNameAsString();
    bool inctype = m_IncompatibleNames.find(str) != m_IncompatibleNames.end();
    if (inctype) Log() << str <<" Using Directive : Incompatible Type\n";
    return inctype;
  }
  bool ForwardDeclPrinter::ContainsIncompatibleName(TemplateParameterList* Params){
    for (unsigned i = 0, e = Params->size(); i != e; ++i) {
      const Decl *Param = Params->getParam(i);
      if (const TemplateTypeParmDecl *TTP =
        dyn_cast<TemplateTypeParmDecl>(Param)) {
          if (TTP->hasDefaultArgument() ) {
            if (m_IncompatibleNames.find(TTP->getName())
                    !=m_IncompatibleNames.end()){
              return true;
            }
          }
      }
      if (const NonTypeTemplateParmDecl *NTTP =
        dyn_cast<NonTypeTemplateParmDecl>(Param)) {

          if (NTTP->hasDefaultArgument() ) {
            QualType type = NTTP->getType();
            if (isIncompatibleType(type)){
              return true;
            }
            Expr* expr = NTTP->getDefaultArgument();
            expr = expr->IgnoreImpCasts();
            if (DeclRefExpr* dre = dyn_cast<DeclRefExpr>(expr)){
              std::string str = dre->getDecl()->getQualifiedNameAsString();
              if (str.find("::") != std::string::npos)
                return true; // TODO: Find proper solution
            }
          }
      }
    }
    return false;
  }

  bool ForwardDeclPrinter::shouldSkip(ClassTemplateDecl *D) {
    if (ContainsIncompatibleName(D->getTemplateParameters())
            || shouldSkip(D->getTemplatedDecl())){
      Log() << D->getName() <<" Class Template : Incompatible Type\n";
      m_IncompatibleNames.insert(D->getName());
      return true;
    }
    return false;
  }
  bool ForwardDeclPrinter::shouldSkip(FunctionTemplateDecl* D){
    bool inctype = ContainsIncompatibleName(D->getTemplateParameters());
    bool func =  shouldSkip(D->getTemplatedDecl());
    bool hasdef = hasDefaultArgument(D->getTemplatedDecl());
    if (inctype || func || hasdef ) {
      if (D->getDeclName().isIdentifier()){
        m_IncompatibleNames.insert(D->getName());
        if (hasdef) Log() << D->getName() << " Function Template : Has default argument\n";
      }
      return true;
    }
    return false;
  }

  bool ForwardDeclPrinter::shouldSkip(UsingDecl *D) {
    if (m_IncompatibleNames.find(D->getName())
            != m_IncompatibleNames.end()) {
      Log() << D->getName() <<" Using Decl : Incompatible Type\n";
      return true;
    }
    return false;
  }
  bool ForwardDeclPrinter::shouldSkip(TypeAliasTemplateDecl *D) {
    m_IncompatibleNames.insert(D->getName());
    return true;
  }

  void ForwardDeclPrinter::printStats() {
    Log() << m_SkipCounter << " decls skipped out of " << m_TotalDecls << "\n";
  }
}//end namespace cling
