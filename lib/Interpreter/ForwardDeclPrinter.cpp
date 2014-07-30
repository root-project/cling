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

  static QualType GetBaseType(QualType T) {
    // FIXME: This should be on the Type class!
    QualType BaseType = T;
    while (!BaseType->isSpecifierType()) {
      if (isa<TypedefType>(BaseType))
        break;
      else if (const PointerType* PTy = BaseType->getAs<PointerType>())
        BaseType = PTy->getPointeeType();
      else if (const BlockPointerType *BPy = BaseType->getAs<BlockPointerType>())
        BaseType = BPy->getPointeeType();
      else if (const ArrayType* ATy = dyn_cast<ArrayType>(BaseType))
        BaseType = ATy->getElementType();
      else if (const FunctionType* FTy = BaseType->getAs<FunctionType>())
        BaseType = FTy->getReturnType();
      else if (const VectorType *VTy = BaseType->getAs<VectorType>())
        BaseType = VTy->getElementType();
      else if (const ReferenceType *RTy = BaseType->getAs<ReferenceType>())
        BaseType = RTy->getPointeeType();
      else
        llvm_unreachable("Unknown declarator!");
    }
    return BaseType;
  }
  static QualType getDeclType(Decl* D) {
    if (TypedefNameDecl* TDD = dyn_cast<TypedefNameDecl>(D))
      return TDD->getUnderlyingType();
    if (ValueDecl* VD = dyn_cast<ValueDecl>(D))
        return VD->getType();
    return QualType();
  }

  ForwardDeclPrinter::ForwardDeclPrinter(llvm::raw_ostream& Out,
                                         SourceManager& SM,
                                         const Transaction& T,
                                         unsigned Indentation,
                                         bool printMacros)
    : m_Out(Out), m_Policy(clang::PrintingPolicy(clang::LangOptions())),
      m_Indentation(Indentation), m_SMgr(SM), m_SkipFlag(false) {
    m_PrintInstantiation = false;
    m_Policy.SuppressTagKeyword = true;

    // Suppress some unfixable warnings.
    m_Out << "#pragma clang diagnostic ignored \"-Wkeyword-compat\"" << "\n";

    std::vector<std::string> macrodefs;
    if (printMacros) {
      for (auto mit = T.macros_begin(); mit != T.macros_end(); ++mit) {
        Transaction::MacroDirectiveInfo macro = *mit;
        if (macro.m_MD->getKind() == MacroDirective::MD_Define) {
          const MacroInfo* MI = macro.m_MD->getMacroInfo();
          if (MI ->getNumTokens() > 1)
            //FIXME: We can not display function like macros yet
            continue;
          m_Out << "#define " << macro.m_II->getName() << ' ';
          for (unsigned i = 0, e = MI->getNumTokens(); i != e; ++i) {
            const Token &Tok = MI->getReplacementToken(i);
            m_Out << Tok.getName() << ' ';
            macrodefs.push_back(macro.m_II->getName());
          }
          m_Out << '\n';
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
          Visit(*dit);
          printSemiColon();
        }
      }
    }
    if (printMacros) {
      for (auto m : macrodefs) {
        m_Out << "#undef " << m << "\n";
      }
    }

  }

  ForwardDeclPrinter::ForwardDeclPrinter(llvm::raw_ostream &Out,
                                         clang::SourceManager& SM,
                                         const clang::PrintingPolicy& P,
                                         unsigned Indentation)
    : m_Out(Out), m_Policy(clang::PrintingPolicy(clang::LangOptions())),
      m_Indentation(Indentation), m_SMgr(SM), m_SkipFlag(false) {
    m_PrintInstantiation = false;
    m_Policy.SuppressTagKeyword = true;
  }


  llvm::raw_ostream& ForwardDeclPrinter::Indent(unsigned Indentation) {
    for (unsigned i = 0; i != Indentation; ++i)
      m_Out << "  ";
    return m_Out;
  }

  void ForwardDeclPrinter::prettyPrintAttributes(Decl *D, std::string extra) {
//      if (Policy.PolishForDeclaration)
//        return;

//      if (D->hasAttrs()) {
//        AttrVec &Attrs = D->getAttrs();
//        for (AttrVec::const_iterator i=Attrs.begin(), e=Attrs.end(); i!=e; ++i) {
//          Attr *A = *i;
//          A->printPretty(Out, Policy);
//        }
//      }
    //FIXME: This ignores previous attributes
    //Do not simply uncomment the above code
    //In some cases, it prints attribs without the strings at all

    //FIXME: Must print file id or full path

    if (D->getSourceRange().isInvalid())
      return;
    std::string file = m_SMgr.getFilename(D->getLocStart());
    m_Out << " __attribute__((annotate(\""
          << DynamicLibraryManager::normalizePath(file);
    if (!extra.empty())
      m_Out << " " << extra;
    m_Out << "\"))) ";
  }

  void ForwardDeclPrinter::ProcessDeclGroup(SmallVectorImpl<Decl*>& Decls) {
    Indent();
    Decl::printGroup(Decls.data(), Decls.size(), m_Out, m_Policy,
                     m_Indentation);
    m_Out << ";\n";
    Decls.clear();

  }

  void ForwardDeclPrinter::Print(AccessSpecifier AS) {
    switch(AS) {
      case AS_none:      llvm_unreachable("No access specifier!");
      case AS_public:    m_Out << "public"; break;
      case AS_protected: m_Out << "protected"; break;
      case AS_private:   m_Out << "private"; break;
    }
  }

    //----------------------------------------------------------------------------
    // Common C declarations
    //----------------------------------------------------------------------------

  void ForwardDeclPrinter::VisitDeclContext(DeclContext *DC, bool Indent) {
    if (m_Policy.TerseOutput)
      return;
    if (Indent)
      m_Indentation += m_Policy.Indentation;

    SmallVector<Decl*, 2> Decls;
    for (DeclContext::decl_iterator D = DC->decls_begin(), DEnd = DC->decls_end();
         D != DEnd; ++D) {

      // Don't print ObjCIvarDecls, as they are printed when visiting the
      // containing ObjCInterfaceDecl.
      if (isa<ObjCIvarDecl>(*D))
        continue;

      // Skip over implicit declarations in pretty-printing mode.
      if (D->isImplicit())
        continue;

      // The next bits of code handles stuff like "struct {int x;} a,b"; we're
      // forced to merge the declarations because there's no other way to
      // refer to the struct in question.  This limited merging is safe without
      // a bunch of other checks because it only merges declarations directly
      // referring to the tag, not typedefs.
      //
      // Check whether the current declaration should be grouped with a previous
      // unnamed struct.
      QualType CurDeclType = getDeclType(*D);
      if (!Decls.empty() && !CurDeclType.isNull()) {
        QualType BaseType = GetBaseType(CurDeclType);
        if (!BaseType.isNull() && isa<ElaboratedType>(BaseType))
          BaseType = cast<ElaboratedType>(BaseType)->getNamedType();
        if (!BaseType.isNull() && isa<TagType>(BaseType) &&
            cast<TagType>(BaseType)->getDecl() == Decls[0]) {
          Decls.push_back(*D);
          continue;
        }
      }

      // If we have a merged group waiting to be handled, handle it now.
      if (!Decls.empty())
        ProcessDeclGroup(Decls);

      // If the current declaration is an unnamed tag type, save it
      // so we can merge it with the subsequent declaration(s) using it.
      if (isa<TagDecl>(*D) && !cast<TagDecl>(*D)->getIdentifier()) {
        Decls.push_back(*D);
        continue;
      }

      if (isa<AccessSpecDecl>(*D)) {
        m_Indentation -= m_Policy.Indentation;
        this->Indent();
        Print(D->getAccess());
        m_Out << ":\n";
        m_Indentation += m_Policy.Indentation;
        continue;
      }

      this->Indent();
      Visit(*D);

      const char *Terminator = 0;
      if (isa<OMPThreadPrivateDecl>(*D))
        Terminator = 0;
      else if (isa<FunctionDecl>(*D) &&
               cast<FunctionDecl>(*D)->isThisDeclarationADefinition())
        Terminator = 0;
      else if (isa<ObjCMethodDecl>(*D) && cast<ObjCMethodDecl>(*D)->getBody())
        Terminator = 0;
      else if (isa<NamespaceDecl>(*D) || isa<LinkageSpecDecl>(*D) ||
               isa<ObjCImplementationDecl>(*D) ||
               isa<ObjCInterfaceDecl>(*D) ||
               isa<ObjCProtocolDecl>(*D) ||
               isa<ObjCCategoryImplDecl>(*D) ||
               isa<ObjCCategoryDecl>(*D))
        Terminator = 0;
      else if (isa<EnumConstantDecl>(*D)) {
        DeclContext::decl_iterator Next = D;
        ++Next;
        if (Next != DEnd)
          Terminator = ",";
      } else
        Terminator = ";";

      if (Terminator)
        m_Out << Terminator;
      m_Out << "\n";
    }

    if (!Decls.empty())
      ProcessDeclGroup(Decls);

    if (Indent)
      m_Indentation -= m_Policy.Indentation;
  }

  void ForwardDeclPrinter::VisitTranslationUnitDecl(TranslationUnitDecl *D) {
    VisitDeclContext(D, false);
  }

  void ForwardDeclPrinter::VisitTypedefDecl(TypedefDecl *D) {
    if (!m_Policy.SuppressSpecifiers) {
      m_Out << "typedef ";

    if (D->isModulePrivate())
      m_Out << "__module_private__ ";
    }
    D->getTypeSourceInfo()->getType().print(m_Out, m_Policy, D->getName());
    prettyPrintAttributes(D);
//      Indent() << ";\n";
  }

  void ForwardDeclPrinter::VisitTypeAliasDecl(TypeAliasDecl *D) {
      /*FIXME: Ugly Hack*/
//      if(!D->getLexicalDeclContext()->isNamespace()
//              && !D->getLexicalDeclContext()->isFileContext())
//          return;
    m_Out << "using " << *D;
    prettyPrintAttributes(D);
    m_Out << " = " << D->getTypeSourceInfo()->getType().getAsString(m_Policy);
//      Indent() << ";\n";
  }

  void ForwardDeclPrinter::VisitEnumDecl(EnumDecl *D) {
    if (D->getName().size() == 0) {
      m_SkipFlag = true;
      return;
    }

    if (!m_Policy.SuppressSpecifiers && D->isModulePrivate())
      m_Out << "__module_private__ ";
    m_Out << "enum ";
    prettyPrintAttributes(D,std::to_string(D->isFixed()));
    if (D->isScoped()) {
      if (D->isScopedUsingClassTag())
        m_Out << "class ";
      else
        m_Out << "struct ";
    }
    m_Out << *D;

//      if (D->isFixed())
    m_Out << " : " << D->getIntegerType().stream(m_Policy);

//      if (D->isCompleteDefinition()) {
//        Out << " {\n";
//        VisitDeclContext(D);
//        Indent() << "};\n";
//      }
  }

  void ForwardDeclPrinter::VisitRecordDecl(RecordDecl *D) {
    if (!m_Policy.SuppressSpecifiers && D->isModulePrivate())
      m_Out << "__module_private__ ";
    m_Out << D->getKindName();
    prettyPrintAttributes(D);
    if (D->getIdentifier())
      m_Out << ' ' << *D;

//    if (D->isCompleteDefinition()) {
//      Out << " {\n";
//      VisitDeclContext(D);
//      Indent() << "}";
//    }
  }

  void ForwardDeclPrinter::VisitEnumConstantDecl(EnumConstantDecl *D) {
    m_Out << *D;
    if (Expr *Init = D->getInitExpr()) {
      m_Out << " = ";
      Init->printPretty(m_Out, 0, m_Policy, m_Indentation);
    }
  }

  void ForwardDeclPrinter::VisitFunctionDecl(FunctionDecl *D) {
    if (shouldSkipFunction(D)) {
      m_SkipFlag = true;
      return;
    }
    CXXConstructorDecl *CDecl = dyn_cast<CXXConstructorDecl>(D);
    CXXConversionDecl *ConversionDecl = dyn_cast<CXXConversionDecl>(D);
      /*FIXME:Ugly Hack*/
//      if (CDecl||ConversionDecl)
//          return;

    if (!m_Policy.SuppressSpecifiers) {
      switch (D->getStorageClass()) {
      case SC_None: break;
      case SC_Extern: m_Out << "extern "; break;
      case SC_Static: m_Out << "static "; break;
      case SC_PrivateExtern: m_Out << "__private_extern__ "; break;
      case SC_Auto: case SC_Register: case SC_OpenCLWorkGroupLocal:
        llvm_unreachable("invalid for functions");
      }

      if (D->isInlineSpecified())  m_Out << "inline ";
      if (D->isVirtualAsWritten()) m_Out << "virtual ";
      if (D->isModulePrivate())    m_Out << "__module_private__ ";
      if (D->isConstexpr() && !D->isExplicitlyDefaulted())
        m_Out << "constexpr ";
      if ((CDecl && CDecl->isExplicitSpecified()) ||
          (ConversionDecl && ConversionDecl->isExplicit()))
        m_Out << "explicit ";
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
        ForwardDeclPrinter ParamPrinter(POut, m_SMgr, SubPolicy, m_Indentation);
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
            m_Out << Proto;
            Proto.clear();
            HasInitializerList = true;
          } else
            m_Out << ", ";

          if (BMInitializer->isAnyMemberInitializer()) {
            FieldDecl *FD = BMInitializer->getAnyMember();
            m_Out << *FD;
          } else {
            m_Out << QualType(BMInitializer->getBaseClass(), 0).getAsString(m_Policy);
          }

          m_Out << "(";
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
              SimpleInit->printPretty(m_Out, 0, m_Policy, m_Indentation);
            else {
              for (unsigned I = 0; I != NumArgs; ++I) {
                if (isa<CXXDefaultArgExpr>(Args[I]))
                  break;

                if (I)
                  m_Out << ", ";
                Args[I]->printPretty(m_Out, 0, m_Policy, m_Indentation);
              }
            }
          }
          m_Out << ")";
          if (BMInitializer->isPackExpansion())
            m_Out << "...";
        }
      }
      else if (!ConversionDecl && !isa<CXXDestructorDecl>(D)) {
        if (FT && FT->hasTrailingReturn()) {
          m_Out << "auto " << Proto << " -> ";
          Proto.clear();
        }
        AFT->getReturnType().print(m_Out, m_Policy, Proto);
        Proto.clear();
      }
      m_Out << Proto;
    }
    else {
      Ty.print(m_Out, m_Policy, Proto);
    }

    prettyPrintAttributes(D);

    if (D->isPure())
      m_Out << " = 0";
    else if (D->isDeletedAsWritten())
      m_Out << " = delete";
    else if (D->isExplicitlyDefaulted())
      m_Out << " = default";
    else if (D->doesThisDeclarationHaveABody() && !m_Policy.TerseOutput) {
      if (!D->hasPrototype() && D->getNumParams()) {
        // This is a K&R function definition, so we need to print the
        // parameters.
        m_Out << '\n';
        ForwardDeclPrinter ParamPrinter(m_Out, m_SMgr, SubPolicy,
                                        m_Indentation);
        m_Indentation += m_Policy.Indentation;
        for (unsigned i = 0, e = D->getNumParams(); i != e; ++i) {
          Indent();
          ParamPrinter.VisitParmVarDecl(D->getParamDecl(i));
          m_Out << ";\n";
        }
        m_Indentation -= m_Policy.Indentation;
      } else
        m_Out << ' ';

      //    D->getBody()->printPretty(Out, 0, SubPolicy, Indentation);

    }
  }

  void ForwardDeclPrinter::VisitFriendDecl(FriendDecl *D) {
//      if (TypeSourceInfo *TSI = D->getFriendType()) {
//        unsigned NumTPLists = D->getFriendTypeNumTemplateParameterLists();
//        for (unsigned i = 0; i < NumTPLists; ++i)
//          PrintTemplateParameters(D->getFriendTypeTemplateParameterList(i));
//        Out << "friend ";
//        Out << " " << TSI->getType().getAsString(Policy);
//      }
//      else if (FunctionDecl *FD =
//          dyn_cast<FunctionDecl>(D->getFriendDecl())) {
//        Out << "friend ";
//        VisitFunctionDecl(FD);
//      }
//      else if (FunctionTemplateDecl *FTD =
//               dyn_cast<FunctionTemplateDecl>(D->getFriendDecl())) {
//        Out << "friend ";
//        VisitFunctionTemplateDecl(FTD);
//      }
//      else if (ClassTemplateDecl *CTD =
//               dyn_cast<ClassTemplateDecl>(D->getFriendDecl())) {
//        Out << "friend ";
//        VisitRedeclarableTemplateDecl(CTD);
//      }
  }

  void ForwardDeclPrinter::VisitFieldDecl(FieldDecl *D) {
    if (!m_Policy.SuppressSpecifiers && D->isMutable())
      m_Out << "mutable ";
    if (!m_Policy.SuppressSpecifiers && D->isModulePrivate())
      m_Out << "__module_private__ ";
    m_Out << D->getASTContext().getUnqualifiedObjCPointerType(D->getType()).
      stream(m_Policy, D->getName());

    if (D->isBitField()) {
      m_Out << " : ";
      D->getBitWidth()->printPretty(m_Out, 0, m_Policy, m_Indentation);
    }

    Expr *Init = D->getInClassInitializer();
    if (!m_Policy.SuppressInitializers && Init) {
      if (D->getInClassInitStyle() == ICIS_ListInit)
        m_Out << " ";
      else
        m_Out << " = ";
      Init->printPretty(m_Out, 0, m_Policy, m_Indentation);
    }
    prettyPrintAttributes(D);
  }

  void ForwardDeclPrinter::VisitLabelDecl(LabelDecl *D) {
    m_Out << *D << ":";
  }


  void ForwardDeclPrinter::VisitVarDecl(VarDecl *D) {
    if(D->getStorageClass() == SC_Static
       || hasNestedNameSpecifier(D->getType())
       || D->isDefinedOutsideFunctionOrMethod()) {
      m_SkipFlag = true;
      return;
    }
//    if(D->isDefinedOutsideFunctionOrMethod() && !(D->getStorageClass() == SC_Extern))
//      Out << "extern ";


    if (!m_Policy.SuppressSpecifiers) {
      StorageClass SC = D->getStorageClass();
      if (SC != SC_None)
        m_Out << VarDecl::getStorageClassSpecifierString(SC) << " ";

      switch (D->getTSCSpec()) {
      case TSCS_unspecified:
        break;
      case TSCS___thread:
        m_Out << "__thread ";
        break;
      case TSCS__Thread_local:
        m_Out << "_Thread_local ";
        break;
      case TSCS_thread_local:
        m_Out << "thread_local ";
        break;
      }

      if (D->isModulePrivate())
        m_Out << "__module_private__ ";
    }

    QualType T = D->getTypeSourceInfo()
      ? D->getTypeSourceInfo()->getType()
      : D->getASTContext().getUnqualifiedObjCPointerType(D->getType());

    //FIXME: It prints restrict as restrict
    //which is not valid C++
    //Should be __restrict
    //So, we ignore restrict here
    T.removeLocalRestrict();
    T.print(m_Out, m_Policy, D->getName());
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
      if (!ImplicitInit) {
        if ((D->getInitStyle() == VarDecl::CallInit)
            && !isa<ParenListExpr>(Init))
          m_Out << "(";
        else if (D->getInitStyle() == VarDecl::CInit) {
          if(!D->isDefinedOutsideFunctionOrMethod())
            m_Out << " = "; //Comment for skipping default function args
        }
        if (!D->isDefinedOutsideFunctionOrMethod())
          //Comment for skipping default function args
          Init->printPretty(m_Out, 0, m_Policy, m_Indentation);
      if ((D->getInitStyle() == VarDecl::CallInit) && !isa<ParenListExpr>(Init))
        m_Out << ")";
      }
    }

    if(D->isDefinedOutsideFunctionOrMethod())
      prettyPrintAttributes(D);
  }

  void ForwardDeclPrinter::VisitParmVarDecl(ParmVarDecl *D) {
    VisitVarDecl(D);
  }

  void ForwardDeclPrinter::VisitFileScopeAsmDecl(FileScopeAsmDecl *D) {
    m_Out << "__asm (";
    D->getAsmString()->printPretty(m_Out, 0, m_Policy, m_Indentation);
    m_Out << ")";
  }

  void ForwardDeclPrinter::VisitImportDecl(ImportDecl *D) {
    m_Out << "@import " << D->getImportedModule()->getFullModuleName()
          << ";\n";
  }

  void ForwardDeclPrinter::VisitStaticAssertDecl(StaticAssertDecl *D) {
    m_Out << "static_assert(";
    D->getAssertExpr()->printPretty(m_Out, 0, m_Policy, m_Indentation);
    m_Out << ", ";
    D->getMessage()->printPretty(m_Out, 0, m_Policy, m_Indentation);
    m_Out << ")";
  }

  //----------------------------------------------------------------------------
  // C++ declarations
  //----------------------------------------------------------------------------
  void ForwardDeclPrinter::VisitNamespaceDecl(NamespaceDecl *D) {
    if (D->isInline())
      m_Out << "inline ";
    m_Out << "namespace " << *D << " {\n";
//      VisitDeclContext(D);
    for (auto dit=D->decls_begin();dit!=D->decls_end();++dit) {
      Visit(*dit);
      printSemiColon();
    }
    Indent() << "}\n";
    m_SkipFlag = true;
  }

  void ForwardDeclPrinter::VisitUsingDirectiveDecl(UsingDirectiveDecl *D) {
    m_Out << "using namespace ";
    if (D->getQualifier())
      D->getQualifier()->print(m_Out, m_Policy);
    m_Out << *D->getNominatedNamespaceAsWritten();
  }

  void ForwardDeclPrinter::VisitNamespaceAliasDecl(NamespaceAliasDecl *D) {
    m_Out << "namespace " << *D << " = ";
    if (D->getQualifier())
      D->getQualifier()->print(m_Out, m_Policy);
    m_Out << *D->getAliasedNamespace();
  }

  void ForwardDeclPrinter::VisitEmptyDecl(EmptyDecl *D) {
//    prettyPrintAttributes(D);
      m_SkipFlag = true;
  }

  void ForwardDeclPrinter::VisitCXXRecordDecl(CXXRecordDecl *D) {
    if (D->getNameAsString().size() == 0) {
        m_SkipFlag = true;
        return;
    }

    if (!m_Policy.SuppressSpecifiers && D->isModulePrivate())
      m_Out << "__module_private__ ";
    m_Out << D->getKindName();

    if (D->isCompleteDefinition())
      prettyPrintAttributes(D);
    if (D->getIdentifier())
      m_Out << ' ' << *D ;

    //  if (D->isCompleteDefinition()) {
    //    // Print the base classes
    //    if (D->getNumBases()) {
    //      Out << " : ";
    //      for (CXXRecordDecl::base_class_iterator Base = D->bases_begin(),
    //             BaseEnd = D->bases_end(); Base != BaseEnd; ++Base) {
    //        if (Base != D->bases_begin())
    //          Out << ", ";

    //        if (Base->isVirtual())
    //          Out << "virtual ";

    //        AccessSpecifier AS = Base->getAccessSpecifierAsWritten();
    //        if (AS != AS_none)
    //          Print(AS);
    //        Out << " " << Base->getType().getAsString(Policy);

    //        if (Base->isPackExpansion())
    //          Out << "...";
    //      }
    //    }

    //    // Print the class definition
    //    // FIXME: Doesn't print access specifiers, e.g., "public:"
    //    Out << " {\n";
    //    VisitDeclContext(D);
    //    Indent() << "}";
    //  }
    m_Out << ";\n";
    m_SkipFlag = true;
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

    m_Out << "extern \"" << l << "\" ";
    if (D->hasBraces()) {
      m_Out << "{\n";
//      VisitDeclContext(D); //To skip weird typedefs and struct definitions
      for (auto it = D->decls_begin(); it != D->decls_end(); ++it) {
        Visit(*it);
        printSemiColon();
      }
      Indent() << "}";
    } else {
      m_Out << "{\n"; // print braces anyway, as the decl may end up getting skipped
      Visit(*D->decls_begin());
      m_Out << ";}\n";
    }
  }

  void ForwardDeclPrinter::PrintTemplateParameters(const TemplateParameterList *Params,
                                              const TemplateArgumentList *Args) {
    assert(Params);
    assert(!Args || Params->size() == Args->size());

    m_Out << "template <";

    for (unsigned i = 0, e = Params->size(); i != e; ++i) {
      if (i != 0)
        m_Out << ", ";

      const Decl *Param = Params->getParam(i);
      if (const TemplateTypeParmDecl *TTP =
          dyn_cast<TemplateTypeParmDecl>(Param)) {

        if (TTP->wasDeclaredWithTypename())
          m_Out << "typename ";
        else
          m_Out << "class ";

        if (TTP->isParameterPack())
          m_Out << "... ";

        m_Out << *TTP;

        if (Args) {
          m_Out << " = ";
          Args->get(i).print(m_Policy, m_Out);
        }
        else if (TTP->hasDefaultArgument() &&
                 !TTP->defaultArgumentWasInherited()) {
          m_Out << " = ";
          m_Out << TTP->getDefaultArgument().getAsString(m_Policy);
        }
      }
      else if (const NonTypeTemplateParmDecl *NTTP =
               dyn_cast<NonTypeTemplateParmDecl>(Param)) {
        m_Out << NTTP->getType().getAsString(m_Policy);
        if (NTTP->isParameterPack() && !isa<PackExpansionType>(NTTP->getType()))
          m_Out << "...";

        if (IdentifierInfo *Name = NTTP->getIdentifier()) {
          m_Out << ' ';
          m_Out << Name->getName();
        }

        if (Args) {
          m_Out << " = ";
          Args->get(i).print(m_Policy, m_Out);
        }
        else if (NTTP->hasDefaultArgument() &&
                 !NTTP->defaultArgumentWasInherited()) {
          m_Out << " = ";
          NTTP->getDefaultArgument()->printPretty(m_Out, 0, m_Policy,
                                                  m_Indentation);
        }
      }
      else if (const TemplateTemplateParmDecl *TTPD =
               dyn_cast<TemplateTemplateParmDecl>(Param)) {
        VisitTemplateDecl(TTPD);
        // FIXME: print the default argument, if present.
      }
    }

    m_Out << "> ";
  }

  void ForwardDeclPrinter::VisitTemplateDecl(const TemplateDecl *D) {

    PrintTemplateParameters(D->getTemplateParameters());

    if (const TemplateTemplateParmDecl *TTP =
          dyn_cast<TemplateTemplateParmDecl>(D)) {
      m_Out << "class ";
    if (TTP->isParameterPack())
      m_Out << "...";
    m_Out << D->getName();
    }
    else {
      Visit(D->getTemplatedDecl());
    }
  }

  void ForwardDeclPrinter::VisitFunctionTemplateDecl(FunctionTemplateDecl *D) {
    if (shouldSkipFunction(D->getAsFunction())) {
      m_SkipFlag = true;
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

    return VisitRedeclarableTemplateDecl(D);
  }

  void ForwardDeclPrinter::VisitClassTemplateDecl(ClassTemplateDecl *D) {
    if (D->getName().size() == 0 ) {
      m_SkipFlag = true;
      return;
    }

    if (m_PrintInstantiation) {
      TemplateParameterList *Params = D->getTemplateParameters();
      for (ClassTemplateDecl::spec_iterator I = D->spec_begin(),
             E = D->spec_end(); I != E; ++I) {
        PrintTemplateParameters(Params, &(*I)->getTemplateArgs());
        Visit(*I);
        m_Out << '\n';
      }
    }

    return VisitRedeclarableTemplateDecl(D);
  }

  void ForwardDeclPrinter::
  VisitClassTemplateSpecializationDecl(ClassTemplateSpecializationDecl* D) {
    //D->dump();
  }

  void ForwardDeclPrinter::printSemiColon(bool flag) {
    if (flag) {
      if (!m_SkipFlag)
        m_Out << ";\n";
      else
        m_SkipFlag = false;
    }
    else m_Out << ";\n";
  }

  bool ForwardDeclPrinter::hasNestedNameSpecifier(QualType q) {
    //FIXME: This is a workaround and filters out many acceptable cases
    std::string str = q.getAsString();
    return str.find("::") != std::string::npos;

//    if (const ElaboratedType *E = dyn_cast<ElaboratedType>(q)) {
//      NestedNameSpecifier* NNS = E->getQualifier();
//      if (!NNS)
//        return false;
//      return NNS->isDependent() || NNS->isInstantiationDependent();
//    }
//    return false;

//          std::string str = q.getAsString();
//          if(str.find("::") != std::string::npos)
//            llvm::outs() << q.getAsString() << " " << q.getTypePtr()->getTypeClassName()<<"\n";

    return false;
  }

  bool ForwardDeclPrinter::isOperator(FunctionDecl *D) {
    return D->getNameAsString().find("operator") == 0;
  }

  bool ForwardDeclPrinter::shouldSkipFunction(FunctionDecl *D) {
    bool param = false;
    //will be true if any of the params turn out to have nested types

    for (unsigned i = 0, e = D->getNumParams(); i != e; ++i) {
      if (hasNestedNameSpecifier(D->getParamDecl(i)->getType()))
        param = true;
    }

    if (D->getNameAsString().size() == 0
        || D->getNameAsString()[0] == '_'
        || D->getStorageClass() == SC_Static
        || D->isCXXClassMember()
        || hasNestedNameSpecifier(D->getReturnType())
        || param
        || isOperator(D) )
      return true;
    return false;
  }
}//end namespace cling
