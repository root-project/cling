#include "ForwardDeclPrinter.h"

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

  raw_ostream& ForwardDeclPrinter::Indent(unsigned Indentation) {
    for (unsigned i = 0; i != Indentation; ++i)
      Out << "  ";
    return Out;
  }
  void ForwardDeclPrinter::prettyPrintAttributes(Decl *D) {
//      if (Policy.PolishForDeclaration)
//        return;

//      if (D->hasAttrs()) {
//        AttrVec &Attrs = D->getAttrs();
//        for (AttrVec::const_iterator i=Attrs.begin(), e=Attrs.end(); i!=e; ++i) {
//          Attr *A = *i;
//          A->printPretty(Out, Policy);
//        }
//      }
    Out << " __attribute__((annotate(\""
        << m_SMgr.getFilename(D->getSourceRange().getBegin()) << "\"))) ";
  }

  void ForwardDeclPrinter::ProcessDeclGroup(SmallVectorImpl<Decl*>& Decls) {
    this->Indent();
    Decl::printGroup(Decls.data(), Decls.size(), Out, Policy, Indentation);
    Out << ";\n";
    Decls.clear();

  }

  void ForwardDeclPrinter::Print(AccessSpecifier AS) {
    switch(AS) {
      case AS_none:      llvm_unreachable("No access specifier!");
      case AS_public:    Out << "public"; break;
      case AS_protected: Out << "protected"; break;
      case AS_private:   Out << "private"; break;
    }
  }

    //----------------------------------------------------------------------------
    // Common C declarations
    //----------------------------------------------------------------------------

  void ForwardDeclPrinter::VisitDeclContext(DeclContext *DC, bool Indent) {
    if (Policy.TerseOutput)
      return;
    if (Indent)
      Indentation += Policy.Indentation;

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
        Indentation -= Policy.Indentation;
        this->Indent();
        Print(D->getAccess());
        Out << ":\n";
        Indentation += Policy.Indentation;
        continue;
    }

    this->Indent();
    Visit(*D);

    // FIXME: Need to be able to tell the FwdPrinter when
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
        Out << Terminator;
    Out << "\n";
    }

    if (!Decls.empty())
        ProcessDeclGroup(Decls);

    if (Indent)
        Indentation -= Policy.Indentation;
  }

  void ForwardDeclPrinter::VisitTranslationUnitDecl(TranslationUnitDecl *D) {
      VisitDeclContext(D, false);
  }

  void ForwardDeclPrinter::VisitTypedefDecl(TypedefDecl *D) {
    if (!Policy.SuppressSpecifiers) {
      Out << "typedef ";

    if (D->isModulePrivate())
      Out << "__module_private__ ";
    }
    D->getTypeSourceInfo()->getType().print(Out, Policy, D->getName());
    prettyPrintAttributes(D);
//      Indent() << ";\n";
  }

  void ForwardDeclPrinter::VisitTypeAliasDecl(TypeAliasDecl *D) {
      /*FIXME: Ugly Hack*/
//      if(!D->getLexicalDeclContext()->isNamespace()
//              && !D->getLexicalDeclContext()->isFileContext())
//          return;
    Out << "using " << *D;
    prettyPrintAttributes(D);
    Out << " = " << D->getTypeSourceInfo()->getType().getAsString(Policy);
//      Indent() << ";\n";
  }

  void ForwardDeclPrinter::VisitEnumDecl(EnumDecl *D) {
    if (D->getName().size() == 0)
      return;

    if (!Policy.SuppressSpecifiers && D->isModulePrivate())
      Out << "__module_private__ ";
    Out << "enum ";
    prettyPrintAttributes(D);
    if (D->isScoped()) {
      if (D->isScopedUsingClassTag())
        Out << "class ";
      else
        Out << "struct ";
    }
    Out << *D;

//      if (D->isFixed())
    Out << " : " << D->getIntegerType().stream(Policy);

//      if (D->isCompleteDefinition()) {
//        Out << " {\n";
//        VisitDeclContext(D);
//        Indent() << "};\n";
//      }


    Indent() << ";\n";
  }

  void ForwardDeclPrinter::VisitRecordDecl(RecordDecl *D) {
    if (!Policy.SuppressSpecifiers && D->isModulePrivate())
      Out << "__module_private__ ";
    Out << D->getKindName();
    prettyPrintAttributes(D);
    if (D->getIdentifier())
      Out << ' ' << *D;

//    if (D->isCompleteDefinition()) {
//      Out << " {\n";
//      VisitDeclContext(D);
//      Indent() << "}";
//    }
  }

  void ForwardDeclPrinter::VisitEnumConstantDecl(EnumConstantDecl *D) {
    Out << *D;
    if (Expr *Init = D->getInitExpr()) {
      Out << " = ";
      Init->printPretty(Out, 0, Policy, Indentation);
    }
  }

  void ForwardDeclPrinter::VisitFunctionDecl(FunctionDecl *D) {
    if (D->getNameAsString().size() == 0 || D->getNameAsString()[0] == '_')
      return;
    if (D->getStorageClass() == SC_Static)
      return;
     /*FIXME:Ugly Hack: should idealy never be triggerred */
    if (D->isCXXClassMember()) {
      return;
    }

    CXXConstructorDecl *CDecl = dyn_cast<CXXConstructorDecl>(D);
      CXXConversionDecl *ConversionDecl = dyn_cast<CXXConversionDecl>(D);
      /*FIXME:Ugly Hack*/
//      if (CDecl||ConversionDecl)
//          return;

    if (!Policy.SuppressSpecifiers) {
      switch (D->getStorageClass()) {
        case SC_None: break;
        case SC_Extern: Out << "extern "; break;
        case SC_Static: Out << "static "; break;
        case SC_PrivateExtern: Out << "__private_extern__ "; break;
        case SC_Auto: case SC_Register: case SC_OpenCLWorkGroupLocal:
          llvm_unreachable("invalid for functions");
      }

      if (D->isInlineSpecified())  Out << "inline ";
      if (D->isVirtualAsWritten()) Out << "virtual ";
      if (D->isModulePrivate())    Out << "__module_private__ ";
      if (D->isConstexpr() && !D->isExplicitlyDefaulted()) Out << "constexpr ";
      if ((CDecl && CDecl->isExplicitSpecified()) ||
          (ConversionDecl && ConversionDecl->isExplicit()))
        Out << "explicit ";
    }

    PrintingPolicy SubPolicy(Policy);
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
        ForwardDeclPrinter ParamPrinter(POut, m_SMgr, SubPolicy, Indentation);
        for (unsigned i = 0, e = D->getNumParams(); i != e; ++i) {
          if (i) POut << ", ";
          ParamPrinter.VisitParmVarDecl(D->getParamDecl(i));
        }

        if (FT->isVariadic()) {
          if (D->getNumParams()) POut << ", ";
          POut << "...";
        }
      } else if (D->doesThisDeclarationHaveABody() && !D->hasPrototype()) {
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
                                               Indentation);
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
              Out << Proto;
              Proto.clear();
              HasInitializerList = true;
            } else
              Out << ", ";

            if (BMInitializer->isAnyMemberInitializer()) {
              FieldDecl *FD = BMInitializer->getAnyMember();
              Out << *FD;
            } else {
              Out << QualType(BMInitializer->getBaseClass(), 0).getAsString(Policy);
            }

            Out << "(";
            if (!BMInitializer->getInit()) {
              // Nothing to print
            } else {
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
                SimpleInit->printPretty(Out, 0, Policy, Indentation);
              else {
                for (unsigned I = 0; I != NumArgs; ++I) {
                  if (isa<CXXDefaultArgExpr>(Args[I]))
                    break;

                  if (I)
                    Out << ", ";
                  Args[I]->printPretty(Out, 0, Policy, Indentation);
                }
              }
            }
            Out << ")";
            if (BMInitializer->isPackExpansion())
              Out << "...";
          }
        } else if (!ConversionDecl && !isa<CXXDestructorDecl>(D)) {
          if (FT && FT->hasTrailingReturn()) {
            Out << "auto " << Proto << " -> ";
            Proto.clear();
          }
          AFT->getReturnType().print(Out, Policy, Proto);
          Proto.clear();
        }
        Out << Proto;
      } else {
        Ty.print(Out, Policy, Proto);
      }

      prettyPrintAttributes(D);

      if (D->isPure())
        Out << " = 0";
      else if (D->isDeletedAsWritten())
        Out << " = delete";
      else if (D->isExplicitlyDefaulted())
        Out << " = default";
      else if (D->doesThisDeclarationHaveABody() && !Policy.TerseOutput) {
        if (!D->hasPrototype() && D->getNumParams()) {
          // This is a K&R function definition, so we need to print the
          // parameters.
          Out << '\n';
          ForwardDeclPrinter ParamPrinter(Out,m_SMgr, SubPolicy, Indentation);
          Indentation += Policy.Indentation;
          for (unsigned i = 0, e = D->getNumParams(); i != e; ++i) {
            Indent();
            ParamPrinter.VisitParmVarDecl(D->getParamDecl(i));
            Out << ";\n";
          }
          Indentation -= Policy.Indentation;
        } else
       Out << ' ';

    //    D->getBody()->printPretty(Out, 0, SubPolicy, Indentation);

    }
//      Out << " __attribute__((annotate(\""
//          << m_SMgr.getFilename(D->getSourceRange().getBegin())<< "\"))) ";
//      Out <<";\n";
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
    if (!Policy.SuppressSpecifiers && D->isMutable())
      Out << "mutable ";
    if (!Policy.SuppressSpecifiers && D->isModulePrivate())
      Out << "__module_private__ ";
    Out << D->getASTContext().getUnqualifiedObjCPointerType(D->getType()).
        stream(Policy, D->getName());

    if (D->isBitField()) {
      Out << " : ";
      D->getBitWidth()->printPretty(Out, 0, Policy, Indentation);
    }

    Expr *Init = D->getInClassInitializer();
    if (!Policy.SuppressInitializers && Init) {
      if (D->getInClassInitStyle() == ICIS_ListInit)
        Out << " ";
      else
        Out << " = ";
      Init->printPretty(Out, 0, Policy, Indentation);
    }
    prettyPrintAttributes(D);
  }

  void ForwardDeclPrinter::VisitLabelDecl(LabelDecl *D) {
    Out << *D << ":";
  }


  void ForwardDeclPrinter::VisitVarDecl(VarDecl *D) {
    //FIXME:Ugly hack
    if(D->getStorageClass() == SC_Static) {
      return;
    }
    if(D->isDefinedOutsideFunctionOrMethod() && !(D->getStorageClass() == SC_Extern))
      Out << "extern ";

    if (!Policy.SuppressSpecifiers) {
      StorageClass SC = D->getStorageClass();
      if (SC != SC_None)
        Out << VarDecl::getStorageClassSpecifierString(SC) << " ";

      switch (D->getTSCSpec()) {
        case TSCS_unspecified:
          break;
        case TSCS___thread:
          Out << "__thread ";
          break;
        case TSCS__Thread_local:
          Out << "_Thread_local ";
          break;
        case TSCS_thread_local:
          Out << "thread_local ";
          break;
      }

      if (D->isModulePrivate())
        Out << "__module_private__ ";
    }

    QualType T = D->getTypeSourceInfo()
      ? D->getTypeSourceInfo()->getType()
      : D->getASTContext().getUnqualifiedObjCPointerType(D->getType());

    //FIXME: It prints restrict as restrict
    //which is not valid C++
    //Should be __restrict
    //So, we ignore restrict here
    T.removeLocalRestrict();
    T.print(Out, Policy, D->getName());
    T.addRestrict();

    Expr *Init = D->getInit();
    if (!Policy.SuppressInitializers && Init) {
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
      if ((D->getInitStyle() == VarDecl::CallInit) && !isa<ParenListExpr>(Init))
        Out << "(";
      else if (D->getInitStyle() == VarDecl::CInit) {
//            Out << " = "; //FOR skipping default function args
      }
//          Init->printPretty(Out, 0, Policy, Indentation);//FOR skipping defalt function args
      if ((D->getInitStyle() == VarDecl::CallInit) && !isa<ParenListExpr>(Init))
        Out << ")";
      }
    }
    if(D->isDefinedOutsideFunctionOrMethod())
      prettyPrintAttributes(D);
  }

  void ForwardDeclPrinter::VisitParmVarDecl(ParmVarDecl *D) {
    VisitVarDecl(D);
  }

  void ForwardDeclPrinter::VisitFileScopeAsmDecl(FileScopeAsmDecl *D) {
    Out << "__asm (";
    D->getAsmString()->printPretty(Out, 0, Policy, Indentation);
    Out << ")";
  }

  void ForwardDeclPrinter::VisitImportDecl(ImportDecl *D) {
    Out << "@import " << D->getImportedModule()->getFullModuleName()
        << ";\n";
  }

  void ForwardDeclPrinter::VisitStaticAssertDecl(StaticAssertDecl *D) {
    Out << "static_assert(";
    D->getAssertExpr()->printPretty(Out, 0, Policy, Indentation);
    Out << ", ";
    D->getMessage()->printPretty(Out, 0, Policy, Indentation);
    Out << ")";
  }

    //----------------------------------------------------------------------------
    // C++ declarations
    //----------------------------------------------------------------------------
  void ForwardDeclPrinter::VisitNamespaceDecl(NamespaceDecl *D) {
    if (D->isInline())
      Out << "inline ";
    Out << "namespace " << *D << " {\n";
//      VisitDeclContext(D);
    for(auto dit=D->decls_begin();dit!=D->decls_end();++dit) {
      this->Visit(*dit);
      Out << ";\n";
    }

    Indent() << "}\n";
  }

  void ForwardDeclPrinter::VisitUsingDirectiveDecl(UsingDirectiveDecl *D) {
    Out << "using namespace ";
    if (D->getQualifier())
      D->getQualifier()->print(Out, Policy);
    Out << *D->getNominatedNamespaceAsWritten();
  }

  void ForwardDeclPrinter::VisitNamespaceAliasDecl(NamespaceAliasDecl *D) {
    Out << "namespace " << *D << " = ";
    if (D->getQualifier())
      D->getQualifier()->print(Out, Policy);
    Out << *D->getAliasedNamespace();
  }

  void ForwardDeclPrinter::VisitEmptyDecl(EmptyDecl *D) {
//    prettyPrintAttributes(D);
  }

  void ForwardDeclPrinter::VisitCXXRecordDecl(CXXRecordDecl *D) {

    if(ClassDeclNames.find(D->getNameAsString()) != ClassDeclNames.end()
          /*|| D->getName().startswith("_")*/)
      return;

    if (D->getNameAsString().size() == 0)
      return;

    if (!Policy.SuppressSpecifiers && D->isModulePrivate())
      Out << "__module_private__ ";
    Out << D->getKindName();
    Out << " __attribute__((annotate(\""
        << m_SMgr.getFilename(D->getSourceRange().getBegin()) << "\"))) ";
    if (D->getIdentifier())
      Out << ' ' << *D ;

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
//      Out << ";\n";
    ClassDeclNames.insert(D->getNameAsString());
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

    Out << "extern \"" << l << "\" ";
    if (D->hasBraces()) {
      Out << "{\n";
      VisitDeclContext(D);
      Indent() << "}";
    } else
      Visit(*D->decls_begin());
  }

  void ForwardDeclPrinter::PrintTemplateParameters(const TemplateParameterList *Params,
                                              const TemplateArgumentList *Args) {
    assert(Params);
    assert(!Args || Params->size() == Args->size());

    Out << "template <";

    for (unsigned i = 0, e = Params->size(); i != e; ++i) {
      if (i != 0)
        Out << ", ";

      const Decl *Param = Params->getParam(i);
      if (const TemplateTypeParmDecl *TTP =
            dyn_cast<TemplateTypeParmDecl>(Param)) {

        if (TTP->wasDeclaredWithTypename())
          Out << "typename ";
        else
          Out << "class ";

        if (TTP->isParameterPack())
          Out << "... ";

        Out << *TTP;

        if (Args) {
          Out << " = ";
          Args->get(i).print(Policy, Out);
        } else if (TTP->hasDefaultArgument()) {
//            Out << " = ";
//            Out << TTP->getDefaultArgument().getAsString(Policy);
          };
      } else if (const NonTypeTemplateParmDecl *NTTP =
                   dyn_cast<NonTypeTemplateParmDecl>(Param)) {
        Out << NTTP->getType().getAsString(Policy);
        if (NTTP->isParameterPack() && !isa<PackExpansionType>(NTTP->getType()))
          Out << "...";

        if (IdentifierInfo *Name = NTTP->getIdentifier()) {
          Out << ' ';
          Out << Name->getName();
        }

        if (Args) {
          Out << " = ";
          Args->get(i).print(Policy, Out);
        } else if (NTTP->hasDefaultArgument()) {
//            Out << " = ";
//            NTTP->getDefaultArgument()->printPretty(Out, 0, Policy, Indentation);
        }
      } else if (const TemplateTemplateParmDecl *TTPD =
                   dyn_cast<TemplateTemplateParmDecl>(Param)) {
        VisitTemplateDecl(TTPD);
          // FIXME: print the default argument, if present.
      }
    }

    Out << "> ";
  }

  void ForwardDeclPrinter::VisitTemplateDecl(const TemplateDecl *D) {
    PrintTemplateParameters(D->getTemplateParameters());

    if (const TemplateTemplateParmDecl *TTP =
          dyn_cast<TemplateTemplateParmDecl>(D)) {
      Out << "class ";
    if (TTP->isParameterPack())
      Out << "...";
    Out << D->getName();
    } else {
      Visit(D->getTemplatedDecl());
    }
  }

  void ForwardDeclPrinter::VisitFunctionTemplateDecl(FunctionTemplateDecl *D) {
    if(D->getNameAsString().size() == 0 || D->getNameAsString()[0] == '_')
      return;
//    if (D->getStorageClass() == SC_Static)
//      return;
    /*FIXME:Ugly Hack: should idealy never be triggerred */
    if (D->isCXXClassMember())
      return;

    if (PrintInstantiation) {
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
    if(ClassDeclNames.find(D->getNameAsString()) != ClassDeclNames.end()
      || D->getName().size() == 0 )
     return;
    if (PrintInstantiation) {
      TemplateParameterList *Params = D->getTemplateParameters();
      for (ClassTemplateDecl::spec_iterator I = D->spec_begin(),
           E = D->spec_end(); I != E; ++I) {
        PrintTemplateParameters(Params, &(*I)->getTemplateArgs());
        Visit(*I);
        Out << '\n';
      }
    }

    return VisitRedeclarableTemplateDecl(D);
  }
  void ForwardDeclPrinter::VisitClassTemplateSpecializationDecl
        (clang::ClassTemplateSpecializationDecl* D) {

      //D->dump();

  }
}//end namespace cling
