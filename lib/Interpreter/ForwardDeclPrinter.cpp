#include "ForwardDeclPrinter.h"

#include "cling/Interpreter/DynamicLibraryManager.h"
#include "cling/Interpreter/Transaction.h"
#include "cling/Utils/AST.h"
#include "cling/Utils/Output.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Type.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/Sema.h"

#include "llvm/Support/Path.h"

namespace cling {

  using namespace clang;


  ForwardDeclPrinter::ForwardDeclPrinter(llvm::raw_ostream& OutS,
                                         llvm::raw_ostream& LogS,
                                         Preprocessor& P,
                                         ASTContext& Ctx,
                                         const Transaction& T,
                                         unsigned Indentation,
                                         bool printMacros,
                                         IgnoreFilesFunc_t ignoreFiles)
    : m_Policy(clang::PrintingPolicy(clang::LangOptions())), m_Log(LogS),
      m_Indentation(Indentation), m_PP(P), m_SMgr(P.getSourceManager()),
      m_Ctx(Ctx), m_SkipFlag(false), m_IgnoreFile(ignoreFiles) {
    m_PrintInstantiation = false;
    m_Policy.SuppressTagKeyword = true;

    m_Policy.Bool = true; // Avoid printing _Bool instead of bool

    m_StreamStack.push(&OutS);

    const clang::Builtin::Context& BuiltinCtx = m_Ctx.BuiltinInfo;
    for (unsigned i = clang::Builtin::NotBuiltin+1;
         i != clang::Builtin::FirstTSBuiltin; ++i)
      m_BuiltinNames.insert(BuiltinCtx.getName(i));

    for (auto&& BuiltinInfo: m_Ctx.getTargetInfo().getTargetBuiltins())
        m_BuiltinNames.insert(BuiltinInfo.Name);


    // Suppress some unfixable warnings.
    // TODO: Find proper fix for these issues
    Out() << "#pragma clang diagnostic ignored \"-Wkeyword-compat\"" << "\n";
    Out() << "#pragma clang diagnostic ignored \"-Wignored-attributes\"" <<"\n";
    Out() << "#pragma clang diagnostic ignored \"-Wreturn-type-c-linkage\"" <<"\n";
    // Inject a special marker:
    Out() << "extern int __Cling_AutoLoading_Map;\n";

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
          //if (!(*dit)->getDeclContext()->isTranslationUnit())
          //  continue;

          Visit(*dit);
          resetSkip();
        }

      }
    }
    if (printMacros) {
      for (const auto &m : macrodefs) {
        Out() << "#undef " << m << "\n";
      }
    }
  }

  void ForwardDeclPrinter::Visit(clang::QualType QT) {
    QT = utils::TypeName::GetFullyQualifiedType(QT, m_Ctx);
    Visit(QT.getTypePtr());
  }
  void ForwardDeclPrinter::Visit(clang::Decl *D) {
    auto Insert = m_Visited.insert(std::pair<const clang::Decl*, bool>(
                                             getCanonicalOrNamespace(D), true));
    if (!Insert.second) {
      // Already fwd declared or skipped.
      if (!Insert.first->second) {
        // Already skipped before; notify callers.
        skipDecl(D, 0);
      }
      return;
    }
    if (shouldSkip(D)) {
      // shouldSkip() called skipDecl()
      m_Visited[getCanonicalOrNamespace(D)] = false;
    } else {
      clang::DeclVisitor<ForwardDeclPrinter>::Visit(D);
      if (m_SkipFlag) {
        // D was not good, flag it.
        skipDecl(D, "Dependency skipped");
        m_Visited[getCanonicalOrNamespace(D)] = false;
      }
    }
  }

  void ForwardDeclPrinter::printDeclType(llvm::raw_ostream& Stream, QualType T,
                                         StringRef DeclName, bool Pack) {
    // Normally, a PackExpansionType is written as T[3]... (for instance, as a
    // template argument), but if it is the type of a declaration, the ellipsis
    // is placed before the name being declared.
    if (auto *PET = T->getAs<PackExpansionType>()) {
      Pack = true;
      T = PET->getPattern();
    }
    T.print(Stream, m_Policy, (Pack ? "..." : "") + DeclName);
  }

  llvm::raw_ostream& ForwardDeclPrinter::Indent(unsigned Indentation) {
    for (unsigned i = 0; i != Indentation; ++i)
      Out() << "  ";
    return Out();
  }

  void ForwardDeclPrinter::prettyPrintAttributes(Decl *D) {

    if (D->getSourceRange().isInvalid())
      return;

    if (D->hasAttrs() && ! isa<FunctionDecl>(D)) {
      AttrVec &Attrs = D->getAttrs();
      for (AttrVec::const_iterator i=Attrs.begin(), e=Attrs.end(); i != e; ++i) {
        Attr *A = *i;
        if (A->isImplicit() || A->isInherited()
            || A->getKind() == attr::Kind::Final)
          continue;
        //FIXME: Remove when the printing of type_visibility attribute is fixed.
        if (!isa<AnnotateAttr>(A))
          continue;
        A->printPretty(Out(), m_Policy);
      }
    }

     auto &smgr = m_SMgr;
     auto getIncludeFileName = [D, &smgr](PresumedLoc loc) {
       clang::SourceLocation includeLoc =
           smgr.getSpellingLoc(loc.getIncludeLoc());
       bool invalid = true;
       const char* includeText = smgr.getCharacterData(includeLoc, &invalid);
       assert(!invalid && "Invalid source data");
       assert(includeText && "Cannot find #include location");
       // With C++ modules it's possible that we get 'include <header>'
       // instead of just '<header>' here. Let's just skip this text at the
       // start in this case as the '<header>' still has the correct value.
       // FIXME: Once the C++ modules replaced the forward decls, remove this.
       if (D->getASTContext().getLangOpts().Modules &&
           llvm::StringRef(includeText).startswith("include ")) {
         includeText += strlen("include ");
       }

       assert((includeText[0] == '<' || includeText[0] == '"') &&
              "Unexpected #include delimiter");
       char endMarker = includeText[0] == '<' ? '>' : '"';
       ++includeText;
       const char* includeEnd = includeText;
       while (*includeEnd != endMarker && *includeEnd) {
         ++includeEnd;
       }
       assert(includeEnd && "Cannot find end of #include file name");
       return llvm::StringRef(includeText, includeEnd - includeText);
     };

     auto& PP = m_PP;
     auto isDirectlyReacheable = [&PP](llvm::StringRef FileName) {
       const FileEntry* FE = nullptr;
       SourceLocation fileNameLoc;
       bool isAngled = false;
       const DirectoryLookup* FromDir = nullptr;
       const FileEntry* FromFile = nullptr;
       const DirectoryLookup* CurDir = nullptr;

       FE = PP.LookupFile(fileNameLoc, FileName, isAngled, FromDir, FromFile,
                          CurDir, /*SearchPath*/ 0,
                          /*RelativePath*/ 0, /*suggestedModule*/ 0,
                          /*IsMapped*/ 0, /*SkipCache*/ false,
                          /*OpenFile*/ false, /*CacheFail*/ true);
       // Return true if we can '#include' the given filename
       return FE != nullptr;
     };

     SourceLocation spellingLoc = m_SMgr.getSpellingLoc(D->getLocStart());
     // Walk up the include chain.
     PresumedLoc PLoc = m_SMgr.getPresumedLoc(spellingLoc);
     llvm::SmallVector<PresumedLoc, 16> PLocs;
     llvm::SmallVector<StringRef, 16> PLocNames;
     while (!m_IgnoreFile(PLoc)) {
       if (!m_SMgr.getPresumedLoc(PLoc.getIncludeLoc()).isValid()) break;
       PLocs.push_back(PLoc);
       StringRef name(getIncludeFileName(PLoc));

       // We record in PLocNames only the include file names that can be
       // reached directly.  Whenever a #include is parsed in addition to
       // the record include path, the directory where the file containing
       // the #include is located is also added implicitly and temporarily
       // to the include path.  So if the include path is empty and a file
       // is include via a full pathname it can still #include file in its
       // (sub)directory using their relative path.
       // Similarly a file included via a sub-directory of the include path
       // (eg. #include "Product/mainheader.h") can include header files in
       // the same subdirectory without mentioning it
       // (eg. #include "otherheader_in_Product.h")
       // Since we do not (want to) record the actual directory in which is
       // located the header with the #include we are looking at, if the
       // #include is relative to that directory we will not be able to find
       // it back and thus there is no point in recording it.
       if (isDirectlyReacheable(name)) {
         PLocNames.push_back(name);
       }
       PLoc = m_SMgr.getPresumedLoc(PLoc.getIncludeLoc());
    }

    if (PLocs.empty() /* declared in dictionary payload*/)
       return;
    else if (PLocNames.empty()) {
      // In this case, all the header file name are of the 'unreacheable' type,
      // most likely because the first one was related to the linkdef file and
      // the linkdef file was pass using a full path name.
      // We are not (easy) to find it back, nonetheless record it as is, just
      // in case the user add the missing include path at run-time.
      if (PLocs.size() > 1) {
        Out() << " __attribute__((annotate(\"$clingAutoload$";
        Out() << getIncludeFileName(PLocs[0]);
        Out() << "\"))) ";
      }
      Out() << " __attribute__((annotate(\"$clingAutoload$";
      Out() << getIncludeFileName(PLocs[PLocs.size() - 1]);
      Out() << "\"))) ";
      return;
    }

    if (PLocNames.size() > 1) {
      Out() << " __attribute__((annotate(\"$clingAutoload$";
      Out() << PLocNames[0];
      Out() << "\"))) ";
    }
    Out() << " __attribute__((annotate(\"$clingAutoload$";
    Out() << PLocNames[PLocNames.size()-1];
    Out() << "\"))) ";
  }


    //----------------------------------------------------------------------------
    // Common C declarations
    //----------------------------------------------------------------------------


  void ForwardDeclPrinter::VisitTranslationUnitDecl(TranslationUnitDecl *D) {
    //    VisitDeclContext(D, false);
    assert(0 && "ForwardDeclPrinter::VisitTranslationUnitDecl unexpected");
    for (auto it = D->decls_begin(); it != D->decls_end(); ++it) {
      Visit(*it);
    }
  }

  void ForwardDeclPrinter::VisitTypedefDecl(TypedefDecl *D) {
    QualType q = D->getTypeSourceInfo()->getType();
    Visit(q);
    if (m_SkipFlag) {
      skipDecl(D, "Underlying type failed");
      return;
    }

    std::string closeBraces = PrintEnclosingDeclContexts(Out(),
                                                         D->getDeclContext());
    if (!m_Policy.SuppressSpecifiers)
      Out() << "typedef ";
    if (D->isModulePrivate())
      Out() << "__module_private__ ";

    if (q.isRestrictQualified()){
      q.removeLocalRestrict();
      q.print(Out(), m_Policy, "");
      Out() << " __restrict " << D->getName(); //TODO: Find some policy that does this automatically
    }
    else {
      q.print(Out(), m_Policy, D->getName());
    }
    prettyPrintAttributes(D);
    Out() << ';' << closeBraces << '\n';
  }

  void ForwardDeclPrinter::VisitTypeAliasDecl(TypeAliasDecl *D) {
      /*FIXME: Ugly Hack*/
//      if(!D->getLexicalDeclContext()->isNamespace()
//              && !D->getLexicalDeclContext()->isFileContext())
//          return;
    std::string closeBraces = PrintEnclosingDeclContexts(Out(),
                                                         D->getDeclContext());
    Out() << "using " << *D;
    prettyPrintAttributes(D);
    Out() << " = " << D->getTypeSourceInfo()->getType().getAsString(m_Policy)
          << ';' << closeBraces << '\n';
  }

  void ForwardDeclPrinter::VisitEnumDecl(EnumDecl *D) {
    std::string closeBraces = PrintEnclosingDeclContexts(Out(),
                                                         D->getDeclContext());
    if (!m_Policy.SuppressSpecifiers && D->isModulePrivate())
      Out() << "__module_private__ ";
    Out() << "enum ";
    prettyPrintAttributes(D);
    if (D->isScoped()) {
      if (D->isScopedUsingClassTag())
        Out() << "class ";
      else
        Out() << "struct ";
    }
    Out() << *D;

//      if (D->isFixed())
    Out() << " : " << D->getIntegerType().stream(m_Policy)
          << ';' << closeBraces << '\n';
  }

  void ForwardDeclPrinter::VisitRecordDecl(RecordDecl *D) {
    std::string closeBraces;
    bool isTemplatePattern = false;
    if (CXXRecordDecl* CXXRD = dyn_cast<CXXRecordDecl>(D))
      isTemplatePattern = CXXRD->getDescribedClassTemplate();
    if (!isTemplatePattern)
      closeBraces = PrintEnclosingDeclContexts(Out(), D->getDeclContext());
    if (!m_Policy.SuppressSpecifiers && D->isModulePrivate())
      Out() << "__module_private__ ";
    Out() << D->getKindName();
    prettyPrintAttributes(D);
    if (D->getIdentifier())
      Out() << ' ' << *D << ';' << closeBraces << '\n';


//    if (D->isCompleteDefinition()) {
//      Out << " {\n";
//      VisitDeclContext(D);
//      Indent() << "}";
//    }
  }

  void ForwardDeclPrinter::VisitFunctionDecl(FunctionDecl *D) {
    bool hasTrailingReturn = false;

    CXXConstructorDecl *CDecl = dyn_cast<CXXConstructorDecl>(D);
    CXXConversionDecl *ConversionDecl = dyn_cast<CXXConversionDecl>(D);

    Visit(D->getReturnType());
    if (m_SkipFlag) {
      skipDecl(D, "Return type failed");
      return;
    }

    StreamRAII stream(*this);

    std::string closeBraces = PrintEnclosingDeclContexts(Out(),
                                                         D->getDeclContext());
    if (!m_Policy.SuppressSpecifiers) {
      switch (D->getStorageClass()) {
      case SC_None: break;
      case SC_Extern: Out() << "extern "; break;
      case SC_Static: Out() << "static "; break;
      case SC_PrivateExtern: Out() << "__private_extern__ "; break;
      case SC_Auto: case SC_Register:
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
        StreamRAII subStream(*this, &SubPolicy);
        for (unsigned i = 0, e = D->getNumParams(); i != e; ++i) {
          if (i) Out() << ", ";
          Visit(D->getParamDecl(i));
          if (m_SkipFlag) {
            skipDecl(D, "Parameter failed");
            return;
          }
        }

        if (FT->isVariadic()) {
          if (D->getNumParams()) Out() << ", ";
          Out() << "...";
        }
        Proto += subStream.take();
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
        StreamRAII subStream(*this, &SubPolicy);
        m_Indentation += m_Policy.Indentation;
        for (unsigned i = 0, e = D->getNumParams(); i != e; ++i) {
          Indent();
          Visit(D->getParamDecl(i));
          Out() << ";\n";
        }
        m_Indentation -= m_Policy.Indentation;
        if (m_SkipFlag) {
          return;
        }
      } else
        Out() << ' ';
      //    D->getBody()->printPretty(Out, 0, SubPolicy, Indentation);
    }
    Out() << ';' << closeBraces << '\n';
  }

  void ForwardDeclPrinter::VisitFriendDecl(FriendDecl *D) {
  }

  void ForwardDeclPrinter::VisitFieldDecl(FieldDecl *D) {
    std::string closeBraces = PrintEnclosingDeclContexts(Out(),
                                                         D->getDeclContext());
    if (!m_Policy.SuppressSpecifiers && D->isMutable())
      Out() << "mutable ";
    if (!m_Policy.SuppressSpecifiers && D->isModulePrivate())
      Out() << "__module_private__ ";
    Out() << m_Ctx.getUnqualifiedObjCPointerType(D->getType()).
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
    Out() << ';' << closeBraces << '\n';
  }

  void ForwardDeclPrinter::VisitLabelDecl(LabelDecl *D) {
    Out() << *D << ":";
  }


  void ForwardDeclPrinter::VisitVarDecl(VarDecl *D) {
    QualType T = D->getTypeSourceInfo()
      ? D->getTypeSourceInfo()->getType()
      : m_Ctx.getUnqualifiedObjCPointerType(D->getType());

    Visit(T);
    if (m_SkipFlag) {
      skipDecl(D, "Variable type failed.");
      return;
    }

    std::string closeBraces = PrintEnclosingDeclContexts(Out(),
                                                         D->getDeclContext());
    if (D->isDefinedOutsideFunctionOrMethod() && D->getStorageClass() != SC_Extern
        && D->getStorageClass() != SC_Static)
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

    //FIXME: It prints restrict as restrict
    //which is not valid C++
    //Should be __restrict
    //So, we ignore restrict here
    T.removeLocalRestrict();
//    T.print(Out(), m_Policy, D->getName());
    printDeclType(Out(), T,D->getName());
    //    cling::outs()<<D->getName()<<"\n";
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
              printDeclType(Out(), D->getType(),"");
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

    Out() << ';' << closeBraces << '\n';
  }

  void ForwardDeclPrinter::VisitParmVarDecl(ParmVarDecl *D) {
    VisitVarDecl(D);
  }

  void ForwardDeclPrinter::VisitFileScopeAsmDecl(FileScopeAsmDecl *D) {
    std::string closeBraces = PrintEnclosingDeclContexts(Out(),
                                                         D->getDeclContext());
    Out() << "__asm (";
    D->getAsmString()->printPretty(Out(), 0, m_Policy, m_Indentation);
    Out() << ");" << closeBraces << '\n';
  }

  void ForwardDeclPrinter::VisitImportDecl(ImportDecl *D) {
    Out() << "@import " << D->getImportedModule()->getFullModuleName()
          << ";\n";
  }

  void ForwardDeclPrinter::VisitStaticAssertDecl(StaticAssertDecl *D) {
    std::string closeBraces = PrintEnclosingDeclContexts(Out(),
                                                         D->getDeclContext());
    Out() << "static_assert(";
    D->getAssertExpr()->printPretty(Out(), 0, m_Policy, m_Indentation);
    Out() << ", ";
    D->getMessage()->printPretty(Out(), 0, m_Policy, m_Indentation);
    Out() << ");" << closeBraces << '\n';
  }

  //----------------------------------------------------------------------------
  // C++ declarations
  //----------------------------------------------------------------------------
  void ForwardDeclPrinter::VisitNamespaceDecl(NamespaceDecl *D) {

//      VisitDeclContext(D);

    bool haveAnyDecl = false;
    for (auto dit=D->decls_begin();dit!=D->decls_end();++dit) {
      Visit(*dit);
      haveAnyDecl |= !m_SkipFlag;
      m_SkipFlag = false;
    }
    if (!haveAnyDecl) {
      // make sure at least one redecl of this namespace is fwd declared.
      if (D == D->getCanonicalDecl()) {
        PrintNamespaceOpen(Out(), D);
        Out() << "}\n";
      }
    }
  }

  void ForwardDeclPrinter::VisitUsingDirectiveDecl(UsingDirectiveDecl *D) {
    Visit(D->getNominatedNamespace());
    if (m_SkipFlag) {
      skipDecl(D, "Using directive's underlying namespace failed");
      return;
    }

    std::string closeBraces = PrintEnclosingDeclContexts(Out(),
                                                         D->getDeclContext());
    Out() << "using namespace ";
    if (D->getQualifier())
      D->getQualifier()->print(Out(), m_Policy);
    Out() << *D->getNominatedNamespaceAsWritten() << ';' << closeBraces << '\n';
  }

  void ForwardDeclPrinter::VisitUsingDecl(UsingDecl *D) {
    // Visit the shadow decls:
    for (auto Shadow: D->shadows())
      Visit(Shadow);

    if (m_SkipFlag) {
      skipDecl(D, "shadow decl failed");
      return;
    }
    std::string closeBraces = PrintEnclosingDeclContexts(Out(),
                                                         D->getDeclContext());
    D->print(Out(),m_Policy);
    Out() << ';' << closeBraces << '\n';
  }
  void ForwardDeclPrinter::VisitUsingShadowDecl(UsingShadowDecl *D) {
    Visit(D->getTargetDecl());
    if (m_SkipFlag)
      skipDecl(D, "target decl failed.");
  }

  void ForwardDeclPrinter::VisitTypeAliasTemplateDecl(TypeAliasTemplateDecl *D) {
  }

  void ForwardDeclPrinter::VisitNamespaceAliasDecl(NamespaceAliasDecl *D) {
    std::string closeBraces = PrintEnclosingDeclContexts(Out(),
                                                         D->getDeclContext());
    Out() << "namespace " << *D << " = ";
    if (D->getQualifier())
      D->getQualifier()->print(Out(), m_Policy);
    Out() << *D->getAliasedNamespace() << ';' << closeBraces << '\n';
  }

  void ForwardDeclPrinter::VisitEmptyDecl(EmptyDecl *D) {
//    prettyPrintAttributes(D);
  }

  void ForwardDeclPrinter::VisitTagDecl(TagDecl *D) {
    std::string closeBraces = PrintEnclosingDeclContexts(Out(),
                                                         D->getDeclContext());
    if (!m_Policy.SuppressSpecifiers && D->isModulePrivate())
      Out() << "__module_private__ ";
    Out() << D->getKindName();

//    if (D->isCompleteDefinition())
      prettyPrintAttributes(D);
    if (D->getIdentifier())
      Out() << ' ' << *D << ';' << closeBraces << '\n';
  }

  void ForwardDeclPrinter::VisitLinkageSpecDecl(LinkageSpecDecl *D) {
    for (auto it = D->decls_begin(); it != D->decls_end(); ++it) {
      Visit(*it);
      resetSkip();
    }
  }

  void ForwardDeclPrinter::PrintTemplateParameters(llvm::raw_ostream& Stream,
                                                  TemplateParameterList *Params,
                                             const TemplateArgumentList *Args) {
    assert(Params);
    assert(!Args || Params->size() == Args->size());

    Stream << "template <";

    for (unsigned i = 0, e = Params->size(); i != e; ++i) {
      if (i != 0)
        Stream << ", ";

      Decl *Param = Params->getParam(i);
      if (const TemplateTypeParmDecl *TTP =
          dyn_cast<TemplateTypeParmDecl>(Param)) {

        if (TTP->wasDeclaredWithTypename())
          Stream << "typename ";
        else
          Stream << "class ";

        if (TTP->isParameterPack())
          Stream << "...";

        Stream << *TTP;

        QualType ArgQT;
        if (Args) {
           ArgQT = Args->get(i).getAsType();
        }
        else if (TTP->hasDefaultArgument()) {
           ArgQT = TTP->getDefaultArgument();
        }
        if (!ArgQT.isNull()) {
          QualType ArgFQQT
            = utils::TypeName::GetFullyQualifiedType(ArgQT, m_Ctx);
          Visit(ArgFQQT);
          if (m_SkipFlag) {
            skipDecl(0, "type template param default failed");
            return;
          }
          Stream << " = ";
          ArgFQQT.print(Stream, m_Policy);
        }
      }
      else if (const NonTypeTemplateParmDecl *NTTP =
               dyn_cast<NonTypeTemplateParmDecl>(Param)) {
        StringRef Name;
        if (IdentifierInfo *II = NTTP->getIdentifier())
          Name = II->getName();
        printDeclType(Stream, NTTP->getType(), Name, NTTP->isParameterPack());

        if (Args) {
          Stream << " = ";
          Args->get(i).print(m_Policy, Stream);
        }
        else if (NTTP->hasDefaultArgument()) {
          Expr* DefArg = NTTP->getDefaultArgument()->IgnoreImpCasts();
          if (DeclRefExpr* DRE = dyn_cast<DeclRefExpr>(DefArg)) {
            Visit(DRE->getFoundDecl());
            if (m_SkipFlag) {
              skipDecl(0, "expression template param default failed");
              return;
            }
          } else if (isa<IntegerLiteral>(DefArg)
                     || isa<CharacterLiteral>(DefArg)
                     || isa<CXXBoolLiteralExpr>(DefArg)
                     || isa<CXXNullPtrLiteralExpr>(DefArg)
                     || isa<FloatingLiteral>(DefArg)
                     || isa<StringLiteral>(DefArg)) {
            Stream << " = ";
            DefArg->printPretty(Stream, 0, m_Policy, m_Indentation);
          } else {
            skipDecl(0, "expression template param default not a literal");
            return;
          }
        }
      }
      else if (TemplateTemplateParmDecl *TTPD =
               dyn_cast<TemplateTemplateParmDecl>(Param)) {
        Visit(TTPD);
        // FIXME: print the default argument, if present.
      }
    }

    Stream << "> ";
  }

  void ForwardDeclPrinter::VisitRedeclarableTemplateDecl(const RedeclarableTemplateDecl *D) {

    // Find redecl with template default arguments: that's the one
    // we want to forward declare.
    for (const RedeclarableTemplateDecl* RD: D->redecls()) {
      clang::TemplateParameterList* TPL = RD->getTemplateParameters();
      if (TPL->getMinRequiredArguments () < TPL->size())
        D = RD;
    }

    stdstrstream Stream;

    std::string closeBraces;
    if (!isa<TemplateTemplateParmDecl>(D))
      closeBraces = PrintEnclosingDeclContexts(Stream, D->getDeclContext());

    PrintTemplateParameters(Stream, D->getTemplateParameters());
    if (m_SkipFlag) {
      skipDecl(0, "Template parameters failed");
      return;
    }

    if (const TemplateTemplateParmDecl *TTP =
          dyn_cast<TemplateTemplateParmDecl>(D)) {
      Stream << "class ";
      if (TTP->isParameterPack())
        Out() << "...";
      Stream << D->getName();
    }
    else {
      StreamRAII SubStream(*this);
      Visit(D->getTemplatedDecl());
      if (m_SkipFlag) {
         skipDecl(D->getTemplatedDecl(), "Template pattern failed");
         return;
      }
      Stream << SubStream.take(true);
    }
    Out() << Stream.str() << closeBraces << '\n';
  }

  void ForwardDeclPrinter::VisitFunctionTemplateDecl(FunctionTemplateDecl *D) {
    if (m_PrintInstantiation) {
      StreamRAII stream(*this);
      TemplateParameterList *Params = D->getTemplateParameters();
      for (FunctionTemplateDecl::spec_iterator I = D->spec_begin(),
             E = D->spec_end(); I != E; ++I) {
        PrintTemplateParameters(Out(),
                                Params, (*I)->getTemplateSpecializationArgs());
        if (m_SkipFlag) {
          skipDecl(D, "Template parameters failed");
          return;
        }

        Visit(*I);
      }
      if (m_SkipFlag) {
        skipDecl(D, "specialization failed");
        return;
      }
      std::string output = stream.take(true);
      Out() << output;
    }

    return VisitRedeclarableTemplateDecl(D);

  }

  void ForwardDeclPrinter::VisitClassTemplateDecl(ClassTemplateDecl *D) {
    if (m_PrintInstantiation) {
      StreamRAII stream(*this);
      TemplateParameterList *Params = D->getTemplateParameters();
      for (ClassTemplateDecl::spec_iterator I = D->spec_begin(),
             E = D->spec_end(); I != E; ++I) {
        PrintTemplateParameters(Out(), Params, &(*I)->getTemplateArgs());
        if (m_SkipFlag) {
          skipDecl(D, "template parameters failed");
          return;
        }
        Visit(*I);
        if (m_SkipFlag) {
          skipDecl(D, "template instance failed");
          return;
        }
        std::string output = stream.take(true);
        Out() << output;
        Out() << '\n';
      }
    }
    return VisitRedeclarableTemplateDecl(D);
  }

  void ForwardDeclPrinter::
  VisitClassTemplateSpecializationDecl(ClassTemplateSpecializationDecl* D) {
//    if (shouldSkip(D)) {
//      skipDecl();
//      return;
//    }

    const TemplateArgumentList& iargs = D->getTemplateInstantiationArgs();
    for (const TemplateArgument& TA: iargs.asArray()) {
       VisitTemplateArgument(TA);
    }

//    Out() << "template <> ";
//    VisitCXXRecordDecl(D->getCanonicalDecl());

//    Out() << "<";
//    for (unsigned int i=0; i < iargs.size(); ++i){
//      if (iargs[i].getKind() == TemplateArgument::Pack)
//        continue;
//      if (i != 0 )
//        Out() << ", ";
//      iargs[i].print(m_Policy,Out());
//    }
//    Out() << ">";
//    skipDecl(false);

    Visit(D->getSpecializedTemplate());
    //Above code doesn't work properly
    //Must find better and more general way to print specializations
  }


  void ForwardDeclPrinter::Visit(const Type* typ) {
    switch (typ->getTypeClass()) {

#define VISIT_DECL(WHAT, HOW) \
      case clang::Type::WHAT: \
        Visit(static_cast<const clang::WHAT##Type*>(typ)->HOW().getTypePtr()); \
     break
      VISIT_DECL(ConstantArray, getElementType);
      VISIT_DECL(DependentSizedArray, getElementType);
      VISIT_DECL(IncompleteArray, getElementType);
      VISIT_DECL(VariableArray, getElementType);
      VISIT_DECL(Atomic, getValueType);
      VISIT_DECL(Auto, getDeducedType);
      VISIT_DECL(Decltype, getUnderlyingType);
      VISIT_DECL(Paren, getInnerType);
      VISIT_DECL(Pointer, getPointeeType);
      VISIT_DECL(LValueReference, getPointeeType);
      VISIT_DECL(RValueReference, getPointeeType);
      VISIT_DECL(TypeOf, getUnderlyingType);
      VISIT_DECL(Elaborated, getNamedType);
      VISIT_DECL(UnaryTransform, getUnderlyingType);
#undef VISIT_DECL

    case clang::Type::DependentName:
      {
        VisitNestedNameSpecifier(static_cast<const DependentNameType*>(typ)
                                 ->getQualifier());
      }
      break;

    case clang::Type::MemberPointer:
      {
        const MemberPointerType* MPT
          = static_cast<const MemberPointerType*>(typ);
        Visit(MPT->getPointeeType().getTypePtr());
        Visit(MPT->getClass());
      }
      break;

    case clang::Type::Enum:
      // intentional fall-through
    case clang::Type::Record:
      Visit(static_cast<const clang::TagType*>(typ)->getDecl());
      break;

    case clang::Type::TemplateSpecialization:
      {
        const TemplateSpecializationType* TST
          = static_cast<const TemplateSpecializationType*>(typ);
        for (const TemplateArgument& TA: *TST) {
          VisitTemplateArgument(TA);
        }
        VisitTemplateName(TST->getTemplateName());
      }
      break;

    case clang::Type::Typedef:
      Visit(static_cast<const TypedefType*>(typ)->getDecl());
      break;

    case clang::Type::TemplateTypeParm:
      Visit(static_cast<const TemplateTypeParmType*>(typ)->getDecl());
      break;

    case clang::Type::Builtin:
      // Nothing to do.
      break;
    case clang::Type::TypeOfExpr:
      // Nothing to do.
      break;

    default:
      Log() << "addDeclsToTransactionForType: Unexpected "
            << typ->getTypeClassName() << '\n';
      break;
    }
  }

  void ForwardDeclPrinter::VisitTemplateArgument(const TemplateArgument& TA) {
    switch (TA.getKind()) {
    case clang::TemplateArgument::Type:
      Visit(TA.getAsType().getTypePtr());
      break;
    case clang::TemplateArgument::Declaration:
      Visit(TA.getAsDecl());
      break;
    case clang::TemplateArgument::Template:
      VisitTemplateName(TA.getAsTemplateOrTemplatePattern());
      break;
    case clang::TemplateArgument::Pack:
      for (const auto& arg : TA.pack_elements())
        VisitTemplateArgument(arg);
      break;
    case clang::TemplateArgument::Expression:
      {
        Expr* TAExpr = TA.getAsExpr();
        if (CastExpr* CastExpr = dyn_cast<clang::CastExpr>(TAExpr))
          TAExpr = CastExpr->getSubExpr();
        if (DeclRefExpr* DRE = dyn_cast<DeclRefExpr>(TAExpr)) {
          Visit(DRE->getFoundDecl());
          if (m_SkipFlag) {
            return;
          }
        }
      }
      break;
    default:
      Log() << "Visit(Type*): Unexpected TemplateSpecializationType "
            << TA.getKind() << '\n';
      break;
    }
  }

  void ForwardDeclPrinter::VisitTemplateName(const clang::TemplateName& TN) {
    switch (TN.getKind()) {
    case clang::TemplateName::Template:
      Visit(TN.getAsTemplateDecl());
      break;
    case clang::TemplateName::QualifiedTemplate:
      Visit(TN.getAsQualifiedTemplateName()->getTemplateDecl());
      break;
    case clang::TemplateName::DependentTemplate:
      VisitNestedNameSpecifier(TN.getAsDependentTemplateName()->getQualifier());
      break;
    case clang::TemplateName::SubstTemplateTemplateParm:
      VisitTemplateName(TN.getAsSubstTemplateTemplateParm()->getReplacement());
      break;
    case clang::TemplateName::SubstTemplateTemplateParmPack:
      VisitTemplateArgument(TN.getAsSubstTemplateTemplateParmPack()->getArgumentPack());
      break;
    default:
      Log() << "VisitTemplateName: Unexpected kind " << TN.getKind() << '\n';
      break;
    }
  }

  void ForwardDeclPrinter::VisitNestedNameSpecifier(
                                        const clang::NestedNameSpecifier* NNS) {
    if (const clang::NestedNameSpecifier* Prefix = NNS->getPrefix())
      VisitNestedNameSpecifier(Prefix);

    switch (NNS->getKind()) {
    case clang::NestedNameSpecifier::Namespace:
      Visit(NNS->getAsNamespace());
      break;
    case clang::NestedNameSpecifier::TypeSpec: // fall-through:
    case clang::NestedNameSpecifier::TypeSpecWithTemplate:
      // We cannot fwd declare nested types.
      skipDecl(0, "NestedNameSpec TypeSpec/TypeSpecWithTemplate");
      break;
    default:
      Log() << "VisitNestedNameSpecifier: Unexpected kind "
            << NNS->getKind() << '\n';
      skipDecl(0, 0);
      break;
   };
  }

  bool ForwardDeclPrinter::isOperator(FunctionDecl *D) {
    //TODO: Find a better check for this
    return D->getNameAsString().find("operator") == 0;
  }

  bool ForwardDeclPrinter::hasDefaultArgument(FunctionDecl *D) {
    auto N = D->getNumParams();
    for (unsigned int i=0; i < N; ++i) {
      if (D->getParamDecl(i)->hasDefaultArg())
        return true;
    }
    return false;
  }

  bool ForwardDeclPrinter::shouldSkipImpl(FunctionDecl *D) {
    //FIXME: setDeletedAsWritten can be called from the
        //InclusionDiretctive callback.
        //Implement that if important functions are marked so.
        //Not important, as users do not need hints
        //about using Deleted functions
    if (D->getIdentifier() == 0
        || D->getNameAsString()[0] == '_'
        || D->getStorageClass() == SC_Static
        || D->isCXXClassMember()
        || isOperator(D)
        || D->isDeleted()
        || D->isDeletedAsWritten()) {
      return true;
    }

    return false;
  }

  bool ForwardDeclPrinter::shouldSkipImpl(FunctionTemplateDecl *D) {
    return shouldSkipImpl(D->getTemplatedDecl());
  }

  bool ForwardDeclPrinter::shouldSkipImpl(TagDecl *D) {
    return !D->getIdentifier();
  }

  bool ForwardDeclPrinter::shouldSkipImpl(VarDecl *D) {
    if (D->getType().isConstant(m_Ctx)) {
      Log() << D->getName() <<" Var : Const\n";
      m_Visited[D->getCanonicalDecl()] = false;
      return true;
    }
    return false;
  }

  bool ForwardDeclPrinter::shouldSkipImpl(EnumDecl *D) {
    if (!D->getIdentifier()){
      D->printName(Log());
      Log() << "Enum: Empty name\n";
      return true;
    }
    return false;
  }

  void ForwardDeclPrinter::skipDecl(Decl* D, const char* Reason) {
    m_SkipFlag = true;
    if (Reason) {
      if (D)
        Log() << D->getDeclKindName() << " " << getNameIfPossible(D) << " ";
      Log() << Reason << '\n';
    }
  }

  bool ForwardDeclPrinter::shouldSkipImpl(ClassTemplateSpecializationDecl *D) {
    if (llvm::isa<ClassTemplatePartialSpecializationDecl>(D)) {
      //TODO: How to print partial specializations?
      return true;
    }
    return false;
  }

  bool ForwardDeclPrinter::shouldSkipImpl(UsingDirectiveDecl *D) {
    if (shouldSkipImpl(D->getNominatedNamespace())) {
      Log() << D->getNameAsString() <<" Using Directive : Incompatible Type\n";
      return true;
    }
    return false;
  }

  bool ForwardDeclPrinter::shouldSkipImpl(TypeAliasTemplateDecl *D) {
    D->printName(Log());
    Log() << " TypeAliasTemplateDecl: Always Skipped\n";
    return true;
  }

  void ForwardDeclPrinter::printStats() {
    size_t bad = 0;
    for (auto&& i: m_Visited)
      if (!i.second)
        ++bad;

    Log() << bad << " decls skipped out of " << m_Visited.size() << "\n";
  }

  void ForwardDeclPrinter::PrintNamespaceOpen(llvm::raw_ostream& Stream,
                                              const NamespaceDecl* ND) {
    if (ND->isInline())
      Stream << "inline ";
    Stream << "namespace " << *ND << '{';
  }

  void ForwardDeclPrinter::PrintLinkageOpen(llvm::raw_ostream& Stream,
                                            const LinkageSpecDecl* LSD) {
    assert((LSD->getLanguage() == LinkageSpecDecl::lang_cxx ||
            LSD->getLanguage() == LinkageSpecDecl::lang_c) &&
           "Unknown linkage spec!");
    Stream << "extern \"C";
    if (LSD->getLanguage() == LinkageSpecDecl::lang_cxx) {
      Stream << "++";
    }
    Stream << "\" {";
  }


  std::string ForwardDeclPrinter::PrintEnclosingDeclContexts(llvm::raw_ostream& Stream,
                                                             const DeclContext* DC) {
    // Return closing "} } } }"...
    SmallVector<const DeclContext*, 16> DeclCtxs;
    for(; DC && !DC->isTranslationUnit(); DC = DC->getParent()) {
      if (!isa<NamespaceDecl>(DC) && !isa<LinkageSpecDecl>(DC)) {
        Log() << "Skipping unhandled " << DC->getDeclKindName() << '\n';
        skipDecl(0, 0);
        return "";
      }
      DeclCtxs.push_back(DC);
    }

    for (auto I = DeclCtxs.rbegin(), E = DeclCtxs.rend(); I != E; ++I) {
      if (const NamespaceDecl* ND = dyn_cast<NamespaceDecl>(*I))
        PrintNamespaceOpen(Stream, ND);
      else if (const LinkageSpecDecl* LSD = dyn_cast<LinkageSpecDecl>(*I))
        PrintLinkageOpen(Stream, LSD);
    }
    return std::string(DeclCtxs.size(), '}');
  }
}//end namespace cling
