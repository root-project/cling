###############################################################################
#
#                           The Cling Interpreter
#
# Cling Packaging Tool (CPT)
#
# tools/packaging/dist-files.mk: Makefile script having list of binaries,
# libraries, and headers to include in the package.
#
# TODO: Add documentation here on how to generate/maintain this file
#
# Author: Anirudha Bose <ani07nov@gmail.com>
#
# This file is dual-licensed: you can choose to license it under the University
# of Illinois Open Source License or the GNU Lesser General Public License. See
# LICENSE.TXT for details.
#
###############################################################################

BIN_FILES := \
  bin/cling@EXEEXT@ 
# CAUTION: The trailing space above is needed. DO NOT delete.

DOCS_FILES := \
  docs/llvm/html/cling/cling.html \
  docs/llvm/html/cling/manpage.css \
  \
  docs/llvm/ps/cling.ps 
# CAUTION: The trailing space above is needed. DO NOT delete.
  
INCLUDE_CLANG_FILES := \
  include/Analysis/Analyses/CFGReachabilityAnalysis.h \
  include/Analysis/Analyses/Consumed.h \
  include/Analysis/Analyses/Dominators.h \
  include/Analysis/Analyses/FormatString.h \
  include/Analysis/Analyses/LiveVariables.h \
  include/clang/Analysis/Analyses/PostOrderCFGView.h \
  include/clang/Analysis/Analyses/PseudoConstantAnalysis.h \
  include/clang/Analysis/Analyses/ReachableCode.h \
  include/clang/Analysis/Analyses/ThreadSafety.h \
  include/clang/Analysis/Analyses/UninitializedValues.h \
  include/clang/Analysis/AnalysisContext.h \
  include/clang/Analysis/AnalysisDiagnostic.h \
  include/clang/Analysis/CallGraph.h \
  include/clang/Analysis/CFG.h \
  include/clang/Analysis/CFGStmtMap.h \
  \
  include/clang/Analysis/DomainSpecific/CocoaConventions.h \
  include/clang/Analysis/DomainSpecific/ObjCNoReturn.h \
  \
  include/clang/Analysis/FlowSensitive/DataflowSolver.h \
  include/clang/Analysis/FlowSensitive/DataflowValues.h \
  include/clang/Analysis/ProgramPoint.h \
  \
  include/clang/Analysis/Support/BumpVector.h \
  \
  include/clang/ARCMigrate/ARCMTActions.h \
  include/clang/ARCMigrate/ARCMT.h \
  include/clang/ARCMigrate/FileRemapper.h \
  \
  include/clang/AST/APValue.h \
  include/clang/AST/ASTConsumer.h \
  include/clang/AST/ASTContext.h \
  include/clang/AST/ASTDiagnostic.h \
  include/clang/AST/ASTFwd.h \
  include/clang/AST/AST.h \
  include/clang/AST/ASTImporter.h \
  include/clang/AST/ASTLambda.h \
  include/clang/AST/ASTMutationListener.h \
  include/clang/AST/ASTTypeTraits.h \
  include/clang/AST/ASTUnresolvedSet.h \
  include/clang/AST/ASTVector.h \
  include/clang/AST/AttrDump.inc \
  include/clang/AST/Attr.h \
  include/clang/AST/AttrImpl.inc \
  include/clang/AST/AttrIterator.h \
  \
  include/clang/AST/AttrVisitor.inc \
  include/clang/AST/BaseSubobject.h \
  include/clang/AST/BuiltinTypes.def \
  include/clang/AST/CanonicalType.h \
  include/clang/AST/CharUnits.h \
  include/clang/AST/CommentBriefParser.h \
  include/clang/AST/CommentCommandInfo.inc \
  include/clang/AST/CommentCommandList.inc \
  include/clang/AST/CommentCommandTraits.h \
  include/clang/AST/CommentDiagnostic.h \
  include/clang/AST/Comment.h \
  include/clang/AST/CommentHTMLNamedCharacterReferences.inc \
  include/clang/AST/CommentHTMLTags.inc \
  include/clang/AST/CommentHTMLTagsProperties.inc \
  include/clang/AST/CommentLexer.h \
  include/clang/AST/CommentNodes.inc \
  include/clang/AST/CommentParser.h \
  include/clang/AST/CommentSema.h \
  include/clang/AST/CommentVisitor.h \
  include/clang/AST/CXXInheritance.h \
  include/clang/AST/DataRecursiveASTVisitor.h \
  include/clang/AST/DeclAccessPair.h \
  include/clang/AST/DeclarationName.h \
  include/clang/AST/DeclBase.h \
  include/clang/AST/DeclContextInternals.h \
  include/clang/AST/DeclCXX.h \
  include/clang/AST/DeclFriend.h \
  include/clang/AST/DeclGroup.h \
  include/clang/AST/Decl.h \
  include/clang/AST/DeclLookups.h \
  include/clang/AST/DeclNodes.inc \
  include/clang/AST/DeclObjC.h \
  include/clang/AST/DeclOpenMP.h \
  include/clang/AST/DeclTemplate.h \
  include/clang/AST/DeclVisitor.h \
  include/clang/AST/DependentDiagnostic.h \
  include/clang/AST/EvaluatedExprVisitor.h \
  include/clang/AST/ExprCXX.h \
  include/clang/AST/Expr.h \
  include/clang/AST/ExprObjC.h \
  include/clang/AST/ExternalASTSource.h \
  include/clang/AST/GlobalDecl.h \
  include/clang/AST/Mangle.h \
  include/clang/AST/MangleNumberingContext.h \
  include/clang/AST/NestedNameSpecifier.h \
  include/clang/AST/NSAPI.h \
  include/clang/AST/OpenMPClause.h \
  include/clang/AST/OperationKinds.h \
  include/clang/AST/ParentMap.h \
  include/clang/AST/PrettyPrinter.h \
  include/clang/AST/RawCommentList.h \
  include/clang/AST/RecordLayout.h \
  include/clang/AST/RecursiveASTVisitor.h \
  include/clang/AST/Redeclarable.h \
  include/clang/AST/SelectorLocationsKind.h \
  include/clang/AST/StmtCXX.h \
  include/clang/AST/StmtGraphTraits.h \
  include/clang/AST/Stmt.h \
  include/clang/AST/StmtIterator.h \
  include/clang/AST/StmtNodes.inc \
  include/clang/AST/StmtObjC.h \
  include/clang/AST/StmtOpenMP.h \
  include/clang/AST/StmtVisitor.h \
  include/clang/AST/TemplateBase.h \
  include/clang/AST/TemplateName.h \
  include/clang/AST/Type.h \
  include/clang/AST/TypeLoc.h \
  include/clang/AST/TypeLocNodes.def \
  include/clang/AST/TypeLocVisitor.h \
  include/clang/AST/TypeNodes.def \
  include/clang/AST/TypeOrdering.h \
  include/clang/AST/TypeVisitor.h \
  include/clang/AST/UnresolvedSet.h \
  include/clang/AST/VTableBuilder.h \
  include/clang/AST/VTTBuilder.h \
  \
  include/clang/ASTMatchers/ASTMatchers.h \
  include/clang/ASTMatchers/ASTMatchersInternal.h \
  include/clang/ASTMatchers/ASTMatchersMacros.h \
  include/clang/ASTMatchers/ASTMatchFinder.h \
  \
  include/clang/ASTMatchers/Dynamic/Diagnostics.h \
  include/clang/ASTMatchers/Dynamic/Parser.h \
  include/clang/ASTMatchers/Dynamic/Registry.h \
  include/clang/ASTMatchers/Dynamic/VariantValue.h \
  \
  include/clang/Basic/ABI.h \
  include/clang/Basic/AddressSpaces.h \
  include/clang/Basic/AllDiagnostics.h \
  include/clang/Basic/arm_neon.inc \
  include/clang/Basic/AttrKinds.h \
  include/clang/Basic/AttrList.inc \
  include/clang/Basic/BuiltinsAArch64.def \
  include/clang/Basic/BuiltinsARM.def \
  include/clang/Basic/Builtins.def \
  include/clang/Basic/Builtins.h \
  include/clang/Basic/BuiltinsHexagon.def \
  include/clang/Basic/BuiltinsMips.def \
  include/clang/Basic/BuiltinsNVPTX.def \
  include/clang/Basic/BuiltinsPPC.def \
  include/clang/Basic/BuiltinsX86.def \
  include/clang/Basic/BuiltinsXCore.def \
  include/clang/Basic/CapturedStmt.h \
  include/clang/Basic/CharInfo.h \
  include/clang/Basic/CommentOptions.h \
  include/clang/Basic/DiagnosticAnalysisKinds.inc \
  include/clang/Basic/DiagnosticASTKinds.inc \
  include/clang/Basic/DiagnosticCategories.h \
  include/clang/Basic/DiagnosticCommentKinds.inc \
  include/clang/Basic/DiagnosticCommonKinds.inc \
  include/clang/Basic/DiagnosticDriverKinds.inc \
  include/clang/Basic/DiagnosticFrontendKinds.inc \
  include/clang/Basic/DiagnosticGroups.inc \
  include/clang/Basic/Diagnostic.h \
  include/clang/Basic/DiagnosticIDs.h \
  include/clang/Basic/DiagnosticIndexName.inc \
  include/clang/Basic/DiagnosticLexKinds.inc \
  include/clang/Basic/DiagnosticOptions.def \
  include/clang/Basic/DiagnosticOptions.h \
  include/clang/Basic/DiagnosticParseKinds.inc \
  include/clang/Basic/DiagnosticSemaKinds.inc \
  include/clang/Basic/DiagnosticSerializationKinds.inc \
  include/clang/Basic/ExceptionSpecificationType.h \
  include/clang/Basic/ExpressionTraits.h \
  include/clang/Basic/FileManager.h \
  include/clang/Basic/FileSystemOptions.h \
  include/clang/Basic/FileSystemStatCache.h \
  include/clang/Basic/IdentifierTable.h \
  include/clang/Basic/Lambda.h \
  include/clang/Basic/LangOptions.def \
  include/clang/Basic/LangOptions.h \
  include/clang/Basic/Linkage.h \
  include/clang/Basic/LLVM.h \
  include/clang/Basic/MacroBuilder.h \
  include/clang/Basic/Module.h \
  include/clang/Basic/ObjCRuntime.h \
  include/clang/Basic/OnDiskHashTable.h \
  include/clang/Basic/OpenCLExtensions.def \
  include/clang/Basic/OpenMPKinds.def \
  include/clang/Basic/OpenMPKinds.h \
  include/clang/Basic/OperatorKinds.def \
  include/clang/Basic/OperatorKinds.h \
  include/clang/Basic/OperatorPrecedence.h \
  include/clang/Basic/PartialDiagnostic.h \
  include/clang/Basic/PlistSupport.h \
  include/clang/Basic/PrettyStackTrace.h \
  include/clang/Basic/Sanitizers.def \
  include/clang/Basic/SourceLocation.h \
  include/clang/Basic/SourceManager.h \
  include/clang/Basic/SourceManagerInternals.h \
  include/clang/Basic/Specifiers.h \
  include/clang/Basic/TargetBuiltins.h \
  include/clang/Basic/TargetCXXABI.h \
  include/clang/Basic/TargetInfo.h \
  include/clang/Basic/TargetOptions.h \
  include/clang/Basic/TemplateKinds.h \
  include/clang/Basic/TokenKinds.def \
  include/clang/Basic/TokenKinds.h \
  include/clang/Basic/TypeTraits.h \
  include/clang/Basic/Version.h \
  include/clang/Basic/Version.inc \
  include/clang/Basic/VersionTuple.h \
  include/clang/Basic/VirtualFileSystem.h \
  include/clang/Basic/Visibility.h \
  \
  include/clang/CodeGen/BackendUtil.h \
  include/clang/CodeGen/CGFunctionInfo.h \
  include/clang/CodeGen/CodeGenABITypes.h \
  include/clang/CodeGen/CodeGenAction.h \
  include/clang/CodeGen/ModuleBuilder.h \
  \
  include/clang/Config/config.h \
  \
  include/clang/Driver/Action.h \
  include/clang/Driver/CC1AsOptions.h \
  include/clang/Driver/CC1AsOptions.inc \
  include/clang/Driver/Compilation.h \
  include/clang/Driver/DriverDiagnostic.h \
  include/clang/Driver/Driver.h \
  include/clang/Driver/Job.h \
  include/clang/Driver/Multilib.h \
  include/clang/Driver/Options.h \
  include/clang/Driver/Options.inc \
  include/clang/Driver/Phases.h \
  include/clang/Driver/SanitizerArgs.h \
  include/clang/Driver/ToolChain.h \
  include/clang/Driver/Tool.h \
  include/clang/Driver/Types.def \
  include/clang/Driver/Types.h \
  include/clang/Driver/Util.h \
  \
  include/clang/Edit/Commit.h \
  include/clang/Edit/EditedSource.h \
  include/clang/Edit/EditsReceiver.h \
  include/clang/Edit/FileOffset.h \
  include/clang/Edit/Rewriters.h \
  \
  include/clang/Format/Format.h \
  \
  include/clang/Frontend/ASTConsumers.h \
  include/clang/Frontend/ASTUnit.h \
  include/clang/Frontend/ChainedDiagnosticConsumer.h \
  include/clang/Frontend/ChainedIncludesSource.h \
  include/clang/Frontend/CodeGenOptions.def \
  include/clang/Frontend/CodeGenOptions.h \
  include/clang/Frontend/CommandLineSourceLoc.h \
  include/clang/Frontend/CompilerInstance.h \
  include/clang/Frontend/CompilerInvocation.h \
  include/clang/Frontend/DependencyOutputOptions.h \
  include/clang/Frontend/DiagnosticRenderer.h \
  include/clang/Frontend/FrontendAction.h \
  include/clang/Frontend/FrontendActions.h \
  include/clang/Frontend/FrontendDiagnostic.h \
  include/clang/Frontend/FrontendOptions.h \
  include/clang/Frontend/FrontendPluginRegistry.h \
  include/clang/Frontend/LangStandard.h \
  include/clang/Frontend/LangStandards.def \
  include/clang/Frontend/LayoutOverrideSource.h \
  include/clang/Frontend/LogDiagnosticPrinter.h \
  include/clang/Frontend/MigratorOptions.h \
  include/clang/Frontend/MultiplexConsumer.h \
  include/clang/Frontend/PreprocessorOutputOptions.h \
  include/clang/Frontend/SerializedDiagnosticPrinter.h \
  include/clang/Frontend/TextDiagnosticBuffer.h \
  include/clang/Frontend/TextDiagnostic.h \
  include/clang/Frontend/TextDiagnosticPrinter.h \
  include/clang/Frontend/Utils.h \
  include/clang/Frontend/VerifyDiagnosticConsumer.h \
  \
  include/clang/FrontendTool/Utils.h \
  \
  include/clang/Index/CommentToXML.h \
  include/clang/Index/USRGeneration.h \
  \
  include/clang/Lex/AttrSpellings.inc \
  include/clang/Lex/CodeCompletionHandler.h \
  include/clang/Lex/DirectoryLookup.h \
  include/clang/Lex/ExternalPreprocessorSource.h \
  include/clang/Lex/HeaderMap.h \
  include/clang/Lex/HeaderSearch.h \
  include/clang/Lex/HeaderSearchOptions.h \
  include/clang/Lex/LexDiagnostic.h \
  include/clang/Lex/Lexer.h \
  include/clang/Lex/LiteralSupport.h \
  include/clang/Lex/MacroArgs.h \
  include/clang/Lex/MacroInfo.h \
  include/clang/Lex/ModuleLoader.h \
  include/clang/Lex/ModuleMap.h \
  include/clang/Lex/MultipleIncludeOpt.h \
  include/clang/Lex/PPCallbacks.h \
  include/clang/Lex/PPConditionalDirectiveRecord.h \
  include/clang/Lex/Pragma.h \
  include/clang/Lex/PreprocessingRecord.h \
  include/clang/Lex/Preprocessor.h \
  include/clang/Lex/PreprocessorLexer.h \
  include/clang/Lex/PreprocessorOptions.h \
  include/clang/Lex/PTHLexer.h \
  include/clang/Lex/PTHManager.h \
  include/clang/Lex/ScratchBuffer.h \
  include/clang/Lex/TokenConcatenation.h \
  include/clang/Lex/Token.h \
  include/clang/Lex/TokenLexer.h \
  \
  include/clang/Parse/AttrParserStringSwitches.inc \
  include/clang/Parse/ParseAST.h \
  include/clang/Parse/ParseDiagnostic.h \
  include/clang/Parse/Parser.h \
  include/clang/Parse/RAIIObjectsForParser.h \
  \
  include/clang/Rewrite/Core/DeltaTree.h \
  include/clang/Rewrite/Core/HTMLRewrite.h \
  include/clang/Rewrite/Core/Rewriter.h \
  include/clang/Rewrite/Core/RewriteRope.h \
  include/clang/Rewrite/Core/TokenRewriter.h \
  \
  include/clang/Rewrite/Frontend/ASTConsumers.h \
  include/clang/Rewrite/Frontend/FixItRewriter.h \
  include/clang/Rewrite/Frontend/FrontendActions.h \
  include/clang/Rewrite/Frontend/Rewriters.h \
  \
  include/clang/Sema/AnalysisBasedWarnings.h \
  include/clang/Sema/AttributeList.h \
  include/clang/Sema/AttrParsedAttrImpl.inc \
  include/clang/Sema/AttrParsedAttrKinds.inc \
  include/clang/Sema/AttrParsedAttrList.inc \
  include/clang/Sema/AttrSpellingListIndex.inc \
  include/clang/Sema/AttrTemplateInstantiate.inc \
  include/clang/Sema/CodeCompleteConsumer.h \
  include/clang/Sema/CodeCompleteOptions.h \
  include/clang/Sema/CXXFieldCollector.h \
  include/clang/Sema/DeclSpec.h \
  include/clang/Sema/DelayedDiagnostic.h \
  include/clang/Sema/Designator.h \
  include/clang/Sema/ExternalSemaSource.h \
  include/clang/Sema/IdentifierResolver.h \
  include/clang/Sema/Initialization.h \
  include/clang/Sema/LocInfoType.h \
  include/clang/Sema/Lookup.h \
  include/clang/Sema/MultiplexExternalSemaSource.h \
  include/clang/Sema/ObjCMethodList.h \
  include/clang/Sema/Overload.h \
  include/clang/Sema/Ownership.h \
  include/clang/Sema/ParsedTemplate.h \
  include/clang/Sema/PrettyDeclStackTrace.h \
  include/clang/Sema/Scope.h \
  include/clang/Sema/ScopeInfo.h \
  include/clang/Sema/SemaConsumer.h \
  include/clang/Sema/SemaDiagnostic.h \
  include/clang/Sema/SemaFixItUtils.h \
  include/clang/Sema/Sema.h \
  include/clang/Sema/SemaInternal.h \
  include/clang/Sema/SemaLambda.h \
  include/clang/Sema/TemplateDeduction.h \
  include/clang/Sema/Template.h \
  include/clang/Sema/TypoCorrection.h \
  include/clang/Sema/Weak.h \
  \
  include/clang/Serialization/ASTBitCodes.h \
  include/clang/Serialization/ASTDeserializationListener.h \
  include/clang/Serialization/ASTReader.h \
  include/clang/Serialization/ASTWriter.h \
  include/clang/Serialization/AttrPCHRead.inc \
  include/clang/Serialization/AttrPCHWrite.inc \
  include/clang/Serialization/ContinuousRangeMap.h \
  include/clang/Serialization/GlobalModuleIndex.h \
  include/clang/Serialization/Module.h \
  include/clang/Serialization/ModuleManager.h \
  include/clang/Serialization/SerializationDiagnostic.h \
  \
  include/clang/StaticAnalyzer/Checkers/ClangCheckers.h \
  include/clang/StaticAnalyzer/Checkers/LocalCheckers.h \
  include/clang/StaticAnalyzer/Checkers/ObjCRetainCount.h \
  \
  include/clang/StaticAnalyzer/Core/Analyses.def \
  include/clang/StaticAnalyzer/Core/AnalyzerOptions.h \
  \
  include/clang/StaticAnalyzer/Core/BugReporter/BugReporter.h \
  include/clang/StaticAnalyzer/Core/BugReporter/BugReporterVisitor.h \
  include/clang/StaticAnalyzer/Core/BugReporter/BugType.h \
  include/clang/StaticAnalyzer/Core/BugReporter/CommonBugCategories.h \
  include/clang/StaticAnalyzer/Core/BugReporter/PathDiagnostic.h \
  include/clang/StaticAnalyzer/Core/Checker.h \
  include/clang/StaticAnalyzer/Core/CheckerManager.h \
  include/clang/StaticAnalyzer/Core/CheckerOptInfo.h \
  include/clang/StaticAnalyzer/Core/CheckerRegistry.h \
  include/clang/StaticAnalyzer/Core/PathDiagnosticConsumers.h \
  \
  include/clang/StaticAnalyzer/Core/PathSensitive/AnalysisManager.h \
  include/clang/StaticAnalyzer/Core/PathSensitive/APSIntType.h \
  include/clang/StaticAnalyzer/Core/PathSensitive/BasicValueFactory.h \
  include/clang/StaticAnalyzer/Core/PathSensitive/BlockCounter.h \
  include/clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h \
  include/clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h \
  include/clang/StaticAnalyzer/Core/PathSensitive/CheckerHelpers.h \
  include/clang/StaticAnalyzer/Core/PathSensitive/ConstraintManager.h \
  include/clang/StaticAnalyzer/Core/PathSensitive/CoreEngine.h \
  include/clang/StaticAnalyzer/Core/PathSensitive/DynamicTypeInfo.h \
  include/clang/StaticAnalyzer/Core/PathSensitive/Environment.h \
  include/clang/StaticAnalyzer/Core/PathSensitive/ExplodedGraph.h \
  include/clang/StaticAnalyzer/Core/PathSensitive/ExprEngine.h \
  include/clang/StaticAnalyzer/Core/PathSensitive/FunctionSummary.h \
  include/clang/StaticAnalyzer/Core/PathSensitive/MemRegion.h \
  include/clang/StaticAnalyzer/Core/PathSensitive/ProgramState_Fwd.h \
  include/clang/StaticAnalyzer/Core/PathSensitive/ProgramState.h \
  include/clang/StaticAnalyzer/Core/PathSensitive/ProgramStateTrait.h \
  include/clang/StaticAnalyzer/Core/PathSensitive/Store.h \
  include/clang/StaticAnalyzer/Core/PathSensitive/StoreRef.h \
  include/clang/StaticAnalyzer/Core/PathSensitive/SubEngine.h \
  include/clang/StaticAnalyzer/Core/PathSensitive/SummaryManager.h \
  include/clang/StaticAnalyzer/Core/PathSensitive/SValBuilder.h \
  include/clang/StaticAnalyzer/Core/PathSensitive/SVals.h \
  include/clang/StaticAnalyzer/Core/PathSensitive/SymbolManager.h \
  include/clang/StaticAnalyzer/Core/PathSensitive/TaintManager.h \
  include/clang/StaticAnalyzer/Core/PathSensitive/TaintTag.h \
  include/clang/StaticAnalyzer/Core/PathSensitive/WorkList.h \
  \
  include/clang/StaticAnalyzer/Frontend/AnalysisConsumer.h \
  include/clang/StaticAnalyzer/Frontend/CheckerRegistration.h \
  include/clang/StaticAnalyzer/Frontend/FrontendActions.h \
  \
  include/clang/Tooling/ArgumentsAdjusters.h \
  include/clang/Tooling/CommonOptionsParser.h \
  include/clang/Tooling/CompilationDatabase.h \
  include/clang/Tooling/CompilationDatabasePluginRegistry.h \
  include/clang/Tooling/FileMatchTrie.h \
  include/clang/Tooling/JSONCompilationDatabase.h \
  include/clang/Tooling/RefactoringCallbacks.h \
  include/clang/Tooling/Refactoring.h \
  include/clang/Tooling/ReplacementsYaml.h \
  include/clang/Tooling/Tooling.h 
# CAUTION: The trailing space above is needed. DO NOT delete.

INCLUDE_CLANG_C_FILES := \
  include/clang-c/BuildSystem.h \
  include/clang-c/CXCompilationDatabase.h \
  include/clang-c/CXErrorCode.h \
  include/clang-c/CXString.h \
  include/clang-c/Index.h \
  include/clang-c/Platform.h 
# CAUTION: The trailing space above is needed. DO NOT delete.

INCLUDE_CLING_FILES := \
  include/cling/Interpreter/CIFactory.h \
  include/cling/Interpreter/ClangInternalState.h \
  include/cling/Interpreter/ClingOptions.h \
  include/cling/Interpreter/ClingOptions.inc \
  include/cling/Interpreter/CompilationOptions.h \
  include/cling/Interpreter/CValuePrinter.h \
  include/cling/Interpreter/DynamicExprInfo.h \
  include/cling/Interpreter/DynamicLibraryManager.h \
  include/cling/Interpreter/DynamicLookupLifetimeHandler.h \
  include/cling/Interpreter/DynamicLookupRuntimeUniverse.h \
  include/cling/Interpreter/InterpreterCallbacks.h \
  include/cling/Interpreter/Interpreter.h \
  include/cling/Interpreter/InvocationOptions.h \
  include/cling/Interpreter/LookupHelper.h \
  include/cling/Interpreter/RuntimeException.h \
  include/cling/Interpreter/RuntimeUniverse.h \
  include/cling/Interpreter/Transaction.h \
  include/cling/Interpreter/Value.h \
  \
  include/cling/MetaProcessor/MetaProcessor.h \
  \
  include/cling/UserInterface/CompilationException.h \
  include/cling/UserInterface/UserInterface.h \
  \
  include/cling/Utils/AST.h 
# CAUTION: The trailing space above is needed. DO NOT delete.

INCLUDE_LLVM_FILES := \
  include/llvm/ADT/APFloat.h \
  include/llvm/ADT/APInt.h \
  include/llvm/ADT/APSInt.h \
  include/llvm/ADT/ArrayRef.h \
  include/llvm/ADT/BitVector.h \
  include/llvm/ADT/DAGDeltaAlgorithm.h \
  include/llvm/ADT/DeltaAlgorithm.h \
  include/llvm/ADT/DenseMap.h \
  include/llvm/ADT/DenseMapInfo.h \
  include/llvm/ADT/DenseSet.h \
  include/llvm/ADT/DepthFirstIterator.h \
  include/llvm/ADT/edit_distance.h \
  include/llvm/ADT/EquivalenceClasses.h \
  include/llvm/ADT/FoldingSet.h \
  include/llvm/ADT/GraphTraits.h \
  include/llvm/ADT/Hashing.h \
  include/llvm/ADT/ilist.h \
  include/llvm/ADT/ilist_node.h \
  include/llvm/ADT/ImmutableIntervalMap.h \
  include/llvm/ADT/ImmutableList.h \
  include/llvm/ADT/ImmutableMap.h \
  include/llvm/ADT/ImmutableSet.h \
  include/llvm/ADT/IndexedMap.h \
  include/llvm/ADT/IntEqClasses.h \
  include/llvm/ADT/IntervalMap.h \
  include/llvm/ADT/IntrusiveRefCntPtr.h \
  include/llvm/ADT/MapVector.h \
  include/llvm/ADT/None.h \
  include/llvm/ADT/Optional.h \
  include/llvm/ADT/OwningPtr.h \
  include/llvm/ADT/PackedVector.h \
  include/llvm/ADT/PointerIntPair.h \
  include/llvm/ADT/PointerUnion.h \
  include/llvm/ADT/polymorphic_ptr.h \
  include/llvm/ADT/PostOrderIterator.h \
  include/llvm/ADT/PriorityQueue.h \
  include/llvm/ADT/SCCIterator.h \
  include/llvm/ADT/ScopedHashTable.h \
  include/llvm/ADT/SetOperations.h \
  include/llvm/ADT/SetVector.h \
  include/llvm/ADT/SmallBitVector.h \
  include/llvm/ADT/SmallPtrSet.h \
  include/llvm/ADT/SmallSet.h \
  include/llvm/ADT/SmallString.h \
  include/llvm/ADT/SmallVector.h \
  include/llvm/ADT/SparseBitVector.h \
  include/llvm/ADT/SparseMultiSet.h \
  include/llvm/ADT/SparseSet.h \
  include/llvm/ADT/Statistic.h \
  include/llvm/ADT/STLExtras.h \
  include/llvm/ADT/StringExtras.h \
  include/llvm/ADT/StringMap.h \
  include/llvm/ADT/StringRef.h \
  include/llvm/ADT/StringSet.h \
  include/llvm/ADT/StringSwitch.h \
  include/llvm/ADT/TinyPtrVector.h \
  include/llvm/ADT/Triple.h \
  include/llvm/ADT/Twine.h \
  include/llvm/ADT/UniqueVector.h \
  include/llvm/ADT/ValueMap.h \
  include/llvm/ADT/VariadicFunction.h \
  \
  include/llvm/Analysis/AliasAnalysis.h \
  include/llvm/Analysis/AliasSetTracker.h \
  include/llvm/Analysis/BlockFrequencyImpl.h \
  include/llvm/Analysis/BlockFrequencyInfo.h \
  include/llvm/Analysis/BranchProbabilityInfo.h \
  include/llvm/Analysis/CallGraph.h \
  include/llvm/Analysis/CallGraphSCCPass.h \
  include/llvm/Analysis/CallPrinter.h \
  include/llvm/Analysis/CaptureTracking.h \
  include/llvm/Analysis/CFG.h \
  include/llvm/Analysis/CFGPrinter.h \
  include/llvm/Analysis/CodeMetrics.h \
  include/llvm/Analysis/ConstantFolding.h \
  include/llvm/Analysis/ConstantsScanner.h \
  include/llvm/Analysis/DependenceAnalysis.h \
  include/llvm/Analysis/DominanceFrontier.h \
  include/llvm/Analysis/DomPrinter.h \
  include/llvm/Analysis/DOTGraphTraitsPass.h \
  include/llvm/Analysis/FindUsedTypes.h \
  include/llvm/Analysis/InlineCost.h \
  include/llvm/Analysis/InstructionSimplify.h \
  include/llvm/Analysis/Interval.h \
  include/llvm/Analysis/IntervalIterator.h \
  include/llvm/Analysis/IntervalPartition.h \
  include/llvm/Analysis/IVUsers.h \
  include/llvm/Analysis/LazyCallGraph.h \
  include/llvm/Analysis/LazyValueInfo.h \
  include/llvm/Analysis/LibCallAliasAnalysis.h \
  include/llvm/Analysis/LibCallSemantics.h \
  include/llvm/Analysis/Lint.h \
  include/llvm/Analysis/Loads.h \
  include/llvm/Analysis/LoopInfo.h \
  include/llvm/Analysis/LoopInfoImpl.h \
  include/llvm/Analysis/LoopIterator.h \
  include/llvm/Analysis/LoopPass.h \
  include/llvm/Analysis/MemoryBuiltins.h \
  include/llvm/Analysis/MemoryDependenceAnalysis.h \
  include/llvm/Analysis/Passes.h \
  include/llvm/Analysis/PHITransAddr.h \
  include/llvm/Analysis/PostDominators.h \
  include/llvm/Analysis/PtrUseVisitor.h \
  include/llvm/Analysis/RegionInfo.h \
  include/llvm/Analysis/RegionIterator.h \
  include/llvm/Analysis/RegionPass.h \
  include/llvm/Analysis/RegionPrinter.h \
  include/llvm/Analysis/ScalarEvolutionExpander.h \
  include/llvm/Analysis/ScalarEvolutionExpressions.h \
  include/llvm/Analysis/ScalarEvolution.h \
  include/llvm/Analysis/ScalarEvolutionNormalization.h \
  include/llvm/Analysis/SparsePropagation.h \
  include/llvm/Analysis/TargetTransformInfo.h \
  include/llvm/Analysis/Trace.h \
  include/llvm/Analysis/ValueTracking.h \
  \
  include/llvm/AsmParser/Parser.h \
  include/llvm/AutoUpgrade.h \
  \
  include/llvm/Bitcode/BitCodes.h \
  include/llvm/Bitcode/BitcodeWriterPass.h \
  include/llvm/Bitcode/BitstreamReader.h \
  include/llvm/Bitcode/BitstreamWriter.h \
  include/llvm/Bitcode/LLVMBitCodes.h \
  include/llvm/Bitcode/ReaderWriter.h \
  \
  include/llvm/CodeGen/Analysis.h \
  include/llvm/CodeGen/AsmPrinter.h \
  include/llvm/CodeGen/CalcSpillWeights.h \
  include/llvm/CodeGen/CallingConvLower.h \
  include/llvm/CodeGen/CommandFlags.h \
  include/llvm/CodeGen/DAGCombine.h \
  include/llvm/CodeGen/DFAPacketizer.h \
  include/llvm/CodeGen/EdgeBundles.h \
  include/llvm/CodeGen/FastISel.h \
  include/llvm/CodeGen/FunctionLoweringInfo.h \
  include/llvm/CodeGen/GCMetadata.h \
  include/llvm/CodeGen/GCMetadataPrinter.h \
  include/llvm/CodeGen/GCs.h \
  include/llvm/CodeGen/GCStrategy.h \
  include/llvm/CodeGen/IntrinsicLowering.h \
  include/llvm/CodeGen/ISDOpcodes.h \
  include/llvm/CodeGen/JITCodeEmitter.h \
  include/llvm/CodeGen/LatencyPriorityQueue.h \
  include/llvm/CodeGen/LexicalScopes.h \
  include/llvm/CodeGen/LinkAllAsmWriterComponents.h \
  include/llvm/CodeGen/LinkAllCodegenComponents.h \
  include/llvm/CodeGen/LiveIntervalAnalysis.h \
  include/llvm/CodeGen/LiveInterval.h \
  include/llvm/CodeGen/LiveIntervalUnion.h \
  include/llvm/CodeGen/LivePhysRegs.h \
  include/llvm/CodeGen/LiveRangeEdit.h \
  include/llvm/CodeGen/LiveRegMatrix.h \
  include/llvm/CodeGen/LiveStackAnalysis.h \
  include/llvm/CodeGen/LiveVariables.h \
  include/llvm/CodeGen/MachineBasicBlock.h \
  include/llvm/CodeGen/MachineBlockFrequencyInfo.h \
  include/llvm/CodeGen/MachineBranchProbabilityInfo.h \
  include/llvm/CodeGen/MachineCodeEmitter.h \
  include/llvm/CodeGen/MachineCodeInfo.h \
  include/llvm/CodeGen/MachineConstantPool.h \
  include/llvm/CodeGen/MachineDominators.h \
  include/llvm/CodeGen/MachineFrameInfo.h \
  include/llvm/CodeGen/MachineFunctionAnalysis.h \
  include/llvm/CodeGen/MachineFunction.h \
  include/llvm/CodeGen/MachineFunctionPass.h \
  include/llvm/CodeGen/MachineInstrBuilder.h \
  include/llvm/CodeGen/MachineInstrBundle.h \
  include/llvm/CodeGen/MachineInstr.h \
  include/llvm/CodeGen/MachineJumpTableInfo.h \
  include/llvm/CodeGen/MachineLoopInfo.h \
  include/llvm/CodeGen/MachineMemOperand.h \
  include/llvm/CodeGen/MachineModuleInfo.h \
  include/llvm/CodeGen/MachineModuleInfoImpls.h \
  include/llvm/CodeGen/MachineOperand.h \
  include/llvm/CodeGen/MachinePassRegistry.h \
  include/llvm/CodeGen/MachinePostDominators.h \
  include/llvm/CodeGen/MachineRegisterInfo.h \
  include/llvm/CodeGen/MachineRelocation.h \
  include/llvm/CodeGen/MachineScheduler.h \
  include/llvm/CodeGen/MachineSSAUpdater.h \
  include/llvm/CodeGen/MachineTraceMetrics.h \
  include/llvm/CodeGen/MachORelocation.h \
  include/llvm/CodeGen/Passes.h \
  \
  include/llvm/CodeGen/PBQP/Graph.h \
  include/llvm/CodeGen/PBQP/HeuristicBase.h \
  \
  include/llvm/CodeGen/PBQP/Heuristics/Briggs.h \
  include/llvm/CodeGen/PBQP/HeuristicSolver.h \
  include/llvm/CodeGen/PBQP/Math.h \
  include/llvm/CodeGen/PBQP/Solution.h \
  include/llvm/CodeGen/PseudoSourceValue.h \
  include/llvm/CodeGen/RegAllocPBQP.h \
  include/llvm/CodeGen/RegAllocRegistry.h \
  include/llvm/CodeGen/RegisterClassInfo.h \
  include/llvm/CodeGen/RegisterPressure.h \
  include/llvm/CodeGen/RegisterScavenging.h \
  include/llvm/CodeGen/ResourcePriorityQueue.h \
  include/llvm/CodeGen/RuntimeLibcalls.h \
  include/llvm/CodeGen/ScheduleDAG.h \
  include/llvm/CodeGen/ScheduleDAGInstrs.h \
  include/llvm/CodeGen/ScheduleDFS.h \
  include/llvm/CodeGen/ScheduleHazardRecognizer.h \
  include/llvm/CodeGen/SchedulerRegistry.h \
  include/llvm/CodeGen/ScoreboardHazardRecognizer.h \
  include/llvm/CodeGen/SelectionDAG.h \
  include/llvm/CodeGen/SelectionDAGISel.h \
  include/llvm/CodeGen/SelectionDAGNodes.h \
  include/llvm/CodeGen/SlotIndexes.h \
  include/llvm/CodeGen/StackMapLivenessAnalysis.h \
  include/llvm/CodeGen/StackMaps.h \
  include/llvm/CodeGen/StackProtector.h \
  include/llvm/CodeGen/TargetLoweringObjectFileImpl.h \
  include/llvm/CodeGen/TargetSchedule.h \
  include/llvm/CodeGen/ValueTypes.h \
  include/llvm/CodeGen/ValueTypes.td \
  include/llvm/CodeGen/VirtRegMap.h \
  \
  include/llvm/Config/AsmParsers.def \
  include/llvm/Config/AsmPrinters.def \
  include/llvm/Config/config.h \
  include/llvm/Config/Disassemblers.def \
  include/llvm/Config/llvm-config.h \
  include/llvm/Config/Targets.def \
  \
  include/llvm/DebugInfo/DIContext.h \
  include/llvm/DebugInfo/DWARFFormValue.h \
  include/llvm/DebugInfo.h \
  include/llvm/DIBuilder.h \
  \
  include/llvm/ExecutionEngine/ExecutionEngine.h \
  include/llvm/ExecutionEngine/GenericValue.h \
  include/llvm/ExecutionEngine/Interpreter.h \
  include/llvm/ExecutionEngine/JITEventListener.h \
  include/llvm/ExecutionEngine/JIT.h \
  include/llvm/ExecutionEngine/JITMemoryManager.h \
  include/llvm/ExecutionEngine/MCJIT.h \
  include/llvm/ExecutionEngine/ObjectBuffer.h \
  include/llvm/ExecutionEngine/ObjectCache.h \
  include/llvm/ExecutionEngine/ObjectImage.h \
  include/llvm/ExecutionEngine/OProfileWrapper.h \
  include/llvm/ExecutionEngine/RTDyldMemoryManager.h \
  include/llvm/ExecutionEngine/RuntimeDyld.h \
  include/llvm/ExecutionEngine/SectionMemoryManager.h \
  include/llvm/GVMaterializer.h \
  include/llvm/InitializePasses.h \
  include/llvm/InstVisitor.h \
  \
  include/llvm/IR/Argument.h \
  include/llvm/IR/AssemblyAnnotationWriter.h \
  include/llvm/IR/Attributes.h \
  include/llvm/IR/BasicBlock.h \
  include/llvm/IR/CallingConv.h \
  include/llvm/IR/Constant.h \
  include/llvm/IR/Constants.h \
  include/llvm/IR/DataLayout.h \
  include/llvm/IR/DerivedTypes.h \
  include/llvm/IR/DiagnosticInfo.h \
  include/llvm/IR/DiagnosticPrinter.h \
  include/llvm/IR/Dominators.h \
  include/llvm/IR/Function.h \
  include/llvm/IR/GlobalAlias.h \
  include/llvm/IR/GlobalValue.h \
  include/llvm/IR/GlobalVariable.h \
  include/llvm/IR/InlineAsm.h \
  include/llvm/IR/InstrTypes.h \
  include/llvm/IR/Instruction.def \
  include/llvm/IR/Instruction.h \
  include/llvm/IR/Instructions.h \
  include/llvm/IR/IntrinsicInst.h \
  include/llvm/IR/IntrinsicsAArch64.td \
  include/llvm/IR/IntrinsicsARM.td \
  include/llvm/IR/Intrinsics.gen \
  include/llvm/IR/Intrinsics.h \
  include/llvm/IR/IntrinsicsHexagon.td \
  include/llvm/IR/IntrinsicsMips.td \
  include/llvm/IR/IntrinsicsNVVM.td \
  include/llvm/IR/IntrinsicsPowerPC.td \
  include/llvm/IR/IntrinsicsR600.td \
  include/llvm/IR/Intrinsics.td \
  include/llvm/IR/IntrinsicsX86.td \
  include/llvm/IR/IntrinsicsXCore.td \
  include/llvm/IR/IRBuilder.h \
  include/llvm/IR/IRPrintingPasses.h \
  include/llvm/IR/LegacyPassManager.h \
  include/llvm/IR/LegacyPassManagers.h \
  include/llvm/IR/LLVMContext.h \
  include/llvm/IR/Mangler.h \
  include/llvm/IR/MDBuilder.h \
  include/llvm/IR/Metadata.h \
  include/llvm/IR/Module.h \
  include/llvm/IR/OperandTraits.h \
  include/llvm/IR/Operator.h \
  include/llvm/IR/PassManager.h \
  include/llvm/IR/SymbolTableListTraits.h \
  include/llvm/IR/TypeBuilder.h \
  include/llvm/IR/TypeFinder.h \
  include/llvm/IR/Type.h \
  include/llvm/IR/Use.h \
  include/llvm/IR/User.h \
  include/llvm/IR/Value.h \
  include/llvm/IR/ValueSymbolTable.h \
  include/llvm/IR/Verifier.h \
  \
  include/llvm/IRReader/IRReader.h \
  \
  include/llvm/LineEditor/LineEditor.h \
  include/llvm/LinkAllIR.h \
  include/llvm/LinkAllPasses.h \
  include/llvm/Linker.h \
  \
  include/llvm/LTO/LTOCodeGenerator.h \
  include/llvm/LTO/LTOModule.h \
  \
  include/llvm/MC/MachineLocation.h \
  include/llvm/MC/MCAsmBackend.h \
  include/llvm/MC/MCAsmInfoCOFF.h \
  include/llvm/MC/MCAsmInfoDarwin.h \
  include/llvm/MC/MCAsmInfoELF.h \
  include/llvm/MC/MCAsmInfo.h \
  include/llvm/MC/MCAsmLayout.h \
  include/llvm/MC/MCAssembler.h \
  include/llvm/MC/MCAtom.h \
  include/llvm/MC/MCCodeEmitter.h \
  include/llvm/MC/MCCodeGenInfo.h \
  include/llvm/MC/MCContext.h \
  include/llvm/MC/MCDirectives.h \
  include/llvm/MC/MCDisassembler.h \
  include/llvm/MC/MCDwarf.h \
  include/llvm/MC/MCELF.h \
  include/llvm/MC/MCELFObjectWriter.h \
  include/llvm/MC/MCELFStreamer.h \
  include/llvm/MC/MCELFSymbolFlags.h \
  include/llvm/MC/MCExpr.h \
  include/llvm/MC/MCExternalSymbolizer.h \
  include/llvm/MC/MCFixedLenDisassembler.h \
  include/llvm/MC/MCFixup.h \
  include/llvm/MC/MCFixupKindInfo.h \
  include/llvm/MC/MCFunction.h \
  include/llvm/MC/MCInstBuilder.h \
  include/llvm/MC/MCInst.h \
  include/llvm/MC/MCInstPrinter.h \
  include/llvm/MC/MCInstrAnalysis.h \
  include/llvm/MC/MCInstrDesc.h \
  include/llvm/MC/MCInstrInfo.h \
  include/llvm/MC/MCInstrItineraries.h \
  include/llvm/MC/MCLabel.h \
  include/llvm/MC/MCMachObjectWriter.h \
  include/llvm/MC/MCMachOSymbolFlags.h \
  include/llvm/MC/MCModule.h \
  include/llvm/MC/MCModuleYAML.h \
  include/llvm/MC/MCObjectDisassembler.h \
  include/llvm/MC/MCObjectFileInfo.h \
  include/llvm/MC/MCObjectStreamer.h \
  include/llvm/MC/MCObjectSymbolizer.h \
  include/llvm/MC/MCObjectWriter.h \
  \
  include/llvm/MC/MCParser/AsmCond.h \
  include/llvm/MC/MCParser/AsmLexer.h \
  include/llvm/MC/MCParser/MCAsmLexer.h \
  include/llvm/MC/MCParser/MCAsmParserExtension.h \
  include/llvm/MC/MCParser/MCAsmParser.h \
  include/llvm/MC/MCParser/MCParsedAsmOperand.h \
  include/llvm/MC/MCRegisterInfo.h \
  include/llvm/MC/MCRelocationInfo.h \
  include/llvm/MC/MCSchedule.h \
  include/llvm/MC/MCSectionCOFF.h \
  include/llvm/MC/MCSectionELF.h \
  include/llvm/MC/MCSection.h \
  include/llvm/MC/MCSectionMachO.h \
  include/llvm/MC/MCStreamer.h \
  include/llvm/MC/MCSubtargetInfo.h \
  include/llvm/MC/MCSymbol.h \
  include/llvm/MC/MCSymbolizer.h \
  include/llvm/MC/MCTargetAsmParser.h \
  include/llvm/MC/MCValue.h \
  include/llvm/MC/MCWin64EH.h \
  include/llvm/MC/MCWinCOFFObjectWriter.h \
  include/llvm/MC/SectionKind.h \
  include/llvm/MC/SubtargetFeature.h \
  \
  include/llvm/Object/Archive.h \
  include/llvm/Object/Binary.h \
  include/llvm/Object/COFF.h \
  include/llvm/Object/COFFYAML.h \
  include/llvm/Object/ELF.h \
  include/llvm/Object/ELFObjectFile.h \
  include/llvm/Object/ELFTypes.h \
  include/llvm/Object/ELFYAML.h \
  include/llvm/Object/Error.h \
  include/llvm/Object/IRObjectFile.h \
  include/llvm/Object/MachO.h \
  include/llvm/Object/MachOUniversal.h \
  include/llvm/Object/ObjectFile.h \
  include/llvm/Object/RelocVisitor.h \
  include/llvm/Object/SymbolicFile.h \
  include/llvm/Object/YAML.h \
  \
  include/llvm/Option/Arg.h \
  include/llvm/Option/ArgList.h \
  include/llvm/Option/Option.h \
  include/llvm/Option/OptParser.td \
  include/llvm/Option/OptSpecifier.h \
  include/llvm/Option/OptTable.h \
  include/llvm/PassAnalysisSupport.h \
  include/llvm/Pass.h \
  include/llvm/PassManager.h \
  include/llvm/PassRegistry.h \
  include/llvm/PassSupport.h \
  \
  include/llvm/Support/AIXDataTypesFix.h \
  include/llvm/Support/AlignOf.h \
  include/llvm/Support/Allocator.h \
  include/llvm/Support/ARMBuildAttributes.h \
  include/llvm/Support/ARMEHABI.h \
  include/llvm/Support/ArrayRecycler.h \
  include/llvm/Support/Atomic.h \
  include/llvm/Support/BlockFrequency.h \
  include/llvm/Support/BranchProbability.h \
  include/llvm/Support/CallSite.h \
  include/llvm/Support/Capacity.h \
  include/llvm/Support/Casting.h \
  include/llvm/Support/CBindingWrapping.h \
  include/llvm/Support/CFG.h \
  include/llvm/Support/circular_raw_ostream.h \
  include/llvm/Support/CodeGen.h \
  include/llvm/Support/COFF.h \
  include/llvm/Support/CommandLine.h \
  include/llvm/Support/Compiler.h \
  include/llvm/Support/Compression.h \
  include/llvm/Support/ConstantFolder.h \
  include/llvm/Support/ConstantRange.h \
  include/llvm/Support/ConvertUTF.h \
  include/llvm/Support/CrashRecoveryContext.h \
  include/llvm/Support/DataExtractor.h \
  include/llvm/Support/DataFlow.h \
  include/llvm/Support/DataStream.h \
  include/llvm/Support/DataTypes.h \
  include/llvm/Support/Debug.h \
  include/llvm/Support/DebugLoc.h \
  include/llvm/Support/Disassembler.h \
  include/llvm/Support/DOTGraphTraits.h \
  include/llvm/Support/Dwarf.h \
  include/llvm/Support/DynamicLibrary.h \
  include/llvm/Support/ELF.h \
  include/llvm/Support/Endian.h \
  include/llvm/Support/Errno.h \
  include/llvm/Support/ErrorHandling.h \
  include/llvm/Support/ErrorOr.h \
  include/llvm/Support/FEnv.h \
  include/llvm/Support/FileOutputBuffer.h \
  include/llvm/Support/FileSystem.h \
  include/llvm/Support/FileUtilities.h \
  include/llvm/Support/Format.h \
  include/llvm/Support/FormattedStream.h \
  include/llvm/Support/GCOV.h \
  include/llvm/Support/GenericDomTreeConstruction.h \
  include/llvm/Support/GenericDomTree.h \
  include/llvm/Support/GetElementPtrTypeIterator.h \
  include/llvm/Support/GraphWriter.h \
  include/llvm/Support/Host.h \
  include/llvm/Support/IncludeFile.h \
  include/llvm/Support/InstIterator.h \
  include/llvm/Support/LeakDetector.h \
  include/llvm/Support/LEB128.h \
  include/llvm/Support/LineIterator.h \
  include/llvm/Support/Locale.h \
  include/llvm/Support/LockFileManager.h \
  include/llvm/Support/MachO.h \
  include/llvm/Support/ManagedStatic.h \
  include/llvm/Support/MathExtras.h \
  include/llvm/Support/MD5.h \
  include/llvm/Support/MemoryBuffer.h \
  include/llvm/Support/Memory.h \
  include/llvm/Support/MemoryObject.h \
  include/llvm/Support/MutexGuard.h \
  include/llvm/Support/Mutex.h \
  include/llvm/Support/NoFolder.h \
  include/llvm/Support/OutputBuffer.h \
  include/llvm/Support/PassNameParser.h \
  include/llvm/Support/Path.h \
  include/llvm/Support/PatternMatch.h \
  include/llvm/Support/PluginLoader.h \
  include/llvm/Support/PointerLikeTypeTraits.h \
  include/llvm/Support/PredIteratorCache.h \
  include/llvm/Support/PrettyStackTrace.h \
  include/llvm/Support/Process.h \
  include/llvm/Support/Program.h \
  include/llvm/Support/raw_os_ostream.h \
  include/llvm/Support/raw_ostream.h \
  include/llvm/Support/Recycler.h \
  include/llvm/Support/RecyclingAllocator.h \
  include/llvm/Support/Regex.h \
  include/llvm/Support/Registry.h \
  include/llvm/Support/RegistryParser.h \
  include/llvm/Support/RWMutex.h \
  include/llvm/Support/SaveAndRestore.h \
  include/llvm/Support/Signals.h \
  include/llvm/Support/SMLoc.h \
  include/llvm/Support/Solaris.h \
  include/llvm/Support/SourceMgr.h \
  include/llvm/Support/StreamableMemoryObject.h \
  include/llvm/Support/StringPool.h \
  include/llvm/Support/StringRefMemoryObject.h \
  include/llvm/Support/SwapByteOrder.h \
  include/llvm/Support/system_error.h \
  include/llvm/Support/SystemUtils.h \
  include/llvm/Support/TargetFolder.h \
  include/llvm/Support/TargetRegistry.h \
  include/llvm/Support/TargetSelect.h \
  include/llvm/Support/Threading.h \
  include/llvm/Support/ThreadLocal.h \
  include/llvm/Support/Timer.h \
  include/llvm/Support/TimeValue.h \
  include/llvm/Support/ToolOutputFile.h \
  include/llvm/Support/type_traits.h \
  include/llvm/Support/UnicodeCharRanges.h \
  include/llvm/Support/Unicode.h \
  include/llvm/Support/Valgrind.h \
  include/llvm/Support/ValueHandle.h \
  include/llvm/Support/Watchdog.h \
  include/llvm/Support/Win64EH.h \
  include/llvm/Support/YAMLParser.h \
  include/llvm/Support/YAMLTraits.h \
  \
  include/llvm/TableGen/Error.h \
  include/llvm/TableGen/Main.h \
  include/llvm/TableGen/Record.h \
  include/llvm/TableGen/StringMatcher.h \
  include/llvm/TableGen/StringToOffsetTable.h \
  include/llvm/TableGen/TableGenBackend.h \
  \
  include/llvm/Target/CostTable.h \
  include/llvm/Target/TargetCallingConv.h \
  include/llvm/Target/TargetCallingConv.td \
  include/llvm/Target/TargetFrameLowering.h \
  include/llvm/Target/TargetInstrInfo.h \
  include/llvm/Target/TargetIntrinsicInfo.h \
  include/llvm/Target/TargetItinerary.td \
  include/llvm/Target/TargetJITInfo.h \
  include/llvm/Target/TargetLibraryInfo.h \
  include/llvm/Target/TargetLowering.h \
  include/llvm/Target/TargetLoweringObjectFile.h \
  include/llvm/Target/TargetMachine.h \
  include/llvm/Target/TargetOpcodes.h \
  include/llvm/Target/TargetOptions.h \
  include/llvm/Target/TargetRegisterInfo.h \
  include/llvm/Target/TargetSchedule.td \
  include/llvm/Target/TargetSelectionDAGInfo.h \
  include/llvm/Target/TargetSelectionDAG.td \
  include/llvm/Target/TargetSubtargetInfo.h \
  include/llvm/Target/Target.td \
  \
  include/llvm/Transforms/Instrumentation.h \
  \
  include/llvm/Transforms/IPO/InlinerPass.h \
  include/llvm/Transforms/IPO/PassManagerBuilder.h \
  include/llvm/Transforms/IPO.h \
  include/llvm/Transforms/ObjCARC.h \
  include/llvm/Transforms/Scalar.h \
  \
  include/llvm/Transforms/Utils/ASanStackFrameLayout.h \
  include/llvm/Transforms/Utils/BasicBlockUtils.h \
  include/llvm/Transforms/Utils/BuildLibCalls.h \
  include/llvm/Transforms/Utils/BypassSlowDivision.h \
  include/llvm/Transforms/Utils/Cloning.h \
  include/llvm/Transforms/Utils/CmpInstAnalysis.h \
  include/llvm/Transforms/Utils/CodeExtractor.h \
  include/llvm/Transforms/Utils/GlobalStatus.h \
  include/llvm/Transforms/Utils/IntegerDivision.h \
  include/llvm/Transforms/Utils/Local.h \
  include/llvm/Transforms/Utils/LoopUtils.h \
  include/llvm/Transforms/Utils/ModuleUtils.h \
  include/llvm/Transforms/Utils/PromoteMemToReg.h \
  include/llvm/Transforms/Utils/SimplifyIndVar.h \
  include/llvm/Transforms/Utils/SimplifyLibCalls.h \
  include/llvm/Transforms/Utils/SpecialCaseList.h \
  include/llvm/Transforms/Utils/SSAUpdater.h \
  include/llvm/Transforms/Utils/SSAUpdaterImpl.h \
  include/llvm/Transforms/Utils/UnifyFunctionExitNodes.h \
  include/llvm/Transforms/Utils/UnrollLoop.h \
  include/llvm/Transforms/Utils/ValueMapper.h \
  include/llvm/Transforms/Vectorize.h 
# CAUTION: The trailing space above is needed. DO NOT delete.

INCLUDE_LLVM_C_FILES := \
  include/llvm-c/Analysis.h \
  include/llvm-c/BitReader.h \
  include/llvm-c/BitWriter.h \
  include/llvm-c/Core.h \
  include/llvm-c/Disassembler.h \
  include/llvm-c/ExecutionEngine.h \
  include/llvm-c/Initialization.h \
  include/llvm-c/IRReader.h \
  include/llvm-c/Linker.h \
  include/llvm-c/LinkTimeOptimizer.h \
  include/llvm-c/lto.h \
  include/llvm-c/Object.h \
  include/llvm-c/Support.h \
  include/llvm-c/Target.h \
  include/llvm-c/TargetMachine.h \
  \
  include/llvm-c/Transforms/IPO.h \
  include/llvm-c/Transforms/PassManagerBuilder.h \
  include/llvm-c/Transforms/Scalar.h \
  include/llvm-c/Transforms/Vectorize.h 
# CAUTION: The trailing space above is needed. DO NOT delete.

LIB_CLANG_HEADERS := \
  lib/clang/@CLANG_VERSION@/include/altivec.h \
  lib/clang/@CLANG_VERSION@/include/ammintrin.h \
  lib/clang/@CLANG_VERSION@/include/arm_neon.h \
  lib/clang/@CLANG_VERSION@/include/avx2intrin.h \
  lib/clang/@CLANG_VERSION@/include/avxintrin.h \
  lib/clang/@CLANG_VERSION@/include/bmi2intrin.h \
  lib/clang/@CLANG_VERSION@/include/bmiintrin.h \
  lib/clang/@CLANG_VERSION@/include/cpuid.h \
  lib/clang/@CLANG_VERSION@/include/emmintrin.h \
  lib/clang/@CLANG_VERSION@/include/f16cintrin.h \
  lib/clang/@CLANG_VERSION@/include/float.h \
  lib/clang/@CLANG_VERSION@/include/fma4intrin.h \
  lib/clang/@CLANG_VERSION@/include/fmaintrin.h \
  lib/clang/@CLANG_VERSION@/include/immintrin.h \
  lib/clang/@CLANG_VERSION@/include/Intrin.h \
  lib/clang/@CLANG_VERSION@/include/iso646.h \
  lib/clang/@CLANG_VERSION@/include/limits.h \
  lib/clang/@CLANG_VERSION@/include/lzcntintrin.h \
  lib/clang/@CLANG_VERSION@/include/mm3dnow.h \
  lib/clang/@CLANG_VERSION@/include/mmintrin.h \
  lib/clang/@CLANG_VERSION@/include/mm_malloc.h \
  lib/clang/@CLANG_VERSION@/include/module.map \
  lib/clang/@CLANG_VERSION@/include/nmmintrin.h \
  lib/clang/@CLANG_VERSION@/include/pmmintrin.h \
  lib/clang/@CLANG_VERSION@/include/popcntintrin.h \
  lib/clang/@CLANG_VERSION@/include/prfchwintrin.h \
  lib/clang/@CLANG_VERSION@/include/rdseedintrin.h \
  lib/clang/@CLANG_VERSION@/include/rtmintrin.h \
  lib/clang/@CLANG_VERSION@/include/shaintrin.h \
  lib/clang/@CLANG_VERSION@/include/smmintrin.h \
  lib/clang/@CLANG_VERSION@/include/stdalign.h \
  lib/clang/@CLANG_VERSION@/include/stdarg.h \
  lib/clang/@CLANG_VERSION@/include/stdbool.h \
  lib/clang/@CLANG_VERSION@/include/stddef.h \
  lib/clang/@CLANG_VERSION@/include/stdint.h \
  lib/clang/@CLANG_VERSION@/include/stdnoreturn.h \
  lib/clang/@CLANG_VERSION@/include/tbmintrin.h \
  lib/clang/@CLANG_VERSION@/include/tgmath.h \
  lib/clang/@CLANG_VERSION@/include/tmmintrin.h \
  lib/clang/@CLANG_VERSION@/include/unwind.h \
  lib/clang/@CLANG_VERSION@/include/varargs.h \
  lib/clang/@CLANG_VERSION@/include/__wmmintrin_aes.h \
  lib/clang/@CLANG_VERSION@/include/wmmintrin.h \
  lib/clang/@CLANG_VERSION@/include/__wmmintrin_pclmul.h \
  lib/clang/@CLANG_VERSION@/include/x86intrin.h \
  lib/clang/@CLANG_VERSION@/include/xmmintrin.h \
  lib/clang/@CLANG_VERSION@/include/xopintrin.h 
# CAUTION: The trailing space above is needed. DO NOT delete.

SHARE_FILES := \
  share/man/man1/cling.1 
# CAUTION: The trailing space above is needed. DO NOT delete.

INCLUDE_FILES := \
  $(INCLUDE_CLANG_FILES) \
  $(INCLUDE_CLANG_C_FILES) \
  $(INCLUDE_CLING_FILES) \
  $(INCLUDE_LLVM_FILES) \
  $(INCLUDE_LLVM_C_FILES)

LIB_FILES := \
  $(LIB_CLANG_HEADERS)

DIST_FILES := \
  $(BIN_FILES) \
  $(DOCS_FILES) \
  $(INCLUDE_FILES) \
  $(LIB_FILES) \
  $(SHARE_FILES)
