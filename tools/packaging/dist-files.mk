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

BIN_FILES_1 := \
  cling 
# CAUTION: The trailing space above is needed. DO NOT delete.

BIN_FILES := \
  $(addprefix @prefix@/bin/, $(BIN_FILES_1))

DOCS_FILES_1 := \
  llvm/html/cling/cling.html \
  llvm/html/cling/manpage.css \
  \
  llvm/ps/cling.ps 
# CAUTION: The trailing space above is needed. DO NOT delete.
  
DOCS_FILES := \
  $(addprefix @prefix@/docs/, $(DOCS_FILES_1))

INCLUDE_CLANG_FILES_1 := \
  Analysis/Analyses/CFGReachabilityAnalysis.h \
  Analysis/Analyses/Consumed.h \
  Analysis/Analyses/Dominators.h \
  Analysis/Analyses/FormatString.h \
  Analysis/Analyses/LiveVariables.h \
  Analysis/Analyses/PostOrderCFGView.h \
  Analysis/Analyses/PseudoConstantAnalysis.h \
  Analysis/Analyses/ReachableCode.h \
  Analysis/Analyses/ThreadSafety.h \
  Analysis/Analyses/UninitializedValues.h \
  Analysis/AnalysisContext.h \
  Analysis/AnalysisDiagnostic.h \
  Analysis/CallGraph.h \
  Analysis/CFG.h \
  Analysis/CFGStmtMap.h \
  \
  Analysis/DomainSpecific/CocoaConventions.h \
  Analysis/DomainSpecific/ObjCNoReturn.h \
  \
  Analysis/FlowSensitive/DataflowSolver.h \
  Analysis/FlowSensitive/DataflowValues.h \
  Analysis/ProgramPoint.h \
  \
  Analysis/Support/BumpVector.h \
  \
  ARCMigrate/ARCMTActions.h \
  ARCMigrate/ARCMT.h \
  ARCMigrate/FileRemapper.h \
  \
  AST/APValue.h \
  AST/ASTConsumer.h \
  AST/ASTContext.h \
  AST/ASTDiagnostic.h \
  AST/ASTFwd.h \
  AST/AST.h \
  AST/ASTImporter.h \
  AST/ASTLambda.h \
  AST/ASTMutationListener.h \
  AST/ASTTypeTraits.h \
  AST/ASTUnresolvedSet.h \
  AST/ASTVector.h \
  AST/AttrDump.inc \
  AST/Attr.h \
  AST/AttrImpl.inc \
  AST/AttrIterator.h \
  \
  AST/AttrVisitor.inc \
  AST/BaseSubobject.h \
  AST/BuiltinTypes.def \
  AST/CanonicalType.h \
  AST/CharUnits.h \
  AST/CommentBriefParser.h \
  AST/CommentCommandInfo.inc \
  AST/CommentCommandList.inc \
  AST/CommentCommandTraits.h \
  AST/CommentDiagnostic.h \
  AST/Comment.h \
  AST/CommentHTMLNamedCharacterReferences.inc \
  AST/CommentHTMLTags.inc \
  AST/CommentHTMLTagsProperties.inc \
  AST/CommentLexer.h \
  AST/CommentNodes.inc \
  AST/CommentParser.h \
  AST/CommentSema.h \
  AST/CommentVisitor.h \
  AST/CXXInheritance.h \
  AST/DataRecursiveASTVisitor.h \
  AST/DeclAccessPair.h \
  AST/DeclarationName.h \
  AST/DeclBase.h \
  AST/DeclContextInternals.h \
  AST/DeclCXX.h \
  AST/DeclFriend.h \
  AST/DeclGroup.h \
  AST/Decl.h \
  AST/DeclLookups.h \
  AST/DeclNodes.inc \
  AST/DeclObjC.h \
  AST/DeclOpenMP.h \
  AST/DeclTemplate.h \
  AST/DeclVisitor.h \
  AST/DependentDiagnostic.h \
  AST/EvaluatedExprVisitor.h \
  AST/ExprCXX.h \
  AST/Expr.h \
  AST/ExprObjC.h \
  AST/ExternalASTSource.h \
  AST/GlobalDecl.h \
  AST/Mangle.h \
  AST/MangleNumberingContext.h \
  AST/NestedNameSpecifier.h \
  AST/NSAPI.h \
  AST/OpenMPClause.h \
  AST/OperationKinds.h \
  AST/ParentMap.h \
  AST/PrettyPrinter.h \
  AST/RawCommentList.h \
  AST/RecordLayout.h \
  AST/RecursiveASTVisitor.h \
  AST/Redeclarable.h \
  AST/SelectorLocationsKind.h \
  AST/StmtCXX.h \
  AST/StmtGraphTraits.h \
  AST/Stmt.h \
  AST/StmtIterator.h \
  AST/StmtNodes.inc \
  AST/StmtObjC.h \
  AST/StmtOpenMP.h \
  AST/StmtVisitor.h \
  AST/TemplateBase.h \
  AST/TemplateName.h \
  AST/Type.h \
  AST/TypeLoc.h \
  AST/TypeLocNodes.def \
  AST/TypeLocVisitor.h \
  AST/TypeNodes.def \
  AST/TypeOrdering.h \
  AST/TypeVisitor.h \
  AST/UnresolvedSet.h \
  AST/VTableBuilder.h \
  AST/VTTBuilder.h \
  \
  ASTMatchers/ASTMatchers.h \
  ASTMatchers/ASTMatchersInternal.h \
  ASTMatchers/ASTMatchersMacros.h \
  ASTMatchers/ASTMatchFinder.h \
  \
  ASTMatchers/Dynamic/Diagnostics.h \
  ASTMatchers/Dynamic/Parser.h \
  ASTMatchers/Dynamic/Registry.h \
  ASTMatchers/Dynamic/VariantValue.h \
  \
  Basic/ABI.h \
  Basic/AddressSpaces.h \
  Basic/AllDiagnostics.h \
  Basic/arm_neon.inc \
  Basic/AttrKinds.h \
  Basic/AttrList.inc \
  Basic/BuiltinsAArch64.def \
  Basic/BuiltinsARM.def \
  Basic/Builtins.def \
  Basic/Builtins.h \
  Basic/BuiltinsHexagon.def \
  Basic/BuiltinsMips.def \
  Basic/BuiltinsNVPTX.def \
  Basic/BuiltinsPPC.def \
  Basic/BuiltinsX86.def \
  Basic/BuiltinsXCore.def \
  Basic/CapturedStmt.h \
  Basic/CharInfo.h \
  Basic/CommentOptions.h \
  Basic/DiagnosticAnalysisKinds.inc \
  Basic/DiagnosticASTKinds.inc \
  Basic/DiagnosticCategories.h \
  Basic/DiagnosticCommentKinds.inc \
  Basic/DiagnosticCommonKinds.inc \
  Basic/DiagnosticDriverKinds.inc \
  Basic/DiagnosticFrontendKinds.inc \
  Basic/DiagnosticGroups.inc \
  Basic/Diagnostic.h \
  Basic/DiagnosticIDs.h \
  Basic/DiagnosticIndexName.inc \
  Basic/DiagnosticLexKinds.inc \
  Basic/DiagnosticOptions.def \
  Basic/DiagnosticOptions.h \
  Basic/DiagnosticParseKinds.inc \
  Basic/DiagnosticSemaKinds.inc \
  Basic/DiagnosticSerializationKinds.inc \
  Basic/ExceptionSpecificationType.h \
  Basic/ExpressionTraits.h \
  Basic/FileManager.h \
  Basic/FileSystemOptions.h \
  Basic/FileSystemStatCache.h \
  Basic/IdentifierTable.h \
  Basic/Lambda.h \
  Basic/LangOptions.def \
  Basic/LangOptions.h \
  Basic/Linkage.h \
  Basic/LLVM.h \
  Basic/MacroBuilder.h \
  Basic/Module.h \
  Basic/ObjCRuntime.h \
  Basic/OnDiskHashTable.h \
  Basic/OpenCLExtensions.def \
  Basic/OpenMPKinds.def \
  Basic/OpenMPKinds.h \
  Basic/OperatorKinds.def \
  Basic/OperatorKinds.h \
  Basic/OperatorPrecedence.h \
  Basic/PartialDiagnostic.h \
  Basic/PlistSupport.h \
  Basic/PrettyStackTrace.h \
  Basic/Sanitizers.def \
  Basic/SourceLocation.h \
  Basic/SourceManager.h \
  Basic/SourceManagerInternals.h \
  Basic/Specifiers.h \
  Basic/TargetBuiltins.h \
  Basic/TargetCXXABI.h \
  Basic/TargetInfo.h \
  Basic/TargetOptions.h \
  Basic/TemplateKinds.h \
  Basic/TokenKinds.def \
  Basic/TokenKinds.h \
  Basic/TypeTraits.h \
  Basic/Version.h \
  Basic/Version.inc \
  Basic/VersionTuple.h \
  Basic/VirtualFileSystem.h \
  Basic/Visibility.h \
  \
  CodeGen/BackendUtil.h \
  CodeGen/CGFunctionInfo.h \
  CodeGen/CodeGenABITypes.h \
  CodeGen/CodeGenAction.h \
  CodeGen/ModuleBuilder.h \
  \
  Config/config.h \
  \
  Driver/Action.h \
  Driver/CC1AsOptions.h \
  Driver/CC1AsOptions.inc \
  Driver/Compilation.h \
  Driver/DriverDiagnostic.h \
  Driver/Driver.h \
  Driver/Job.h \
  Driver/Multilib.h \
  Driver/Options.h \
  Driver/Options.inc \
  Driver/Phases.h \
  Driver/SanitizerArgs.h \
  Driver/ToolChain.h \
  Driver/Tool.h \
  Driver/Types.def \
  Driver/Types.h \
  Driver/Util.h \
  \
  Edit/Commit.h \
  Edit/EditedSource.h \
  Edit/EditsReceiver.h \
  Edit/FileOffset.h \
  Edit/Rewriters.h \
  \
  Format/Format.h \
  \
  Frontend/ASTConsumers.h \
  Frontend/ASTUnit.h \
  Frontend/ChainedDiagnosticConsumer.h \
  Frontend/ChainedIncludesSource.h \
  Frontend/CodeGenOptions.def \
  Frontend/CodeGenOptions.h \
  Frontend/CommandLineSourceLoc.h \
  Frontend/CompilerInstance.h \
  Frontend/CompilerInvocation.h \
  Frontend/DependencyOutputOptions.h \
  Frontend/DiagnosticRenderer.h \
  Frontend/FrontendAction.h \
  Frontend/FrontendActions.h \
  Frontend/FrontendDiagnostic.h \
  Frontend/FrontendOptions.h \
  Frontend/FrontendPluginRegistry.h \
  Frontend/LangStandard.h \
  Frontend/LangStandards.def \
  Frontend/LayoutOverrideSource.h \
  Frontend/LogDiagnosticPrinter.h \
  Frontend/MigratorOptions.h \
  Frontend/MultiplexConsumer.h \
  Frontend/PreprocessorOutputOptions.h \
  Frontend/SerializedDiagnosticPrinter.h \
  Frontend/TextDiagnosticBuffer.h \
  Frontend/TextDiagnostic.h \
  Frontend/TextDiagnosticPrinter.h \
  Frontend/Utils.h \
  Frontend/VerifyDiagnosticConsumer.h \
  \
  FrontendTool/Utils.h \
  \
  Index/CommentToXML.h \
  Index/USRGeneration.h \
  \
  Lex/AttrSpellings.inc \
  Lex/CodeCompletionHandler.h \
  Lex/DirectoryLookup.h \
  Lex/ExternalPreprocessorSource.h \
  Lex/HeaderMap.h \
  Lex/HeaderSearch.h \
  Lex/HeaderSearchOptions.h \
  Lex/LexDiagnostic.h \
  Lex/Lexer.h \
  Lex/LiteralSupport.h \
  Lex/MacroArgs.h \
  Lex/MacroInfo.h \
  Lex/ModuleLoader.h \
  Lex/ModuleMap.h \
  Lex/MultipleIncludeOpt.h \
  Lex/PPCallbacks.h \
  Lex/PPConditionalDirectiveRecord.h \
  Lex/Pragma.h \
  Lex/PreprocessingRecord.h \
  Lex/Preprocessor.h \
  Lex/PreprocessorLexer.h \
  Lex/PreprocessorOptions.h \
  Lex/PTHLexer.h \
  Lex/PTHManager.h \
  Lex/ScratchBuffer.h \
  Lex/TokenConcatenation.h \
  Lex/Token.h \
  Lex/TokenLexer.h \
  \
  Parse/AttrParserStringSwitches.inc \
  Parse/ParseAST.h \
  Parse/ParseDiagnostic.h \
  Parse/Parser.h \
  Parse/RAIIObjectsForParser.h \
  \
  Rewrite/Core/DeltaTree.h \
  Rewrite/Core/HTMLRewrite.h \
  Rewrite/Core/Rewriter.h \
  Rewrite/Core/RewriteRope.h \
  Rewrite/Core/TokenRewriter.h \
  \
  Rewrite/Frontend/ASTConsumers.h \
  Rewrite/Frontend/FixItRewriter.h \
  Rewrite/Frontend/FrontendActions.h \
  Rewrite/Frontend/Rewriters.h \
  \
  Sema/AnalysisBasedWarnings.h \
  Sema/AttributeList.h \
  Sema/AttrParsedAttrImpl.inc \
  Sema/AttrParsedAttrKinds.inc \
  Sema/AttrParsedAttrList.inc \
  Sema/AttrSpellingListIndex.inc \
  Sema/AttrTemplateInstantiate.inc \
  Sema/CodeCompleteConsumer.h \
  Sema/CodeCompleteOptions.h \
  Sema/CXXFieldCollector.h \
  Sema/DeclSpec.h \
  Sema/DelayedDiagnostic.h \
  Sema/Designator.h \
  Sema/ExternalSemaSource.h \
  Sema/IdentifierResolver.h \
  Sema/Initialization.h \
  Sema/LocInfoType.h \
  Sema/Lookup.h \
  Sema/MultiplexExternalSemaSource.h \
  Sema/ObjCMethodList.h \
  Sema/Overload.h \
  Sema/Ownership.h \
  Sema/ParsedTemplate.h \
  Sema/PrettyDeclStackTrace.h \
  Sema/Scope.h \
  Sema/ScopeInfo.h \
  Sema/SemaConsumer.h \
  Sema/SemaDiagnostic.h \
  Sema/SemaFixItUtils.h \
  Sema/Sema.h \
  Sema/SemaInternal.h \
  Sema/SemaLambda.h \
  Sema/TemplateDeduction.h \
  Sema/Template.h \
  Sema/TypoCorrection.h \
  Sema/Weak.h \
  \
  Serialization/ASTBitCodes.h \
  Serialization/ASTDeserializationListener.h \
  Serialization/ASTReader.h \
  Serialization/ASTWriter.h \
  Serialization/AttrPCHRead.inc \
  Serialization/AttrPCHWrite.inc \
  Serialization/ContinuousRangeMap.h \
  Serialization/GlobalModuleIndex.h \
  Serialization/Module.h \
  Serialization/ModuleManager.h \
  Serialization/SerializationDiagnostic.h \
  \
  StaticAnalyzer/Checkers/ClangCheckers.h \
  StaticAnalyzer/Checkers/LocalCheckers.h \
  StaticAnalyzer/Checkers/ObjCRetainCount.h \
  \
  StaticAnalyzer/Core/Analyses.def \
  StaticAnalyzer/Core/AnalyzerOptions.h \
  \
  StaticAnalyzer/Core/BugReporter/BugReporter.h \
  StaticAnalyzer/Core/BugReporter/BugReporterVisitor.h \
  StaticAnalyzer/Core/BugReporter/BugType.h \
  StaticAnalyzer/Core/BugReporter/CommonBugCategories.h \
  StaticAnalyzer/Core/BugReporter/PathDiagnostic.h \
  StaticAnalyzer/Core/Checker.h \
  StaticAnalyzer/Core/CheckerManager.h \
  StaticAnalyzer/Core/CheckerOptInfo.h \
  StaticAnalyzer/Core/CheckerRegistry.h \
  StaticAnalyzer/Core/PathDiagnosticConsumers.h \
  \
  StaticAnalyzer/Core/PathSensitive/AnalysisManager.h \
  StaticAnalyzer/Core/PathSensitive/APSIntType.h \
  StaticAnalyzer/Core/PathSensitive/BasicValueFactory.h \
  StaticAnalyzer/Core/PathSensitive/BlockCounter.h \
  StaticAnalyzer/Core/PathSensitive/CallEvent.h \
  StaticAnalyzer/Core/PathSensitive/CheckerContext.h \
  StaticAnalyzer/Core/PathSensitive/CheckerHelpers.h \
  StaticAnalyzer/Core/PathSensitive/ConstraintManager.h \
  StaticAnalyzer/Core/PathSensitive/CoreEngine.h \
  StaticAnalyzer/Core/PathSensitive/DynamicTypeInfo.h \
  StaticAnalyzer/Core/PathSensitive/Environment.h \
  StaticAnalyzer/Core/PathSensitive/ExplodedGraph.h \
  StaticAnalyzer/Core/PathSensitive/ExprEngine.h \
  StaticAnalyzer/Core/PathSensitive/FunctionSummary.h \
  StaticAnalyzer/Core/PathSensitive/MemRegion.h \
  StaticAnalyzer/Core/PathSensitive/ProgramState_Fwd.h \
  StaticAnalyzer/Core/PathSensitive/ProgramState.h \
  StaticAnalyzer/Core/PathSensitive/ProgramStateTrait.h \
  StaticAnalyzer/Core/PathSensitive/Store.h \
  StaticAnalyzer/Core/PathSensitive/StoreRef.h \
  StaticAnalyzer/Core/PathSensitive/SubEngine.h \
  StaticAnalyzer/Core/PathSensitive/SummaryManager.h \
  StaticAnalyzer/Core/PathSensitive/SValBuilder.h \
  StaticAnalyzer/Core/PathSensitive/SVals.h \
  StaticAnalyzer/Core/PathSensitive/SymbolManager.h \
  StaticAnalyzer/Core/PathSensitive/TaintManager.h \
  StaticAnalyzer/Core/PathSensitive/TaintTag.h \
  StaticAnalyzer/Core/PathSensitive/WorkList.h \
  \
  StaticAnalyzer/Frontend/AnalysisConsumer.h \
  StaticAnalyzer/Frontend/CheckerRegistration.h \
  StaticAnalyzer/Frontend/FrontendActions.h \
  \
  Tooling/ArgumentsAdjusters.h \
  Tooling/CommonOptionsParser.h \
  Tooling/CompilationDatabase.h \
  Tooling/CompilationDatabasePluginRegistry.h \
  Tooling/FileMatchTrie.h \
  Tooling/JSONCompilationDatabase.h \
  Tooling/RefactoringCallbacks.h \
  Tooling/Refactoring.h \
  Tooling/ReplacementsYaml.h \
  Tooling/Tooling.h 
# CAUTION: The trailing space above is needed. DO NOT delete.

INCLUDE_CLANG_FILES := \
  $(addprefix @prefix@/include/clang/, $(INCLUDE_CLANG_FILES_1))

INCLUDE_CLANG_C_FILES_1 := \
  BuildSystem.h \
  CXCompilationDatabase.h \
  CXErrorCode.h \
  CXString.h \
  Index.h \
  Platform.h 
# CAUTION: The trailing space above is needed. DO NOT delete.

INCLUDE_CLANG_C_FILES := \
  $(addprefix @prefix@/include/clang-c/, $(INCLUDE_CLANG_C_FILES_1))

INCLUDE_CLING_FILES := \
  Interpreter/CIFactory.h \
  Interpreter/ClangInternalState.h \
  Interpreter/ClingOptions.h \
  Interpreter/ClingOptions.inc \
  Interpreter/CompilationOptions.h \
  Interpreter/CValuePrinter.h \
  Interpreter/DynamicExprInfo.h \
  Interpreter/DynamicLibraryManager.h \
  Interpreter/DynamicLookupLifetimeHandler.h \
  Interpreter/DynamicLookupRuntimeUniverse.h \
  Interpreter/InterpreterCallbacks.h \
  Interpreter/Interpreter.h \
  Interpreter/InvocationOptions.h \
  Interpreter/LookupHelper.h \
  Interpreter/RuntimeException.h \
  Interpreter/RuntimeUniverse.h \
  Interpreter/Transaction.h \
  Interpreter/Value.h \
  \
  MetaProcessor/MetaProcessor.h \
  \
  UserInterface/CompilationException.h \
  UserInterface/UserInterface.h \
  \
  Utils/AST.h 
# CAUTION: The trailing space above is needed. DO NOT delete.

INCLUDE_CLING_FILES := \
  $(addprefix @prefix@/include/cling/, $(INCLUDE_CLING_FILES_1))

INCLUDE_LLVM_FILES_1 := \
  ADT/APFloat.h \
  ADT/APInt.h \
  ADT/APSInt.h \
  ADT/ArrayRef.h \
  ADT/BitVector.h \
  ADT/DAGDeltaAlgorithm.h \
  ADT/DeltaAlgorithm.h \
  ADT/DenseMap.h \
  ADT/DenseMapInfo.h \
  ADT/DenseSet.h \
  ADT/DepthFirstIterator.h \
  ADT/edit_distance.h \
  ADT/EquivalenceClasses.h \
  ADT/FoldingSet.h \
  ADT/GraphTraits.h \
  ADT/Hashing.h \
  ADT/ilist.h \
  ADT/ilist_node.h \
  ADT/ImmutableIntervalMap.h \
  ADT/ImmutableList.h \
  ADT/ImmutableMap.h \
  ADT/ImmutableSet.h \
  ADT/IndexedMap.h \
  ADT/IntEqClasses.h \
  ADT/IntervalMap.h \
  ADT/IntrusiveRefCntPtr.h \
  ADT/MapVector.h \
  ADT/None.h \
  ADT/Optional.h \
  ADT/OwningPtr.h \
  ADT/PackedVector.h \
  ADT/PointerIntPair.h \
  ADT/PointerUnion.h \
  ADT/polymorphic_ptr.h \
  ADT/PostOrderIterator.h \
  ADT/PriorityQueue.h \
  ADT/SCCIterator.h \
  ADT/ScopedHashTable.h \
  ADT/SetOperations.h \
  ADT/SetVector.h \
  ADT/SmallBitVector.h \
  ADT/SmallPtrSet.h \
  ADT/SmallSet.h \
  ADT/SmallString.h \
  ADT/SmallVector.h \
  ADT/SparseBitVector.h \
  ADT/SparseMultiSet.h \
  ADT/SparseSet.h \
  ADT/Statistic.h \
  ADT/STLExtras.h \
  ADT/StringExtras.h \
  ADT/StringMap.h \
  ADT/StringRef.h \
  ADT/StringSet.h \
  ADT/StringSwitch.h \
  ADT/TinyPtrVector.h \
  ADT/Triple.h \
  ADT/Twine.h \
  ADT/UniqueVector.h \
  ADT/ValueMap.h \
  ADT/VariadicFunction.h \
  \
  Analysis/AliasAnalysis.h \
  Analysis/AliasSetTracker.h \
  Analysis/BlockFrequencyImpl.h \
  Analysis/BlockFrequencyInfo.h \
  Analysis/BranchProbabilityInfo.h \
  Analysis/CallGraph.h \
  Analysis/CallGraphSCCPass.h \
  Analysis/CallPrinter.h \
  Analysis/CaptureTracking.h \
  Analysis/CFG.h \
  Analysis/CFGPrinter.h \
  Analysis/CodeMetrics.h \
  Analysis/ConstantFolding.h \
  Analysis/ConstantsScanner.h \
  Analysis/DependenceAnalysis.h \
  Analysis/DominanceFrontier.h \
  Analysis/DomPrinter.h \
  Analysis/DOTGraphTraitsPass.h \
  Analysis/FindUsedTypes.h \
  Analysis/InlineCost.h \
  Analysis/InstructionSimplify.h \
  Analysis/Interval.h \
  Analysis/IntervalIterator.h \
  Analysis/IntervalPartition.h \
  Analysis/IVUsers.h \
  Analysis/LazyCallGraph.h \
  Analysis/LazyValueInfo.h \
  Analysis/LibCallAliasAnalysis.h \
  Analysis/LibCallSemantics.h \
  Analysis/Lint.h \
  Analysis/Loads.h \
  Analysis/LoopInfo.h \
  Analysis/LoopInfoImpl.h \
  Analysis/LoopIterator.h \
  Analysis/LoopPass.h \
  Analysis/MemoryBuiltins.h \
  Analysis/MemoryDependenceAnalysis.h \
  Analysis/Passes.h \
  Analysis/PHITransAddr.h \
  Analysis/PostDominators.h \
  Analysis/PtrUseVisitor.h \
  Analysis/RegionInfo.h \
  Analysis/RegionIterator.h \
  Analysis/RegionPass.h \
  Analysis/RegionPrinter.h \
  Analysis/ScalarEvolutionExpander.h \
  Analysis/ScalarEvolutionExpressions.h \
  Analysis/ScalarEvolution.h \
  Analysis/ScalarEvolutionNormalization.h \
  Analysis/SparsePropagation.h \
  Analysis/TargetTransformInfo.h \
  Analysis/Trace.h \
  Analysis/ValueTracking.h \
  \
  AsmParser/Parser.h \
  AutoUpgrade.h \
  \
  Bitcode/BitCodes.h \
  Bitcode/BitcodeWriterPass.h \
  Bitcode/BitstreamReader.h \
  Bitcode/BitstreamWriter.h \
  Bitcode/LLVMBitCodes.h \
  Bitcode/ReaderWriter.h \
  \
  CodeGen/Analysis.h \
  CodeGen/AsmPrinter.h \
  CodeGen/CalcSpillWeights.h \
  CodeGen/CallingConvLower.h \
  CodeGen/CommandFlags.h \
  CodeGen/DAGCombine.h \
  CodeGen/DFAPacketizer.h \
  CodeGen/EdgeBundles.h \
  CodeGen/FastISel.h \
  CodeGen/FunctionLoweringInfo.h \
  CodeGen/GCMetadata.h \
  CodeGen/GCMetadataPrinter.h \
  CodeGen/GCs.h \
  CodeGen/GCStrategy.h \
  CodeGen/IntrinsicLowering.h \
  CodeGen/ISDOpcodes.h \
  CodeGen/JITCodeEmitter.h \
  CodeGen/LatencyPriorityQueue.h \
  CodeGen/LexicalScopes.h \
  CodeGen/LinkAllAsmWriterComponents.h \
  CodeGen/LinkAllCodegenComponents.h \
  CodeGen/LiveIntervalAnalysis.h \
  CodeGen/LiveInterval.h \
  CodeGen/LiveIntervalUnion.h \
  CodeGen/LivePhysRegs.h \
  CodeGen/LiveRangeEdit.h \
  CodeGen/LiveRegMatrix.h \
  CodeGen/LiveStackAnalysis.h \
  CodeGen/LiveVariables.h \
  CodeGen/MachineBasicBlock.h \
  CodeGen/MachineBlockFrequencyInfo.h \
  CodeGen/MachineBranchProbabilityInfo.h \
  CodeGen/MachineCodeEmitter.h \
  CodeGen/MachineCodeInfo.h \
  CodeGen/MachineConstantPool.h \
  CodeGen/MachineDominators.h \
  CodeGen/MachineFrameInfo.h \
  CodeGen/MachineFunctionAnalysis.h \
  CodeGen/MachineFunction.h \
  CodeGen/MachineFunctionPass.h \
  CodeGen/MachineInstrBuilder.h \
  CodeGen/MachineInstrBundle.h \
  CodeGen/MachineInstr.h \
  CodeGen/MachineJumpTableInfo.h \
  CodeGen/MachineLoopInfo.h \
  CodeGen/MachineMemOperand.h \
  CodeGen/MachineModuleInfo.h \
  CodeGen/MachineModuleInfoImpls.h \
  CodeGen/MachineOperand.h \
  CodeGen/MachinePassRegistry.h \
  CodeGen/MachinePostDominators.h \
  CodeGen/MachineRegisterInfo.h \
  CodeGen/MachineRelocation.h \
  CodeGen/MachineScheduler.h \
  CodeGen/MachineSSAUpdater.h \
  CodeGen/MachineTraceMetrics.h \
  CodeGen/MachORelocation.h \
  CodeGen/Passes.h \
  \
  CodeGen/PBQP/Graph.h \
  CodeGen/PBQP/HeuristicBase.h \
  \
  CodeGen/PBQP/Heuristics/Briggs.h \
  CodeGen/PBQP/HeuristicSolver.h \
  CodeGen/PBQP/Math.h \
  CodeGen/PBQP/Solution.h \
  CodeGen/PseudoSourceValue.h \
  CodeGen/RegAllocPBQP.h \
  CodeGen/RegAllocRegistry.h \
  CodeGen/RegisterClassInfo.h \
  CodeGen/RegisterPressure.h \
  CodeGen/RegisterScavenging.h \
  CodeGen/ResourcePriorityQueue.h \
  CodeGen/RuntimeLibcalls.h \
  CodeGen/ScheduleDAG.h \
  CodeGen/ScheduleDAGInstrs.h \
  CodeGen/ScheduleDFS.h \
  CodeGen/ScheduleHazardRecognizer.h \
  CodeGen/SchedulerRegistry.h \
  CodeGen/ScoreboardHazardRecognizer.h \
  CodeGen/SelectionDAG.h \
  CodeGen/SelectionDAGISel.h \
  CodeGen/SelectionDAGNodes.h \
  CodeGen/SlotIndexes.h \
  CodeGen/StackMapLivenessAnalysis.h \
  CodeGen/StackMaps.h \
  CodeGen/StackProtector.h \
  CodeGen/TargetLoweringObjectFileImpl.h \
  CodeGen/TargetSchedule.h \
  CodeGen/ValueTypes.h \
  CodeGen/ValueTypes.td \
  CodeGen/VirtRegMap.h \
  \
  Config/AsmParsers.def \
  Config/AsmPrinters.def \
  Config/config.h \
  Config/Disassemblers.def \
  Config/llvm-config.h \
  Config/Targets.def \
  \
  DebugInfo/DIContext.h \
  DebugInfo/DWARFFormValue.h \
  DebugInfo.h \
  DIBuilder.h \
  \
  ExecutionEngine/ExecutionEngine.h \
  ExecutionEngine/GenericValue.h \
  ExecutionEngine/Interpreter.h \
  ExecutionEngine/JITEventListener.h \
  ExecutionEngine/JIT.h \
  ExecutionEngine/JITMemoryManager.h \
  ExecutionEngine/MCJIT.h \
  ExecutionEngine/ObjectBuffer.h \
  ExecutionEngine/ObjectCache.h \
  ExecutionEngine/ObjectImage.h \
  ExecutionEngine/OProfileWrapper.h \
  ExecutionEngine/RTDyldMemoryManager.h \
  ExecutionEngine/RuntimeDyld.h \
  ExecutionEngine/SectionMemoryManager.h \
  GVMaterializer.h \
  InitializePasses.h \
  InstVisitor.h \
  \
  IR/Argument.h \
  IR/AssemblyAnnotationWriter.h \
  IR/Attributes.h \
  IR/BasicBlock.h \
  IR/CallingConv.h \
  IR/Constant.h \
  IR/Constants.h \
  IR/DataLayout.h \
  IR/DerivedTypes.h \
  IR/DiagnosticInfo.h \
  IR/DiagnosticPrinter.h \
  IR/Dominators.h \
  IR/Function.h \
  IR/GlobalAlias.h \
  IR/GlobalValue.h \
  IR/GlobalVariable.h \
  IR/InlineAsm.h \
  IR/InstrTypes.h \
  IR/Instruction.def \
  IR/Instruction.h \
  IR/Instructions.h \
  IR/IntrinsicInst.h \
  IR/IntrinsicsAArch64.td \
  IR/IntrinsicsARM.td \
  IR/Intrinsics.gen \
  IR/Intrinsics.h \
  IR/IntrinsicsHexagon.td \
  IR/IntrinsicsMips.td \
  IR/IntrinsicsNVVM.td \
  IR/IntrinsicsPowerPC.td \
  IR/IntrinsicsR600.td \
  IR/Intrinsics.td \
  IR/IntrinsicsX86.td \
  IR/IntrinsicsXCore.td \
  IR/IRBuilder.h \
  IR/IRPrintingPasses.h \
  IR/LegacyPassManager.h \
  IR/LegacyPassManagers.h \
  IR/LLVMContext.h \
  IR/Mangler.h \
  IR/MDBuilder.h \
  IR/Metadata.h \
  IR/Module.h \
  IR/OperandTraits.h \
  IR/Operator.h \
  IR/PassManager.h \
  IR/SymbolTableListTraits.h \
  IR/TypeBuilder.h \
  IR/TypeFinder.h \
  IR/Type.h \
  IR/Use.h \
  IR/User.h \
  IR/Value.h \
  IR/ValueSymbolTable.h \
  IR/Verifier.h \
  \
  IRReader/IRReader.h \
  \
  LineEditor/LineEditor.h \
  LinkAllIR.h \
  LinkAllPasses.h \
  Linker.h \
  \
  LTO/LTOCodeGenerator.h \
  LTO/LTOModule.h \
  \
  MC/MachineLocation.h \
  MC/MCAsmBackend.h \
  MC/MCAsmInfoCOFF.h \
  MC/MCAsmInfoDarwin.h \
  MC/MCAsmInfoELF.h \
  MC/MCAsmInfo.h \
  MC/MCAsmLayout.h \
  MC/MCAssembler.h \
  MC/MCAtom.h \
  MC/MCCodeEmitter.h \
  MC/MCCodeGenInfo.h \
  MC/MCContext.h \
  MC/MCDirectives.h \
  MC/MCDisassembler.h \
  MC/MCDwarf.h \
  MC/MCELF.h \
  MC/MCELFObjectWriter.h \
  MC/MCELFStreamer.h \
  MC/MCELFSymbolFlags.h \
  MC/MCExpr.h \
  MC/MCExternalSymbolizer.h \
  MC/MCFixedLenDisassembler.h \
  MC/MCFixup.h \
  MC/MCFixupKindInfo.h \
  MC/MCFunction.h \
  MC/MCInstBuilder.h \
  MC/MCInst.h \
  MC/MCInstPrinter.h \
  MC/MCInstrAnalysis.h \
  MC/MCInstrDesc.h \
  MC/MCInstrInfo.h \
  MC/MCInstrItineraries.h \
  MC/MCLabel.h \
  MC/MCMachObjectWriter.h \
  MC/MCMachOSymbolFlags.h \
  MC/MCModule.h \
  MC/MCModuleYAML.h \
  MC/MCObjectDisassembler.h \
  MC/MCObjectFileInfo.h \
  MC/MCObjectStreamer.h \
  MC/MCObjectSymbolizer.h \
  MC/MCObjectWriter.h \
  \
  MC/MCParser/AsmCond.h \
  MC/MCParser/AsmLexer.h \
  MC/MCParser/MCAsmLexer.h \
  MC/MCParser/MCAsmParserExtension.h \
  MC/MCParser/MCAsmParser.h \
  MC/MCParser/MCParsedAsmOperand.h \
  MC/MCRegisterInfo.h \
  MC/MCRelocationInfo.h \
  MC/MCSchedule.h \
  MC/MCSectionCOFF.h \
  MC/MCSectionELF.h \
  MC/MCSection.h \
  MC/MCSectionMachO.h \
  MC/MCStreamer.h \
  MC/MCSubtargetInfo.h \
  MC/MCSymbol.h \
  MC/MCSymbolizer.h \
  MC/MCTargetAsmParser.h \
  MC/MCValue.h \
  MC/MCWin64EH.h \
  MC/MCWinCOFFObjectWriter.h \
  MC/SectionKind.h \
  MC/SubtargetFeature.h \
  \
  Object/Archive.h \
  Object/Binary.h \
  Object/COFF.h \
  Object/COFFYAML.h \
  Object/ELF.h \
  Object/ELFObjectFile.h \
  Object/ELFTypes.h \
  Object/ELFYAML.h \
  Object/Error.h \
  Object/IRObjectFile.h \
  Object/MachO.h \
  Object/MachOUniversal.h \
  Object/ObjectFile.h \
  Object/RelocVisitor.h \
  Object/SymbolicFile.h \
  Object/YAML.h \
  \
  Option/Arg.h \
  Option/ArgList.h \
  Option/Option.h \
  Option/OptParser.td \
  Option/OptSpecifier.h \
  Option/OptTable.h \
  PassAnalysisSupport.h \
  Pass.h \
  PassManager.h \
  PassRegistry.h \
  PassSupport.h \
  \
  Support/AIXDataTypesFix.h \
  Support/AlignOf.h \
  Support/Allocator.h \
  Support/ARMBuildAttributes.h \
  Support/ARMEHABI.h \
  Support/ArrayRecycler.h \
  Support/Atomic.h \
  Support/BlockFrequency.h \
  Support/BranchProbability.h \
  Support/CallSite.h \
  Support/Capacity.h \
  Support/Casting.h \
  Support/CBindingWrapping.h \
  Support/CFG.h \
  Support/circular_raw_ostream.h \
  Support/CodeGen.h \
  Support/COFF.h \
  Support/CommandLine.h \
  Support/Compiler.h \
  Support/Compression.h \
  Support/ConstantFolder.h \
  Support/ConstantRange.h \
  Support/ConvertUTF.h \
  Support/CrashRecoveryContext.h \
  Support/DataExtractor.h \
  Support/DataFlow.h \
  Support/DataStream.h \
  Support/DataTypes.h \
  Support/Debug.h \
  Support/DebugLoc.h \
  Support/Disassembler.h \
  Support/DOTGraphTraits.h \
  Support/Dwarf.h \
  Support/DynamicLibrary.h \
  Support/ELF.h \
  Support/Endian.h \
  Support/Errno.h \
  Support/ErrorHandling.h \
  Support/ErrorOr.h \
  Support/FEnv.h \
  Support/FileOutputBuffer.h \
  Support/FileSystem.h \
  Support/FileUtilities.h \
  Support/Format.h \
  Support/FormattedStream.h \
  Support/GCOV.h \
  Support/GenericDomTreeConstruction.h \
  Support/GenericDomTree.h \
  Support/GetElementPtrTypeIterator.h \
  Support/GraphWriter.h \
  Support/Host.h \
  Support/IncludeFile.h \
  Support/InstIterator.h \
  Support/LeakDetector.h \
  Support/LEB128.h \
  Support/LICENSE.TXT \
  Support/LineIterator.h \
  Support/Locale.h \
  Support/LockFileManager.h \
  Support/MachO.h \
  Support/ManagedStatic.h \
  Support/MathExtras.h \
  Support/MD5.h \
  Support/MemoryBuffer.h \
  Support/Memory.h \
  Support/MemoryObject.h \
  Support/MutexGuard.h \
  Support/Mutex.h \
  Support/NoFolder.h \
  Support/OutputBuffer.h \
  Support/PassNameParser.h \
  Support/Path.h \
  Support/PatternMatch.h \
  Support/PluginLoader.h \
  Support/PointerLikeTypeTraits.h \
  Support/PredIteratorCache.h \
  Support/PrettyStackTrace.h \
  Support/Process.h \
  Support/Program.h \
  Support/raw_os_ostream.h \
  Support/raw_ostream.h \
  Support/Recycler.h \
  Support/RecyclingAllocator.h \
  Support/Regex.h \
  Support/Registry.h \
  Support/RegistryParser.h \
  Support/RWMutex.h \
  Support/SaveAndRestore.h \
  Support/Signals.h \
  Support/SMLoc.h \
  Support/Solaris.h \
  Support/SourceMgr.h \
  Support/StreamableMemoryObject.h \
  Support/StringPool.h \
  Support/StringRefMemoryObject.h \
  Support/SwapByteOrder.h \
  Support/system_error.h \
  Support/SystemUtils.h \
  Support/TargetFolder.h \
  Support/TargetRegistry.h \
  Support/TargetSelect.h \
  Support/Threading.h \
  Support/ThreadLocal.h \
  Support/Timer.h \
  Support/TimeValue.h \
  Support/ToolOutputFile.h \
  Support/type_traits.h \
  Support/UnicodeCharRanges.h \
  Support/Unicode.h \
  Support/Valgrind.h \
  Support/ValueHandle.h \
  Support/Watchdog.h \
  Support/Win64EH.h \
  Support/YAMLParser.h \
  Support/YAMLTraits.h \
  \
  TableGen/Error.h \
  TableGen/Main.h \
  TableGen/Record.h \
  TableGen/StringMatcher.h \
  TableGen/StringToOffsetTable.h \
  TableGen/TableGenBackend.h \
  \
  Target/CostTable.h \
  Target/TargetCallingConv.h \
  Target/TargetCallingConv.td \
  Target/TargetFrameLowering.h \
  Target/TargetInstrInfo.h \
  Target/TargetIntrinsicInfo.h \
  Target/TargetItinerary.td \
  Target/TargetJITInfo.h \
  Target/TargetLibraryInfo.h \
  Target/TargetLowering.h \
  Target/TargetLoweringObjectFile.h \
  Target/TargetMachine.h \
  Target/TargetOpcodes.h \
  Target/TargetOptions.h \
  Target/TargetRegisterInfo.h \
  Target/TargetSchedule.td \
  Target/TargetSelectionDAGInfo.h \
  Target/TargetSelectionDAG.td \
  Target/TargetSubtargetInfo.h \
  Target/Target.td \
  \
  Transforms/Instrumentation.h \
  \
  Transforms/IPO/InlinerPass.h \
  Transforms/IPO/PassManagerBuilder.h \
  Transforms/IPO.h \
  Transforms/ObjCARC.h \
  Transforms/Scalar.h \
  \
  Transforms/Utils/ASanStackFrameLayout.h \
  Transforms/Utils/BasicBlockUtils.h \
  Transforms/Utils/BuildLibCalls.h \
  Transforms/Utils/BypassSlowDivision.h \
  Transforms/Utils/Cloning.h \
  Transforms/Utils/CmpInstAnalysis.h \
  Transforms/Utils/CodeExtractor.h \
  Transforms/Utils/GlobalStatus.h \
  Transforms/Utils/IntegerDivision.h \
  Transforms/Utils/Local.h \
  Transforms/Utils/LoopUtils.h \
  Transforms/Utils/ModuleUtils.h \
  Transforms/Utils/PromoteMemToReg.h \
  Transforms/Utils/SimplifyIndVar.h \
  Transforms/Utils/SimplifyLibCalls.h \
  Transforms/Utils/SpecialCaseList.h \
  Transforms/Utils/SSAUpdater.h \
  Transforms/Utils/SSAUpdaterImpl.h \
  Transforms/Utils/UnifyFunctionExitNodes.h \
  Transforms/Utils/UnrollLoop.h \
  Transforms/Utils/ValueMapper.h \
  Transforms/Vectorize.h 
# CAUTION: The trailing space above is needed. DO NOT delete.

INCLUDE_LLVM_FILES := \
  $(addprefix @prefix@/include/llvm/, $(INCLUDE_LLVM_FILES_1))

INCLUDE_LLVM_C_FILES_1 := \
  Analysis.h \
  BitReader.h \
  BitWriter.h \
  Core.h \
  Disassembler.h \
  ExecutionEngine.h \
  Initialization.h \
  IRReader.h \
  Linker.h \
  LinkTimeOptimizer.h \
  lto.h \
  Object.h \
  Support.h \
  Target.h \
  TargetMachine.h \
  \
  Transforms/IPO.h \
  Transforms/PassManagerBuilder.h \
  Transforms/Scalar.h \
  Transforms/Vectorize.h 
# CAUTION: The trailing space above is needed. DO NOT delete.

INCLUDE_LLVM_C_FILES := \
  $(addprefix @prefix@/include/llvm-c/, $(INCLUDE_LLVM_C_FILES_1))

LIB_CLANG_HEADERS_1 := \
  clang/3.5/include/altivec.h \
  clang/3.5/include/ammintrin.h \
  clang/3.5/include/arm_neon.h \
  clang/3.5/include/avx2intrin.h \
  clang/3.5/include/avxintrin.h \
  clang/3.5/include/bmi2intrin.h \
  clang/3.5/include/bmiintrin.h \
  clang/3.5/include/cpuid.h \
  clang/3.5/include/emmintrin.h \
  clang/3.5/include/f16cintrin.h \
  clang/3.5/include/float.h \
  clang/3.5/include/fma4intrin.h \
  clang/3.5/include/fmaintrin.h \
  clang/3.5/include/immintrin.h \
  clang/3.5/include/Intrin.h \
  clang/3.5/include/iso646.h \
  clang/3.5/include/limits.h \
  clang/3.5/include/lzcntintrin.h \
  clang/3.5/include/mm3dnow.h \
  clang/3.5/include/mmintrin.h \
  clang/3.5/include/mm_malloc.h \
  clang/3.5/include/module.map \
  clang/3.5/include/nmmintrin.h \
  clang/3.5/include/pmmintrin.h \
  clang/3.5/include/popcntintrin.h \
  clang/3.5/include/prfchwintrin.h \
  clang/3.5/include/rdseedintrin.h \
  clang/3.5/include/rtmintrin.h \
  clang/3.5/include/shaintrin.h \
  clang/3.5/include/smmintrin.h \
  clang/3.5/include/stdalign.h \
  clang/3.5/include/stdarg.h \
  clang/3.5/include/stdbool.h \
  clang/3.5/include/stddef.h \
  clang/3.5/include/stdint.h \
  clang/3.5/include/stdnoreturn.h \
  clang/3.5/include/tbmintrin.h \
  clang/3.5/include/tgmath.h \
  clang/3.5/include/tmmintrin.h \
  clang/3.5/include/unwind.h \
  clang/3.5/include/varargs.h \
  clang/3.5/include/__wmmintrin_aes.h \
  clang/3.5/include/wmmintrin.h \
  clang/3.5/include/__wmmintrin_pclmul.h \
  clang/3.5/include/x86intrin.h \
  clang/3.5/include/xmmintrin.h \
  clang/3.5/include/xopintrin.h 
# CAUTION: The trailing space above is needed. DO NOT delete.

LIB_CLANG_HEADERS := \
  $(addprefix @prefix@/lib/, $(LIB_CLANG_HEADERS_1))

SHARE_FILES_1 := \
  man/man1/cling.1 
# CAUTION: The trailing space above is needed. DO NOT delete.

  
SHARE_FILES := \
  $(addprefix @prefix@/share/, $(SHARE_FILES_1))

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
