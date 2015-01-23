//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vvasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_COMPILATION_OPTIONS
#define CLING_COMPILATION_OPTIONS

namespace cling {

  ///\brief Options controlling the incremental compilation. Describe the set of
  /// custom AST consumers to be enabled/disabled.
  ///
  class CompilationOptions {
  public:
    ///\brief Whether or not to extract the declarations out from the processed
    /// input.
    ///
    unsigned DeclarationExtraction : 1;

    ///\brief Whether or not to print the result of the run input
    ///
    /// 0 -> Disabled; 1 -> Enabled; 2 -> Auto;
    ///
    unsigned ValuePrinting : 2;
    enum ValuePrinting { VPDisabled, VPEnabled, VPAuto };

    ///\brief Whether or not to return result from an execution.
    ///
    unsigned ResultEvaluation: 1;

    ///\brief Whether or not to extend the static scope with new information
    /// about the names available only at runtime
    ///
    unsigned DynamicScoping : 1;

    ///\brief Whether or not to print debug information on the fly
    ///
    unsigned Debug : 1;

    ///\brief Whether or not to generate executable (LLVM IR) code for the input
    /// or to cache the incoming declarations in a queue
    ///
    unsigned CodeGeneration : 1;

    ///\brief When generating executable, select whether to generate all
    /// the code (when false) or just the code needed when the input is
    /// describing code coming from an existing library.
    unsigned CodeGenerationForModule : 1;

    ///\brief Prompt input can look weird for the compiler, e.g.
    /// void __cling_prompt() { sin(0.1); } // warning: unused function call
    /// This flag suppresses these warnings; it should be set whenever input
    /// is wrapped.
    unsigned IgnorePromptDiags : 1;

    CompilationOptions() {
      DeclarationExtraction = 0;
      ValuePrinting = VPDisabled;
      ResultEvaluation = 0;
      DynamicScoping = 0;
      Debug = 0;
      CodeGeneration = 1;
      CodeGenerationForModule = 0;
      IgnorePromptDiags = 0;
    }

    bool operator==(CompilationOptions Other) const {
      return
        DeclarationExtraction == Other.DeclarationExtraction &&
        ValuePrinting         == Other.ValuePrinting &&
        ResultEvaluation      == Other.ResultEvaluation &&
        DynamicScoping        == Other.DynamicScoping &&
        Debug                 == Other.Debug &&
        CodeGeneration        == Other.CodeGeneration &&
        CodeGenerationForModule == Other.CodeGenerationForModule &&
        IgnorePromptDiags     == Other.IgnorePromptDiags;
    }

    bool operator!=(CompilationOptions Other) const {
      return
        DeclarationExtraction != Other.DeclarationExtraction ||
        ValuePrinting         != Other.ValuePrinting ||
        ResultEvaluation      != Other.ResultEvaluation ||
        DynamicScoping        != Other.DynamicScoping ||
        Debug                 != Other.Debug ||
        CodeGeneration        != Other.CodeGeneration ||
        CodeGenerationForModule != Other.CodeGenerationForModule ||
        IgnorePromptDiags     != Other.IgnorePromptDiags;
    }
  };
} // end namespace cling
#endif // CLING_COMPILATION_OPTIONS
