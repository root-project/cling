//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id: IncrementalParser.h 42072 2011-11-16 19:27:35Z vvassilev $
// author:  Vassil Vassilev <vvasilev@cern.ch>
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

    CompilationOptions() {
      DeclarationExtraction = 1;
      ValuePrinting = VPAuto;
      ResultEvaluation = 0;
      DynamicScoping = 0;
      Debug = 0;
      CodeGeneration = 1;
    }
  };
} // end namespace cling
#endif // CLING_COMPILATION_OPTIONS
