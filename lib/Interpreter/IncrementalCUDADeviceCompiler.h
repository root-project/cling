//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Simeon Ehrig <s.ehrig@hzdr.de>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_INCREMENTAL_CUDA_DEVICE_JIT_H
#define CLING_INCREMENTAL_CUDA_DEVICE_JIT_H

#include "clang/Frontend/CodeGenOptions.h"
#include "llvm/ADT/SmallVector.h"

#include <string>
#include <system_error>
#include <vector>

namespace cling {
  class InvocationOptions;
  class Transaction;
  class Interpreter;
} // namespace cling

namespace clang {
  class CompilerInstance;
  class HeaderSearchOptions;
  class LangOptions;
} // namespace clang

namespace llvm {
  class StringRef;
}

namespace cling {

  ///\brief The class is responsible for generating CUDA device code in
  /// cuda fatbinary form during the runtime. The PTX code is compiled by a
  /// second interpreter instance. The external NVIDIA tool fatbin is used with
  /// llvm::sys::ExecuteAndWait to generate a fatbinary file.
  ///
  class IncrementalCUDADeviceCompiler {

    ///\brief Contains the arguments for the cling nvptx and the nvidia
    /// fatbinary tool.
    struct CUDACompilerArgs {
      const std::string cppStdVersion;
      const std::string optLevel;
      const std::string smVersion;
      const std::string ptxSmVersion;
      const std::string fatbinSmVersion;
      ///\brief Argument for the fatbinary tool, which is depend, if the OS is
      /// 32 bit or 64 bit.
      const std::string fatbinArch;
      ///\brief True, if the flag -v is set.
      const bool verbose;
      ///\brief True, if the flag -g is set.
      const bool debug;
      ///\brief A list Arguments, which will passed to the clang nvptx.
      const std::vector<std::string> additionalPtxOpt;
      ///\brief A list Arguments, which will passed to the fatbinary tool.
      const std::vector<std::string> fatbinaryOpt;

      CUDACompilerArgs(std::string cppStdVersion, std::string optLevel,
                       std::string smVersion, std::string ptxSmVersion,
                       std::string fatbinSmVersion, std::string fatbinArch,
                       bool verbose, bool debug,
                       std::vector<std::string> additionalPtxOpt,
                       std::vector<std::string> fatbinaryOpt)
          : cppStdVersion(cppStdVersion), optLevel(optLevel),
            smVersion(smVersion), ptxSmVersion(ptxSmVersion),
            fatbinSmVersion(fatbinSmVersion), fatbinArch(fatbinArch),
            verbose(verbose), debug(debug), additionalPtxOpt(additionalPtxOpt),
            fatbinaryOpt(fatbinaryOpt) {}
    };

    std::unique_ptr<CUDACompilerArgs> m_CuArgs;

    ///\brief Interpreter instance with target NVPTX which compiles the input to
    ///LLVM IR. Then the LLVM IR is compiled to PTX via an additional backend.
    std::unique_ptr<Interpreter> m_PTX_interp;

    ///\brief Is true if the second interpreter instance was created and the
    /// NVIDIA fatbin tool was found.
    bool m_Init = false;

    ///\brief Path to the folder, where all files will put in. Ordinary the tmp
    /// folder. Have to end with a separator. Can be empty.
    const std::string m_FilePath;
    ///\brief Path to the fatbin file, which will used by the CUDACodeGen.
    const std::string m_FatbinFilePath;
    ///\brief Path to the PTX file.
    const std::string m_PTXFilePath;

    ///\brief Path to the NIVDIA tool fatbinary.
    std::string m_FatbinaryPath;

    ///\brief Find the path of the NIVDIA tool fatbinary.
    ///
    ///\param [in] invocationOptions - Can contains a custom path to the cuda
    ///       toolkit
    ///
    ///\returns True, whether the fatbinary was found.
    bool findToolchain(const cling::InvocationOptions& invocationOptions);

    ///\brief Add the include paths from the interpreter runtime to a argument
    /// list.
    ///
    ///\param [in,out] argv - The include commands will append to the argv
    /// vector.
    ///\param [in] headerSearchOptions - Contains information about all include
    /// paths.
    void addHeaderSearchPathFlags(
        std::vector<std::string>& argv,
        const std::shared_ptr<clang::HeaderSearchOptions> headerSearchOptions);

    ///\brief Compiles a PTX file from the current input. The PTX code is
    /// written to cling.ptx.
    ///
    ///\returns True, if the new cling.ptx was compiled.
    bool generatePTX();

    ///\brief Start the NVIDIA tool fatbinary. Generate a fatbin file
    /// from the cling.ptx. The fatbin code will write to the path of
    /// m_FatbinFilePath.
    ///
    ///\returns True, if the fatbinary tool returns 0.
    bool generateFatbinary();

    ///\brief The function set the values of m_CuArgs.
    ///
    ///\param [in] langOpts - The LangOptions of the CompilerInstance.
    ///\param [in] invocationOptions - The invocationOptions of the interpreter.
    ///\param [in] intprOptLevel - The optimization level of the interpreter.
    ///\param [in] debugInfo - The debugInfo of the CompilerInstance.
    void setCuArgs(const clang::LangOptions& langOpts,
                   const cling::InvocationOptions& invocationOptions,
                   const int intprOptLevel,
                   const clang::codegenoptions::DebugInfoKind debugInfo);

  public:
    ///\brief Constructor for IncrementalCUDADeviceCompiler
    ///
    ///\param [in] filePath - All files will generated in the folder of the
    ///       filePath, except the fatbin file, if it have another path. Have
    ///       to end with a separator. Can be empty.
    ///\param [in] optLevel - The optimization level of the interpreter
    /// instance.
    ///       The value will be copied, because a change of it is not allowed.
    ///\param [in] invocationOptions - Contains values for the arguments of
    ///       the interpreter instance and the NVIDIA tool fatbinary.
    ///\param [in] CI - Will be used for m_CuArgs and the include path handling.
    IncrementalCUDADeviceCompiler(
        const std::string& filePath, const int optLevel,
        const cling::InvocationOptions& invocationOptions,
        const clang::CompilerInstance& CI);

    ///\brief Generate an new fatbin file with the path in
    /// CudaGpuBinaryFileNames.
    /// It will add the content of input, to the existing source code, which was
    /// passed to compileDeviceCode, before.
    ///
    ///\param [in] input - The input directly from the UI. Attention, the string
    /// must not be wrapped or transformed.
    ///
    ///\returns True, if all stages of generating fatbin runs right and a new
    /// fatbin file is written.
    bool compileDeviceCode(const llvm::StringRef input);

    ///\brief Print some information of the IncrementalCUDADeviceCompiler to
    /// llvm::outs(). For Example the paths of the files and tools.
    void dump();
  };

} // namespace cling

#endif // CLING_INCREMENTAL_CUDA_DEVICE_JIT_H
