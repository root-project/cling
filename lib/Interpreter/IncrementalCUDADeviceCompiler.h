//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Simeon Ehrig <simeonehrig@web.de>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_INCREMENTAL_CUDA_DEVICE_JIT_H
#define CLING_INCREMENTAL_CUDA_DEVICE_JIT_H

#include "llvm/ADT/SmallVector.h"

#include <string>
#include <vector>

namespace cling{
    class InvocationOptions;
}

namespace clang {
    class CodeGenOptions;
}

namespace llvm {
    class StringRef;
}

namespace cling {

  ///\brief The class is responsible for generating CUDA device code in
  /// cuda fatbinary form during the runtime. It works with
  /// llvm::sys::ExecuteAndWait.
  /// 
  class IncrementalCUDADeviceCompiler {
    /// FIXME : Add handling of new included Headers. The include commands can
    /// be added by the prompt or via .L .

    ///\brief The counter responsible to generate a chain of .cu source files
    /// and .cu.pch files.
    unsigned int m_Counter;

    ///\brief Is true if all necessary files have been generated and clang and 
    /// cuda NVIDIA fatbinary are found.
    bool m_Init;

    ///\brief Path to the folder, where all files will put in. Ordinary the tmp
    /// folder. Have to end with a separator. Can be empty.
    std::string m_FilePath;
    ///\brief Path to the fatbin file, which will used by the CUDACodeGen.
    std::string m_FatbinFilePath;
    ///\brief Path to a empty dummy.cu file. The file is necessary to generate
    /// PTX code from the pch files.
    std::string m_DummyCUPath;
    ///\brief Path to the PTX file. Will be reused for every PTX generation.
    std::string m_PTXFilePath;
    ///\brief Will be used to generate .cu and .cu.pch files.
    std::string m_GenericFileName;
    ///\brief The SM-Level describes, which functions are possible in the code 
    /// and on the gpu. Just a number [1-7][0-9].
    std::string m_SMLevel;

    ///\brief Path to the clang++ compiler, which will used to compile the pch
    /// files and the PTX code. Should be in same folder, as the cling.
    std::string m_ClangPath;
    ///\brief Path to the NIVDIA tool fatbinary.
    std::string m_FatbinaryPath;

    ///\brief Contains the include commands for the cling runtime headers.
    llvm::SmallVector<std::string, 256> m_ClingHeaders;
    ///\brief Argument for the fatbinary tool, which is depend, if the OS is
    /// 32 bit or 64 bit.
    std::string m_FatbinArch;

    ///\brief Generate the dummy.cu file and set the paths of m_PTXFilePath and
    /// m_GenericFileName.
    ///
    ///\returns True, if it created a dummy.cu file.
    bool generateHelperFiles();

    ///\brief Find the path of the clang and the NIVDIA tool fatbinary. Clang
    /// have to be in the same folder as cling.
    ///
    ///\param [in] invocationOptions - Can contains a custom path to the cuda
    ///       toolkit
    ///
    ///\returns True, whether clang and fatbinary was found.
    bool searchCompilingTools(cling::InvocationOptions & invocationOptions);

    ///\brief Add the include path commands (-I...) to a argument list. The path
    /// points to the cling runtime headers.
    ///
    ///\param [in,out] argv - The include commands will append to the argv vector.
    void addClingHeaders(llvm::SmallVectorImpl<const char*> & argv);

    ///\brief Start an clang compiler with nvptx backend. Read the content of
    /// cling.cu and compile it to a new PCH file. If predecessor PCH file is
    /// existing, it will included.
    ///
    ///\returns True, if the clang returns 0.
    bool generatePCH();

    ///\brief Start an clang compiler with nvptx backend. Generate a PTX file
    /// from the latest PCH file. The PTX code will write to cling.ptx.
    ///
    ///\returns True, if the clang returns 0.
    bool generatePTX();

    ///\brief Start the NVIDIA tool fatbinary. Generate a fatbin file
    /// from the cling.ptx. The fatbin code will write to the path of
    /// m_FatbinFilePath.
    ///
    ///\returns True, if the fatbinary tool returns 0.
    bool generateFatbinaryInternal();

  public:
    ///\brief Constructor for IncrementalCUDADeviceCompiler
    ///
    ///\param [in] filePath - All files will generated in the folder of the
    ///       filePath, except the fatbin file, if it have another path. Have
    ///       to end with a separator. Can be empty.
    ///\param [in] CudaGpuBinaryFileNames - Path to the fatbin file. Must not
    ///       be empty.
    ///\param [in] invocationOptions - Contains values for the arguments of
    ///       clang and the NVIDIA tool fatbinary.
    ///\param [in] clingHeaders - Contains the paths to the cling runtime
    ///       headers with include command (-I).
    IncrementalCUDADeviceCompiler(std::string filePath, 
                                  std::string & CudaGpuBinaryFileNames,
                                  cling::InvocationOptions & invocationOptions,
                                  const llvm::SmallVectorImpl<std::string> & clingHeaders);

    ///\brief Generate an new fatbin file with the path in CudaGpuBinaryFileNames.
    /// It will add the content of input, to the existing source code, which was
    /// passed to generateFatbinary, before.
    ///
    ///\param [in] input - New source code. The function can select, if code
    ///       is relevant for the device side. Have to be valid CUDA C++ code.
    ///
    ///\returns True, if all stages of generating fatbin runs right and a new
    /// fatbin file is written.
    bool generateFatbinary(llvm::StringRef input);

    ///\brief Print some information of the IncrementalCUDADeviceCompiler to
    /// llvm::outs(). For Example the paths of the files and tools.
    void dump();

  };

} // end cling

#endif // CLING_INCREMENTAL_CUDA_DEVICE_JIT_H
