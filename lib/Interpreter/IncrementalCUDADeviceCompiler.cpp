//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Simeon Ehrig <simeonehrig@web.de>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "IncrementalCUDADeviceCompiler.h"

#include "cling/Interpreter/InvocationOptions.h"
#include "cling/Utils/Paths.h"
#include "clang/Lex/HeaderSearchOptions.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Path.h"
#include "llvm/ADT/Triple.h"

#include <string>

namespace cling {


  IncrementalCUDADeviceCompiler::IncrementalCUDADeviceCompiler(
      std::string filePath,
      std::string & CudaGpuBinaryFileNames,
      cling::InvocationOptions & invocationOptions,
      std::shared_ptr<clang::HeaderSearchOptions> headerSearchOptions)
     : m_Counter(0),
       m_FilePath(filePath),
       m_FatbinFilePath(CudaGpuBinaryFileNames),
       // We get for example sm_20 from the cling arguments and have to shrink to
       // 20.
       m_SMLevel(invocationOptions.CompilerOpts.CUDAGpuArch.empty() ? "20" :
         invocationOptions.CompilerOpts.CUDAGpuArch.substr(3) ),
       m_HeaderSearchOptions(headerSearchOptions) {
    assert(!CudaGpuBinaryFileNames.empty() && "CudaGpuBinaryFileNames can't be empty");

    m_Init = generateHelperFiles();
    m_Init = m_Init && searchCompilingTools(invocationOptions);

    llvm::Triple hostTarget(llvm::sys::getDefaultTargetTriple());
    m_FatbinArch = hostTarget.isArch64Bit() ? "-64" : "-32";
  }

  bool IncrementalCUDADeviceCompiler::generateHelperFiles(){
    // Generate an empty dummy.cu file.
    m_DummyCUPath = m_FilePath + "dummy.cu";
    std::error_code EC;
    llvm::raw_fd_ostream dummyCU(m_DummyCUPath, EC, llvm::sys::fs::F_Text);
    if(EC){ 
      llvm::errs() << "Could not open file: " << EC.message();
      return false;
    }
    dummyCU.close();

    m_PTXFilePath = m_FilePath + "cling.ptx";
    m_GenericFileName = m_FilePath + "cling";
    return true;
  }

  bool IncrementalCUDADeviceCompiler::searchCompilingTools(cling::InvocationOptions & invocationOptions){
    // Search after clang in the folder of cling.
    llvm::SmallString<128> cwd;
    llvm::sys::fs::current_path(cwd);
    cwd.append(llvm::sys::path::get_separator());
    cwd.append("clang++");
    m_ClangPath = cwd.c_str();
    // Check, if clang is existing and executable.
    if(!llvm::sys::fs::can_execute(m_ClangPath)){
      llvm::errs() << "Error: " << m_ClangPath << " not existing or executable!\n";
      return false;
    }

    // Use the custom CUDA toolkit path, if it set via cling argument.
    if(!invocationOptions.CompilerOpts.CUDAPath.empty()){
      m_FatbinaryPath = invocationOptions.CompilerOpts.CUDAPath + "/bin/fatbinary";
      if(!llvm::sys::fs::can_execute(m_FatbinaryPath)){
        llvm::errs() << "Error: " << m_FatbinaryPath << " not existing or executable!\n";
        return false;
      }
    }else{
      // Search after fatbinary on the system.
      if (llvm::ErrorOr<std::string> fatbinary = 
            llvm::sys::findProgramByName("fatbinary")) {
        llvm::SmallString<256> fatbinaryAbsolutePath;
        llvm::sys::fs::real_path(*fatbinary, fatbinaryAbsolutePath);
        m_FatbinaryPath = fatbinaryAbsolutePath.c_str();
      } else {
        llvm::errs() << "Error: nvidia tool fatbinary not found!\n" <<
          "Please add the cuda /bin path to PATH or set the toolkit path via --cuda-path argument.\n";
        return false;
      }
    }

    return true;
  }

  void IncrementalCUDADeviceCompiler::addHeaders(
      llvm::SmallVectorImpl<std::string> & argv){
    for(clang::HeaderSearchOptions::Entry e : m_HeaderSearchOptions->UserEntries){
      if(e.Group == clang::frontend::IncludeDirGroup::Angled)
        argv.push_back("-I" + e.Path);
    }
  }

  bool IncrementalCUDADeviceCompiler::generateFatbinary(llvm::StringRef input){
    if(!m_Init){
      llvm::errs() << "Error: Initializiation of CUDA Device Code Compiler failed\n";
      return false;
    }

    // Write the (CUDA) C++ source code to a file.
    std::error_code EC;
    llvm::raw_fd_ostream cuFile(m_GenericFileName + std::to_string(m_Counter)
                                + ".cu", EC, llvm::sys::fs::F_Text);
    if (EC) {
      llvm::errs() << "Could not open file: " << EC.message();
      return false;
    }
    cuFile << input;
    cuFile.close();

    if(!generatePCH()){
      return false;
    }

    if(!generatePTX()){
      return false;
    }

    if(!generateFatbinaryInternal()){
      return false;
    }

    ++m_Counter;
    return true;
  }

  bool IncrementalCUDADeviceCompiler::generatePCH() {
    // clang++ -std=c++14 -S -Xclang -emit-pch ${clingHeaders} cling[0-9].cu
    // -D__CLING__ -o cling[0-9].cu.pch ${ | -include-pch cling[0-9].cu.pch }
    // --cuda-gpu-arch=sm_${m_smLevel} -pthread --cuda-device-only
    llvm::SmallVector<const char*, 256> argv;

    // First argument have to be the program name.
    argv.push_back(m_ClangPath.c_str());

    // FIXME: Should replaced by the arguments of the cling instance.
    argv.push_back("-std=c++14");
    argv.push_back("-S");
    argv.push_back("-Xclang");
    argv.push_back("-emit-pch");
    llvm::SmallVector<std::string, 256> headers;
    addHeaders(headers);
    for(std::string & s : headers)
      argv.push_back(s.c_str());
    // Is necessary for the cling runtime header.
    argv.push_back("-D__CLING__");
    std::string cuFilePath = m_GenericFileName + std::to_string(m_Counter)
                             + ".cu";
    argv.push_back(cuFilePath.c_str());
    argv.push_back("-o");
    std::string outputname = m_GenericFileName + std::to_string(m_Counter)
                             +".cu.pch";
    argv.push_back(outputname.c_str());
    // If a previos file exist, include it.
    std::string previousFile;
    if(m_Counter){
      previousFile = m_GenericFileName + std::to_string(m_Counter-1) +".cu.pch";
      argv.push_back("-include-pch");
      argv.push_back(previousFile.c_str());
    }
    // FIXME: Should replaced by the arguments of the cling instance.
    std::string smString = "--cuda-gpu-arch=sm_" + m_SMLevel;
    argv.push_back(smString.c_str());
    argv.push_back("-pthread");
    argv.push_back("--cuda-device-only");

    // Argv list have to finish with a nullptr.
    argv.push_back(nullptr);

    std::string executionError;
    int res = llvm::sys::ExecuteAndWait(m_ClangPath.c_str(), argv.data(),
                                        nullptr, {}, 0, 0, &executionError);

    if(res){
      llvm::errs() << "error at launching clang instance to generate PCH file\n"
                   << executionError << "\n";
      return false;
    }

    return true;
  }

  bool cling::IncrementalCUDADeviceCompiler::generatePTX() {
    // clang++ -std=c++14 -S dummy.cu -o cling.ptx -include-pch cling[0-9].cu.pch
    // --cuda-gpu-arch=sm_${m_smLevel} -pthread --cuda-device-only
    llvm::SmallVector<const char*, 128> argv;

    // First argument have to be the program name.
    argv.push_back(m_ClangPath.c_str());

    // FIXME: Should replaced by the arguments of the cling instance.
    argv.push_back("-std=c++14");
    argv.push_back("-S");
    argv.push_back(m_DummyCUPath.c_str());
    argv.push_back("-o");
    argv.push_back(m_PTXFilePath.c_str());
    argv.push_back("-include-pch");
    std::string pchFile = m_GenericFileName + std::to_string(m_Counter) +".cu.pch";
    argv.push_back(pchFile.c_str());
    // FIXME: Should replaced by the arguments of the cling instance.
    std::string smString = "--cuda-gpu-arch=sm_" + m_SMLevel;
    argv.push_back(smString.c_str());
    argv.push_back("-pthread");
    argv.push_back("--cuda-device-only");

    // Argv list have to finish with a nullptr.
    argv.push_back(nullptr);

    std::string executionError;
    int res = llvm::sys::ExecuteAndWait(m_ClangPath.c_str(), argv.data(),
                                        nullptr, {}, 0, 0, &executionError);

    if(res){
      llvm::errs() << "error at launching clang instance to generate ptx code"
                   << "\n" << executionError << "\n";
      return false;
    }

    return true;
  }

  bool IncrementalCUDADeviceCompiler::generateFatbinaryInternal() {
    // fatbinary --cuda [-32 | -64] --create cling.fatbin
    // --image=profile=compute_${m_smLevel},file=cling.ptx
    llvm::SmallVector<const char*, 128> argv;

    // First argument have to be the program name.
    argv.push_back(m_FatbinaryPath.c_str());

    argv.push_back("--cuda");
    argv.push_back(m_FatbinArch.c_str());
    argv.push_back("--create");
    argv.push_back(m_FatbinFilePath.c_str());
    std::string ptxCode = "--image=profile=compute_"+ m_SMLevel
                          + ",file=" + m_PTXFilePath;
    argv.push_back(ptxCode.c_str());

    // Argv list have to finish with a nullptr.
    argv.push_back(nullptr);

    std::string executionError;
    int res = llvm::sys::ExecuteAndWait(m_FatbinaryPath.c_str(), argv.data(),
                                        nullptr, {}, 0, 0, &executionError);

    if(res){
      llvm::errs() << "error at launching fatbin" << "\n" << executionError << "\n";
      return false;
    }

    return true;
  }

  void IncrementalCUDADeviceCompiler::addIncludePath(llvm::StringRef pathStr,
                                                     bool leadingIncludeCommand){
    if(leadingIncludeCommand) {
      m_Headers.push_back(pathStr);
    } else {
      m_Headers.push_back("-I" + std::string(pathStr.data()));
    }
  }

  void IncrementalCUDADeviceCompiler::addIncludePaths(
      const llvm::SmallVectorImpl<std::string> & headers,
      bool leadingIncludeCommand){
    if(leadingIncludeCommand){
      m_Headers.append(headers.begin(), headers.end());
    } else {
      for(std::string header : headers){
        m_Headers.push_back("-I" + header);
      }
    }
  }

  void IncrementalCUDADeviceCompiler::dump(){
    llvm::outs() << "current counter: " << m_Counter << "\n" <<
                    "CUDA device compiler is valid: " << m_Init << "\n" <<
                    "file path: " << m_FilePath << "\n" <<
                    "fatbin file path: " << m_FatbinFilePath << "\n" <<
                    "dummy.cu file path: " << m_DummyCUPath << "\n" <<
                    "cling.ptx file path: " << m_PTXFilePath << "\n" <<
                    "generic file path: " << m_GenericFileName << "[0-9]*.cu{.pch}\n" <<
                    "clang++ path: " << m_ClangPath << "\n" <<
                    "nvidia fatbinary path: " << m_FatbinaryPath << "\n";
  }

} // end namespace cling
