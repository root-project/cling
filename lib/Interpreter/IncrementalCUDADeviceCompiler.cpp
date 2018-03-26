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
#include "clang/Frontend/CompilerInstance.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Path.h"
#include "llvm/ADT/Triple.h"

#include <string>

#define PCHMODE 0

namespace cling {

  IncrementalCUDADeviceCompiler::IncrementalCUDADeviceCompiler(
      std::string filePath,
      int optLevel,
      cling::InvocationOptions & invocationOptions,
      clang::CompilerInstance * CI)
     : m_Counter(0),
       m_FilePath(filePath){
    if(CI->getCodeGenOpts().CudaGpuBinaryFileNames.empty()){
      llvm::errs() << "Error: CudaGpuBinaryFileNames can't be empty\n";
      m_Init = false;
    } else {
      m_FatbinFilePath = CI->getCodeGenOpts().CudaGpuBinaryFileNames[0];
      m_Init = true;
    }

    m_Init = m_Init && generateHelperFiles();
    m_Init = m_Init && searchCompilingTools(invocationOptions);
    setCuArgs(CI->getLangOpts(), invocationOptions, optLevel,
              CI->getCodeGenOpts().getDebugInfo());

    m_HeaderSearchOptions = CI->getHeaderSearchOptsPtr();
  }

  void IncrementalCUDADeviceCompiler::setCuArgs(
      clang::LangOptions & langOpts,
      cling::InvocationOptions & invocationOptions,
      int & optLevel, clang::codegenoptions::DebugInfoKind debugInfo){
    // Set the c++ standard. Just one condition is possible.
    if(langOpts.CPlusPlus11)
      m_CuArgs.cppStdVersion = "-std=c++11";
    if(langOpts.CPlusPlus14)
      m_CuArgs.cppStdVersion = "-std=c++14";
    if(langOpts.CPlusPlus1z)
      m_CuArgs.cppStdVersion = "-std=c++1z";
    if(langOpts.CPlusPlus2a)
      m_CuArgs.cppStdVersion = "-std=c++2a";

    m_CuArgs.optLevel = "-O" + std::to_string(optLevel);

    if(!invocationOptions.CompilerOpts.CUDAGpuArch.empty()){
      m_CuArgs.ptxSmVersion = "--cuda-gpu-arch="
                              + invocationOptions.CompilerOpts.CUDAGpuArch;
      m_CuArgs.fatbinSmVersion = "--image=profile=compute_"
                              + invocationOptions.CompilerOpts.CUDAGpuArch.substr(3);
    }

    //The generating of the fatbin file is depend of the architecture of the host.
    llvm::Triple hostTarget(llvm::sys::getDefaultTargetTriple());
    m_CuArgs.fatbinArch = hostTarget.isArch64Bit() ? "-64" : "-32";

    m_CuArgs.verbose = invocationOptions.Verbose();

    // FIXME : Should not reduce the fine granulated debug options to a simple.
    // -g
    if(debugInfo == clang::codegenoptions::DebugLineTablesOnly ||
       debugInfo == clang::codegenoptions::LimitedDebugInfo ||
       debugInfo == clang::codegenoptions::FullDebugInfo)
      m_CuArgs.debug = true;

    // FIXME : Cling has problems to detect this arguments.
    /*
    if(langOpts.CUDADeviceFlushDenormalsToZero)
      m_CuArgs.additionalPtxOpt.push_back("-fcuda-flush-denormals-to-zero");
    if(langOpts.CUDADeviceApproxTranscendentals)
      m_CuArgs.additionalPtxOpt.push_back("-fcuda-approx-transcendentals");
    if(langOpts.CUDAAllowVariadicFunctions)
      m_CuArgs.additionalPtxOpt.push_back("-fcuda-allow-variadic-functions");
    */

    m_CuArgs.fatbinaryOpt = invocationOptions.CompilerOpts.CUDAFatbinaryArgs;
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

  bool IncrementalCUDADeviceCompiler::searchCompilingTools(
      cling::InvocationOptions & invocationOptions){
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

#if PCHMODE == 0
    llvm::sys::fs::remove(m_GenericFileName + std::to_string(m_Counter)
                             +".cu.pch");
#endif

    ++m_Counter;
    return true;
  }

  bool IncrementalCUDADeviceCompiler::generatePCH() {
    // clang++ -std=c++xx -Ox -S -Xclang -emit-pch ${clingHeaders} cling[0-9].cu
    // -D__CLING__ -o cling[0-9].cu.pch [-include-pch cling[0-9].cu.pch]
    // --cuda-gpu-arch=sm_[1-7][0-9] -pthread --cuda-device-only [-v] [-g]
    // ${m_CuArgs.additionalPtxOpt}
    llvm::SmallVector<const char*, 256> argv;

    // First argument have to be the program name.
    argv.push_back(m_ClangPath.c_str());

    argv.push_back(m_CuArgs.cppStdVersion.c_str());
    argv.push_back(m_CuArgs.optLevel.c_str());
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
#if PCHMODE == 1
    std::string previousFile;
    if(m_Counter){
      previousFile = m_GenericFileName + std::to_string(m_Counter-1) +".cu.pch";
      argv.push_back("-include-pch");
      argv.push_back(previousFile.c_str());
    }
#else
    std::vector<std::string> previousFiles;
    if(m_Counter){
      for(unsigned int i = 0; i <= m_Counter-1; ++i){
        previousFiles.push_back(m_GenericFileName + std::to_string(i) +".cu");
        argv.push_back("-include");
        argv.push_back(previousFiles[i].c_str());
      }
    }
#endif
    argv.push_back(m_CuArgs.ptxSmVersion.c_str());
    argv.push_back("-pthread");
    argv.push_back("--cuda-device-only");
    if(m_CuArgs.verbose)
      argv.push_back("-v");
    if(m_CuArgs.debug)
      argv.push_back("-g");
    for(std::string & s : m_CuArgs.additionalPtxOpt){
      argv.push_back(s.c_str());
    }

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
    // clang++ -std=c++xx -Ox -S dummy.cu -o cling.ptx -include-pch
    // cling[0-9].cu.pch --cuda-gpu-arch=sm_xx -pthread --cuda-device-only [-v]
    // [-g] ${m_CuArgs.additionalPtxOpt}
    llvm::SmallVector<const char*, 128> argv;

    // First argument have to be the program name.
    argv.push_back(m_ClangPath.c_str());

    argv.push_back(m_CuArgs.cppStdVersion.c_str());
    argv.push_back(m_CuArgs.optLevel.c_str());
    argv.push_back("-S");
    argv.push_back(m_DummyCUPath.c_str());
    argv.push_back("-o");
    argv.push_back(m_PTXFilePath.c_str());
    argv.push_back("-include-pch");
    std::string pchFile = m_GenericFileName + std::to_string(m_Counter) +".cu.pch";
    argv.push_back(pchFile.c_str());
    argv.push_back(m_CuArgs.ptxSmVersion.c_str());
    argv.push_back("-pthread");
    argv.push_back("--cuda-device-only");
    if(m_CuArgs.verbose)
      argv.push_back("-v");
    if(m_CuArgs.debug)
      argv.push_back("-g");
    for(std::string & s : m_CuArgs.additionalPtxOpt){
      argv.push_back(s.c_str());
    }

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
    // --image=profile=compute_xx,file=cling.ptx [-g] ${m_CuArgs.fatbinaryOpt}
    llvm::SmallVector<const char*, 128> argv;

    // First argument have to be the program name.
    argv.push_back(m_FatbinaryPath.c_str());

    argv.push_back("--cuda");
    argv.push_back(m_CuArgs.fatbinArch.c_str());
    argv.push_back("--create");
    argv.push_back(m_FatbinFilePath.c_str());
    std::string ptxCode = m_CuArgs.fatbinSmVersion
                          + ",file=" + m_PTXFilePath;
    argv.push_back(ptxCode.c_str());
    if(m_CuArgs.debug)
      argv.push_back("-g");
    for(std::string & s : m_CuArgs.fatbinaryOpt){
      argv.push_back(s.c_str());
    }

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

  void IncrementalCUDADeviceCompiler::dump(){
    llvm::outs() << "current counter: " << m_Counter << "\n" <<
                    "CUDA device compiler is valid: " << m_Init << "\n" <<
                    "file path: " << m_FilePath << "\n" <<
                    "fatbin file path: " << m_FatbinFilePath << "\n" <<
                    "dummy.cu file path: " << m_DummyCUPath << "\n" <<
                    "cling.ptx file path: " << m_PTXFilePath << "\n" <<
                    "generic file path: " << m_GenericFileName
                    << "[0-9]*.cu{.pch}\n" <<
                    "clang++ path: " << m_ClangPath << "\n" <<
                    "nvidia fatbinary path: " << m_FatbinaryPath << "\n" <<
                    "m_CuArgs c++ standard: " << m_CuArgs.cppStdVersion << "\n" <<
                    "m_CuArgs opt level: " << m_CuArgs.optLevel << "\n" <<
                    "m_CuArgs SM level for clang nvptx: "
                    << m_CuArgs.ptxSmVersion << "\n" <<
                    "m_CuArgs SM level for fatbinary: "
                    << m_CuArgs.fatbinSmVersion << "\n" <<
                    "m_CuArgs fatbinary architectur: "
                    << m_CuArgs.fatbinArch << "\n" <<
                    "m_CuArgs verbose: " << m_CuArgs.verbose << "\n" <<
                    "m_CuArgs debug: " << m_CuArgs.debug << "\n";
     llvm::outs() << "m_CuArgs additional clang nvptx options: ";
     for(std::string & s : m_CuArgs.additionalPtxOpt){
       llvm::outs() << s << " ";
     }
     llvm::outs() << "\n";
     llvm::outs() << "m_CuArgs additional fatbinary options: ";
     for(std::string & s : m_CuArgs.fatbinaryOpt){
       llvm::outs() << s << " ";
     }
     llvm::outs() << "\n";
  }

} // end namespace cling
