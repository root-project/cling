//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Simeon Ehrig <s.ehrig@hzdr.de>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "IncrementalCUDADeviceCompiler.h"

#include "cling/Interpreter/InvocationOptions.h"
#include "cling/Interpreter/Transaction.h"
#include "cling/Utils/Paths.h"

#include "clang/AST/Decl.h"
#include "clang/AST/DeclGroup.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/HeaderSearchOptions.h"

#include "llvm/ADT/Triple.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <string>
#include <system_error>

// The clang nvptx jit has an growing AST-Tree. At runtime, continuously new
// statements will append to the AST. To improve the compiletime, the existing
// AST will save as PCH-file. The new statements will append via source code
// files. A bug in clang avoids, that more than 4 statements can append to the
// PCH. If the flag is true, it improves the compiletime but it crash after the
// fifth iteration. https://bugs.llvm.org/show_bug.cgi?id=37167
#define PCHMODE 0

namespace cling {

  IncrementalCUDADeviceCompiler::IncrementalCUDADeviceCompiler(
      const std::string& filePath, const int optLevel,
      const cling::InvocationOptions& invocationOptions,
      const clang::CompilerInstance& CI)
      : m_FilePath(filePath),
        m_FatbinFilePath(CI.getCodeGenOpts().CudaGpuBinaryFileNames.empty()
                             ? ""
                             : CI.getCodeGenOpts().CudaGpuBinaryFileNames[0]),
        m_DummyCUPath(m_FilePath + "dummy.cu"),
        m_PTXFilePath(m_FilePath + "cling.ptx"),
        m_GenericFileName(m_FilePath + "cling") {
    if (m_FatbinFilePath.empty()) {
      llvm::errs() << "Error: CudaGpuBinaryFileNames can't be empty\n";
      return;
    }

    if (!generateHelperFiles()) return;
    if (!findToolchain(invocationOptions)) return;
    setCuArgs(CI.getLangOpts(), invocationOptions, optLevel,
              CI.getCodeGenOpts().getDebugInfo());

    m_HeaderSearchOptions = CI.getHeaderSearchOptsPtr();

    // This code will write to the first .cu-file. It is necessary that some
    // cling generated code can be handled.
    const std::string initialCUDADeviceCode =
        "extern void setValueNoAlloc(void* vpI, void* vpV, void* vpQT, char "
        "vpOn, float value);\n"
        "extern void setValueNoAlloc(void* vpI, void* vpV, void* vpQT, char "
        "vpOn, double value);\n"
        "extern void setValueNoAlloc(void* vpI, void* vpV, void* vpQT, char "
        "vpOn, long double value);\n"
        "extern void setValueNoAlloc(void* vpI, void* vpV, void* vpQT, char "
        "vpOn, unsigned long long value);\n"
        "extern void setValueNoAlloc(void* vpI, void* vpV, void* vpQT, char "
        "vpOn, const void* value);\n";

    std::error_code EC;
    llvm::raw_fd_ostream cuFile(m_FilePath + "cling0.cu", EC,
                                llvm::sys::fs::F_Text);
    if (EC) {
      llvm::errs() << "Could not open " << m_FilePath << "cling0.cu"
                   << EC.message() << "\n";
      return;
    }

    cuFile << initialCUDADeviceCode;
    cuFile.close();

    m_Init = true;
  }

  void IncrementalCUDADeviceCompiler::setCuArgs(
      const clang::LangOptions& langOpts,
      const cling::InvocationOptions& invocationOptions,
      const int intprOptLevel,
      const clang::codegenoptions::DebugInfoKind debugInfo) {

    std::string cppStdVersion;
    // Set the c++ standard. Just one condition is possible.
    if (langOpts.CPlusPlus11) cppStdVersion = "-std=c++11";
    if (langOpts.CPlusPlus14) cppStdVersion = "-std=c++14";
    if (langOpts.CPlusPlus1z) cppStdVersion = "-std=c++1z";
    if (langOpts.CPlusPlus2a) cppStdVersion = "-std=c++2a";

    if (cppStdVersion.empty())
      llvm::errs()
          << "IncrementalCUDADeviceCompiler: No valid c++ standard is set.\n";

    const std::string optLevel = "-O" + std::to_string(intprOptLevel);

    std::string ptxSmVersion = "--cuda-gpu-arch=sm_20";
    std::string fatbinSmVersion = "--image=profile=compute_20";
    if (!invocationOptions.CompilerOpts.CUDAGpuArch.empty()) {
      ptxSmVersion =
          "--cuda-gpu-arch=" + invocationOptions.CompilerOpts.CUDAGpuArch;
      fatbinSmVersion = "--image=profile=compute_" +
                        invocationOptions.CompilerOpts.CUDAGpuArch.substr(3);
    }

    // The generating of the fatbin file is depend of the architecture of the
    // host.
    llvm::Triple hostTarget(llvm::sys::getDefaultTargetTriple());
    const std::string fatbinArch = hostTarget.isArch64Bit() ? "-64" : "-32";

    // FIXME : Should not reduce the fine granulated debug options to a simple.
    // -g
    bool debug = false;
    if (debugInfo == clang::codegenoptions::DebugLineTablesOnly ||
        debugInfo == clang::codegenoptions::LimitedDebugInfo ||
        debugInfo == clang::codegenoptions::FullDebugInfo)
      debug = true;

    // FIXME : Cling has problems to detect these arguments.
    /*
    if(langOpts.CUDADeviceFlushDenormalsToZero)
      m_CuArgs.additionalPtxOpt.push_back("-fcuda-flush-denormals-to-zero");
    if(langOpts.CUDADeviceApproxTranscendentals)
      m_CuArgs.additionalPtxOpt.push_back("-fcuda-approx-transcendentals");
    if(langOpts.CUDAAllowVariadicFunctions)
      m_CuArgs.additionalPtxOpt.push_back("-fcuda-allow-variadic-functions");
    */
    std::vector<std::string> additionalPtxOpt;

    m_CuArgs.reset(new IncrementalCUDADeviceCompiler::CUDACompilerArgs(
        cppStdVersion, optLevel, ptxSmVersion, fatbinSmVersion, fatbinArch,
        invocationOptions.Verbose(), debug, additionalPtxOpt,
        invocationOptions.CompilerOpts.CUDAFatbinaryArgs));
  }

  bool IncrementalCUDADeviceCompiler::generateHelperFiles() {
    // Generate an empty dummy.cu file.
    std::error_code EC;
    llvm::raw_fd_ostream dummyCU(m_DummyCUPath, EC, llvm::sys::fs::F_Text);
    if (EC) {
      llvm::errs() << "Could not open " << m_DummyCUPath << ": " << EC.message()
                   << "\n";
      return false;
    }
    dummyCU.close();

    return true;
  }

  bool IncrementalCUDADeviceCompiler::findToolchain(
      const cling::InvocationOptions& invocationOptions) {
    // Search after clang in the folder of cling.
    llvm::SmallString<128> cwd;
    // get folder of the cling executable to find the clang which is contained
    // in cling
    // nullptr is ok, if we are the main and not a shared library
    cwd.append(llvm::sys::path::parent_path(llvm::sys::fs::getMainExecutable(
        invocationOptions.CompilerOpts.Remaining[0], (void*)&cwd)));
    cwd.append(llvm::sys::path::get_separator());
    cwd.append("clang++");
    m_ClangPath = cwd.c_str();
    // Check, if clang is existing and executable.
    if (!llvm::sys::fs::can_execute(m_ClangPath)) {
      llvm::errs() << "Error: " << m_ClangPath
                   << " not existing or executable!\n";
      return false;
    }

    // Use the custom CUDA toolkit path, if it set via cling argument.
    if (!invocationOptions.CompilerOpts.CUDAPath.empty()) {
      m_FatbinaryPath =
          invocationOptions.CompilerOpts.CUDAPath + "/bin/fatbinary";
      if (!llvm::sys::fs::can_execute(m_FatbinaryPath)) {
        llvm::errs() << "Error: " << m_FatbinaryPath
                     << " not existing or executable!\n";
        return false;
      }
    } else {
      // Search after fatbinary on the system.
      if (llvm::ErrorOr<std::string> fatbinary =
              llvm::sys::findProgramByName("fatbinary")) {
        llvm::SmallString<256> fatbinaryAbsolutePath;
        llvm::sys::fs::real_path(*fatbinary, fatbinaryAbsolutePath);
        m_FatbinaryPath = fatbinaryAbsolutePath.c_str();
      } else {
        llvm::errs() << "Error: nvidia tool fatbinary not found!\n"
                     << "Please add the cuda /bin path to PATH or set the "
                        "toolkit path via --cuda-path argument.\n";
        return false;
      }
    }

    return true;
  }

  void IncrementalCUDADeviceCompiler::addHeaderSearchPathFlags(
      llvm::SmallVectorImpl<std::string>& argv) {
    for (clang::HeaderSearchOptions::Entry e :
         m_HeaderSearchOptions->UserEntries) {
      if (e.Group == clang::frontend::IncludeDirGroup::Quoted) {
        argv.push_back("-iquote");
        argv.push_back(e.Path);
      }

      if (e.Group == clang::frontend::IncludeDirGroup::Angled)
        argv.push_back("-I" + e.Path);
    }
  }

  bool IncrementalCUDADeviceCompiler::compileDeviceCode(
      const llvm::StringRef input, const cling::Transaction* const T) {
    if (!m_Init) {
      llvm::errs()
          << "Error: Initializiation of CUDA Device Code Compiler failed\n";
      return false;
    }

    const unsigned int counter = getCounterCopy();

    // Write the (CUDA) C++ source code to a file.
    std::error_code EC;
    llvm::raw_fd_ostream cuFile(m_GenericFileName + std::to_string(counter) +
                                    ".cu",
                                EC, llvm::sys::fs::F_Text);
    if (EC) {
      llvm::errs() << "Could not open "
                   << m_GenericFileName + std::to_string(counter)
                   << ".cu: " << EC.message() << "\n";
      return false;
    }
    // This variable prevent, that the input and the code from the transaction
    // will be written to the .cu-file.
    bool foundUnwrappedDecl = false;

    assert(T != nullptr && "transaction can't be missing");

    // Search after statements, which are unwrapped. The conditions are, that
    // the source code comes from the prompt (getWrapperFD()) and has the type
    // kCCIHandleTopLevelDecl.
    if (T->getWrapperFD()) {
      // Template specialization declaration will be save two times at a
      // transaction. Once with the type
      // kCCIHandleCXXImplicitFunctionInstantiation and once with the type
      // kCCIHandleTopLevelDecl. To avoid sending a template specialization to
      // the clang nvptx and causing a
      // explicit-specialization-after-instantiation-error it have to check,
      // which kCCIHandleTopLevelDecl declaration is also a
      // kCCIHandleCXXImplicitFunctionInstantiation declaration.
      std::vector<clang::Decl*> implFunc;
      for (auto iDCI = T->decls_begin(), eDCI = T->decls_end(); iDCI != eDCI;
           ++iDCI)
        if (iDCI->m_Call == Transaction::ConsumerCallInfo::
                                kCCIHandleCXXImplicitFunctionInstantiation)
          for (clang::Decl* decl : iDCI->m_DGR)
            implFunc.push_back(decl);

      for (auto iDCI = T->decls_begin(), eDCI = T->decls_end(); iDCI != eDCI;
           ++iDCI) {
        if (iDCI->m_Call ==
            Transaction::ConsumerCallInfo::kCCIHandleTopLevelDecl) {
          for (clang::Decl* decl : iDCI->m_DGR) {
            if (std::find(implFunc.begin(), implFunc.end(), decl) ==
                implFunc.end()) {
              foundUnwrappedDecl = true;
              decl->print(cuFile);
              // The c++ code has no whitespace and semicolon at the end.
              cuFile << ";\n";
            }
          }
        }
      }
    }

    if (!foundUnwrappedDecl) {
      cuFile << input;
    }

    cuFile.close();

    if (!generatePCH() || !generatePTX() || !generateFatbinary()) {
      saveFaultyCUfile();
      return false;
    }

#if PCHMODE == 0
    llvm::sys::fs::remove(m_GenericFileName + std::to_string(counter) +
                          ".cu.pch");
#endif

    ++m_Counter;
    return true;
  }

  bool IncrementalCUDADeviceCompiler::generatePCH() {
    const unsigned int counter = getCounterCopy();

    // clang++ -std=c++xx -Ox -S -Xclang -emit-pch ${clingHeaders} cling[0-9].cu
    // -D__CLING__ -o cling[0-9].cu.pch [-include-pch cling[0-9].cu.pch]
    // --cuda-gpu-arch=sm_[1-7][0-9] -pthread --cuda-device-only [-v] [-g]
    // ${m_CuArgs->additionalPtxOpt}
    llvm::SmallVector<std::string, 256> argv;

    // First argument have to be the program name.
    argv.push_back(m_ClangPath);

    argv.push_back(m_CuArgs->cppStdVersion);
    argv.push_back(m_CuArgs->optLevel);
    argv.push_back("-S");
    argv.push_back("-Xclang");
    argv.push_back("-emit-pch");
    addHeaderSearchPathFlags(argv);
    // Is necessary for the cling runtime header.
    argv.push_back("-D__CLING__");
    argv.push_back(m_GenericFileName + std::to_string(counter) + ".cu");
    argv.push_back("-o");
    argv.push_back(m_GenericFileName + std::to_string(counter) + ".cu.pch");
    // If a previos file exist, include it.
#if PCHMODE == 1
    if (counter) {
      argv.push_back("-include-pch");
      argv.push_back(m_GenericFileName + std::to_string(counter - 1) +
                     ".cu.pch");
    }
#else
    if (counter) {
      for (unsigned int i = 0; i <= counter - 1; ++i) {
        argv.push_back("-include");
        argv.push_back(m_GenericFileName + std::to_string(i) + ".cu");
      }
    }
#endif
    argv.push_back(m_CuArgs->ptxSmVersion);
    argv.push_back("-pthread");
    argv.push_back("--cuda-device-only");
    if (m_CuArgs->verbose) argv.push_back("-v");
    if (m_CuArgs->debug) argv.push_back("-g");
    for (const std::string& s : m_CuArgs->additionalPtxOpt) {
      argv.push_back(s.c_str());
    }

    argv.push_back("-Wno-unused-value");

    std::vector<const char*> argvChar;
    argvChar.resize(argv.size() + 1);

    std::transform(argv.begin(), argv.end(), argvChar.begin(),
                   [&](const std::string& s) { return s.c_str(); });

    // Argv list have to finish with a nullptr.
    argvChar.push_back(nullptr);

    std::string executionError;
    int res = llvm::sys::ExecuteAndWait(m_ClangPath.c_str(), argvChar.data(),
                                        nullptr, {}, 0, 0, &executionError);

    if (res) {
      llvm::errs() << "cling::IncrementalCUDADeviceCompiler::generatePCH(): "
                      "error compiling PCH file:\n"
                   << m_ClangPath;
      for (const char* c : argvChar)
        llvm::errs() << " " << c;
      llvm::errs() << '\n' << executionError << "\n";
      return false;
    }

    return true;
  }

  bool cling::IncrementalCUDADeviceCompiler::generatePTX() {
    const unsigned int counter = getCounterCopy();

    // clang++ -std=c++xx -Ox -S dummy.cu -o cling.ptx -include-pch
    // cling[0-9].cu.pch --cuda-gpu-arch=sm_xx -pthread --cuda-device-only [-v]
    // [-g] ${m_CuArgs->additionalPtxOpt}
    llvm::SmallVector<std::string, 128> argv;

    // First argument have to be the program name.
    argv.push_back(m_ClangPath);

    argv.push_back(m_CuArgs->cppStdVersion);
    argv.push_back(m_CuArgs->optLevel);
    argv.push_back("-S");
    argv.push_back(m_DummyCUPath);
    argv.push_back("-o");
    argv.push_back(m_PTXFilePath);
    argv.push_back("-include-pch");
    argv.push_back(m_GenericFileName + std::to_string(counter) + ".cu.pch");
    argv.push_back(m_CuArgs->ptxSmVersion);
    argv.push_back("-pthread");
    argv.push_back("--cuda-device-only");
    if (m_CuArgs->verbose) argv.push_back("-v");
    if (m_CuArgs->debug) argv.push_back("-g");
    for (const std::string& s : m_CuArgs->additionalPtxOpt) {
      argv.push_back(s.c_str());
    }

    std::vector<const char*> argvChar;
    argvChar.resize(argv.size() + 1);

    std::transform(argv.begin(), argv.end(), argvChar.begin(),
                   [&](const std::string& s) { return s.c_str(); });

    // Argv list have to finish with a nullptr.
    argvChar.push_back(nullptr);

    std::string executionError;
    int res = llvm::sys::ExecuteAndWait(m_ClangPath.c_str(), argvChar.data(),
                                        nullptr, {}, 0, 0, &executionError);

    if (res) {
      llvm::errs() << "cling::IncrementalCUDADeviceCompiler::generatePTX(): "
                      "error compiling PCH file:\n"
                   << m_ClangPath;
      for (const char* c : argvChar)
        llvm::errs() << " " << c;
      llvm::errs() << '\n' << executionError << "\n";
      return false;
    }

    return true;
  }

  bool IncrementalCUDADeviceCompiler::generateFatbinary() {
    // fatbinary --cuda [-32 | -64] --create cling.fatbin
    // --image=profile=compute_xx,file=cling.ptx [-g] ${m_CuArgs->fatbinaryOpt}
    llvm::SmallVector<std::string, 128> argv;

    // First argument have to be the program name.
    argv.push_back(m_FatbinaryPath);

    argv.push_back("--cuda");
    argv.push_back(m_CuArgs->fatbinArch);
    argv.push_back("--create");
    argv.push_back(m_FatbinFilePath);
    argv.push_back(m_CuArgs->fatbinSmVersion + ",file=" + m_PTXFilePath);
    if (m_CuArgs->debug) argv.push_back("-g");
    for (const std::string& s : m_CuArgs->fatbinaryOpt) {
      argv.push_back(s.c_str());
    }

    std::vector<const char*> argvChar;
    argvChar.resize(argv.size() + 1);

    std::transform(argv.begin(), argv.end(), argvChar.begin(),
                   [&](const std::string& s) { return s.c_str(); });

    // Argv list have to finish with a nullptr.
    argvChar.push_back(nullptr);

    std::string executionError;
    int res =
        llvm::sys::ExecuteAndWait(m_FatbinaryPath.c_str(), argvChar.data(),
                                  nullptr, {}, 0, 0, &executionError);

    if (res) {
      llvm::errs() << "cling::IncrementalCUDADeviceCompiler::generateFatbinary("
                      "): error compiling PCH file:\n"
                   << m_ClangPath;
      for (const char* c : argvChar)
        llvm::errs() << " " << c;
      llvm::errs() << '\n' << executionError << "\n";
      return false;
    }

    return true;
  }

  void IncrementalCUDADeviceCompiler::dump() {
    llvm::outs() << "current counter: " << getCounterCopy() << "\n"
                 << "CUDA device compiler is valid: " << m_Init << "\n"
                 << "file path: " << m_FilePath << "\n"
                 << "fatbin file path: " << m_FatbinFilePath << "\n"
                 << "dummy.cu file path: " << m_DummyCUPath << "\n"
                 << "cling.ptx file path: " << m_PTXFilePath << "\n"
                 << "generic file path: " << m_GenericFileName
                 << "[0-9]*.cu{.pch}\n"
                 << "clang++ path: " << m_ClangPath << "\n"
                 << "nvidia fatbinary path: " << m_FatbinaryPath << "\n"
                 << "m_CuArgs c++ standard: " << m_CuArgs->cppStdVersion << "\n"
                 << "m_CuArgs opt level: " << m_CuArgs->optLevel << "\n"
                 << "m_CuArgs SM level for clang nvptx: "
                 << m_CuArgs->ptxSmVersion << "\n"
                 << "m_CuArgs SM level for fatbinary: "
                 << m_CuArgs->fatbinSmVersion << "\n"
                 << "m_CuArgs fatbinary architectur: " << m_CuArgs->fatbinArch
                 << "\n"
                 << "m_CuArgs verbose: " << m_CuArgs->verbose << "\n"
                 << "m_CuArgs debug: " << m_CuArgs->debug << "\n";
    llvm::outs() << "m_CuArgs additional clang nvptx options: ";
    for (const std::string& s : m_CuArgs->additionalPtxOpt) {
      llvm::outs() << s << " ";
    }
    llvm::outs() << "\n";
    llvm::outs() << "m_CuArgs additional fatbinary options: ";
    for (const std::string& s : m_CuArgs->fatbinaryOpt) {
      llvm::outs() << s << " ";
    }
    llvm::outs() << "\n";
  }

  std::error_code IncrementalCUDADeviceCompiler::saveFaultyCUfile() {
    const unsigned int counter = getCounterCopy();
    unsigned int faultFileCounter = 0;

    // Construct the file path of the current .cu file without extension.
    std::string originalCU = m_GenericFileName + std::to_string(counter);

    // counter (= m_Counter) will just increased, if the compiling get right. So
    // we need a second counter, if two or more following files fails.
    std::string faultyCU;
    do {
      faultFileCounter += 1;
      faultyCU =
          originalCU + "_fault" + std::to_string(faultFileCounter) + ".cu";
    } while (llvm::sys::fs::exists(faultyCU));

    // orginial: cling[counter].cu
    // faulty file: cling[counter]_fault[faultFileCounter].cu
    return llvm::sys::fs::rename(originalCU + ".cu", faultyCU);
  }

} // end namespace cling
