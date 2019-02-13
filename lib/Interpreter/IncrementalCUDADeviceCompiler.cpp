//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Simeon Ehrig <s.ehrig@hzdr.de>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "IncrementalCUDADeviceCompiler.h"

#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/InvocationOptions.h"
#include "cling/Interpreter/Transaction.h"

#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/HeaderSearchOptions.h"

#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>

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
        m_PTXFilePath(m_FilePath + "cling.ptx") {
    if (m_FatbinFilePath.empty()) {
      llvm::errs() << "Error: CudaGpuBinaryFileNames can't be empty\n";
      return;
    }

    if (!findToolchain(invocationOptions)) return;
    setCuArgs(CI.getLangOpts(), invocationOptions, optLevel,
              CI.getCodeGenOpts().getDebugInfo());

    // cling -std=c++xx -Ox -x cuda -S --cuda-gpu-arch=sm_xx --cuda-device-only
    // ${include headers} ${-I/paths} [-v] [-g] ${m_CuArgs->additionalPtxOpt}
    std::vector<std::string> argv = {"cling",
                                     m_CuArgs->cppStdVersion.c_str(),
                                     m_CuArgs->optLevel.c_str(),
                                     "-x",
                                     "cuda",
                                     "-S",
                                     m_CuArgs->ptxSmVersion.c_str(),
                                     "--cuda-device-only"};
    addHeaderSearchPathFlags(argv, CI.getHeaderSearchOptsPtr());
    if (m_CuArgs->verbose) argv.push_back("-v");
    if (m_CuArgs->debug) argv.push_back("-g");
    argv.insert(argv.end(), m_CuArgs->additionalPtxOpt.begin(),
                m_CuArgs->additionalPtxOpt.end());

    std::vector<const char*> argvChar;
    argvChar.resize(argv.size() + 1);

    std::transform(argv.begin(), argv.end(), argvChar.begin(),
                   [&](const std::string& s) { return s.c_str(); });

    // argv list have to finish with a nullptr.
    argvChar.push_back(nullptr);

    // create incremental compiler instance
    m_PTX_interp.reset(new Interpreter(argvChar.size(), argvChar.data()));

    if (!m_PTX_interp) {
      llvm::errs() << "Could not create PTX interpreter instance\n";
      return;
    }

    // initialize NVPTX backend
    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXTargetMC();
    LLVMInitializeNVPTXAsmPrinter();

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

    std::string smVersion = "sm_20";
    std::string ptxSmVersion = "--cuda-gpu-arch=sm_20";
    std::string fatbinSmVersion = "--image=profile=compute_20";
    if (!invocationOptions.CompilerOpts.CUDAGpuArch.empty()) {
      smVersion = invocationOptions.CompilerOpts.CUDAGpuArch;
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

    // search for defines (-Dmacros=value) in the args and add them to the PTX
    // compiler args
    for (const char* arg : invocationOptions.CompilerOpts.Remaining) {
      std::string s = arg;
      if (s.compare(0, 2, "-D") == 0) additionalPtxOpt.push_back(s);
    }

    m_CuArgs.reset(new IncrementalCUDADeviceCompiler::CUDACompilerArgs(
        cppStdVersion, optLevel, smVersion, ptxSmVersion, fatbinSmVersion,
        fatbinArch, invocationOptions.Verbose(), debug, additionalPtxOpt,
        invocationOptions.CompilerOpts.CUDAFatbinaryArgs));
  }

  bool IncrementalCUDADeviceCompiler::findToolchain(
      const cling::InvocationOptions& invocationOptions) {
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
      std::vector<std::string>& argv,
      const std::shared_ptr<clang::HeaderSearchOptions> headerSearchOptions) {
    for (clang::HeaderSearchOptions::Entry e :
         headerSearchOptions->UserEntries) {
      if (e.Group == clang::frontend::IncludeDirGroup::Quoted) {
        argv.push_back("-iquote");
        argv.push_back(e.Path);
      }

      if (e.Group == clang::frontend::IncludeDirGroup::Angled)
        argv.push_back("-I" + e.Path);
    }
  }

  bool IncrementalCUDADeviceCompiler::compileDeviceCode(
      const llvm::StringRef input) {
    if (!m_Init) {
      llvm::errs()
          << "Error: Initializiation of CUDA Device Code Compiler failed\n";
      return false;
    }

    Interpreter::CompilationResult CR = m_PTX_interp->process(input);

    if (CR == Interpreter::CompilationResult::kFailure) {
      llvm::errs() << "failed at compile ptx code\n";
      return false;
    }

    // for example blocks which are not closed
    if (CR == Interpreter::CompilationResult::kMoreInputExpected) return true;

    if (!generatePTX() || !generateFatbinary()) return false;

    return true;
  }

  bool cling::IncrementalCUDADeviceCompiler::generatePTX() {
    std::error_code EC;
    llvm::raw_fd_ostream os(m_PTXFilePath, EC, llvm::sys::fs::F_None);
    if (EC) {
      llvm::errs() << "ERROR: cannot generate file " << m_PTXFilePath << "\n";
      return false;
    }

    std::shared_ptr<llvm::Module> module =
        m_PTX_interp->getLastTransaction()->getModule();

    std::string error;
    auto Target =
        llvm::TargetRegistry::lookupTarget(module->getTargetTriple(), error);

    if (!Target) {
      llvm::errs() << error;
      return 1;
    }

    // is not important, because PTX does not use any object format
    llvm::Optional<llvm::Reloc::Model> RM =
        llvm::Optional<llvm::Reloc::Model>(llvm::Reloc::Model::PIC_);

    llvm::TargetOptions TO = llvm::TargetOptions();

    llvm::TargetMachine* targetMachine =
        Target->createTargetMachine(module->getTargetTriple(),
                                    m_CuArgs->smVersion, "", TO, RM);
    module->setDataLayout(targetMachine->createDataLayout());

    llvm::SmallString<1024> ptx_code;
    llvm::raw_svector_ostream dest(ptx_code);

    llvm::legacy::PassManager pass;
    // it's important to use the type assembler
    // object file is not supported and do not make sense
    auto FileType = llvm::TargetMachine::CGFT_AssemblyFile;

    if (targetMachine->addPassesToEmitFile(pass, os, FileType)) {
      llvm::errs() << "TargetMachine can't emit assembler code";
      return 1;
    }

    return pass.run(*module);
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
                      "): error compiling fatbinary file:\n"
                   << m_FatbinaryPath;
      for (const char* c : argvChar)
        llvm::errs() << " " << c;
      llvm::errs() << '\n' << executionError << "\n";
      return false;
    }

    return true;
  }

  void IncrementalCUDADeviceCompiler::dump() {
    llvm::outs() << "CUDA device compiler is valid: " << m_Init << "\n"
                 << "file path: " << m_FilePath << "\n"
                 << "fatbin file path: " << m_FatbinFilePath << "\n"
                 << "cling.ptx file path: " << m_PTXFilePath << "\n"
                 << "nvidia fatbinary path: " << m_FatbinaryPath << "\n"
                 << "m_CuArgs c++ standard: " << m_CuArgs->cppStdVersion << "\n"
                 << "m_CuArgs opt level: " << m_CuArgs->optLevel << "\n"
                 << "m_CuArgs SM level general: " << m_CuArgs->smVersion << "\n"
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

} // end namespace cling
