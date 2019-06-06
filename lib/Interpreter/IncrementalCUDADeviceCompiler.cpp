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

#include "clang/Basic/TargetOptions.h"
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
#include <bitset>
#include <string>
#include <system_error>

namespace cling {

  IncrementalCUDADeviceCompiler::IncrementalCUDADeviceCompiler(
      const std::string& filePath, const int optLevel,
      const cling::InvocationOptions& invocationOptions,
      const clang::CompilerInstance& CI)
      : m_FilePath(filePath),
        m_FatbinFilePath(CI.getCodeGenOpts().CudaGpuBinaryFileNames.empty()
                             ? ""
                             : CI.getCodeGenOpts().CudaGpuBinaryFileNames[0]) {
    if (m_FatbinFilePath.empty()) {
      llvm::errs() << "Error: CudaGpuBinaryFileNames can't be empty\n";
      return;
    }

    setCuArgs(CI.getLangOpts(), invocationOptions, optLevel,
              CI.getCodeGenOpts().getDebugInfo(),
              llvm::Triple(CI.getTargetOpts().Triple));

    // cling -std=c++xx -Ox -x cuda -S --cuda-gpu-arch=sm_xx --cuda-device-only
    // ${include headers} ${-I/paths} [-v] [-g] ${m_CuArgs->additionalPtxOpt}
    std::vector<std::string> argv = {
        "cling",
        m_CuArgs->cppStdVersion.c_str(),
        m_CuArgs->optLevel.c_str(),
        "-x",
        "cuda",
        "-S",
        std::string("--cuda-gpu-arch=sm_")
            .append(std::to_string(m_CuArgs->smVersion)),
        "--cuda-device-only"};

    addHeaderSearchPathFlags(argv, CI.getHeaderSearchOptsPtr());

    if (m_CuArgs->verbose)
      argv.push_back("-v");
    if (m_CuArgs->debug)
      argv.push_back("-g");
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
      const clang::codegenoptions::DebugInfoKind debugInfo,
      const llvm::Triple hostTriple) {
    std::string cppStdVersion;
    // Set the c++ standard. Just one condition is possible.
    if (langOpts.CPlusPlus11)
      cppStdVersion = "-std=c++11";
    if (langOpts.CPlusPlus14)
      cppStdVersion = "-std=c++14";
    if (langOpts.CPlusPlus1z)
      cppStdVersion = "-std=c++1z";
    if (langOpts.CPlusPlus2a)
      cppStdVersion = "-std=c++2a";

    if (cppStdVersion.empty())
      llvm::errs()
          << "IncrementalCUDADeviceCompiler: No valid c++ standard is set.\n";

    const std::string optLevel = "-O" + std::to_string(intprOptLevel);

    uint32_t smVersion = 20;
    if (!invocationOptions.CompilerOpts.CUDAGpuArch.empty()) {
      llvm::StringRef(invocationOptions.CompilerOpts.CUDAGpuArch)
          .drop_front(3 /* sm_ */)
          .getAsInteger(10, smVersion);
    }

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
      if (s.compare(0, 2, "-D") == 0)
        additionalPtxOpt.push_back(s);
    }

    enum FatBinFlags {
      AddressSize64 = 0x01,
      HasDebugInfo = 0x02,
      ProducerCuda = 0x04,
      HostLinux = 0x10,
      HostMac = 0x20,
      HostWindows = 0x40
    };

    uint32_t fatbinFlags = FatBinFlags::ProducerCuda;
    if (debug)
      fatbinFlags |= FatBinFlags::HasDebugInfo;

    if (hostTriple.isArch64Bit())
      fatbinFlags |= FatBinFlags::AddressSize64;

    if (hostTriple.isOSWindows())
      fatbinFlags |= FatBinFlags::HostWindows;
    else if (hostTriple.isOSDarwin())
      fatbinFlags |= FatBinFlags::HostMac;
    else
      fatbinFlags |= FatBinFlags::HostLinux;

    m_CuArgs.reset(new IncrementalCUDADeviceCompiler::CUDACompilerArgs(
        cppStdVersion, optLevel, hostTriple, smVersion, fatbinFlags,
        invocationOptions.Verbose(), debug, additionalPtxOpt));
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
    // delete compiled PTX code of last input
    m_PTX_code = "";

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

    llvm::TargetMachine* targetMachine = Target->createTargetMachine(
        module->getTargetTriple(),
        std::string("sm_").append(std::to_string(m_CuArgs->smVersion)), "", TO,
        RM);
    module->setDataLayout(targetMachine->createDataLayout());

    llvm::raw_svector_ostream dest(m_PTX_code);

    llvm::legacy::PassManager pass;
    // it's important to use the type assembler
    // object file is not supported and do not make sense
    auto FileType = llvm::TargetMachine::CGFT_AssemblyFile;

    if (targetMachine->addPassesToEmitFile(pass, dest, FileType)) {
      llvm::errs() << "TargetMachine can't emit assembler code";
      return 1;
    }

    return pass.run(*module);
  }

  bool IncrementalCUDADeviceCompiler::generateFatbinary() {
    // FIXME: At the moment the fatbin code must be writen to a file so that
    // CodeGen can use it. This should be replaced by a in-memory solution
    // (e.g. virtual file).
    std::error_code EC;
    llvm::raw_fd_ostream os(m_FatbinFilePath, EC, llvm::sys::fs::F_None);
    if (EC) {
      llvm::errs() << "ERROR: cannot generate file " << m_FatbinFilePath
                   << "\n";
      return false;
    }

    // implementation ist copied from clangJIT
    // (https://github.com/hfinkel/llvm-project-cxxjit/)

    // The outer header of the fat binary is documented in the CUDA
    // fatbinary.h header. As mentioned there, the overall size must be a
    // multiple of eight, and so we must make sure that the PTX is.
    while (m_PTX_code.size() % 7)
      m_PTX_code += ' ';
    m_PTX_code += '\0';

    // NVIDIA, unfortunatly, does not provide full documentation on their
    // fatbin format. There is some information on the outer header block in
    // the CUDA fatbinary.h header. Also, it is possible to figure out more
    // about the format by creating fatbins using the provided utilities
    // and then observing what cuobjdump reports about the resulting files.
    // There are some other online references which shed light on the format,
    // including https://reviews.llvm.org/D8397 and FatBinaryContext.{cpp,h}
    // from the GPU Ocelot project (https://github.com/gtcasl/gpuocelot).

    struct FatBinHeader {
      uint32_t Magic;      // 0x00
      uint16_t Version;    // 0x04
      uint16_t HeaderSize; // 0x06
      uint32_t DataSize;   // 0x08
      uint32_t unknown0c;  // 0x0c
    public:
      FatBinHeader(uint32_t DataSize)
          : Magic(0xba55ed50), Version(1), HeaderSize(sizeof(*this)),
            DataSize(DataSize), unknown0c(0) {}
    };

    struct FatBinFileHeader {
      uint16_t Kind;             // 0x00
      uint16_t unknown02;        // 0x02
      uint32_t HeaderSize;       // 0x04
      uint32_t DataSize;         // 0x08
      uint32_t unknown0c;        // 0x0c
      uint32_t CompressedSize;   // 0x10
      uint32_t SubHeaderSize;    // 0x14
      uint16_t VersionMinor;     // 0x18
      uint16_t VersionMajor;     // 0x1a
      uint32_t CudaArch;         // 0x1c
      uint32_t unknown20;        // 0x20
      uint32_t unknown24;        // 0x24
      uint32_t Flags;            // 0x28
      uint32_t unknown2c;        // 0x2c
      uint32_t unknown30;        // 0x30
      uint32_t unknown34;        // 0x34
      uint32_t UncompressedSize; // 0x38
      uint32_t unknown3c;        // 0x3c
      uint32_t unknown40;        // 0x40
      uint32_t unknown44;        // 0x44
      FatBinFileHeader(uint32_t DataSize, uint32_t CudaArch, uint32_t Flags)
          : Kind(1 /*PTX*/), unknown02(0x0101), HeaderSize(sizeof(*this)),
            DataSize(DataSize), unknown0c(0), CompressedSize(0),
            SubHeaderSize(HeaderSize - 8), VersionMinor(2), VersionMajor(4),
            CudaArch(CudaArch), unknown20(0), unknown24(0), Flags(Flags),
            unknown2c(0), unknown30(0), unknown34(0), UncompressedSize(0),
            unknown3c(0), unknown40(0), unknown44(0) {}
    };

    FatBinFileHeader fatBinFileHeader(m_PTX_code.size(), m_CuArgs->smVersion,
                                      m_CuArgs->fatbinFlags);
    FatBinHeader fatBinHeader(m_PTX_code.size() + fatBinFileHeader.HeaderSize);

    os.write((char*)&fatBinHeader, fatBinHeader.HeaderSize);
    os.write((char*)&fatBinFileHeader, fatBinFileHeader.HeaderSize);
    os << m_PTX_code;

    return true;
  }

  void IncrementalCUDADeviceCompiler::dump() {
    llvm::outs() << "CUDA device compiler is valid: " << m_Init << "\n"
                 << "file path: " << m_FilePath << "\n"
                 << "fatbin file path: " << m_FatbinFilePath << "\n"
                 << "m_CuArgs c++ standard: " << m_CuArgs->cppStdVersion << "\n"
                 << "m_CuArgs opt level: " << m_CuArgs->optLevel << "\n"
                 << "m_CuArgs host triple: " << m_CuArgs->hostTriple.str()
                 << "\n"
                 << "m_CuArgs Nvidia SM Version: " << m_CuArgs->smVersion
                 << "\n"
                 << "m_CuArgs Fatbin Flags (see "
                    "IncrementalCUDADeviceCompiler::setCuArgs()): "
                 << std::bitset<7>(m_CuArgs->fatbinFlags).to_string() << "\n"
                 << "m_CuArgs verbose: " << m_CuArgs->verbose << "\n"
                 << "m_CuArgs debug: " << m_CuArgs->debug << "\n";
    llvm::outs() << "m_CuArgs additional clang nvptx options: ";
    for (const std::string& s : m_CuArgs->additionalPtxOpt) {
      llvm::outs() << s << " ";
    }
    llvm::outs() << "\n";
  }

} // end namespace cling
