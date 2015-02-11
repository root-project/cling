//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vvasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include <memory>

namespace llvm {
  class Function;
  class LLVMContext;
  class Module;
  class PassManagerBuilder;

  namespace legacy {
    class FunctionPassManager;
    class PassManager;
  }
}

namespace clang{
  class CodeGenOptions;
  class LangOptions;
  class TargetOptions;
}

namespace cling {
  ///\brief Runs passes on IR. Remove once we can migrate from ModuleBuilder to
  /// what's in clang's CodeGen/BackendUtil.
  class BackendPasses {
    std::unique_ptr<llvm::legacy::PassManager> m_MPM;
    std::unique_ptr<llvm::PassManagerBuilder> m_PMBuilder;
    bool m_CodeGenOptsVerifyModule;

    void CreatePasses(const clang::CodeGenOptions &CGOpts,
                      const clang::TargetOptions &TOpts,
                      const clang::LangOptions &LOpts);

  public:
    BackendPasses(const clang::CodeGenOptions &CGOpts,
                  const clang::TargetOptions &TOpts,
                  const clang::LangOptions &LOpts);
    ~BackendPasses();

    void runOnModule(llvm::Module& M);
  };
}
