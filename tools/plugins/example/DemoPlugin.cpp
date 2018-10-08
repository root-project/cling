//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vvasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Lex/Preprocessor.h"

struct PluginConsumer : public clang::ASTConsumer {
  virtual bool HandleTopLevelDecl(clang::DeclGroupRef DGR) {
    llvm::outs() << "PluginConsumer::HandleTopLevelDecl\n";
    return true; // Happiness
  }
};

template<typename ConsumerType>
class Action : public clang::PluginASTAction {
protected:
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance&,
                    llvm::StringRef /*InFile*/) override {
    llvm::outs() << "Action::CreateASTConsumer\n";
    return std::unique_ptr<clang::ASTConsumer>(new ConsumerType());
  }

  bool ParseArgs(const clang::CompilerInstance &,
                 const std::vector<std::string>&) override {
    llvm::outs() << "Action::ParseArgs\n";
    // return false; // Tells clang not to create the plugin.
    return true; // Happiness
  }

  PluginASTAction::ActionType getActionType() override {
    return AddBeforeMainAction;
  }
};

// Define a pragma handler for #pragma demoplugin
class DemoPluginPragmaHandler : public clang::PragmaHandler {
public:
  DemoPluginPragmaHandler() : clang::PragmaHandler("demoplugin") { }
  void HandlePragma(clang::Preprocessor &PP,
                    clang::PragmaIntroducerKind Introducer,
                    clang::Token &PragmaTok) {
    llvm::outs() << "DemoPluginPragmaHandler::HandlePragma\n";
    // Handle the pragma
  }
};

// Register the PluginASTAction in the registry.
static clang::FrontendPluginRegistry::Add<Action<PluginConsumer> >
X("DemoPlugin", "Used to test plugin mechanisms in cling.");

// Register the DemoPluginPragmaHandler in the registry.
static clang::PragmaHandlerRegistry::Add<DemoPluginPragmaHandler>
Y("demoplugin","Used to test plugin-attached pragrams in cling.");
