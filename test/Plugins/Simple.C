// RUN: cat %s | %cling -fplugin=%cling_obj_root/tools/plugins/example/libclingDemoPlugin%shlibext | FileCheck %s

// CHECK:Action::ParseArgs
// CHECK-NEXT:Action::CreateASTConsumer

int dummy = 15;

// CHECK-NEXT:PluginConsumer::HandleTopLevelDecl

#pragma demoplugin

// CHECK:DemoPluginPragmaHandler::HandlePragma

.q
