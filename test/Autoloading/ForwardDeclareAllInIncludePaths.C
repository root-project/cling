//RUN: cat %s | %cling -Xclang -verify "-DCLING=\" %cling \"" | FileCheck %s
//RUN: %rm /tmp/__cling_fwd_*
//XFAIL:*
#include "cling/Interpreter/Interpreter.h"

#include "llvm/ADT/SmallVector.h"

#include <string>
#include <regex>

#include "dirent.h" // For lightweight dir access.

typedef llvm::SmallVector<std::string, 16> Includes;
Includes includePaths;
gCling->GetIncludePaths(includePaths, /*system=*/true, /*withflags=*/false);
DIR *dir;
struct dirent *ent;
static std::regex dirsToIgnore("(.*)/backward");
// Limited regexp support, cannot do file by file.
std::string filesToIgnore = "bfd.h;bfdlink.h;";
// LLVM/Clang internals, cannot be directly included:
filesToIgnore += "varargs.h;wmmintrin.h;altivec.h;";
filesToIgnore += "shaintrin.h;xopintrin.h;lzcntintrin.h;rdseedintrin.h;";
filesToIgnore += "pmmintrin.h;tmmintrin.h;ia32intrin.h;bmiintrin.h;arm_neon.h;";
filesToIgnore += "prfchwintrin.h;fmaintrin.h;bmi2intrin.h;ammintrin.h;";
filesToIgnore += "module.modulemap;smmintrin.h;Intrin.h;avxintrin.h;tbmintrin.h;";
filesToIgnore += "f16cintrin.h;nmmintrin.h;__wmmintrin_aes.h;popcntintrin.h;";
filesToIgnore += "__wmmintrin_pclmul.h;rtmintrin.h;fma4intrin.h;avx2intrin.h;";
// Wrong setups, i.e not self-contained header files:
filesToIgnore += "dlg_colors.h;dialog.h;plugin-api.h;regexp.h;etip.h;dlg_keys.h;";
filesToIgnore += "cursesw.h;cursesm.h;cursesf.h;cursslk.h;cursesapp.h;term_entry.h;";
filesToIgnore += "cursesp.h;ft2build.h;shared_mutex;ciso646;cxxabi.h;";

// AUX:
filesToIgnore += "Makefile;CMakeLists.txt;";

.rawInput 1
bool has_suffix(const std::string &str, const std::string &suffix) {
    return str.size() >= suffix.size() &&
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}
.rawInput 0

std::string fwdDeclFile;
std::string nestedCling = CLING; nestedCling += " -Xclang -verify ";
std::string sourceCode;
for (int i = 0; i < 1 /*includePaths.size()*/; ++i) { // We know STL is first.
  if (std::regex_match(includePaths[i], dirsToIgnore))
    continue;
  if ((dir = opendir(includePaths[i].c_str())) != NULL) {
    /* print all the files and directories within directory */
    while ((ent = readdir(dir)) != NULL) {
      if (filesToIgnore.find(ent->d_name) != std::string::npos)
        continue;
      if (includePaths[i] == "." && !has_suffix(ent->d_name, ".h"))
        continue;
      if (ent->d_type == DT_REG && strcmp(ent->d_name, ".") && strcmp(ent->d_name, "..")) {
        fwdDeclFile = "/tmp/__cling_fwd_"; fwdDeclFile += ent->d_name;
        gCling->GenerateAutoloadingMap(ent->d_name, fwdDeclFile);
        // Run it in separate cling and assert it went all fine:
        sourceCode = " \"//expected-no-diagnostics\"";
        sourceCode += " '#include \"" + fwdDeclFile + "\"'";
        sourceCode += std::string(" '#include \"") + ent->d_name + "\"'";

        if (system((nestedCling + sourceCode).c_str()))
          printf("fail: %s\n", (nestedCling + sourceCode).c_str());
        //printf("%s\n", ent->d_name);
        else puts("pass\n");
      }
    }
    closedir(dir);
  }
 }
//CHECK-NOT: {{fail}}
//expected-no-diagnostics
.q
