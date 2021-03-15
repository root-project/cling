// RUN: cd "%cling_obj_root/../clang/test" && python "%src_include_dir/../../../utils/lit/lit.py" -v -sv --param clang_site_config="%cling_obj_root/../clang/test/lit.site.cfg" "%/p/../../../clang/test/Sema/address-constant.c"

// REQUIRES: vanilla-cling

// Current SemaCXX failures:
// Expected Passes    : 392
// Expected Failures  : 1
// Unexpected Failures: 184

// Current Sema failures:
// Expected Passes    : 420
// Unsupported Tests  : 2
// Unexpected Failures: 74
