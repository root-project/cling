#ifndef PREFIX
#error "Define PREFIX prior to including this file!"
#endif

/////////
// Prefixes

#define COMMA ,
PREFIX(prefix_0, {0})
PREFIX(prefix_1, {"-" COMMA 0})
PREFIX(prefix_3, {"-" COMMA "--" COMMA 0})
PREFIX(prefix_2, {"--" COMMA 0})
#undef COMMA

#ifndef OPTION
#error "Define OPTION prior to including this file!"
#endif

OPTION(prefix_0, "<input>", INPUT, Input, INVALID, INVALID, 0, 0, 0, 0, 0)
OPTION(prefix_0, "<unknown>", UNKNOWN, Unknown, INVALID, INVALID, 0, 0, 0, 0, 0)
OPTION(prefix_2, "errorout", _errorout, Flag, INVALID, INVALID, 0, 0, 0,
       "Do not recover from input errors", 0)
OPTION(prefix_3, "help", help, Flag, INVALID, INVALID, 0, 0, 0,
       "Print this help text", 0)
OPTION(prefix_1, "L", L, JoinedOrSeparate, INVALID, INVALID, 0, 0, 0,
       "Add directory to library search path", "<directory>")
// Re-implement to forward to our help
OPTION(prefix_1, "l", l, JoinedOrSeparate, INVALID, INVALID, 0, 0, 0,
       "Load a library before prompt", "<library>")
OPTION(prefix_2, "metastr=", _metastr_EQ, Joined, INVALID, INVALID, 0, 0, 0,
       "Set the meta command tag, default '.'", 0)
OPTION(prefix_2, "metastr", _metastr, Separate, INVALID, INVALID, 0, 0, 0,
       "Set the meta command tag, default '.'", 0)
OPTION(prefix_2, "nologo", _nologo, Flag, INVALID, INVALID, 0, 0, 0,
       "Do not show startup-banner", 0)
OPTION(prefix_3, "version", version, Flag, INVALID, INVALID, 0, 0, 0,
       "Print the compiler version", 0)
OPTION(prefix_1, "v", v, Flag, INVALID, INVALID, 0, 0, 0,
       "Enable verbose output", 0)
