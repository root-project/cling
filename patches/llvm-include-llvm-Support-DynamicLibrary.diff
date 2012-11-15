Index: include/llvm/Support/DynamicLibrary.h
===================================================================
--- include/llvm/Support/DynamicLibrary.h	(revision 47319)
+++ include/llvm/Support/DynamicLibrary.h	(working copy)
@@ -96,6 +96,9 @@ namespace sys {
     /// libraries.
     /// @brief Add searchable symbol/value pair.
     static void AddSymbol(StringRef symbolName, void *symbolValue);
+
+    bool operator<(const DynamicLibrary& Other) const { return Data < Other.Data; }
+    bool operator==(const DynamicLibrary& Other) const { return Data == Other.Data; }
   };
 
 } // End sys namespace
