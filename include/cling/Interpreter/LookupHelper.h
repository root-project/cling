//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vvasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_LOOKUP_HELPER_H
#define CLING_LOOKUP_HELPER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/SmallVector.h"

#include <array>
#include <map>
#include <memory>

namespace clang {
  class ClassTemplateDecl;
  class Decl;
  class Expr;
  class FunctionDecl;
  class FunctionTemplateDecl;
  class Parser;
  class QualType;
  class Type;
  class ValueDecl;
}

namespace llvm {
  template<typename T, unsigned N> class SmallVector;
}

namespace cling {
  class Interpreter;
  class Transaction;

  ///\brief Reflection information query interface. The class performs lookups
  /// in the currently loaded information in the AST, using the same Parser,
  /// Sema and Preprocessor objects.
  ///
  class LookupHelper {
  public:
    enum StringType {
      kStdString,
      kWCharString,
      kUTF16Str,
      kUTF32Str,
      kNumCachedStrings,
      kNotAString = kNumCachedStrings,
    };
    enum DiagSetting {
      NoDiagnostics,
      WithDiagnostics
    };
  private:
    std::unique_ptr<clang::Parser> m_Parser;
    Interpreter* m_Interpreter; // we do not own.
    std::array<const clang::Type*, kNumCachedStrings> m_StringTy = {};
    /// A map containing the hash of the lookup buffer. This allows us to avoid
    /// allocating memory for parsing when we know nothing has changed. Used by
    /// StartParsingRAII.
    // We do not want to include "clang/Basic/SourceLocation.h" because it makes
    // the use of this header at runtime significantly slower.
    // We really need llvm::hash_code->clang::FileID mapping but we put opaque
    // source location as unsigned to compute the FileID when needed.
    std::map<llvm::hash_code, unsigned> m_ParseBufferCache;
    /// Number of times we hit the cache.
    unsigned m_CacheHits = 0;
    /// Number of times we missed the cache.
    unsigned m_TotalParseRequests = 0;
    /// If we are called recursively.
    bool IsRecursivelyRunning = false;

  public:
    LookupHelper(clang::Parser* P, Interpreter* interp);
    ~LookupHelper();

    ///\brief Lookup a type by name, starting from the global
    /// namespace.
    ///
    ///\param [in] typeName - The type to lookup.
    ///\param [in] diagOnOff - Whether to diagnose lookup failures.
    ///\returns On a failed lookup retval.isNull() will be true.
    ///
    clang::QualType findType(llvm::StringRef typeName,
                             DiagSetting diagOnOff) const;

    ///\brief Lookup a class declaration by name, starting from the global
    /// namespace, also handles struct, union, namespace, and enum.
    ///
    ///\param [in] className   - The name of the class, struct, union,
    ///                          namespace, or enum to lookup.
    ///\param [in] diagOnOff - Whether to diagnose lookup failures.
    ///\param [out] resultType - The type of the class, struct, union,
    ///                          or enum to lookup; NULL otherwise.
    ///\param [in] instantiateTemplate - When true, will attempt to instantiate
    ///                          a class template satisfying the rquest.
    ///\returns The found declaration or null.
    ///
    const clang::Decl* findScope(llvm::StringRef className,
                                 DiagSetting diagOnOff,
                                 const clang::Type** resultType = 0,
                                 bool instantiateTemplate = true) const;


    ///\brief Lookup a class template declaration by name, starting from
    /// the global namespace, also handles struct, union, namespace, and enum.
    ///
    ///\param [in] Name   - The name of the class template to lookup.
    ///\param [in] diagOnOff - Whether to diagnose lookup failures.
    ///\returns The found declaration or null.
    ///
    const clang::ClassTemplateDecl*
    findClassTemplate(llvm::StringRef Name, DiagSetting diagOnOff) const;

    ///\brief Lookup a data member based on its Decl(Context), name.
    ///
    ///\param [in] scopeDecl - the scope (namespace or tag) that is searched for
    ///   the function.
    ///\param [in] dataName  - the name of the data member to find.
    ///\param [in] diagOnOff - whether to diagnose lookup failures.
    ///\returns The value/data member found or null.
    const clang::ValueDecl* findDataMember(const clang::Decl* scopeDecl,
                                           llvm::StringRef dataName,
                                           DiagSetting diagOnOff) const;

    ///\brief Lookup a function template based on its Decl(Context), name.
    ///
    ///\param [in] scopeDecl - the scope (namespace or tag) that is searched for
    ///   the function.
    ///\param [in] templateName  - the name of the function template to find.
    ///\param [in] objectIsConst - if true search fo function that can
    ///   be called on a const object ; default to false.
    ///\param [in] diagOnOff - whether to diagnose lookup failures.
    ///\returns The function template found or null.
    const clang::FunctionTemplateDecl*
    findFunctionTemplate(const clang::Decl* scopeDecl,
                         llvm::StringRef templateName,
                         DiagSetting diagOnOff,
                         bool objectIsConst = false) const;


    ///\brief Lookup a function based on its Decl(Context), name (return any
    ///function that matches the name (and constness if requested).
    ///
    ///\param [in] scopeDecl - the scope (namespace or tag) that is searched for
    ///   the function.
    ///\param [in] funcName  - the name of the function to find.
    ///\param [in] objectIsConst - if true search fo function that can
    ///   be called on a const object ; default to false.
    ///\param [in] diagOnOff - whether to diagnose lookup failures.
    ///\returns The function found or null.
    const clang::FunctionDecl*
    findAnyFunction(const clang::Decl* scopeDecl, llvm::StringRef funcName,
                    DiagSetting diagOnOff, bool objectIsConst = false) const;

    ///\brief Lookup a function based on its Decl(Context), name and parameters.
    ///
    ///\param [in] scopeDecl - the scope (namespace or tag) that is searched for
    ///   the function.
    ///\param [in] funcName  - the name of the function to find.
    ///\param [in] funcProto - the function parameter list (without enclosing
    ///   parantheses). Example: "size_t,int".
    ///\param [in] objectIsConst - if true search fo function that can
    ///   be called on a const object ; default to false.
    ///\param [in] diagOnOff - whether to diagnose lookup failures.
    ///\returns The function found or null.
    const clang::FunctionDecl*
    findFunctionProto(const clang::Decl* scopeDecl, llvm::StringRef funcName,
                      llvm::StringRef funcProto, DiagSetting diagOnOff,
                      bool objectIsConst = false) const;

    const clang::FunctionDecl*
    findFunctionArgs(const clang::Decl* scopeDecl, llvm::StringRef funcName,
                     llvm::StringRef funcArgs, DiagSetting diagOnOff,
                     bool objectIsConst = false) const;

    ///\brief Lookup a function based on its Decl(Context), name and parameters.
    ///
    ///\param [in] scopeDecl - the scope (namespace or tag) that is searched for
    ///   the function.
    ///\param [in] funcName  - the name of the function to find.
    ///\param [in] funcProto - the list of types of the function parameters
    ///\param [in] objectIsConst - if true search fo function that can
    ///   be called on a const object ; default to false.
    ///\param [in] diagOnOff - whether to diagnose lookup failures.
    ///\returns The function found or null.
     const clang::FunctionDecl*
     findFunctionProto(const clang::Decl* scopeDecl, llvm::StringRef funcName,
                       const llvm::SmallVectorImpl<clang::QualType>& funcProto,
                       DiagSetting diagOnOff,
                       bool objectIsConst = false) const;


    ///\brief Lookup a function based on its Decl(Context), name and parameters.
    ///   where the result if any must have exactly the arguments requested.
    ///
    ///\param[in] scopeDecl - the scope (namespace or tag) that is searched for
    ///   the function.
    ///\param[in] funcName  - the name of the function to find.
    ///\param[in] funcProto - the function parameter list (without enclosing
    ///   parantheses). Example: "size_t,int".
    ///\param[in] objectIsConst - if true search fo function that can
    ///   be called on a const object ; default to false.
    ///\param [in] diagOnOff - Whether to diagnose lookup failures.
    ///\returns The function found or null.
    const clang::FunctionDecl*
    matchFunctionProto(const clang::Decl* scopeDecl, llvm::StringRef funcName,
                       llvm::StringRef funcProto, DiagSetting diagOnOff,
                       bool objectIsConst) const;

    ///\brief Lookup a function based on its Decl(Context), name and parameters.
    ///   where the result if any must have exactly the arguments requested.
    ///
    ///\param[in] scopeDecl - the scope (namespace or tag) that is searched for
    ///   the function.
    ///\param[in] funcName  - the name of the function to find.
    ///\param[in] funcProto - the list of types of the function parameters
    ///\param[in] objectIsConst - if true search fo function that can
    ///   be called on a const object ; default to false.
    ///\param [in] diagOnOff - whether to diagnose lookup failures.
    ///\returns The function found or null.
    const clang::FunctionDecl*
    matchFunctionProto(const clang::Decl* scopeDecl, llvm::StringRef funcName,
                       const llvm::SmallVectorImpl<clang::QualType>& funcProto,
                       DiagSetting diagOnOff, bool objectIsConst) const;

    ///\brief Lookup given argument list and return each argument as an
    /// expression.
    ///
    ///\param[in] argList - The string representation of the argument list.
    ///\param[out] argExprs - The corresponding expressions to the argList.
    ///\param [in] diagOnOff - whether to diagnose lookup failures.
    ///
    void findArgList(llvm::StringRef argList,
                     llvm::SmallVectorImpl<clang::Expr*>& argExprs,
                     DiagSetting diagOnOff) const;


    ///\brief Test whether a function with the given name exists.
    ///
    ///\param [in] scopeDecl - scope in which to look for the function.
    ///\param [in] funcName - name of the function to look for.
    ///\param [in] diagOnOff - whether to diagnose lookup failures.
    bool hasFunction(const clang::Decl* scopeDecl, llvm::StringRef funcName,
                     DiagSetting diagOnOff) const;

    ///\brief Retrieve the StringType of given Type.
    StringType getStringType(const clang::Type* Type);

    void printStats() const;
    friend class StartParsingRAII;
  };

} // end namespace

#endif // CLING_LOOKUP_HELPER_H
