//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Axel Naumann <axel@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_INCREMENTAL_PARSER_H
#define CLING_INCREMENTAL_PARSER_H

#include "clang/Basic/SourceLocation.h"

#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <vector>

namespace llvm {
  struct GenericValue;
  class MemoryBuffer;
}

namespace clang {
  class CodeGenerator;
  class CompilerInstance;
  class FileID;
  class Parser;
}


namespace cling {
  class CompilationOptions;
  class CIFactory;
  class DeclCollector;
  class ExecutionContext;
  class Interpreter;
  class Transaction;
  class TransactionTransformer;

  ///\brief Responsible for the incremental parsing and compilation of input.
  ///
  /// The class manages the entire process of compilation line-by-line by 
  /// appending the compiled delta to clang'a AST. It provides basic operations
  /// on the already compiled code. See cling::Transaction class.
  ///
  class IncrementalParser {
  private:
    // our interpreter context
    // FIXME: Get rid of that back reference to the interpreter.
    Interpreter* m_Interpreter;

    // compiler instance.
    llvm::OwningPtr<clang::CompilerInstance> m_CI;

    // parser (incremental)
    llvm::OwningPtr<clang::Parser> m_Parser;

    // One buffer for each command line, owner by the source file manager
    std::vector<llvm::MemoryBuffer*> m_MemoryBuffers;

    // file ID of the memory buffer
    clang::FileID m_VirtualFileID;

    // CI owns it
    DeclCollector* m_Consumer;

    ///\brief The head of the single list of transactions. 
    ///
    const Transaction* m_FirstTransaction;

    ///\brief The last transaction
    ///
    Transaction* m_LastTransaction;

    ///\brief Code generator
    ///
    llvm::OwningPtr<clang::CodeGenerator> m_CodeGen;

    ///\brief Contains the transaction transformers.
    ///
    llvm::SmallVector<TransactionTransformer*, 6> m_TTransformers;

  public:
    enum EParseResult {
      kSuccess,
      kSuccessWithWarnings,
      kFailed
    };
    IncrementalParser(Interpreter* interp, int argc, const char* const *argv,
                      const char* llvmdir);
    ~IncrementalParser();

    void Initialize();
    clang::CompilerInstance* getCI() const { return m_CI.get(); }
    clang::Parser* getParser() const { return m_Parser.get(); }
    clang::CodeGenerator* getCodeGenerator() const { return m_CodeGen.get(); }
    bool hasCodeGenerator() const { return m_CodeGen.get(); }

    /// \{
    /// \name Transaction Support

    ///\brief Starts a transaction.
    ///
    Transaction* beginTransaction(const CompilationOptions& Opts);

    ///\brief Finishes a transaction.
    ///
    Transaction* endTransaction() const;

    ///\brief Commits a transaction if it was compete. I.e pipes it 
    /// through the consumer chain, including codegen.
    ///
    void commitTransaction(Transaction* T);

    ///\brief Reverts the AST into its previous state.
    ///
    /// If one of the declarations caused error in clang it is rolled back from
    /// the AST. This is essential feature for the error recovery subsystem.
    ///
    ///\param[in] T - The transaction to be reverted from the AST
    ///
    void rollbackTransaction(Transaction* T) const; 

    ///\brief Returns the first transaction the incremental parser saw.
    ///
    const Transaction* getFirstTransaction() const { 
      return m_FirstTransaction; 
    }

    ///\brief Returns the last transaction the incremental parser saw.
    ///
    Transaction* getLastTransaction() { 
      return m_LastTransaction; 
    }

    ///\brief Returns the last transaction the incremental parser saw.
    ///
    const Transaction* getLastTransaction() const { 
      return m_LastTransaction; 
    }

    ///\brief Returns the list of transactions seen by the interpreter.
    /// Intentionally makes a copy - that function is meant to be use for debug
    /// purposes.
    ///
    std::vector<const Transaction*> getAllTransactions();

    /// \}

    ///\brief Compiles the given input with the given compilation options.
    ///
    ///\param[in] input - The code to compile.
    ///\param[in] Opts - The compilation options to use.
    ///\returns the declarations that were compiled.
    ///
    Transaction* Compile(llvm::StringRef input, const CompilationOptions& Opts);

    ///\brief Parses the given input without calling the custom consumers and 
    /// code generation.
    ///
    /// I.e changes to the decls in the transaction commiting it will cause 
    /// different executable code.
    ///
    ///\param[in] input - The code to parse.
    ///\param[in] Opts - The compilation options to use.
    ///\returns The transaction corresponding to the input.
    ///
    Transaction* Parse(llvm::StringRef input, const CompilationOptions& Opts);

    void unloadTransaction(Transaction* T);

  private:
    void CreateSLocOffsetGenerator();
    EParseResult ParseInternal(llvm::StringRef input);
  };
} // end namespace cling
#endif // CLING_INCREMENTAL_PARSER_H
