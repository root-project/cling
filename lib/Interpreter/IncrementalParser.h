//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id: 2e7bf01c5cf0048b0e6353b5ba55d09cc0961993 $
// author:  Axel Naumann <axel@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_INCREMENTAL_PARSER_H
#define CLING_INCREMENTAL_PARSER_H

#include "clang/Basic/SourceLocation.h"

#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <vector>
#include <deque>

namespace llvm {
  struct GenericValue;
  class MemoryBuffer;
}

namespace clang {
  class CodeGenerator;
  class CompilerInstance;
  class DeclGroupRef;
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
  class TransactionPool;
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
    std::deque<llvm::MemoryBuffer*> m_MemoryBuffers;

    // file ID of the memory buffer
    clang::FileID m_VirtualFileID;

    // CI owns it
    DeclCollector* m_Consumer;

    ///\brief The storage for our transactions.
    ///
    /// We don't need the elements to be contiguous in memory, that is why we
    /// don't use std::vector. We don't need to copy the elements every time the
    /// capacity is exceeded.
    ///
    std::deque<Transaction*> m_Transactions;

    ///\brief Code generator
    ///
    llvm::OwningPtr<clang::CodeGenerator> m_CodeGen;

    ///\brief Contains the transaction AST transformers.
    ///
    llvm::SmallVector<TransactionTransformer*, 6> m_ASTTransformers;

    ///\brief Contains the transaction IR transformers.
    ///
    llvm::SmallVector<TransactionTransformer*, 2> m_IRTransformers;

    ///\brief Pool of reusable block-allocated transactions.
    ///
    llvm::OwningPtr<TransactionPool> m_TransactionPool;

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
    Transaction* endTransaction(Transaction* T);

    ///\brief Commits a transaction if it was complete. I.e pipes it 
    /// through the consumer chain, including codegen.
    ///
    ///\param[in] T - the transaction to be committed
    ///
    void commitTransaction(Transaction* T);

    ///\brief Runs the consumers (e.g. CodeGen) on a transaction.
    ///
    ///\param[in] T - the transaction to be consumed
    ///
    void codeGenTransaction(Transaction* T);

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
      if (!m_Transactions.size())
        return 0;
      return m_Transactions.front();
    }

    ///\brief Returns the last transaction the incremental parser saw.
    ///
    Transaction* getLastTransaction() {
      if (!m_Transactions.size())
        return 0;
      return m_Transactions.back();
    }

    ///\brief Returns the last transaction the incremental parser saw.
    ///
    const Transaction* getLastTransaction() const {
      if (!m_Transactions.size())
        return 0;
      return m_Transactions.back();
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
    void printTransactionStructure() const;

    ///\brief Adds a UsedAttr to all decls in the transaction.
    ///
    ///\param[in] T - the transaction for which all decls will get a UsedAttr.
    ///
    void markWholeTransactionAsUsed(Transaction* T) const;

    ///\brief Runs the static initializers created by codegening a transaction.
    ///
    ///\param[in] T - the transaction for which to run the initializers.
    ///
    bool runStaticInitOnTransaction(Transaction* T) const;

  private:
    ///\brief Runs AST transformers on a transaction.
    ///
    ///\param[in] T - the transaction to be transformed.
    ///
    void transformTransactionAST(Transaction* T);

    ///\brief Runs IR transformers on a transaction.
    ///
    ///\param[in] T - the transaction to be transformed.
    ///
    bool transformTransactionIR(Transaction* T) const;

    ///\brief Initializes a virtual file, which will be able to produce valid
    /// source locations, with the proper offsets.
    ///
    void initializeVirtualFile();

    ///\brief The work horse for parsing. It queries directly clang.
    ///
    ///\param[in] input - The incremental input that needs to be parsed.
    ///
    EParseResult ParseInternal(llvm::StringRef input);

    ///\brief Return true if this decl (which comes from an AST file) should
    /// not be sent to CodeGen. The module is assumed to describe the contents 
    /// of a library; symbols inside the library must thus not be reemitted /
    /// duplicated by CodeGen.
    ///
    bool shouldIgnore(clang::DeclGroupRef DGR) const;
  };
} // end namespace cling
#endif // CLING_INCREMENTAL_PARSER_H
