//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Axel Naumann <axel@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_INCREMENTAL_PARSER_H
#define CLING_INCREMENTAL_PARSER_H

#include "clang/Basic/SourceLocation.h"

#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <vector>
#include <deque>
#include <memory>

namespace llvm {
  struct GenericValue;
  class MemoryBuffer;
  class Module;
}

namespace clang {
  class ASTConsumer;
  class CodeGenerator;
  class CompilerInstance;
  class DiagnosticConsumer;
  class Decl;
  class FileID;
  class ModuleFileExtension;
  class Parser;
}

namespace cling {
  class CompilationOptions;
  class DeclCollector;
  class ExecutionContext;
  class Interpreter;
  class Transaction;
  class TransactionPool;
  class ASTTransformer;
  class IncrementalCUDADeviceCompiler;

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
    std::unique_ptr<clang::CompilerInstance> m_CI;

    // parser (incremental)
    std::unique_ptr<clang::Parser> m_Parser;

    // One buffer for each command line, owner by the source file manager
    std::deque<std::pair<llvm::MemoryBuffer*, clang::FileID>> m_MemoryBuffers;

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

    ///\brief Number of created modules.
    unsigned m_ModuleNo = 0;

    ///\brief Code generator
    ///
    clang::CodeGenerator* m_CodeGen = nullptr;

    ///\brief Pool of reusable block-allocated transactions.
    ///
    std::unique_ptr<TransactionPool> m_TransactionPool;

    ///\brief DiagnosticConsumer instance
    ///
    std::unique_ptr<clang::DiagnosticConsumer> m_DiagConsumer;

    ///\brief Cling's worker class implementing the compilation of CUDA device code
    ///
    std::unique_ptr<IncrementalCUDADeviceCompiler> m_CUDACompiler;

    using ModuleFileExtensions =
        std::vector<std::shared_ptr<clang::ModuleFileExtension>>;

  public:
    enum EParseResult {
      kSuccess,
      kSuccessWithWarnings,
      kFailed
    };
    typedef llvm::PointerIntPair<Transaction*, 2, EParseResult>
      ParseResultTransaction;
    IncrementalParser(Interpreter* interp, const char* llvmdir,
                      const ModuleFileExtensions& moduleExtensions);
    ~IncrementalParser();

    ///\brief Whether the IncrementalParser is valid.
    ///
    ///\param[in] initialized - check if IncrementalParser has been initialized.
    ///
    bool isValid(bool initialized = true) const;

    bool Initialize(llvm::SmallVectorImpl<ParseResultTransaction>& result,
                    bool isChildInterpreter);
    clang::CompilerInstance* getCI() const { return m_CI.get(); }
    clang::Parser* getParser() const { return m_Parser.get(); }
    clang::CodeGenerator* getCodeGenerator() const { return m_CodeGen; }
    bool hasCodeGenerator() const { return m_CodeGen; }
    clang::SourceLocation getLastMemoryBufferEndLoc() const;

    /// \{
    /// \name Transaction Support

    ///\brief Starts a transaction.
    ///
    Transaction* beginTransaction(const CompilationOptions& Opts);

    ///\brief Finishes a transaction.
    ///
    ParseResultTransaction endTransaction(Transaction* T);

    ///\brief Commits a transaction if it was complete. I.e pipes it
    /// through the consumer chain, including codegen.
    ///
    ///\param[in] PRT - the transaction (ParseResultTransaction) to be
    /// committed
    ///\param[in] ClearDiagClient - Reset the DiagnosticsEngine client or not
    ///
    void commitTransaction(ParseResultTransaction& PRT,
                           bool ClearDiagClient = true);

    ///\brief Runs the consumers (e.g. CodeGen) on a non-parsed transaction.
    ///
    ///\param[in] T - the transaction to be consumed
    ///
    void emitTransaction(Transaction* T);

    ///\brief Remove a Transaction from the collection of Transactions.
    ///
    ///\param[in] T - The transaction to be reverted from the AST
    ///
    void deregisterTransaction(Transaction& T);

    ///\brief Returns the first transaction the incremental parser saw.
    ///
    const Transaction* getFirstTransaction() const {
      if (m_Transactions.empty())
        return 0;
      return m_Transactions.front();
    }

    ///\brief Returns the last transaction the incremental parser saw.
    ///
    Transaction* getLastTransaction() {
      if (m_Transactions.empty())
        return 0;
      return m_Transactions.back();
    }

    ///\brief Returns the last transaction the incremental parser saw.
    ///
    const Transaction* getLastTransaction() const {
      if (m_Transactions.empty())
        return 0;
      return m_Transactions.back();
    }

    ///\brief Returns the most recent transaction with an input line wrapper,
    /// which could well be the current one.
    ///
    const Transaction* getLastWrapperTransaction() const;

    ///\brief Returns the currently active transaction.
    ///
    const Transaction* getCurrentTransaction() const;


    ///\brief Add a user-generated transaction.
    void addTransaction(Transaction* T);

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
    ParseResultTransaction Compile(llvm::StringRef input, const CompilationOptions& Opts);

    void printTransactionStructure() const;

    ///\brief Runs the static initializers created by codegening a transaction.
    ///
    ///\param[in] T - the transaction for which to run the initializers.
    ///
    bool runStaticInitOnTransaction(Transaction* T) const;

    ///\brief Add the trnasformers to the Incremental Parser.
    ///
    void SetTransformers(bool isChildInterpreter);

  private:
    ///\brief Finalizes the consumers (e.g. CodeGen) on a transaction.
    ///
    ///\param[in] T - the transaction to be finalized
    ///
    void codeGenTransaction(Transaction* T);

    ///\brief Initializes a virtual file, which will be able to produce valid
    /// source locations, with the proper offsets.
    ///
    void initializeVirtualFile();

    ///\brief The work horse for parsing. It queries directly clang.
    ///
    ///\param[in] input - The incremental input that needs to be parsed.
    ///
    EParseResult ParseInternal(llvm::StringRef input);

    ///\brief Create a unique name for the next llvm::Module
    ///
    std::string makeModuleName();

    ///\brief Create a new llvm::Module
    ///
    llvm::Module* StartModule();

  };
} // end namespace cling
#endif // CLING_INCREMENTAL_PARSER_H
