//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author: Timur Pocheptsov <Timur.Pocheptsov@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "Display.h"

#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/LookupHelper.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/Type.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/Specifiers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/MacroInfo.h"
#include "clang/Lex/Preprocessor.h"

#include "llvm/Support/Format.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/Path.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <cctype>
#include <limits>
#include <set>

using namespace clang;

namespace cling {

namespace {

typedef DeclContext::decl_iterator decl_iterator;
typedef CXXRecordDecl::base_class_const_iterator base_decl_iterator;

//______________________________________________________________________________
template<class Decl>
bool HasUDT(const Decl* decl)
{
  //Check the type of decl, if it's a CXXRecordDecl or array with base element type of CXXRecordDecl.
  assert(decl != 0 && "HasUDT, 'decl' parameter is null");

  if (const RecordType* const recordType = decl->getType()->template getAs<RecordType>())
    return cast_or_null<CXXRecordDecl>(recordType->getDecl()->getDefinition());

  if (const ArrayType* const arrayType = decl->getType()->getAsArrayTypeUnsafe()) {
    if (const Type* const elType = arrayType->getBaseElementTypeUnsafe()) {
      if (const RecordType* const recordType = elType->getAs<RecordType>())
        return cast_or_null<CXXRecordDecl>(recordType->getDecl()->getDefinition());
    }
  }

  return false;
}

//______________________________________________________________________________
int NumberOfElements(const ArrayType* type)
{
  assert(type != 0 && "NumberOfElements, 'type' parameter is null");

  if (const ConstantArrayType* const arrayType = dyn_cast<ConstantArrayType>(type)) {
    //We can calculate only the size of constant size array.
    //no conv. to int :(
    const int nElements = int(arrayType->getSize().roundToDouble());
    if (nElements <= 0)
      return 0;

    if (const Type* elementType = arrayType->getElementType().getTypePtr()) {
      if (const ArrayType* subArrayType = elementType->getAsArrayTypeUnsafe())
        return nElements* NumberOfElements(subArrayType);
    }

    return nElements;
  } else
    return 0;
}

//______________________________________________________________________________
static void AppendAnyDeclLocation(const CompilerInstance* compiler,
                                  SourceLocation loc,
                                  std::string& textLine,
                                  const char* format,
                                  const char* formatNull,
                                  const char* filenameNull)
{
  assert(compiler != 0 && "AppendAnyDeclLocation, 'compiler' parameter is null");

  llvm::raw_string_ostream rss(textLine);
  llvm::formatted_raw_ostream frss(rss);
  std::string baseName;
  int lineNo = -2;


  if (compiler->hasSourceManager()) {
    const SourceManager &sourceManager = compiler->getSourceManager();
    if (loc.isValid() && sourceManager.isLoadedSourceLocation(loc)) {
      // No line numbers as they would touich disk.
      baseName = llvm::sys::path::filename(sourceManager.getFilename(loc));
      lineNo = -1;
    } else {
      PresumedLoc ploc(sourceManager.getPresumedLoc(loc));
      if (ploc.isValid()) {
        baseName = llvm::sys::path::filename(ploc.getFilename()).str();
        lineNo = (int)ploc.getLine();
      }
    }
  }
  if (lineNo == -2)
    frss<<llvm::format(formatNull, filenameNull);
  else
    frss<<llvm::format(format, baseName.c_str(), lineNo);
}

//______________________________________________________________________________
void AppendClassDeclLocation(const CompilerInstance* compiler, const CXXRecordDecl* classDecl,
                             std::string& textLine, bool verbose)
{
  assert(classDecl != 0 && "AppendClassDeclLocation, 'classDecl' parameter is null");

  //Location has a fixed format - from G__display_class.
  static const char* formatShort = "%-25s%5d";
  static const char* formatVerbose = "FILE: %s LINE: %d";
  const char* format = formatShort;
  if (verbose)
    format = formatVerbose;
  AppendAnyDeclLocation(compiler, classDecl->getLocation(), textLine,
                        format, "%-30s", "");
}

//______________________________________________________________________________
void AppendMemberFunctionLocation(const CompilerInstance* compiler, const Decl* decl,
                                  std::string& textLine)
{
  //Location has a fixed format - from G__display_class.

  assert(compiler != 0 && "AppendMemberFunctionLocation, 'compiler' parameter is null");
  assert(decl != 0 && "AppendMemberFunctionLocation, 'decl' parameter is null");

  llvm::raw_string_ostream rss(textLine);
  llvm::formatted_raw_ostream frss(rss);
  //Location can be actually somewhere in a compiled code.
  const char* const unknownLocation = "(compiled)";
  frss<<llvm::format("%-15s(NA):(NA) 0", unknownLocation);
}

//______________________________________________________________________________
void AppendDeclLocation(const CompilerInstance* compiler, const Decl* decl,
                        std::string& textLine)
{

  AppendAnyDeclLocation(compiler, decl->getLocation(), textLine,
                        "%-15s%4d", "%-15s    ", "compiled");
}

//______________________________________________________________________________
void AppendMacroLocation(const CompilerInstance* compiler, const MacroInfo* macroInfo,
                         std::string& textLine)
{
  assert(macroInfo != 0 && "AppendMacroLocation, 'macroInfo' parameter is null");

  //TODO: check what does location for macro definition really means -
  //macro can be defined many times, what do we have in a TranslationUnit in this case?
  //At the moment this function is similar to AppendDeclLocation.

  AppendAnyDeclLocation(compiler, macroInfo->getDefinitionLoc(), textLine,
                        "%-15s%4d", "%-15s    ", "(unknown)");
}

//______________________________________________________________________________
void AppendClassKeyword(const CXXRecordDecl* classDecl, std::string& name)
{
  assert(classDecl != 0 && "AppendClassKeyword, 'classDecl' parameter is null");

  name += classDecl->getKindName();
  name += ' ';
}

//______________________________________________________________________________
void AppendClassName(const CXXRecordDecl* classDecl, std::string& name)
{
  assert(classDecl != 0 && "AppendClassName, 'classDecl' parameter is null");

  const LangOptions langOpts;
  const PrintingPolicy printingPolicy(langOpts);
  std::string tmp;
  //Name for diagnostic will include template arguments if any.
  llvm::raw_string_ostream stream(tmp);
  classDecl->getNameForDiagnostic(stream,
                                  printingPolicy, /*qualified name=*/true);
  stream.flush();
  name += tmp;
}

//______________________________________________________________________________
void AppendMemberAccessSpecifier(const Decl* memberDecl, std::string& name)
{
  assert(memberDecl != 0 && "AppendMemberAccessSpecifier, 'memberDecl' parameter is 0");

  switch (memberDecl->getAccess()) {
  case AS_private:
    name += "private:";
    break;
  case AS_protected:
    name += "protected:";
    break;
  case AS_public:
  case AS_none://Public or private?
    name += "public:";
  }
}

//______________________________________________________________________________
void AppendConstructorSignature(const CXXConstructorDecl* ctorDecl, std::string& name)
{
  assert(ctorDecl != 0 && "AppendConstructorSignature, 'ctorDecl' parameter is null");

  const QualType type = ctorDecl->getType();
  assert(isa<FunctionType>(type) == true && "AppendConstructorSignature, ctorDecl->getType is not a FunctionType");

  const FunctionType* const aft = type->getAs<FunctionType>();
  const FunctionProtoType* const ft = ctorDecl->hasWrittenPrototype() ?
                                      dyn_cast<FunctionProtoType>(aft) : 0;

  if (ctorDecl->isExplicit())
    name += "explicit ";

  name += ctorDecl->getNameInfo().getAsString();
  name += "(";

  if (ft) {
    llvm::raw_string_ostream stream(name);

    for (unsigned i = 0, e = ctorDecl->getNumParams(); i != e; ++i) {
      if (i)
        stream << ", ";
      ctorDecl->getParamDecl(i)->print(stream, 0, false);//or true?
    }

    if (ft->isVariadic()) {
      if (ctorDecl->getNumParams())
        stream << ", ";
      stream << "...";
    }
  } else if (ctorDecl->doesThisDeclarationHaveABody() && !ctorDecl->hasPrototype()) {
    for (unsigned i = 0, e = ctorDecl->getNumParams(); i != e; ++i) {
      if (i)
        name += ", ";
      name += ctorDecl->getParamDecl(i)->getNameAsString();
    }
  }

  name += ")";
}

//______________________________________________________________________________
void AppendMemberFunctionSignature(const Decl* methodDecl, std::string& name)
{
  assert(methodDecl != 0 && "AppendMemberFunctionSignature, 'methodDecl' parameter is null");
  assert(methodDecl->getKind() != Decl::CXXConstructor && "AppendMemberFunctionSignature, called for a ctor declaration");

  llvm::raw_string_ostream out(name);
  const LangOptions langOpts;
  PrintingPolicy printingPolicy(langOpts);
  printingPolicy.TerseOutput = true;//Do not print the body of an inlined function.
  printingPolicy.SuppressSpecifiers = false; //Show 'static', 'inline', etc.

  methodDecl->print(out, printingPolicy, 0, false);
}

//______________________________________________________________________________
void AppendObjectDeclaration(const Decl* objDecl, const PrintingPolicy& policy,
                             bool printInstantiation, std::string& name)
{
  assert(objDecl != 0 && "AppendObjectDeclaration, 'objDecl' parameter is null");

  llvm::raw_string_ostream out(name);
  objDecl->print(out, policy, 0, printInstantiation);
}

//______________________________________________________________________________
void AppendBaseClassSpecifiers(base_decl_iterator base, std::string& textLine)
{
  if (base->isVirtual())
    textLine += "virtual";

  switch (base->getAccessSpecifier()) {
  case AS_private:
    textLine += "private";
    break;
  case AS_protected:
    textLine += "protected";
    break;
  case AS_public:
  case AS_none://TODO - check this.
    textLine += "public";
  }
}

//______________________________________________________________________________
void AppendClassSize(const CompilerInstance* compiler, const RecordDecl* decl,
                     std::string& textLine)
{
  assert(compiler != 0 && "AppendClassSize, 'compiler' parameter is null");
  assert(decl != 0 && "AppendClassSize, 'decl' parameter is null");

  if (dyn_cast<ClassTemplatePartialSpecializationDecl>(decl)) {
    textLine += "SIZE: (NA)";
    return;
  }

  const ASTRecordLayout& layout = compiler->getASTContext().getASTRecordLayout(decl);

  llvm::raw_string_ostream rss(textLine);
  llvm::formatted_raw_ostream frss(rss);
  frss<<llvm::format("SIZE: %d", int(layout.getSize().getQuantity()));
}

//______________________________________________________________________________
template<class Decl>
void AppendUDTSize(const CompilerInstance* compiler, const Decl* decl, std::string& textLine)
{
  assert(compiler != 0 && "AppendUDTSize, 'compiler' parameter is null");
  assert(decl != 0 && "AppendUDTSize, 'decl' parameter is null");

  std::string formatted;

  {
  llvm::raw_string_ostream rss(formatted);
  llvm::formatted_raw_ostream frss(rss);

  if (const RecordType* const recordType = decl->getType()->template getAs<RecordType>()) {
    if (const RecordDecl* const recordDecl = cast_or_null<RecordDecl>(recordType->getDecl()->getDefinition())) {
      const ASTRecordLayout& layout = compiler->getASTContext().getASTRecordLayout(recordDecl);
      frss<<llvm::format("%d", int(layout.getSize().getQuantity()));
    }
  } else if (const ArrayType* const arrayType = decl->getType()->getAsArrayTypeUnsafe()) {
    if (const Type* const elementType = arrayType->getBaseElementTypeUnsafe()) {
      if (const CXXRecordDecl* const recordDecl = elementType->getAsCXXRecordDecl()) {
        const ASTRecordLayout& layout = compiler->getASTContext().getASTRecordLayout(recordDecl);
        const int baseElementSize = int(layout.getSize().getQuantity());
        const int nElements = NumberOfElements(arrayType);
        if (nElements > 0)
          frss<<llvm::format("%d", nElements * baseElementSize);
      }
    }
  }

  }

  formatted.length() ? textLine += formatted : textLine += "NA";
}

//______________________________________________________________________________
void AppendBaseClassOffset(const CompilerInstance* compiler, const CXXRecordDecl* completeClass,
                           const CXXRecordDecl* baseClass, bool isVirtual, std::string& textLine)
{
  assert(compiler != 0 && "AppendBaseClassOffset, 'compiler' parameter is null");
  assert(completeClass != 0 && "AppendBaseClassOffset, 'completeClass' parameter is null");
  assert(baseClass != 0 && "AppendBaseClassOffset, 'baseClass' parameter is null");

  const ASTRecordLayout& layout = compiler->getASTContext().getASTRecordLayout(completeClass);

  llvm::raw_string_ostream rss(textLine);
  llvm::formatted_raw_ostream frss(rss);

  if (isVirtual)//format is from G__display_classinheritance.
    frss<<llvm::format("0x%-8x", int(layout.getVBaseClassOffset(baseClass).getQuantity()));
  else
    frss<<llvm::format("0x%-8x", int(layout.getBaseClassOffset(baseClass).getQuantity()));
}

//______________________________________________________________________________
void AppendDataMemberOffset(const CompilerInstance* compiler, const CXXRecordDecl* classDecl,
                            const FieldDecl* fieldDecl, std::string& textLine)
{
  assert(compiler != 0 && "AppendDataMemberOffset, 'compiler' parameter is null");
  assert(classDecl != 0 && "AppendDataMemberOffset, 'classDecl' parameter is null");
  assert(fieldDecl != 0 && "AppendDataMemberOffset, 'fieldDecl' parameter is null");

  const ASTRecordLayout& layout = compiler->getASTContext().getASTRecordLayout(classDecl);

  std::string formatted;
  //
  llvm::raw_string_ostream rss(textLine);
  llvm::formatted_raw_ostream frss(rss);
  frss<<llvm::format("0x%-8x", int(layout.getFieldOffset(fieldDecl->getFieldIndex())
                               / std::numeric_limits<unsigned char>::digits));
}

//
//This is a primitive class which does nothing except fprintf for the moment,
//but this can change later.
class FILEPrintHelper {
public:
  FILEPrintHelper(llvm::raw_ostream& stream);

  void Print(const char* msg)const;

private:
  llvm::raw_ostream& fStream;
};

//______________________________________________________________________________
FILEPrintHelper::FILEPrintHelper(llvm::raw_ostream& stream)
             : fStream(stream)
{
  fStream.flush();
}

//______________________________________________________________________________
void FILEPrintHelper::Print(const char* msg)const
{
  assert(msg != 0 && "Print, 'msg' parameter is null");
  // We want to keep stdout and fStream in sync if fStream is different.
  fflush(stdout);
  fStream << msg;
  fStream.flush();
}

//
//Aux. class to traverse translation-unit-declaration/class-declaration.
//

class ClassPrinter {
private:
  enum {
    kBaseTreeShift = 3,
    kMemberTypeShift = 3
  };
public:
  ClassPrinter(llvm::raw_ostream& stream, const class cling::Interpreter* interpreter);

  void DisplayAllClasses()const;
  void DisplayClass(const std::string& className)const;

  void SetVerbose(bool verbose);

  void Reset();
private:

  //These are declarations, which can contain nested class declarations,
  //I have separate function for the case I want to treat them in different ways.
  //Can be only one processDecl actually.

  void ProcessDecl(decl_iterator decl)const;
  void ProcessBlockDecl(decl_iterator decl)const;
  void ProcessFunctionDecl(decl_iterator decl)const;
  void ProcessNamespaceDecl(decl_iterator decl)const;
  void ProcessLinkageSpecDecl(decl_iterator decl)const;
  void ProcessClassDecl(decl_iterator decl)const;
  void ProcessClassTemplateDecl(decl_iterator decl)const;

  template<class Decl>
  void ProcessTypeOfMember(const Decl* decl, unsigned nSpaces)const
  {
    //Extract the type of declaration and process it.
    assert(decl != 0 && "ProcessTypeOfMember, 'decl' parameter is null");

    if (const RecordType* const recordType = decl->getType()->template getAs<RecordType>()) {
      if (const CXXRecordDecl* const classDecl = cast_or_null<CXXRecordDecl>(recordType->getDecl()->getDefinition())) {
        if (fSeenDecls.find(classDecl) == fSeenDecls.end())
          DisplayDataMembers(classDecl, nSpaces);
      }
    } else if (const ArrayType* const arrayType = decl->getType()->getAsArrayTypeUnsafe()) {
      if (const Type* const elType = arrayType->getBaseElementTypeUnsafe()) {
        if (const RecordType* const recordType = elType->getAs<RecordType>()) {
          if (const CXXRecordDecl* classDecl = cast_or_null<CXXRecordDecl>(recordType->getDecl()->getDefinition()))
            if (fSeenDecls.find(classDecl) == fSeenDecls.end())
              DisplayDataMembers(classDecl, nSpaces);
        }
      }
    }
  }

  void DisplayClassDecl(const CXXRecordDecl* classDecl)const;
  void DisplayClassFwdDecl(const CXXRecordDecl* classDecl)const;
  void DisplayBasesAsList(const CXXRecordDecl* classDecl)const;
  void DisplayBasesAsTree(const CXXRecordDecl* classDecl, unsigned nSpaces)const;
  void DisplayMemberFunctions(const CXXRecordDecl* classDecl)const;
  void DisplayDataMembers(const CXXRecordDecl* classDecl, unsigned nSpaces)const;

  FILEPrintHelper fOut;
  const cling::Interpreter* fInterpreter;
  bool fVerbose;

  mutable std::set<const Decl*> fSeenDecls;
};

//______________________________________________________________________________
ClassPrinter::ClassPrinter(llvm::raw_ostream& stream, const cling::Interpreter* interpreter)
           : fOut(stream),
             fInterpreter(interpreter),
             fVerbose(false)
{
  assert(interpreter != 0 && "ClassPrinter, 'compiler' parameter is null");
}


//______________________________________________________________________________
void ClassPrinter::DisplayAllClasses()const
{
  //Just in case asserts were deleted from ctor:
  assert(fInterpreter != 0 && "DisplayAllClasses, fCompiler is null");

  const CompilerInstance* const compiler = fInterpreter->getCI();
  assert(compiler != 0 && "DisplayAllClasses, compiler instance is null");

  const TranslationUnitDecl* const tuDecl = compiler->getASTContext().getTranslationUnitDecl();
  assert(tuDecl != 0 && "DisplayAllClasses, translation unit is empty");

  fOut.Print("List of classes");
  // Could trigger deserialization of decls.
  Interpreter::PushTransactionRAII RAII(const_cast<Interpreter*>(fInterpreter));
  for (decl_iterator decl = tuDecl->decls_begin(); decl != tuDecl->decls_end(); ++decl)
    ProcessDecl(decl);
}

//______________________________________________________________________________
void ClassPrinter::DisplayClass(const std::string& className)const
{
  //Just in case asserts were deleted from ctor:
  assert(fInterpreter != 0 && "DisplayClass, fCompiler is null");

  const cling::LookupHelper &lookupHelper = fInterpreter->getLookupHelper();
  if (const Decl* const decl
      = lookupHelper.findScope(className, cling::LookupHelper::NoDiagnostics)) {
    if (const CXXRecordDecl* const classDecl = dyn_cast<CXXRecordDecl>(decl)) {
      if (classDecl->hasDefinition())
        DisplayClassDecl(classDecl);
      else
        fOut.Print(("The class " + className +
                    " does not have any definition available\n").c_str());
    } else
       fOut.Print(("A " + std::string(decl->getDeclKindName()) + " declaration"
                   " was found for " + className + "\n").c_str());
  } else
    fOut.Print(("Class " + className + " not found\n").c_str());
}

//______________________________________________________________________________
void ClassPrinter::SetVerbose(bool verbose)
{
  fVerbose = verbose;
}

//______________________________________________________________________________
void ClassPrinter::Reset()
{
  fSeenDecls.clear();
}

//______________________________________________________________________________
void ClassPrinter::ProcessDecl(decl_iterator decl)const
{
  //Just in case asserts were deleted from ctor:
  assert(fInterpreter != 0 && "ProcessDecl, fInterpreter is null");
  assert(*decl != 0 && "ProcessDecl, 'decl' parameter is not a valid iterator");

  switch (decl->getKind()) {
  case Decl::Namespace:
    ProcessNamespaceDecl(decl);
    break;
  case Decl::Block:
    ProcessBlockDecl(decl);
    break;
  case Decl::Function:
  case Decl::CXXMethod:
  case Decl::CXXConstructor:
  case Decl::CXXConversion:
  case Decl::CXXDestructor:
    ProcessFunctionDecl(decl);
    break;
  case Decl::LinkageSpec:
    ProcessLinkageSpecDecl(decl);
    break;
  case Decl::ClassTemplate:
    ProcessClassTemplateDecl(decl);
    break;
  case Decl::CXXRecord:
  case Decl::ClassTemplateSpecialization:
  case Decl::ClassTemplatePartialSpecialization:
    ProcessClassDecl(decl);
    break;
  default:
    if (dyn_cast<FunctionDecl>(*decl))
      //decl->getKind() != Decl::Function, but decl has type, inherited from FunctionDecl.
      ProcessFunctionDecl(decl);
    break;
  }
}

//______________________________________________________________________________
void ClassPrinter::ProcessBlockDecl(decl_iterator decl)const
{
  //Just in case asserts were deleted from ctor:
  assert(fInterpreter != 0 && "ProcessBlockDecl, fInterpreter is null");
  assert(*decl != 0 && "ProcessBlockDecl, 'decl' parameter is not a valid iterator");
  assert(decl->getKind() == Decl::Block && "ProcessBlockDecl, decl->getKind() != BlockDecl");

  //Block can contain nested (arbitrary deep) class declarations.
  //Though, I'm not sure if have block in our code.
  const BlockDecl* const blockDecl = dyn_cast<BlockDecl>(*decl);
  assert(blockDecl != 0 && "ProcessBlockDecl, internal error - decl is not a BlockDecl");

  // Could trigger deserialization of decls.
  Interpreter::PushTransactionRAII RAII(const_cast<Interpreter*>(fInterpreter));
  for (decl_iterator it = blockDecl->decls_begin(); it != blockDecl->decls_end(); ++it)
    ProcessDecl(it);
}

//______________________________________________________________________________
void ClassPrinter::ProcessFunctionDecl(decl_iterator decl)const
{
  //Just in case asserts were deleted from ctor:
  assert(fInterpreter != 0 && "ProcessFunctionDecl, fInterpreter is null");
  assert(*decl != 0 && "ProcessFunctionDecl, 'decl' parameter is not a valid iterator");

  //Function can contain class declarations, we have to check this.
  const FunctionDecl* const functionDecl = dyn_cast<FunctionDecl>(*decl);
  assert(functionDecl != 0 && "ProcessFunctionDecl, internal error - decl is not a FunctionDecl");

  // Could trigger deserialization of decls.
  Interpreter::PushTransactionRAII RAII(const_cast<Interpreter*>(fInterpreter));
  for (decl_iterator it = functionDecl->decls_begin(); it != functionDecl->decls_end(); ++it)
    ProcessDecl(it);
}

//______________________________________________________________________________
void ClassPrinter::ProcessNamespaceDecl(decl_iterator decl)const
{
  //Just in case asserts were deleted from ctor:
  assert(fInterpreter != 0 && "ProcessNamespaceDecl, fInterpreter is null");
  assert(*decl != 0 && "ProcessNamespaceDecl, 'decl' parameter is not a valid iterator");
  assert(decl->getKind() == Decl::Namespace && "ProcessNamespaceDecl, decl->getKind() != Namespace");

  //Namespace can contain nested (arbitrary deep) class declarations.
  const NamespaceDecl* const namespaceDecl = dyn_cast<NamespaceDecl>(*decl);
  assert(namespaceDecl != 0 && "ProcessNamespaceDecl, 'decl' parameter is not a NamespaceDecl");

  // Could trigger deserialization of decls.
  Interpreter::PushTransactionRAII RAII(const_cast<Interpreter*>(fInterpreter));
  for (decl_iterator it = namespaceDecl->decls_begin(); it != namespaceDecl->decls_end(); ++it)
    ProcessDecl(it);
}

//______________________________________________________________________________
void ClassPrinter::ProcessLinkageSpecDecl(decl_iterator decl)const
{
  //Just in case asserts were deleted from ctor:
  assert(fInterpreter != 0 && "ProcessLinkageSpecDecl, fInterpreter is null");
  assert(*decl != 0 && "ProcessLinkageSpecDecl, 'decl' parameter is not a valid iterator");

  const LinkageSpecDecl* const linkageSpec = dyn_cast<LinkageSpecDecl>(*decl);
  assert(linkageSpec != 0 && "ProcessLinkageSpecDecl, decl is not a LinkageSpecDecl");

  // Could trigger deserialization of decls.
  Interpreter::PushTransactionRAII RAII(const_cast<Interpreter*>(fInterpreter));
  for (decl_iterator it = linkageSpec->decls_begin(); it != linkageSpec->decls_end(); ++it)
    ProcessDecl(it);
}

//______________________________________________________________________________
void ClassPrinter::ProcessClassDecl(decl_iterator decl)const
{
  assert(fInterpreter != 0 && "ProcessClassDecl, fInterpreter is null");
  assert(*decl != 0 && "ProcessClassDecl, 'decl' parameter is not a valid iterator");

  const CXXRecordDecl* const classDecl = dyn_cast<CXXRecordDecl>(*decl);
  assert(classDecl != 0 && "ProcessClassDecl, internal error, declaration is not a CXXRecordDecl");

  if (!classDecl->hasDefinition()) {
    DisplayClassFwdDecl(classDecl);
    return;
  }

  DisplayClassDecl(classDecl);

  // Could trigger deserialization of decls.
  Interpreter::PushTransactionRAII RAII(const_cast<Interpreter*>(fInterpreter));
  //Now we have to check nested scopes for class declarations.
  for (decl_iterator decl = classDecl->decls_begin(); decl != classDecl->decls_end(); ++decl)
    ProcessDecl(decl);
}

//______________________________________________________________________________
void ClassPrinter::ProcessClassTemplateDecl(decl_iterator decl)const
{
  assert(fInterpreter != 0 && "ProcessClassDecl, fInterpreter is null");
  assert(*decl != 0 && "ProcessClassDecl, 'decl' parameter is not a valid iterator");

  ClassTemplateDecl *templateDecl = dyn_cast<ClassTemplateDecl>(*decl);
  assert(templateDecl != 0 && "ProcessClassTemplateDecl, internal error, declaration is not a ClassTemplateDecl");

  templateDecl = templateDecl->getCanonicalDecl();

  if (!templateDecl->isThisDeclarationADefinition())
    return;

  // Could trigger deserialization of decls.
  Interpreter::PushTransactionRAII RAII(const_cast<Interpreter*>(fInterpreter));
  //Now we have to display all the specialization (/instantiations)
  for (ClassTemplateDecl::spec_iterator spec = templateDecl->spec_begin();
       spec != templateDecl->spec_end(); ++spec)
     ProcessDecl(decl_iterator( *spec ));
}

//______________________________________________________________________________
void ClassPrinter::DisplayClassDecl(const CXXRecordDecl* classDecl)const
{
  assert(classDecl != 0 && "DisplayClassDecl, 'classDecl' parameter is null");
  assert(fInterpreter != 0 && "DisplayClassDecl, fInterpreter is null");

  // Could trigger deserialization of decls.
  Interpreter::PushTransactionRAII RAII(const_cast<Interpreter*>(fInterpreter));

  classDecl = classDecl->getDefinition();
  assert(classDecl != 0 && "DisplayClassDecl, invalid decl - no definition");

  if (fSeenDecls.find(classDecl) != fSeenDecls.end())
    return;
  else
    fSeenDecls.insert(classDecl);

  if (!fVerbose) {
    //Print: source file, line number, class-keyword, qualifies class name, base classes.
    std::string classInfo;

    AppendClassDeclLocation(fInterpreter->getCI(), classDecl, classInfo, false);
    classInfo += "     ";
    AppendClassKeyword(classDecl, classInfo);
    classInfo += ' ';
    AppendClassName(classDecl, classInfo);
    classInfo += ' ';
    //
    fOut.Print(classInfo.c_str());

    DisplayBasesAsList(classDecl);

    fOut.Print("\n");
  } else {
    //Hehe, this line was stolen from CINT.
    fOut.Print("===========================================================================\n");

    std::string classInfo;
    AppendClassKeyword(classDecl, classInfo);
    AppendClassName(classDecl, classInfo);

    fOut.Print(classInfo.c_str());
    fOut.Print("\n");

    classInfo.clear();
    AppendClassSize(fInterpreter->getCI(), classDecl, classInfo);
    classInfo += ' ';
    AppendClassDeclLocation(fInterpreter->getCI(), classDecl, classInfo, true);
    fOut.Print(classInfo.c_str());
    fOut.Print("\n");

    if (classDecl->bases_begin() != classDecl->bases_end())
      fOut.Print("Base classes: --------------------------------------------------------\n");

    DisplayBasesAsTree(classDecl, 0);
    //now list all members.40963410

    fOut.Print("List of member variables --------------------------------------------------\n");
    DisplayDataMembers(classDecl, 0);

    fOut.Print("List of member functions :---------------------------------------------------\n");
    //CINT has a format like %-15s blah-blah.
    fOut.Print("filename     line:size busy function type and name\n");
    DisplayMemberFunctions(classDecl);
  }
}

//______________________________________________________________________________
void ClassPrinter::DisplayClassFwdDecl(const CXXRecordDecl* classDecl)const
{
  assert(classDecl != 0 && "DisplayClassDecl, 'classDecl' parameter is null");
  assert(fInterpreter != 0 && "DisplayClassDecl, fInterpreter is null");

  if (classDecl->isImplicit() || fSeenDecls.find(classDecl) != fSeenDecls.end())
    return;
  else
    fSeenDecls.insert(classDecl);

  if (!fVerbose) {
    //Print: source file, line number, class-keyword, qualifies class name, base classes.
    std::string classInfo;

    AppendClassDeclLocation(fInterpreter->getCI(), classDecl, classInfo, false);
    classInfo += " fwd ";
    AppendClassKeyword(classDecl, classInfo);
    classInfo += ' ';
    AppendClassName(classDecl, classInfo);
    classInfo += ' ';
    //
    fOut.Print(classInfo.c_str());

    fOut.Print("\n");
  } else {
    fOut.Print("===========================================================================\n");

    std::string classInfo("Forwarded ");
    AppendClassKeyword(classDecl, classInfo);
    AppendClassName(classDecl, classInfo);

    fOut.Print(classInfo.c_str());
    fOut.Print("\n");

    classInfo.clear();
    classInfo = "SIZE: n/a ";
    AppendClassDeclLocation(fInterpreter->getCI(), classDecl, classInfo, true);
    fOut.Print(classInfo.c_str());
    fOut.Print("\n");
  }
}

//______________________________________________________________________________
void ClassPrinter::DisplayBasesAsList(const CXXRecordDecl* classDecl)const
{
  assert(fInterpreter != 0 && "DisplayBasesAsList, fInterpreter is null");
  assert(classDecl != 0 && "DisplayBasesAsList, 'classDecl' parameter is 0");
  assert(classDecl->hasDefinition() == true && "DisplayBasesAsList, 'classDecl' is invalid");
  assert(fVerbose == false && "DisplayBasesAsList, called in a verbose output");

  //we print a list of base classes as one line, with access specifiers and 'virtual' if needed.
  std::string bases(": ");
  // Could trigger deserialization of decls.
  Interpreter::PushTransactionRAII RAII(const_cast<Interpreter*>(fInterpreter));
  for (base_decl_iterator baseIt = classDecl->bases_begin(); baseIt != classDecl->bases_end(); ++baseIt) {
    if (baseIt != classDecl->bases_begin())
      bases += ", ";

    const RecordType* const type = baseIt->getType()->getAs<RecordType>();
    if (type) {
      const CXXRecordDecl* const baseDecl = cast<CXXRecordDecl>(type->getDecl()->getDefinition());
      if (baseDecl) {
        AppendBaseClassSpecifiers(baseIt, bases);
        bases += ' ';
        AppendClassName(baseDecl, bases);
      } else
        return;
    } else
      return;
  }

  if (bases.length() > 2) //initial ": "
    fOut.Print(bases.c_str());
}

//______________________________________________________________________________
void ClassPrinter::DisplayBasesAsTree(const CXXRecordDecl* classDecl, unsigned nSpaces)const
{
  assert(classDecl != 0 && "DisplayBasesAsTree, 'classDecl' parameter is null");
  assert(classDecl->hasDefinition() == true && "DisplayBasesAsTree, 'classDecl' is invalid");

  assert(fInterpreter != 0 && "DisplayBasesAsTree, fInterpreter is null");
  assert(fVerbose == true && "DisplayBasesAsTree, call in a simplified output");

  std::string textLine;
  // Could trigger deserialization of decls.
  Interpreter::PushTransactionRAII RAII(const_cast<Interpreter*>(fInterpreter));
  for (base_decl_iterator baseIt = classDecl->bases_begin(); baseIt != classDecl->bases_end(); ++baseIt) {
    textLine.assign(nSpaces, ' ');
    const RecordType* const type = baseIt->getType()->getAs<RecordType>();
    if (type) {
      const CXXRecordDecl* const baseDecl = cast<CXXRecordDecl>(type->getDecl()->getDefinition());
      if (baseDecl) {
        AppendBaseClassOffset(fInterpreter->getCI(), classDecl, baseDecl, baseIt->isVirtual(), textLine);
        textLine += ' ';
        AppendBaseClassSpecifiers(baseIt, textLine);
        textLine += ' ';
        AppendClassName(baseDecl, textLine);
        textLine += '\n';

        fOut.Print(textLine.c_str());

        DisplayBasesAsTree(baseDecl, nSpaces + kBaseTreeShift);

        continue;
      }
    }

    textLine += "<no type info for a base found>\n";
    fOut.Print(textLine.c_str());
  }
}

//______________________________________________________________________________
void ClassPrinter::DisplayMemberFunctions(const CXXRecordDecl* classDecl)const
{
  assert(classDecl != 0 && "DisplayMemberFunctions, 'classDecl' parameter is null");

  typedef CXXRecordDecl::method_iterator method_iterator;
  typedef CXXRecordDecl::ctor_iterator ctor_iterator;

  std::string textLine;

  // Could trigger deserialization of decls.
  Interpreter::PushTransactionRAII RAII(const_cast<Interpreter*>(fInterpreter));
  for (ctor_iterator ctor = classDecl->ctor_begin(); ctor != classDecl->ctor_end(); ++ctor) {
    if (ctor->isImplicit())//Compiler-generated.
      continue;

    textLine.clear();
    AppendMemberFunctionLocation(fInterpreter->getCI(), *ctor, textLine);
    textLine += ' ';
    AppendMemberAccessSpecifier(*ctor, textLine);
    textLine += ' ';
    AppendConstructorSignature(dyn_cast<CXXConstructorDecl>(*ctor), textLine);
    textLine += ";\n";
    fOut.Print(textLine.c_str());
  }

  for (method_iterator method = classDecl->method_begin(); method != classDecl->method_end(); ++method) {
    if (method->getKind() == Decl::CXXConstructor)
      continue;

    if (method->isImplicit())//Compiler-generated.
      continue;

    textLine.clear();
    AppendMemberFunctionLocation(fInterpreter->getCI(), *method, textLine);
    textLine += ' ';
    AppendMemberAccessSpecifier(*method, textLine);
    textLine += ' ';
    AppendMemberFunctionSignature(*method, textLine);
    textLine += ";\n";
    fOut.Print(textLine.c_str());
  }

  //Now, the problem: template member-functions are not in the list of methods.
  //I have to additionally scan class declarations.
  for (decl_iterator decl = classDecl->decls_begin(); decl != classDecl->decls_end(); ++decl) {
    if (decl->getKind() == Decl::FunctionTemplate) {
      const FunctionTemplateDecl* const ftDecl = dyn_cast<FunctionTemplateDecl>(*decl);
      assert(ftDecl != 0 && "DisplayMemberFunctions, decl is not a function template");

      textLine.clear();
      AppendMemberFunctionLocation(fInterpreter->getCI(), *decl, textLine);
      textLine += ' ';
      AppendMemberAccessSpecifier(*decl, textLine);
      textLine += ' ';
      AppendMemberFunctionSignature(*decl, textLine);

      //Unless this is fixed in clang, I have to do a stupid trick here to
      //print constructor-template correctly, otherwise, class name and
      //parameters are omitted (this is also true for clang and normal, non-templated
      //constructors.
      if (const FunctionDecl* const funcDecl = ftDecl->getTemplatedDecl()) {
        if (const CXXConstructorDecl* const ctorDecl = dyn_cast<CXXConstructorDecl>(funcDecl)) {
          textLine += ' ';
          AppendConstructorSignature(ctorDecl, textLine);
        }
      }

      textLine += ";\n";
      fOut.Print(textLine.c_str());
    }
  }
}

//______________________________________________________________________________
void ClassPrinter::DisplayDataMembers(const CXXRecordDecl* classDecl, unsigned nSpaces)const
{
  assert(classDecl != 0 && "DisplayDataMembers, 'classDecl' parameter is null");

  typedef RecordDecl::field_iterator field_iterator;
  typedef EnumDecl::enumerator_iterator enumerator_iterator;

  //
  const LangOptions langOpts;
  PrintingPolicy printingPolicy(langOpts);
  printingPolicy.SuppressSpecifiers = false;
  printingPolicy.SuppressInitializers = true;
  //

  std::string textLine;
  const std::string gap(std::max(nSpaces, 1u), ' ');

  // Could trigger deserialization of decls.
  Interpreter::PushTransactionRAII RAII(const_cast<Interpreter*>(fInterpreter));
  for (field_iterator field = classDecl->field_begin();
       field != classDecl->field_end(); ++field) {
    textLine.clear();
    AppendDeclLocation(fInterpreter->getCI(), *field, textLine);
    textLine += gap;
    AppendDataMemberOffset(fInterpreter->getCI(), classDecl, *field, textLine);
    textLine += ' ';
    AppendMemberAccessSpecifier(*field, textLine);
    textLine += ' ';
    AppendObjectDeclaration(*field, printingPolicy, true, textLine);
    if (HasUDT(*field)) {
      textLine += ", size = ";
      AppendUDTSize(fInterpreter->getCI(), *field, textLine);
      textLine += '\n';
      fOut.Print(textLine.c_str());
      ProcessTypeOfMember(*field, nSpaces + kMemberTypeShift);
    } else {
      textLine += "\n";
      fOut.Print(textLine.c_str());
    }
  }

  //Now the problem: static data members are not fields, enumerators are not fields.
  for (decl_iterator decl = classDecl->decls_begin(); decl != classDecl->decls_end(); ++decl) {
    if (decl->getKind() == Decl::Enum) {
      const EnumDecl* enumDecl = dyn_cast<EnumDecl>(*decl);
      assert(enumDecl != 0 && "DisplayDataMembers, decl->getKind() == Enum, but decl is not a EnumDecl");
      //it's not really clear, if I should really check this.
      if (enumDecl->isComplete() && (enumDecl = enumDecl->getDefinition())) {
        //if (fSeenDecls.find(enumDecl) == fSeenDecls.end()) {
        //  fSeenDecls.insert(enumDecl);
        for (enumerator_iterator enumerator = enumDecl->enumerator_begin();
             enumerator != enumDecl->enumerator_end(); ++enumerator) {
          //
          textLine.clear();
          AppendDeclLocation(fInterpreter->getCI(), *enumerator, textLine);
          textLine += gap;
          textLine += "0x0      ";//offset is meaningless.

          AppendMemberAccessSpecifier(*enumerator, textLine);
          textLine += ' ';
          //{//Block to force flush for stream.
          //llvm::raw_string_ostream stream(textLine);
          const QualType type(enumerator->getType());
          //const LangOptions lo;
          //PrintingPolicy pp(lo);
          //pp.SuppressScope = true;
          //type.print(stream, pp);
          textLine += type.getAsString();
          //}
          textLine += ' ';
          //SuppressInitializer does not help with PrintingPolicy,
          //so I have to use getNameAsString.
          textLine += enumerator->getNameAsString();
          textLine += "\n";
          fOut.Print(textLine.c_str());
        }
        //}
      }
    } else if (decl->getKind() == Decl::Var) {
      const VarDecl* const varDecl = dyn_cast<VarDecl>(*decl);
      assert(varDecl != 0 && "DisplayDataMembers, decl->getKind() == Var, but decl is not a VarDecl");
      if (varDecl->getStorageClass() == SC_Static) {
        //I hope, this is a static data-member :)
        textLine.clear();
        AppendDeclLocation(fInterpreter->getCI(), varDecl, textLine);
        textLine += gap;
        textLine += "0x0      ";//offset is meaningless.
        AppendMemberAccessSpecifier(varDecl, textLine);
        textLine += ' ';
        AppendObjectDeclaration(varDecl, printingPolicy, true, textLine);
        if (HasUDT(varDecl)) {
          textLine += ", size = ";
          AppendUDTSize(fInterpreter->getCI(), varDecl, textLine);
          textLine += '\n';
          fOut.Print(textLine.c_str());
          ProcessTypeOfMember(varDecl, nSpaces + kMemberTypeShift);
        } else {
          textLine += "\n";
          fOut.Print(textLine.c_str());
        }
      }
    }
  }
}

//Aux. class to display global objects, macros (object-like), enumerators.

class GlobalsPrinter {
public:
  GlobalsPrinter(llvm::raw_ostream& stream, const class cling::Interpreter* interpreter);

  void DisplayGlobals()const;
  void DisplayGlobal(const std::string& name)const;

private:

  void DisplayVarDecl(const VarDecl* varDecl)const;
  void DisplayEnumeratorDecl(const EnumConstantDecl* enumerator)const;
  void DisplayObjectLikeMacro(const IdentifierInfo* identifierInfo, const MacroInfo* macroInfo)const;

  FILEPrintHelper fOut;
  const cling::Interpreter* fInterpreter;

  mutable std::set<const Decl*> fSeenDecls;
};

//______________________________________________________________________________
GlobalsPrinter::GlobalsPrinter(llvm::raw_ostream& stream, const cling::Interpreter* interpreter)
           : fOut(stream),
             fInterpreter(interpreter)
{
  assert(interpreter != 0 && "GlobalsPrinter, 'compiler' parameter is null");
}

//______________________________________________________________________________
void GlobalsPrinter::DisplayGlobals()const
{
  typedef EnumDecl::enumerator_iterator enumerator_iterator;
  typedef Preprocessor::macro_iterator macro_iterator;

  assert(fInterpreter != 0 && "DisplayGlobals, fInterpreter is null");

  const CompilerInstance* const compiler = fInterpreter->getCI();
  assert(compiler != 0 && "DisplayGlobals, compiler instance is null");

  const TranslationUnitDecl* const tuDecl = compiler->getASTContext().getTranslationUnitDecl();
  assert(tuDecl != 0 && "DisplayGlobals, translation unit is empty");

  // Could trigger deserialization of decls.
  Interpreter::PushTransactionRAII RAII(const_cast<Interpreter*>(fInterpreter));

  //Try to print global macro definitions (object-like only).
  const Preprocessor& pp = compiler->getPreprocessor();
  for (macro_iterator macro = pp.macro_begin(); macro != pp.macro_end(); ++macro) {
    if (macro->second->getMacroInfo()
        && macro->second->getMacroInfo()->isObjectLike())
      DisplayObjectLikeMacro(macro->first, macro->second->getMacroInfo());
  }

  //TODO: fSeenDecls - should I check that some declaration is already visited?
  //It's obviously that for objects we can have one definition and any number
  //of declarations, should I print them?

  for (decl_iterator decl = tuDecl->decls_begin(); decl != tuDecl->decls_end(); ++decl) {
    if (const VarDecl* const varDecl = dyn_cast<VarDecl>(*decl))
      DisplayVarDecl(varDecl);
    else if (const EnumDecl* enumDecl = dyn_cast<EnumDecl>(*decl)) {
      //it's not really clear, if I should really check this:
      if (enumDecl->isComplete() && (enumDecl = enumDecl->getDefinition())) {
        for (enumerator_iterator enumerator = enumDecl->enumerator_begin();
             enumerator != enumDecl->enumerator_end(); ++enumerator)
          DisplayEnumeratorDecl(*enumerator);
      }
    }
  }
}

//______________________________________________________________________________
void GlobalsPrinter::DisplayGlobal(const std::string& name)const
{
  typedef EnumDecl::enumerator_iterator enumerator_iterator;
  typedef Preprocessor::macro_iterator macro_iterator;

  //TODO: is it ok to compare 'name' with decl->getNameAsString() ??

  assert(fInterpreter != 0 && "DisplayGlobal, fInterpreter is null");

  const CompilerInstance* const compiler = fInterpreter->getCI();
  assert(compiler != 0 && "DisplayGlobal, compiler instance is null");

  const TranslationUnitDecl* const tuDecl = compiler->getASTContext().getTranslationUnitDecl();
  assert(tuDecl != 0 && "DisplayGlobal, translation unit is empty");

  //fSeenDecls.clear();
  bool found = false;

  // Could trigger deserialization of decls.
  Interpreter::PushTransactionRAII RAII(const_cast<Interpreter*>(fInterpreter));
  const Preprocessor& pp = compiler->getPreprocessor();
  for (macro_iterator macro = pp.macro_begin(); macro != pp.macro_end(); ++macro) {
    if (macro->second->getMacroInfo()
        && macro->second->getMacroInfo()->isObjectLike()) {
      if (name == macro->first->getName().data()) {
        DisplayObjectLikeMacro(macro->first, macro->second->getMacroInfo());
        found = true;
      }
    }
  }

  for (decl_iterator decl = tuDecl->decls_begin(); decl != tuDecl->decls_end(); ++decl) {
    if (const VarDecl* const varDecl = dyn_cast<VarDecl>(*decl)) {
      if (varDecl->getNameAsString() == name) {
        DisplayVarDecl(varDecl);
        found = true;
      }
    } else if (const EnumDecl* enumDecl = dyn_cast<EnumDecl>(*decl)) {
      //it's not really clear, if I should really check this:
      if (enumDecl->isComplete() && (enumDecl = enumDecl->getDefinition())) {
        for (enumerator_iterator enumerator = enumDecl->enumerator_begin();
             enumerator != enumDecl->enumerator_end(); ++enumerator) {
          if (enumerator->getNameAsString() == name) {
            DisplayEnumeratorDecl(*enumerator);
            found = true;
          }
        }
      }
    }
  }

  //Do as CINT does:
  if (!found)
    fOut.Print(("Variable " + name + " not found\n").c_str());
}

//______________________________________________________________________________
void GlobalsPrinter::DisplayVarDecl(const VarDecl* varDecl) const
{
  assert(fInterpreter != 0 && "DisplayVarDecl, fInterpreter is null");
  assert(varDecl != 0 && "DisplayVarDecl, 'varDecl' parameter is null");

  const LangOptions langOpts;
  PrintingPolicy printingPolicy(langOpts);
  printingPolicy.SuppressSpecifiers = false;
  printingPolicy.SuppressInitializers = false;

  std::string textLine;

  AppendDeclLocation(fInterpreter->getCI(), varDecl, textLine);

  //TODO:
  //Hehe, it's strange to expect addresses from an AST :)
  //Add it, if you know how, I don't care.
  textLine += " (address: NA) ";

  AppendObjectDeclaration(varDecl, printingPolicy, false, textLine);

  if (HasUDT(varDecl)) {
    textLine += ", size = ";
    AppendUDTSize(fInterpreter->getCI(), varDecl, textLine);
  }

  textLine += "\n";
  fOut.Print(textLine.c_str());
}

//______________________________________________________________________________
void GlobalsPrinter::DisplayEnumeratorDecl(const EnumConstantDecl* enumerator)const
{
  assert(fInterpreter != 0 && "DisplayEnumeratorDecl, fInterpreter is null");
  assert(enumerator != 0 && "DisplayEnumeratorDecl, 'enumerator' parameter is null");

  const LangOptions langOpts;
  PrintingPolicy printingPolicy(langOpts);
  printingPolicy.SuppressInitializers = false;

  std::string textLine;

  AppendDeclLocation(fInterpreter->getCI(), enumerator, textLine);

  textLine += " (address: NA) ";//No address, that's an enumerator.

  const QualType type(enumerator->getType());
  textLine += type.getAsString();
  textLine += ' ';

  AppendObjectDeclaration(enumerator, printingPolicy, false, textLine);

  textLine += "\n";
  fOut.Print(textLine.c_str());
}

//______________________________________________________________________________
void GlobalsPrinter::DisplayObjectLikeMacro(const IdentifierInfo* identifierInfo,
                                            const MacroInfo* macroInfo)const
{
  assert(identifierInfo != 0 && "DisplayObjectLikeMacro, 'identifierInfo' parameter is null");
  assert(macroInfo != 0 && "DisplayObjectLikeMacro, 'macroInfo' parameter is null");

  std::string textLine;

  AppendMacroLocation(fInterpreter->getCI(), macroInfo, textLine);

  textLine += " (address: NA) #define ";//No address exists for a macro definition.

  textLine += identifierInfo->getName().data();

  if (macroInfo->getNumTokens())
    textLine += " =";

  assert(fInterpreter->getCI() != 0 && "DisplayObjectLikeMacro, compiler instance is null");
  const Preprocessor &pp = fInterpreter->getCI()->getPreprocessor();

  for (unsigned i = 0, e = macroInfo->getNumTokens(); i < e; ++i) {
    textLine += ' ';
    textLine += pp.getSpelling(macroInfo->getReplacementToken(i));
  }

  fOut.Print(textLine.c_str());
  fOut.Print("\n");
}

//Aux. class traversing TU and printing namespaces.
class NamespacePrinter {
public:
   NamespacePrinter(llvm::raw_ostream& stream, const Interpreter* interpreter);

   void Print()const;

private:
   void ProcessNamespaceDeclaration(decl_iterator decl,
                                    const std::string& enclosingNamespaceName)const;

   FILEPrintHelper fOut;
   const cling::Interpreter* fInterpreter;
};

//______________________________________________________________________________
NamespacePrinter::NamespacePrinter(llvm::raw_ostream& stream,
                                   const Interpreter* interpreter)
                     : fOut(stream),
                       fInterpreter(interpreter)
{
   assert(interpreter != nullptr &&
          "NamespacePrinter, parameter 'interpreter' is null");
}

//______________________________________________________________________________
void NamespacePrinter::Print()const
{
  assert(fInterpreter != nullptr && "Print, fInterpreter is null");

  const auto compiler = fInterpreter->getCI();
  assert(compiler != nullptr && "Print, compiler instance is null");

  const auto tu = compiler->getASTContext().getTranslationUnitDecl();
  assert(tu != nullptr && "Print, translation unit is null");

  const std::string globalNSName;

  fOut.Print("List of namespaces\n");
  for (auto it = tu->decls_begin(), eIt = tu->decls_end(); it != eIt; ++it) {
    if (it->getKind() == Decl::Namespace || it->getKind() == Decl::NamespaceAlias)
      ProcessNamespaceDeclaration(it, globalNSName);
  }
}

//______________________________________________________________________________
void NamespacePrinter::ProcessNamespaceDeclaration(decl_iterator declIt,
                                const std::string& enclosingNamespaceName)const
{
  assert(fInterpreter != 0 &&
         "ProcessNamespaceDeclaration, fInterpreter is null");
  assert(*declIt != 0 &&
         "ProcessNamespaceDeclaration, parameter 'decl' is not a valid iterator");

  if (const auto nsDecl = dyn_cast<NamespaceDecl>(*declIt)) {
    if (nsDecl->isAnonymousNamespace())
      return;//TODO: invent some name?

    std::string name(enclosingNamespaceName);
    if (enclosingNamespaceName.length())
      name += "::";
    name += nsDecl->getNameAsString();

    if (nsDecl->isOriginalNamespace()) {
      fOut.Print(name.c_str());
      fOut.Print("\n");
    }

    if (const auto ctx = dyn_cast<DeclContext>(*declIt)) {
      for (auto it = ctx->decls_begin(), eIt = ctx->decls_end(); it != eIt; ++it) {
        if (it->getKind() == Decl::Namespace ||
            it->getKind() == Decl::NamespaceAlias)
          ProcessNamespaceDeclaration(it, name);
      }
    }//TODO: else diagnostic?
  } else if (const auto alDecl = dyn_cast<NamespaceAliasDecl>(*declIt)) {
    if (enclosingNamespaceName.length()) {
      fOut.Print((enclosingNamespaceName + "::" +
                  alDecl->getNameAsString()).c_str());
    } else
      fOut.Print(alDecl->getNameAsString().c_str());

    fOut.Print("\n");
  }
}

//Print typedefs.
class TypedefPrinter {
public:
  TypedefPrinter(llvm::raw_ostream& stream, const Interpreter* interpreter);

  void DisplayTypedefs()const;
  void DisplayTypedef(const std::string& name)const;

private:

  void ProcessNestedDeclarations(const DeclContext* decl)const;
  void ProcessDecl(decl_iterator decl) const;

  void DisplayTypedefDecl(TypedefNameDecl* typedefDecl)const;

  FILEPrintHelper fOut;
  const cling::Interpreter* fInterpreter;

  //mutable std::set<const Decl*> fSeenDecls;
};

//______________________________________________________________________________
TypedefPrinter::TypedefPrinter(llvm::raw_ostream& stream, const Interpreter* interpreter)
                  : fOut(stream),
                    fInterpreter(interpreter)
{
  assert(interpreter != 0 && "TypedefPrinter, parameter 'interpreter' is null");
}

//______________________________________________________________________________
void TypedefPrinter::DisplayTypedefs()const
{
  assert(fInterpreter != 0 && "DisplayTypedefs, fInterpreter is null");

  const CompilerInstance* const compiler = fInterpreter->getCI();
  assert(compiler != 0 && "DisplayTypedefs, compiler instance is null");

  const TranslationUnitDecl* const tuDecl = compiler->getASTContext().getTranslationUnitDecl();
  assert(tuDecl != 0 && "DisplayTypedefs, translation unit is empty");

  //fSeenDecls.clear();

  fOut.Print("List of typedefs");
  ProcessNestedDeclarations(tuDecl);
}

//______________________________________________________________________________
void TypedefPrinter::DisplayTypedef(const std::string& typedefName)const
{
  assert(fInterpreter != 0 && "DisplayTypedef, fInterpreter is null");

  const cling::LookupHelper &lookupHelper = fInterpreter->getLookupHelper();
  const QualType type
    = lookupHelper.findType(typedefName, cling::LookupHelper::NoDiagnostics);

  if(!type.isNull()) {
    if (const TypedefType* const typedefType = type->getAs<TypedefType>()) {
      if (typedefType->getDecl()) {
        DisplayTypedefDecl(typedefType->getDecl());
        return;
      } else
        fOut.Print(("A " + std::string(type->getTypeClassName()) + " declaration"
                    " was found for " + typedefName + "\n").c_str());

    }
  }

  fOut.Print(("Type " + typedefName + " is not defined\n").c_str());
}

//______________________________________________________________________________
void TypedefPrinter::ProcessNestedDeclarations(const DeclContext* decl)const
{
  assert(decl != 0 && "ProcessNestedDeclarations, parameter 'decl' is null");
  // Could trigger deserialization of decls.
  Interpreter::PushTransactionRAII RAII(const_cast<Interpreter*>(fInterpreter));
  for (decl_iterator it = decl->decls_begin(), eIt = decl->decls_end(); it != eIt; ++it)
    ProcessDecl(it);
}

//______________________________________________________________________________
void TypedefPrinter::ProcessDecl(decl_iterator decl)const
{
  assert(fInterpreter != 0 && "ProcessDecl, fInterpreter is null");
  assert(*decl != 0 && "ProcessDecl, parameter 'decl' is not a valid iterator");

  switch (decl->getKind()) {
  case Decl::Typedef:
    DisplayTypedefDecl(dyn_cast<TypedefDecl>(*decl));
    break;
  case Decl::Namespace:
  case Decl::Block:
  case Decl::Function:
  case Decl::CXXMethod:
  case Decl::CXXConstructor:
  case Decl::CXXConversion:
  case Decl::CXXDestructor:
  case Decl::LinkageSpec:
  case Decl::CXXRecord:
  case Decl::ClassTemplateSpecialization:
  //case Decl::ClassTemplatePartialSpecialization:
    ProcessNestedDeclarations(dyn_cast<DeclContext>(*decl));
    break;
  default:
    if (FunctionDecl * const funDecl = dyn_cast<FunctionDecl>(*decl))
      ProcessNestedDeclarations(funDecl);
    break;
  }
}

//______________________________________________________________________________
void TypedefPrinter::DisplayTypedefDecl(TypedefNameDecl* typedefDecl)const
{
  assert(typedefDecl != 0
         && "DisplayTypedefDecl, parameter 'typedefDecl' is null");
  assert(fInterpreter != 0 && "DisplayTypedefDecl, fInterpreter is null");

  std::string textLine;
  AppendDeclLocation(fInterpreter->getCI(), typedefDecl, textLine);

  textLine += " typedef ";
  {
    const LangOptions langOpts;
    PrintingPolicy printingPolicy(langOpts);
    printingPolicy.SuppressSpecifiers = false;
    printingPolicy.SuppressInitializers = true;
    printingPolicy.SuppressScope = false;
    printingPolicy.SuppressTagKeyword = true;
    llvm::raw_string_ostream out(textLine);
    typedefDecl->getUnderlyingType().
       getDesugaredType(typedefDecl->getASTContext()).print(out,printingPolicy);
    //Name for diagnostic will include template arguments if any.
    typedefDecl->getNameForDiagnostic(out,
                                      printingPolicy,/*qualified=*/true);
  }

  fOut.Print(textLine.c_str());
  fOut.Print("\n");
}

}//unnamed namespace

//______________________________________________________________________________
void DisplayClasses(llvm::raw_ostream& stream, const Interpreter* interpreter,
                    bool verbose)
{
  assert(interpreter != 0 && "DisplayClasses, 'interpreter' parameter is null");

  ClassPrinter printer(stream, interpreter);
  printer.SetVerbose(verbose);
  printer.DisplayAllClasses();
}

//______________________________________________________________________________
void DisplayClass(llvm::raw_ostream& stream, const Interpreter* interpreter,
                  const char* className, bool verbose)
{
  assert(interpreter != 0 && "DisplayClass, 'interpreter' parameter is null");
  assert(className != 0 && "DisplayClass, 'className' parameter is null");

  while (std::isspace(*className))
    ++className;

  ClassPrinter printer(stream, interpreter);

  if (*className) {
    printer.SetVerbose(verbose);
    printer.DisplayClass(className);
  } else {
    printer.SetVerbose(true);//?
    printer.DisplayAllClasses();
  }
}

//______________________________________________________________________________
void DisplayNamespaces(llvm::raw_ostream &stream, const Interpreter *interpreter)
{
  assert(interpreter != 0 && "DisplayNamespaces, parameter 'interpreter' is null");
  Interpreter::PushTransactionRAII RAII(const_cast<Interpreter*>(interpreter));

  NamespacePrinter printer(stream, interpreter);
  Interpreter::PushTransactionRAII guard(const_cast<Interpreter *>(interpreter));
  printer.Print();
}

//______________________________________________________________________________
void DisplayGlobals(llvm::raw_ostream& stream, const Interpreter* interpreter)
{
  assert(interpreter != 0 && "DisplayGlobals, 'interpreter' parameter is null");

  GlobalsPrinter printer(stream, interpreter);
  // Could trigger deserialization of decls.
  Interpreter::PushTransactionRAII RAII(const_cast<Interpreter*>(interpreter));
  printer.DisplayGlobals();
}

//______________________________________________________________________________
void DisplayGlobal(llvm::raw_ostream& stream, const Interpreter* interpreter,
                   const std::string& name)
{
  assert(interpreter != 0 && "DisplayGlobal, 'interpreter' parameter is null");

  GlobalsPrinter printer(stream, interpreter);
  // Could trigger deserialization of decls.
  Interpreter::PushTransactionRAII RAII(const_cast<Interpreter*>(interpreter));
  printer.DisplayGlobal(name);
}

//______________________________________________________________________________
void DisplayTypedefs(llvm::raw_ostream &stream, const Interpreter *interpreter)
{
   assert(interpreter != 0 && "DisplayTypedefs, parameter 'interpreter' is null");

   TypedefPrinter printer(stream, interpreter);
   // Could trigger deserialization of decls.
   Interpreter::PushTransactionRAII RAII(const_cast<Interpreter*>(interpreter));
   printer.DisplayTypedefs();
}

//______________________________________________________________________________
void DisplayTypedef(llvm::raw_ostream &stream, const Interpreter *interpreter,
                    const std::string &name)
{
   assert(interpreter != 0 && "DisplayTypedef, parameter 'interpreter' is null");

   TypedefPrinter printer(stream, interpreter);
   printer.DisplayTypedef(name);
}


}//namespace cling
