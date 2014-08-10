namespace test { //implicit instantiation
  template<bool B, class T, class F>
  struct conditional { typedef T type; };

  template<class T, class F>
  struct conditional<false, T, F> { typedef F type; };

  template <typename _Tp> using example = typename conditional<true,int,float>::type;

}//end namespace test

namespace test { //nested name specifier
  class HasSubType {
  public:
    class SubType {};
  };
  HasSubType::SubType FunctionUsingSubtype(HasSubType::SubType s){return s;}
  extern HasSubType::SubType variable;//locale::id id

}//end namespace test // This problem is bypassed by skipping types containing "::"

namespace test { //restrict keyword: try include/mmprivate.h and strlcpy.h when fixed
  typedef long ssize_t;
  typedef unsigned int size_t;
  //Has signature of readlink from unistd.h
  extern ssize_t FunctionUsingRestrictPtr (const char *__restrict __path,
             char *__restrict __buf, size_t __len);
}//end namespace test // This is bypassed by forcibly removing restrict from types

// namespace test { //default template arg
//   template <typename T,int MAX=100> class Stack {
//   };
//   Stack<int> FunctionReturningStack(){return Stack<int>();}
// }//end namespace test // Fixed with callback, strip old default args before including new file

// namespace test {
// //#include<tuple> //'tie' function
// //commented out to skip huge output
// } //Fixed bug in VisitFunctionDecl

