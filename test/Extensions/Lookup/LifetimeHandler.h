//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include <cstring>

extern "C" int printf(const char* fmt, ...);

class Alpha {
private:
  const char* m_Var;
public:
  Alpha(): m_Var(0) {printf("Alpha's default ctor called\n");}
  Alpha(const char* n): m_Var(n) {
    printf("Alpha's single arg ctor called %s\n", n);
  }
  Alpha(char* n1, char* n2): m_Var(strcat(n1, n2)) {
    printf("Alpha's double arg ctor called %s\n", m_Var);
  }
  ~Alpha() { printf("Alpha dtor called %s\n", m_Var); }
  const char* getVar() { return m_Var; }
  void printNext(){
    printf("After Alpha is Beta %s\n", m_Var);
  }
};

void LifetimeHandler() {
  int i = 5;
  Alpha my(dep->getVersion());
  my.printNext();
  // Alpha a(dep->getVersion(), dep1->Add10(h->Draw() + 1 + i));
  // Alpha b("abc");
  // b.printName();

  // Alpha a(dep->Call(i));
  // Alpha c(const_cast<char*>(dep->getVersion()), const_cast<char*>(dep1->getVersion()));
  // c.printName();
}
