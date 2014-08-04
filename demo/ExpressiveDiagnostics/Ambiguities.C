//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// The demo shows ambiguities in the virtual tables and the nice, expressive
// and accurate errors from clang.
// Author: Vassil Vassilev <vvasilev@cern.ch>

class Person {
public:
  virtual Person* Clone() const;
};

class Student : public Person {
public:
};

class Teacher : public Person {
public:
};

class TeachingAssistant : public Student, Teacher {
public:
  virtual TeachingAssistant* Clone() const;
};

void Ambiguities() {}
