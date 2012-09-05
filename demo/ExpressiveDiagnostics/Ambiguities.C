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
