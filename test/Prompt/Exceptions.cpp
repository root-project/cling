struct TheStruct {
  char m_Storage[100];
};

TheStruct getStructButThrows() { throw "ABC!"; }
getStructButThrows()
