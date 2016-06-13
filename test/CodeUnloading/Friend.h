class Test {
  friend void bestFriend();
  friend void testFriend();
  friend class FC;
  class FCD {};
  friend class FCD;
  template <class T> friend class FCT;
  template <class T> friend void testFriendT();
};
