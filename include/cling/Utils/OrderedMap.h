//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author: Roman Zulak
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_ORDERED_MAP_H
#define CLING_ORDERED_MAP_H

#include <algorithm>
#include <cassert>
#include <unordered_map>
#include <vector>

namespace cling {
namespace utils {

///\brief Thin wrapper class for tracking the order of insertion into a
/// std::unordered_map.
///
/// Only supports 'emplace' and '[Key]' for insertion of values, and adds an
/// additional parameter to 'erase' so that a mapped value can be moved into
/// local storage before erasing the iterator occurs.
///
template <typename Key, typename Value> class OrderedMap {
  typedef std::unordered_map<Key, Value> map_t;
  // Would this be faster as a std::unoredered_map<Key, size_t> for erasure?
  typedef std::vector<typename map_t::const_iterator> order_t;

  map_t m_Map;
  order_t m_Order;

public:
  typedef typename map_t::iterator iterator;
  typedef typename map_t::const_iterator const_iterator;
  typedef typename map_t::mapped_type mapped_type;

  template <typename... Args>
  std::pair<iterator, bool> emplace(Args&&... args) {
    auto Rval = m_Map.emplace(std::forward<Args>(args)...);
    if (Rval.second) m_Order.emplace_back(Rval.first);
    return Rval;
  }

  Value& operator[](const Key& K) {
    iterator Itr = find(K);
    if (Itr == end()) {
      Itr = m_Map.emplace(K, Value()).first;
      m_Order.emplace_back(Itr);
    }
    return Itr->second;
  }

  Value& operator[](Key&& K) {
    iterator Itr = find(K);
    if (Itr == end()) {
      Itr = m_Map.emplace(K, Value()).first;
      m_Order.emplace_back(Itr);
    }
    return Itr->second;
  }

  ///\brief Erase a mapping from this object.
  ///
  ///\param [in] Itr - The iterator to erase.
  ///\param [out] Move - Move the mapped object to this pointer before erasing.
  ///
  void erase(const_iterator Itr, mapped_type* Move = nullptr) {
    assert(std::find(m_Order.begin(), m_Order.end(), Itr) != m_Order.end());
    for (auto Otr = m_Order.begin(), End = m_Order.end(); Otr != End; ++Otr) {
      if (Itr == *Otr) {
        m_Order.erase(Otr);
        break;
      }
    }
    assert(std::find(m_Order.begin(), m_Order.end(), Itr) == m_Order.end());
    if (Move) *Move = std::move(Itr->second);
    m_Map.erase(Itr);
  }

  iterator find(const Key& K) { return m_Map.find(K); }
  const_iterator find(const Key& K) const { return m_Map.find(K); }

  iterator end() { return m_Map.end(); }
  const_iterator end() const { return m_Map.end(); }

  void swap(OrderedMap& Other) {
    m_Map.swap(Other.m_Map);
    m_Order.swap(Other.m_Order);
  }

  void clear() {
    m_Map.clear();
    m_Order.clear();
  }

  bool empty() const {
    assert(m_Map.empty() == m_Order.empty() && "Not synchronized");
    return m_Order.empty();
  }

  void size() const {
    assert(m_Map.size() == m_Order.size() && "Not synchronized");
    return m_Order.size();
  }

  const order_t& ordered() const { return m_Order; }
  order_t& ordered() { return m_Order; }
};

} // namespace utils
} // namespace cling

#endif // CLING_PLATFORM_H
