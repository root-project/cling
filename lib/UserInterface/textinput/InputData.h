//===--- InputData.h - Normalized Inputs ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines input, abtracted and normalized, whatever the platform
//  or input device.
//
//  Axel Naumann <axel@cern.ch>, 2011-05-12
//===----------------------------------------------------------------------===//

#ifndef TEXTINPUT_INPUTDATA_H
#define TEXTINPUT_INPUTDATA_H

namespace textinput {

  // Normalization of input data.
  class InputData {
  public:
    // Non-character input
    enum EExtendedInput {
      kEIUninitialized,
      kEIHome,
      kEIEnd,
      kEIUp,
      kEIDown,
      kEILeft,
      kEIRight,
      kEIPgUp,
      kEIPgDown,
      kEIBackSpace,
      kEIDel,
      kEIIns,
      kEITab,
      kEIEnter,
      kEIEsc,
      kEIF1,
      kEIF2,
      kEIF3,
      kEIF4,
      kEIF5,
      kEIF6,
      kEIF7,
      kEIF8,
      kEIF9,
      kEIF10,
      kEIF11,
      kEIF12,
      kEIEOF,
      kEIResizeEvent,
      kEIIgnore
    };

    // Input modifier keys, bitset
    enum EInputModifier {
      kModNone = 0,
      kModShift = 1,
      kModCtrl = 2,
      kModAlt = 4
    };

    // Flag (bit) to signal raw input
    enum {
      kIsRaw = 0x80
    };

    InputData(): fExt(kEIUninitialized), fMod(0) {}
    InputData(int ch, char mod = 0): fRaw(ch), fMod(mod | kIsRaw) {}

    bool IsRaw() const { return (fMod & kIsRaw) != 0; }
    int GetRaw() const { return fRaw; }

    EExtendedInput GetExtendedInput() const { return fExt; }
    unsigned char GetModifier() const { return fMod & ~kIsRaw; }

    void SetRaw(char R) { fRaw = R; fMod |= kIsRaw; }
    void SetExtended(EExtendedInput E) { fExt = E;  fMod &= ~kIsRaw; }
    void SetModifier(char M) { fMod = M | (fMod & kIsRaw); }

  private:
    union {
      char fRaw; // raw input character, if kIsRaw & fMod
      EExtendedInput fExt; // non-character input
    };
    unsigned char fMod; // Modifiers, also stores union descriminator (kIsRaw)
  };
}
#endif // TEXTINPUT_INPUTDATA_H
