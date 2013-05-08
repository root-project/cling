// @(#)root/editline:$Id$
// Author: Mary-Louise Gill, 2009

/*************************************************************************
 * Copyright (C) 1995-2011, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "Getline_color.h"

#include <stack>
#include <string>

#include "TClass.h"
#include "TClassTable.h"
#include "TInterpreter.h"
#include "TROOT.h"
#include "textinput/Range.h"
#include "textinput/Text.h"

using namespace std;
using namespace textinput;

namespace {

   Color ColorFromName(const char* name) {
      // Convert a color name to a Color.
      Color ret;

      std::string lowname(name);
      size_t lenname = strlen(name);
      for (size_t i = 0; i < lenname; ++i)
         lowname[i] = tolower(lowname[i]);

      if (lowname.find("bold") != std::string::npos
          || lowname.find("light") != std::string::npos)
         ret.fModifiers |= Color::kModBold;
      if (lowname.find("under") != std::string::npos)
         ret.fModifiers |= Color::kModUnderline;

      size_t poshash = lowname.find('#');
      size_t lenrgb = 0;
      if (poshash != std::string::npos) {
         int endrgb = poshash + 1;
         while ((lowname[endrgb] >= '0' && lowname[endrgb] <= '9')
                || (lowname[endrgb] >= 'a' && lowname[endrgb] <= 'f')) {
            ++endrgb;
         }
         lenrgb = endrgb - poshash - 1;
      }

      int rgb[3] = {0};
      if (lenrgb == 3) {
         for (int i = 0; i < 3; ++i) {
            rgb[i] = lowname[poshash + 1 + i] - '0';
            if (rgb[i] > 9) {
               rgb[i] = rgb[i] + '0' - 'a' + 10;
            }
            rgb[i] *= 16; // only upper 4 bits are set.
         }
         return ret;
      } else if (lenrgb == 6) {
         for (int i = 0; i < 6; ++i) {
            int v = lowname[poshash + 1 + i] - '0';
            if (v > 9) {
               v = v + '0' - 'a' + 10;
            }
            if (i % 2 == 0) {
               v *= 16;
            }
            rgb[i / 2] += v;
         }
         ret.fR = rgb[0];
         ret.fG = rgb[1];
         ret.fB = rgb[2];
         return ret;
      } else {
         if (lowname.find("default") != std::string::npos) {
            return ret;
         }

         static const char* colornames[] = {
            "black", "red", "green", "yellow",
            "blue", "magenta", "cyan", "white", 0
         };
         static const unsigned char colorrgb[][3] = {
            {0,0,0}, {127,0,0}, {0,127,0}, {127,127,0},
            {0,0,127}, {127,0,127}, {0,127,127}, {127,127,127},
            {0}
         };

         for (int i = 0; colornames[i]; ++i) {
            if (lowname.find(colornames[i]) != std::string::npos) {
               int boldify = 0;
               if (ret.fModifiers & Color::kModBold)
                  boldify = 64;
               ret.fR = colorrgb[i][0] + boldify;
               ret.fG = colorrgb[i][1] + boldify;
               ret.fB = colorrgb[i][2] + boldify;
               return ret;
            }
         }
      }
      fprintf(stderr, "Getline_color/ColorFromName: cannot parse color %s!\n", name);
      return Color();
   } // ColorFromName()

   bool IsAlnum_(char c) { return c == '_' || isalnum(c); }
   bool IsAlpha_(char c) { return c == '_' || isalpha(c); }
} // unnamed namespace


ROOT::TextInputColorizer::TextInputColorizer():
   fColorIsDefault(), fPrevBracketColor(kColorNone) {
   // Set the default colors.
   // fColors[kColorNone] stays default initialized.
   fColors[kColorType] = ColorFromName("blue");
   fColors[kColorTabComp] = ColorFromName("magenta");
   fColors[kColorBracket] = ColorFromName("green");
   fColors[kColorBadBracket] = ColorFromName("red");
   fColors[kColorPrompt] = ColorFromName("default");
   fColorIsDefault[kColorPrompt] = true;
}

ROOT::TextInputColorizer::~TextInputColorizer() {
   // pin vtable
}


void ROOT::TextInputColorizer::ExtendRangeAndSetColor(Text& input,
                                                      size_t idx, char col,
                                                      Range& disp) {
   // Utility function that updates the display modification range if the
   // color at index idx is different from what it was before.

   if (fColorIsDefault[(int)col]) {
      // Never mind the color: use use default.
      col = 0;
   }
   if (input.GetColor(idx) != col) {
      input.SetColor(idx, col);
      disp.Extend(idx);
   }
}

bool ROOT::TextInputColorizer::GetColor(char type, Color& col) {
   // Set the Color corresponding to an entry in EColorTypes.
   // Returns false if the type index is out of range.

   if (type < (int)kNumColors) {
      col = fColors[(size_t)type];
      return true;
   }
   col = Color();
   return false;
}

void ROOT::TextInputColorizer::SetColors(const char* colorType,
                                         const char* colorTabComp,
                                         const char* colorBracket,
                                         const char* colorBadBracket,
                                         const char* colorPrompt) {
   // Set the colors of the different items as either
   // #RGB
   // #RRGGBB or
   // color name, optionally prepended by "underline" or "bold"

   fColors[kColorType] = ColorFromName(colorType);
   fColorIsDefault[kColorType] = (fColors[kColorType] == Color());
   fColors[kColorTabComp] = ColorFromName(colorTabComp);
   fColorIsDefault[kColorTabComp] = (fColors[kColorTabComp] == Color());
   fColors[kColorBracket] = ColorFromName(colorBracket);
   fColorIsDefault[kColorBracket] = (fColors[kColorBracket] == Color());
   fColors[kColorBadBracket] = ColorFromName(colorBadBracket);
   fColorIsDefault[kColorBadBracket] = (fColors[kColorBadBracket] == Color());
   fColors[kColorPrompt] = ColorFromName(colorPrompt);
   fColorIsDefault[kColorPrompt] = (fColors[kColorPrompt] == Color());
}

void ROOT::TextInputColorizer::ProcessTextChange(EditorRange& Modification,
                                                 Text& input) {
   // The text has changed; look for word that are types.

   const std::string& text = input.GetText();

   size_t modStart = Modification.fEdit.fStart;
   size_t inputLength = input.length();

   // Find end of modified word:
   size_t modEnd = Modification.fEdit.fLength;
   if (modEnd == (size_t) -1) {
      modEnd = inputLength;
   } else {
      modEnd += modStart;
      if (modEnd > inputLength) {
         modEnd = inputLength;
      } else {
         while (modEnd < inputLength && IsAlnum_(text[modEnd]))
            ++modEnd;
      }
   }

   // Find beginning of modified word. Don't treat
   // "12ull" specially, it will fall out below.
   while (modStart && IsAlnum_(text[modStart])) --modStart;

   // Ignore spaces
   while (modStart < modEnd && isspace(text[modStart]))
      ++modStart;
   while (modEnd > modStart && isspace(text[modEnd]))
      --modStart;

   for (size_t i = modStart; i < modEnd;) {
      // i points to beginning of word here.
      if (isdigit(text[i])) {
         // "12", or "12ull". Default color.
         ExtendRangeAndSetColor(input, i, 0, Modification.fDisplay);
         ++i;
         while (i < modEnd && IsAlnum_(text[i])) {
            ExtendRangeAndSetColor(input, i, 0, Modification.fDisplay);
            ++i;
         }
      } else if (IsAlpha_(text[i])) {
         // identifier, but is it a type?
         size_t wordLen = 1;
         while (i + wordLen < modEnd && IsAlnum_(text[i + wordLen])) {
            ++wordLen;
         }
         std::string word = text.substr(i, wordLen);
         char color = kColorNone;
         if (gROOT->GetListOfTypes()->FindObject(word.c_str())
             || gClassTable->GetDict(word.c_str())
             || gInterpreter->GetClassSharedLibs(word.c_str())
             || gInterpreter->CheckClassInfo(word.c_str(), false /*autoload*/)) {
            color = kColorType;
         }
         for (size_t ic = i; ic < i + wordLen; ++ic) {
            ExtendRangeAndSetColor(input, ic, color, Modification.fDisplay);
         }
         i += wordLen;
      } else {
         size_t wordLen = 1;
         while (i + wordLen < modEnd && !IsAlnum_(text[i + wordLen]))
            ++wordLen;
         for (size_t ic = i; ic < i + wordLen; ++ic) {
            // protect colored parens
            char oldColor = input.GetColor(ic);
            if (oldColor != kColorBracket && oldColor != kColorBadBracket) {
               ExtendRangeAndSetColor(input, ic, kColorNone, Modification.fDisplay);
            }
         }
         i += wordLen;
      }

      // skip trailing whitespace.
      while (i < modEnd && isspace(text[i])) {
         ExtendRangeAndSetColor(input, i, kColorNone, Modification.fDisplay);
         ++i;
      }
   }
}

void ROOT::TextInputColorizer::ProcessPromptChange(Text& prompt) {
   int idx = kColorPrompt;
   if (fColorIsDefault[kColorPrompt]) {
      idx = 0;
   }
   prompt.SetColor(Range::AllText(), idx);
}

void ROOT::TextInputColorizer::ProcessCursorChange(size_t Cursor,
                                                   Text& input,
                                                   Range& DisplayR) {
   // Check each char to see if it is an opening bracket,
   // if so, check for its closing one and color them green.

   static const int numBrackets = 3;
   static const char bTypes[numBrackets][3] = {"()", "{}", "[]"};

   if (input.empty()) return;

   if (fPrevBracketColor != kColorNone) {
      // Remove previous bracket coloring.
      const char* colors = &input.GetColors()[0];
      const char* prevBracket = (const char*) memchr(colors, fPrevBracketColor,
                                                     input.length());
      if (prevBracket) {
         ExtendRangeAndSetColor(input, prevBracket - colors, kColorNone, DisplayR);
         if (fPrevBracketColor == kColorBracket) {
            // There will be two.
            prevBracket = (const char*) memchr(prevBracket, fPrevBracketColor,
                                               input.length() - (prevBracket - colors));
            if (prevBracket) {
               ExtendRangeAndSetColor(input, prevBracket - colors, kColorNone,
                                      DisplayR);
            }
         }
      }
   }

   // locations of brackets
   stack<size_t> locBrackets;
   int foundParenIdx = -1;
   int parenType = 0;
   const std::string& text = input.GetText();

   if (Cursor < input.length()) {
      // check against each bracket type
      for (; parenType < numBrackets; parenType++) {
         // if current char is equal to opening bracket, push onto stack
         if (text[Cursor] == bTypes[parenType][0]) {
            locBrackets.push(Cursor);
            foundParenIdx = 0;
            break;
         } else if (text[Cursor] == bTypes[parenType][1]) {
            locBrackets.push(Cursor);
            foundParenIdx = 1;
            break;
         }
      }
   }

   // current cursor char is not an open bracket, and there is a previous char
   // to check
   if (foundParenIdx == -1 && Cursor > 0) {
      // check previously typed char for being a closing bracket
      --Cursor;
      // check against each bracket type
      parenType = 0;

      for (; parenType < numBrackets; parenType++) {
         // if current char is equal to closing bracket, push onto stack
         if (text[Cursor] == bTypes[parenType][1]) {
            locBrackets.push(Cursor);
            foundParenIdx = 1;
            break;
         }
      }
   }
   // no bracket found on either current or previous char, return.
   if (foundParenIdx == -1) {
      return;
   }

   // terate through remaining letters until find a matching closing bracket
   // if another open bracket of the same type is found, push onto stack
   // and pop on next closing bracket match
   int direction = 1;

   if (foundParenIdx == 1) {
      direction = -1;
   }

   size_t lenLine = input.length();
   // direction == -1: Cursor - 1 to front.
   size_t scanBegin = (Cursor > 0) ? Cursor - 1 : 0;
   size_t scanLast = 0;
   if (direction == 1) {
      // direction == 1: Cursor + 1 to end.
      scanBegin = Cursor + 1;
      scanLast = lenLine - 1;
      if (scanBegin > scanLast) return;
   }
   for (size_t i = scanBegin; true /*avoid "unsigned >= 0" condition*/; i += direction) {
      // if current char is equal to another opening bracket, push onto stack
      if (text[i] == bTypes[parenType][foundParenIdx]) {
         // push index of bracket
         locBrackets.push(i);
      }
      // if current char is equal to closing bracket
      else if (text[i] == bTypes[parenType][1 - foundParenIdx]) {
         // pop previous opening bracket off stack
         locBrackets.pop();
         // if previous opening was the last entry and we are at the cursor, then highlight match
         if (locBrackets.empty()) {
            ExtendRangeAndSetColor(input, i, kColorBracket, DisplayR);
            ExtendRangeAndSetColor(input, Cursor, kColorBracket, DisplayR);
            fPrevBracketColor = kColorBracket;
            break;
         }
      }
      // loop termination check before possible underflow ("--0")
      if (i == scanLast) break;
   }

   if (!locBrackets.empty()) {
      ExtendRangeAndSetColor(input, Cursor, kColorBadBracket, DisplayR);
      fPrevBracketColor = kColorBadBracket;
   }

} // matchParentheses
