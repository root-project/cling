// @(#)root/textinput:$Id$
// Author: Axel Naumann <axel@cern.ch>, 2011

/*************************************************************************
 * Copyright (C) 1995-2011, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_Getline_color
#define ROOT_Getline_color

#include "textinput/Color.h"

namespace ROOT {
   using std::size_t;

   // Colorization interface.
   class TextInputColorizer: public textinput::Colorizer {
   public:
      TextInputColorizer();
      virtual ~TextInputColorizer();
      void ProcessTextChange(textinput::EditorRange& Modification,
                             textinput::Text& input);

      void ProcessPromptChange(textinput::Text& prompt);

      void ProcessCursorChange(size_t Cursor, textinput::Text& input,
                               textinput::Range& DisplayR);
      bool GetColor(char type, textinput::Color& Col);
      char GetInfoColor() const { return (char) kColorTabComp; }


      void SetColors(const char* colorType, const char* colorTabComp,
                     const char* colorBracket, const char* colorBadBracket,
                     const char* colorPrompt);
   private:
      void ExtendRangeAndSetColor(textinput::Text& input, size_t idx,
                                  char col, textinput::Range& disp);

      enum EColorsTypes {
         kColorNone,
         kColorType,
         kColorTabComp,
         kColorBracket,
         kColorBadBracket,
         kColorPrompt,
         kNumColors
      };

      textinput::Color fColors[kNumColors]; // Colors used, indexed by EColorsTypes
      bool fColorIsDefault[kNumColors]; // Whether the fColors entry is the default color.
      EColorsTypes fPrevBracketColor; // previous bracket: None or [Bad]Bracket
   };

}

#endif // ROOT_Getline_color
