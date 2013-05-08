// @(#)root/textinput:$Id$
// Author: Axel Naumann <axel@cern.ch>, 2011-05-21

/*************************************************************************
 * Copyright (C) 1995-2011, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "Getline.h"

#include <string>
#include <sstream>
#include "textinput/TextInput.h"
#include "textinput/TextInputContext.h"
#include "textinput/History.h"
#include "textinput/TerminalDisplay.h"
#include "textinput/StreamReader.h"
#include "textinput/Callbacks.h"
#include "Getline_color.h"
#include "TApplication.h"

extern "C" {
   int (* Gl_in_key)(int ch) = 0;
   int (* Gl_beep_hook)() = 0;
}


using namespace textinput;

namespace {
   // TTabCom adapter.
   class ROOTTabCompletion: public TabCompletion {
   private:
      ROOTTabCompletion(const ROOTTabCompletion&); // not implemented
      ROOTTabCompletion& operator=(const ROOTTabCompletion&); // not implemented
   public:
      ROOTTabCompletion(): fLineBuf(new char[fgLineBufSize]) {}
      virtual ~ROOTTabCompletion() { delete []fLineBuf; }

      // Returns false on error
      bool Complete(Text& line /*in+out*/, size_t& cursor /*in+out*/,
                    EditorRange& r /*out*/,
                    std::vector<std::string>& displayCompletions /*out*/) {
         strlcpy(fLineBuf, line.GetText().c_str(), fgLineBufSize);
         int cursorInt = (int) cursor;
         std::stringstream sstr;
         size_t posFirstChange = gApplication->TabCompletionHook(fLineBuf, &cursorInt, sstr);
         if (posFirstChange == (size_t) -1) {
            // no change
            return true;
         }

         line = std::string(fLineBuf);
         std::string compLine;
         while (std::getline(sstr, compLine)) {
            displayCompletions.push_back(compLine);
         }

         size_t lenLineBuf = strlen(fLineBuf);
         if (posFirstChange == (size_t) -2) {
            // redraw whole line, incl prompt
            r.fEdit.Extend(Range::AllWithPrompt());
            r.fDisplay.Extend(Range::AllWithPrompt());
         } else {
            if (!lenLineBuf) {
               r.fEdit.Extend(Range::AllText());
               r.fDisplay.Extend(Range::AllText());
            } else {
               r.fEdit.Extend(Range(posFirstChange, Range::End()));
               r.fDisplay.Extend(Range(posFirstChange, Range::End()));
            }
         }
         cursor = (size_t)cursorInt;
         line.GetColors().resize(lenLineBuf);
         return true;
      }
   private:
      static const size_t fgLineBufSize;
      char* fLineBuf;
   };
   const size_t ROOTTabCompletion::fgLineBufSize = 16*1024;


   // Helper to define the lifetime of the TextInput singleton.
   class TextInputHolder {
   public:
      TextInputHolder():
         fTextInput(*(fReader = StreamReader::Create()),
                    *(fDisplay = TerminalDisplay::Create()),
                    fgHistoryFile.c_str()) {
         fTextInput.SetColorizer(&fCol);
         fTextInput.SetCompletion(&fTabComp);
         fTextInput.EnableAutoHistAdd(false);
         History* Hist = fTextInput.GetContext()->GetHistory();
         Hist->SetMaxDepth(fgSizeLines);
         Hist->SetPruneLength(fgSaveLines);
      }

      ~TextInputHolder() {
         // Delete allocated objects.
         delete fReader;
         delete fDisplay;
      }

      const char* TakeInput() {
         fTextInput.TakeInput(fInputLine);
         fInputLine += "\n"; // ROOT wants a trailing newline.
         return fInputLine.c_str();
      }

      void SetColors(const char* colorTab, const char* colorTabComp,
                     const char* colorBracket, const char* colorBadBracket,
                     const char* colorPrompt) {
         fCol.SetColors(colorTab, colorTabComp, colorBracket, colorBadBracket,
                        colorPrompt);
      }

      static void SetHistoryFile(const char* hist) {
         fgHistoryFile = hist;
      }
      static void SetHistSize(int size, int save) {
         fgSizeLines = size;
         fgSaveLines = save;
      }

      static TextInputHolder& getHolder() {
         // Controls initialization of static TextInput.
         static TextInputHolder sTIHolder;
         return sTIHolder;
      }

      static TextInput& get() {
         return getHolder().fTextInput;
      }

   private:
      TextInput fTextInput; // The singleton TextInput object.
      Display* fDisplay; // Default TerminalDisplay
      Reader* fReader; // Default StreamReader
      std::string fInputLine; // Taken from TextInput
      ROOT::TextInputColorizer fCol; // Colorizer
      ROOTTabCompletion fTabComp; // Tab completion handler / TTabCom adapter

      // Config values:
      static std::string fgHistoryFile;
      // # History file size, once HistSize is reached remove all but HistSave entries,
      // # set to 0 to turn off command recording.
      // # Can be overridden by environment variable ROOT_HIST=size[:save],
      // # the ":save" part is optional.
      // # Rint.HistSize:         500
      // # Set to -1 for sensible default (80% of HistSize), set to 0 to disable history.
      // # Rint.HistSave:         400
      static int fgSizeLines;
      static int fgSaveLines;
   };

   int TextInputHolder::fgSizeLines = 500;
   int TextInputHolder::fgSaveLines = -1;
   std::string TextInputHolder::fgHistoryFile;
}

/************************ extern "C" part *********************************/

extern "C" {
void
Gl_config(const char* which, int value) {
   if (strcmp(which, "noecho") == 0) {
      TextInputHolder::get().HideInput(value);
   } else {
      // unsupported directive
      printf("Gl_config unsupported: %s ?\n", which);
   }
}

void
Gl_histadd(const char* buf) {
   TextInputHolder::get().AddHistoryLine(buf);
}

/* Wrapper around textinput.
 * Modes: -1 = init, 0 = line mode, 1 = one char at a time mode, 2 = cleanup
 */
const char*
Getlinem(EGetLineMode mode, const char* prompt) {

   if (mode == kCleanUp) {
      TextInputHolder::get().ReleaseInputOutput();
      return 0;
   }

   if (mode == kOneChar) {
      // Check first display: if !TTY, read full line.
      const textinput::Display* disp
         = TextInputHolder::get().GetContext()->GetDisplays()[0];
      const textinput::TerminalDisplay* tdisp = 0;
      if (disp) tdisp = dynamic_cast<const textinput::TerminalDisplay*>(disp);
      if (tdisp && !tdisp->IsTTY()) {
         mode = kLine1;
      }
   }

   if (mode == kInit || mode == kLine1) {
      if (prompt) {
         TextInputHolder::get().SetPrompt(prompt);
      }
      // Trigger attach:
      TextInputHolder::get().Redraw();
      if (mode == kInit) {
         return 0;
      }
      TextInputHolder::get().SetBlockingUntilEOL();
   } else {
      // mode == kOneChar
      if (Gl_in_key) {
         // We really need to go key by key:
         TextInputHolder::get().SetMaxPendingCharsToRead(1);
      } else {
         // Can consume all pending characters
         TextInputHolder::get().SetReadingAllPendingChars();
      }
   }

   TextInput::EReadResult res = TextInputHolder::get().ReadInput();
   if (Gl_in_key) {
      (*Gl_in_key)(TextInputHolder::get().GetLastKey());
   }
   if (res == TextInput::kRRReadEOLDelimiter) {
      return TextInputHolder::getHolder().TakeInput();
   } else if (res == TextInput::kRREOF) {
      // ROOT expects "" and then Gl_eol() returning true
      return "";
   }

   return NULL;
}

const char*
Getline(const char* prompt) {
   // Get a line of user input, showing prompt.
   // Does not return after every character entered, but
   // only returns once the user has hit return.
   // For ROOT Getline.c backward compatibility reasons,
   // the returned value is volatile and will be overwritten
   // by the subsequent call to Getline() or Getlinem(),
   // so copy the string if it needs to stay around.
   // The returned value must not be deleted.
   // The returned string contains a trailing newline '\n'.

   return Getlinem(kLine1, prompt);
}


/******************* Simple C -> C++ forwards *********************************/

void
Gl_histsize(int size, int save) {
   TextInputHolder::SetHistSize(size, save);
}

void
Gl_histinit(const char* file) {
   // Has to be called before constructing TextInputHolder singleton.
   TextInputHolder::SetHistoryFile(file);
}

int
Gl_eof() {
   return TextInputHolder::get().GetReadState() == TextInput::kRREOF;
}

void
Gl_setColors(const char* colorTab, const char* colorTabComp, const char* colorBracket,
             const char* colorBadBracket, const char* colorPrompt) {
   // call to enhance.cxx to set colours
   TextInputHolder::getHolder().SetColors(colorTab, colorTabComp, colorBracket,
                                          colorBadBracket, colorPrompt);
}

/******************** Superseded interface *********************************/

void Gl_setwidth(int /*w*/) {
   // ignored, handled by displays themselves.
}


void Gl_windowchanged() {
   // ignored, handled by displays themselves.
}

} // extern "C"
