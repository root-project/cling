/* @(#)root/textinput:$Id$ */
/* Author: Axel Naumann <axel@cern.ch>, 2011-05-21 */

/*************************************************************************
 * Copyright (C) 1995-2011, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_Getline
#define ROOT_Getline

#ifndef ROOT_DllImport
#include "DllImport.h"
#endif

#ifndef __CINT__
#ifdef __cplusplus
extern "C" {
#endif
#endif

typedef enum { kInit = -1, kLine1, kOneChar, kCleanUp } EGetLineMode;

const char *Getline(const char *prompt);
const char *Getlinem(EGetLineMode mode, const char *prompt);
void Gl_config(const char *which, int value);
void Gl_setwidth(int width);
void Gl_windowchanged();
void Gl_histsize(int size, int save);
void Gl_histinit(const char *file);
void Gl_histadd(const char *buf);
int  Gl_eof();
void Gl_setColors(const char* colorTab, const char* colorTabComp, const char* colorBracket,
                  const char* colorBadBracket, const char* colorPrompt);

R__EXTERN int (*Gl_beep_hook)();
R__EXTERN int (*Gl_in_key)(int key);

#ifndef __CINT__
#ifdef __cplusplus
}
#endif
#endif

#endif
