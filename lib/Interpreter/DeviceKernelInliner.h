//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Simeon Ehrig <s.ehrig@hzdr.de>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_DEVICEKERNELINLINER_H
#define CLING_DEVICEKERNELINLINER_H

#include "ASTTransformer.h"


namespace cling
{

// This ASTTransformer adds an inline attribute to any CUDA __device__ kernel
// that does not have the attribute. Inlining solves a problem caused by
// incremental compilation of PTX code. In a normal compiler, all definitions
// of __global__ and __device__ kernels are in the same translation unit. In
// the incremental compiler, each kernel has its own translation unit. In case
// a __global__ kernel uses a __device__ function, this design caused an error.
// Instead of generating the PTX code of the __device__ kernel in the same file
// as the __global__ kernel, there is only an external declaration of the
// __device__ function. However, PTX does not support an external declaration of
// functions.

class DeviceKernelInliner : public ASTTransformer
{
public:
    DeviceKernelInliner(clang::Sema* S);

    ASTTransformer::Result Transform(clang::Decl * D) override;
};

}

#endif // CLING_DEVICEKERNELINLINER_H
