#pragma once
/*----------------------------------------------------------------------------
 *
 * Name:     err_code()
 *
 * Purpose:  Function to output descriptions of errors for an input error code
 *           and quit a program on an error with a user message
 *
 *
 * RETURN:   echoes the input error code / echos user message and exits
 *----------------------------------------------------------------------------
 */
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#ifdef __cplusplus
 #include <cstdio>
#endif

const char *err_code(cl_int err_in);
void check_error(cl_int err, const char *operation, const char *filename, int line);


#define checkError(E, S) check_error(E,S,__FILE__,__LINE__)

