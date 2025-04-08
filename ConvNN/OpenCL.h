#pragma once
#include "include.h"
#include "util.h"


class OpenCL {


public:

	//static cl::Program clprogram;
	//static cl::CommandQueue clqueue;
	//static cl::Context clcontext;

	static cl_program clprogram;
	static cl_command_queue clqueue;
	static cl_context clcontext;

	static uint32_t initialize_OpenCL();
	static void cleanup_OpenCL();



};



