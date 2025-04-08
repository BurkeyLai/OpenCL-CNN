#include "OpenCL.h"
#include "util.h"
#include "err_code.h"


cl_program OpenCL::clprogram;
cl_context OpenCL::clcontext;
cl_command_queue OpenCL::clqueue;


uint32_t OpenCL::initialize_OpenCL() {
	// get all platforms (drivers), e.g. NVIDIA

	cl_uint num_platforms;
	cl_int err;
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    checkError(err, "Finding platforms");
    if (num_platforms == 0)
    {
        printf("Found 0 platforms!\n");
        return EXIT_FAILURE;
    }
    // Create a list of platform IDs
    cl_platform_id all_platforms[num_platforms];
	cl_device_id device_id;
    err = clGetPlatformIDs(num_platforms, all_platforms, NULL);
    checkError(err, "Getting platforms");

    printf("\nNumber of OpenCL platforms: %d\n", num_platforms);

	// Investigate each platform
    for (int i = 0; i < num_platforms; i++)
    {
        cl_char string[10240] = {0};
        // Print out the platform name
        err = clGetPlatformInfo(all_platforms[i], CL_PLATFORM_NAME, sizeof(string), &string, NULL);
        checkError(err, "Getting platform name");
        printf("Platform: %s\n", string);

        // Print out the platform vendor
        err = clGetPlatformInfo(all_platforms[i], CL_PLATFORM_VENDOR, sizeof(string), &string, NULL);
        checkError(err, "Getting platform vendor");
        printf("Vendor: %s\n", string);

        // Print out the platform OpenCL version
        err = clGetPlatformInfo(all_platforms[i], CL_PLATFORM_VERSION, sizeof(string), &string, NULL);
        checkError(err, "Getting platform OpenCL version");
        printf("Version: %s\n", string);

        // Count the number of devices in the platform
        cl_uint num_devices;
        err = clGetDeviceIDs(all_platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
        checkError(err, "Finding devices");

        // Get the device IDs
        cl_device_id device[num_devices];
        err = clGetDeviceIDs(all_platforms[i], CL_DEVICE_TYPE_ALL, num_devices, device, NULL);
        checkError(err, "Getting devices");
        printf("Number of devices: %d\n", num_devices);

		// Investigate each device
        for (int j = 0; j < num_devices; j++)
        {
            printf("\t-------------------------\n");

            // Get device name
            err = clGetDeviceInfo(device[j], CL_DEVICE_NAME, sizeof(string), &string, NULL);
            checkError(err, "Getting device name");
            printf("\t\tName: %s\n", string);

			// Create a compute context
			// a context is like a "runtime link" to the device and platform;
			// i.e. communication is possible
    		OpenCL::clcontext = clCreateContext(0, 1, &device[j], NULL, NULL, &err);
    		checkError(err, "Creating context");

			device_id = device[j];

			std::cout << "\t\tDevice ID: " << device_id << std::endl;

			break;
		}
	}
	
	
	
	
	// ////////////////////////////////////////////////////
	// std::cout << "\t-------------------------" << std::endl;
	// std::string s;
	// default_device.getInfo(CL_DEVICE_NAME, &s);
	// std::cout << "\t\tName: " << s << std::endl;
	// //default_device.getInfo(CL_DEVICE_OPENCL_C_VERSION, &s);
	// //std::cout << "\t\tVersion: " << s << std::endl;
	// int i;
	// default_device.getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &i);
	// std::cout << "\t\tMax. Compute Units: " << i << std::endl;
	// size_t size;
	// default_device.getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &size);
	// std::cout << "\t\tLocal Memory Size: " << size / 1024 << " KB" << std::endl;
	// default_device.getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &size);
	// std::cout << "\t\tGlobal Memory Size: " << size / (1024 * 1024) << " MB" << std::endl;
	// default_device.getInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE, &size);
	// std::cout << "\t\tMax Alloc Size: " << size / (1024 * 1024) << " MB" << std::endl;
	// default_device.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &size);
	// std::cout << "\t\tMax Work-group Total Size: " << size << std::endl;
	// std::vector<size_t> d;
	// default_device.getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, &d);
	// std::cout << "\t\tMax Work-group Dims: (";
	// for (std::vector<size_t>::iterator st = d.begin(); st != d.end(); st++)
	// 	std::cout << *st << " ";
	// std::cout << "\x08)" << std::endl;
	// std::cout << "\t-------------------------" << std::endl;
	// ////////////////////////////////////////////////////////////


	

	std::string src, src2, src3;
	
	src = util::loadProgram("kernels/kernelheader.cl");
	src2 = util::loadProgram("kernels/fcnn_kernels.cl");
	src3 = util::loadProgram("kernels/conv_kernels.cl");
	src = src + src2 + src3;


	size_t src_len = src.length();
	const char* sources = src.c_str();

	//std::cout << "Length: " << src.length() << std::endl;

	//std::cout <<  sources << std::endl;

	OpenCL::clprogram = clCreateProgramWithSource(OpenCL::clcontext, 1, (const char**)&sources, &src_len, &err);

	std::cout << "Create program success" << std::endl;

	checkError(err, "Creating program with source");
	try {
		err = clBuildProgram(OpenCL::clprogram, 0, NULL, NULL, NULL, NULL);

		std::cout << "Build program success" << std::endl;
	}
	catch (...) {

		size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n%s\n", err_code(err));
        clGetProgramBuildInfo(OpenCL::clprogram, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        return EXIT_FAILURE;
	}
	OpenCL::clqueue = clCreateCommandQueue(OpenCL::clcontext, device_id, 0, &err);
    checkError(err, "Creating command queue");

	return 0;
}

// 添加清理 OpenCL 資源的函數實現
void OpenCL::cleanup_OpenCL() {
    // 釋放命令隊列
    if (OpenCL::clqueue) {
        clReleaseCommandQueue(OpenCL::clqueue);
        OpenCL::clqueue = NULL;
    }
    
    // 釋放程序
    if (OpenCL::clprogram) {
        clReleaseProgram(OpenCL::clprogram);
        OpenCL::clprogram = NULL;
    }
    
    // 釋放上下文
    if (OpenCL::clcontext) {
        clReleaseContext(OpenCL::clcontext);
        OpenCL::clcontext = NULL;
    }
}




