/*******************************************************************************
Vendor: iStamoulias
Associated Filename: Black_Scholes.cpp
Purpose: SDAccel Black-Scholes
Revision History: July 4, 2016

*******************************************************************************
*******************************************************************************/

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <fstream>

#include <time.h>
#include <sys/time.h>

//OpenCL includes
#include "xcl.h"
#include "Black_Scholes.h"

using namespace std;

struct krnl_bufs {
    cl_mem spot;
    cl_mem strike;
    cl_mem time;
    cl_mem sigma;
    cl_mem volatility;
    cl_mem type;
    cl_mem result;
};

void accelerator_init(int argc, char* argv[], xcl_world *world, cl_kernel *krnl, int selKrnl, krnl_bufs *bufs, int block)
{   
    //TARGET_DEVICE macro needs to be passed from gcc command line
    #if defined(SDA_PLATFORM) && !defined(TARGET_DEVICE)
      #define STR_VALUE(arg)      #arg
      #define GET_STRING(name) STR_VALUE(name)
      #define TARGET_DEVICE GET_STRING(SDA_PLATFORM)
    #endif
    //!!!Changed to have fixed TARGET_DEVICE, not to need input arguments!!!
    // Set the targeted device
    const char *target_device_name = TARGET_DEVICE;//"xilinx:adm-pcie-ku3:1ddr:3.0"; // 
    const char *target_vendor = "Xilinx";
	const char *target_kernel;
	if(selKrnl==0)
		target_kernel = "krnl_Black_Scholes";
	else if(selKrnl==1)
		target_kernel = "krnl_Black";
	else
		target_kernel = "krnl_Binomial";
    //!!!Changed to have fixed xclbinFilename, not to need input arguments!!!
    if(argc != 2)
        std::cout << "Usage: " << argv[0] <<" <xclbin>" << std::endl;
    const char* xclbinFilename = argv[1];//"bin_Black_Scholes_cpu_emu.xclbin"; // 
    if(strstr(argv[1], ".xclbin") != NULL) {
    //if(strstr(xclbinFilename, ".xclbin") != NULL) {
        *world = xcl_world_single(CL_DEVICE_TYPE_ACCELERATOR, target_vendor, target_device_name);
        *krnl  = xcl_import_binary(*world, xclbinFilename, target_kernel);
    } else {
        *world = xcl_world_single(CL_DEVICE_TYPE_CPU, NULL, NULL);
        *krnl  = xcl_import_source(*world, xclbinFilename, target_kernel);
    }
    // Create the buffers for the transmittion of the data to the kernel
    size_t vector_size_bytes = sizeof(float) * block;
    bufs->spot       = xcl_malloc(*world, CL_MEM_READ_ONLY,  vector_size_bytes);
    bufs->strike     = xcl_malloc(*world, CL_MEM_READ_ONLY,  vector_size_bytes);
    bufs->time       = xcl_malloc(*world, CL_MEM_READ_ONLY,  vector_size_bytes);
    bufs->sigma      = xcl_malloc(*world, CL_MEM_READ_ONLY,  vector_size_bytes);
    bufs->volatility = xcl_malloc(*world, CL_MEM_READ_ONLY,  vector_size_bytes);
    bufs->type       = xcl_malloc(*world, CL_MEM_READ_ONLY,  vector_size_bytes);
    bufs->result     = xcl_malloc(*world, CL_MEM_WRITE_ONLY, vector_size_bytes);
}

void Black_Scholes_accel(xcl_world world, cl_kernel krnl, krnl_bufs bufs, float *spot, float *strike, float *time, float *sigma, float *volatility, float *result, int amount, int block)
{
    size_t vector_size_bytes = sizeof(float) * block;
    int repeat_trans=floor(amount/block);
    float amount_tmp=amount;
    int index=0;
    while(repeat_trans>0){
	    // Copy input vectors to memory
	    xcl_memcpy_to_device(world,bufs.spot,spot+(index*block),vector_size_bytes);
	    xcl_memcpy_to_device(world,bufs.strike,strike+(index*block),vector_size_bytes);
	    xcl_memcpy_to_device(world,bufs.time,time+(index*block),vector_size_bytes);
	    xcl_memcpy_to_device(world,bufs.sigma,sigma+(index*block),vector_size_bytes);
	    xcl_memcpy_to_device(world,bufs.volatility,volatility+(index*block),vector_size_bytes);
	    // Set the kernel arguments
	    clSetKernelArg(krnl, 0, sizeof(cl_mem), &bufs.spot);
	    clSetKernelArg(krnl, 1, sizeof(cl_mem), &bufs.strike);
	    clSetKernelArg(krnl, 2, sizeof(cl_mem), &bufs.time);
	    clSetKernelArg(krnl, 3, sizeof(cl_mem), &bufs.sigma);
	    clSetKernelArg(krnl, 4, sizeof(cl_mem), &bufs.volatility);
	    clSetKernelArg(krnl, 5, sizeof(cl_mem), &bufs.result);
	    // Launch the kernel
		xcl_run_kernel3d(world, krnl, 1, 1, 1);
	    //unsigned long duration = xcl_run_kernel3d(world, krnl, 1, 1, 1);
	    //std::cout << repeat_trans << " - Duration of kernel execution: "<< duration <<" ns"<< endl;
	    // Copy result to local buffer
	    xcl_memcpy_from_device(world, result+(index*block), bufs.result, vector_size_bytes);
            repeat_trans--;
            index++;
            amount_tmp=amount_tmp-block;
    }
    if(amount_tmp>0){
            size_t vector_fin_size_bytes = sizeof(float) * amount_tmp;
            // Copy final input vectors to memory
	    xcl_memcpy_to_device(world,bufs.spot,spot+(index*block),vector_fin_size_bytes);
	    xcl_memcpy_to_device(world,bufs.strike,strike+(index*block),vector_fin_size_bytes);
	    xcl_memcpy_to_device(world,bufs.time,time+(index*block),vector_fin_size_bytes);
	    xcl_memcpy_to_device(world,bufs.sigma,sigma+(index*block),vector_fin_size_bytes);
	    xcl_memcpy_to_device(world,bufs.volatility,volatility+(index*block),vector_fin_size_bytes);
	    // Set the kernel arguments
	    clSetKernelArg(krnl, 0, sizeof(cl_mem), &bufs.spot);
	    clSetKernelArg(krnl, 1, sizeof(cl_mem), &bufs.strike);
	    clSetKernelArg(krnl, 2, sizeof(cl_mem), &bufs.time);
	    clSetKernelArg(krnl, 3, sizeof(cl_mem), &bufs.sigma);
	    clSetKernelArg(krnl, 4, sizeof(cl_mem), &bufs.volatility);
	    clSetKernelArg(krnl, 5, sizeof(cl_mem), &bufs.result);
	    // Launch the kernel
		xcl_run_kernel3d(world, krnl, 1, 1, 1);
	    //unsigned long duration = xcl_run_kernel3d(world, krnl, 1, 1, 1);
	    //std::cout << repeat_trans << " - Duration of kernel execution: "<< duration <<" ns"<< endl;
	    // Copy final result to local buffer
	    xcl_memcpy_from_device(world, result+(index*block), bufs.result, vector_fin_size_bytes);
    }
}

void accelerator_end(xcl_world *world, cl_kernel *krnl, krnl_bufs *bufs)
{ 
    // Release the memory for the data buffers transmitted to kernel
    clReleaseMemObject(bufs->spot);
    clReleaseMemObject(bufs->strike);
    clReleaseMemObject(bufs->time);
    clReleaseMemObject(bufs->sigma);
    clReleaseMemObject(bufs->volatility);
    clReleaseMemObject(bufs->type);
    clReleaseMemObject(bufs->result);
    // Release kernel
    clReleaseKernel(*krnl);
    xcl_release_world(*world);
}

int main(int argc, char* argv[])
{
    //char cwd[1024];
    //getcwd(cwd, sizeof(cwd));
    //printf("Current working dir: %s\n", cwd);
    /* Open the files with the data and the file for the results */
    ifstream fd_bsfsp_in;
    fd_bsfsp_in.open("../Data/BS_spot.txt", ios::in);
    ifstream fd_bsxsr_in;
    fd_bsxsr_in.open("../Data/BS_strike.txt", ios::in);
    ifstream fd_bsv_in;
    fd_bsv_in.open("../Data/BS_sigma.txt", ios::in);
    ifstream fd_bsr_in;
    fd_bsr_in.open("../Data/BS_volatility.txt", ios::in);
    ifstream fd_bst_in;
    fd_bst_in.open("../Data/BS_time.txt", ios::in);
    ifstream fd_bsres_in;
    fd_bsres_in.open("../Data/BS_res.txt", ios::in);
    ofstream fd_bsres_out;
    fd_bsres_out.open("../Data/BS_res_hw.txt", ios::out);
    ofstream fd_diff_out;
    fd_diff_out.open("../Data/BS_res_diff.txt", ios::out);
    if (!fd_bsfsp_in.is_open())
       cout << "Error opening file BS_spot.txt" << endl;
    if (!fd_bsxsr_in.is_open())
       cout << "Error opening file BS_strike.txt" << endl;
    if (!fd_bsv_in.is_open())
       cout << "Error opening file BS_volatility.txt" << endl;
    if (!fd_bsr_in.is_open())
       cout << "Error opening file BS_sigma.txt" << endl;
    if (!fd_bst_in.is_open())
       cout << "Error opening file BS_time.txt" << endl;
    if (!fd_bsres_in.is_open())
       cout << "Error opening file BS_res.txt" << endl;
    if (!fd_bsres_out.is_open())
       cout << "Error opening file BS_res_hw.txt" << endl;
    if (!fd_diff_out.is_open())
       cout << "Error opening file BS_res_diff.txt" << endl;

 //   ofstream fd_test_out;
 //   fd_test_out.open("../Data/BS_test.txt", ios::out);

    /* The host buffers that all the data from the files are transfered */
    size_t vector_total_size_bytes = sizeof(float) * LENGTH;
    float *source_S0  = (float *) malloc(vector_total_size_bytes);
    float *source_K   = (float *) malloc(vector_total_size_bytes);
    float *source_T   = (float *) malloc(vector_total_size_bytes);
    float *source_r   = (float *) malloc(vector_total_size_bytes);
    float *source_s   = (float *) malloc(vector_total_size_bytes);
    float *result_sim = (float *) malloc(vector_total_size_bytes);
    /* Allocate result buffer on host memory */
    float *result_krnl = (float*) malloc(vector_total_size_bytes);
    /* Read the data and the results from ATHEX locally */
    for(int i=0; i < LENGTH; i++){
    	fd_bsfsp_in >> source_S0[i];
    	fd_bsxsr_in >> source_K[i];
    	fd_bsv_in   >> source_r[i];
    	fd_bsr_in   >> source_s[i];
    	fd_bst_in   >> source_T[i];
    	fd_bsres_in >> result_sim[i];
    }
    /* Close the files with the input data and the results from ATHEX */
    fd_bsfsp_in.close();
    fd_bsxsr_in.close();
    fd_bsv_in.close();
    fd_bsr_in.close();
    fd_bst_in.close();
    fd_bsres_in.close();

    /* OPENCL HOST CODE AREA START */
    xcl_world world;
    cl_kernel krnl;
    krnl_bufs bufs;
    
	struct timeval start, end;
	gettimeofday(&start, NULL);

    accelerator_init(argc, argv, &world, &krnl, 0, &bufs, BLOCK);

	gettimeofday(&end, NULL);
	std::cout << "Duration of initialize: "<< ((end.tv_sec * 1000000 + end.tv_usec)-(start.tv_sec * 1000000 + start.tv_usec)) <<" us"<< endl;
    ///////////////////
	gettimeofday(&start, NULL);

    Black_Scholes_accel(world, krnl, bufs, source_S0, source_K, source_T, source_r, source_s, result_krnl, LENGTH, BLOCK);

	gettimeofday(&end, NULL);
	std::cout << "Duration of execution: "<< ((end.tv_sec * 1000000 + end.tv_usec)-(start.tv_sec * 1000000 + start.tv_usec)) <<" us"<< endl;
    ///////////////////
	gettimeofday(&start, NULL);

    accelerator_end(&world, &krnl, &bufs);

	gettimeofday(&end, NULL);
	std::cout << "Duration of finalize: "<< ((end.tv_sec * 1000000 + end.tv_usec)-(start.tv_sec * 1000000 + start.tv_usec)) <<" us"<< endl;
    ///////////////////

    /* Release the memory for temporary source data buffers on the host */
    free(source_S0);
    free(source_K);
    free(source_T);
    free(source_r);
    free(source_s);
    /* Compare the results of the kernel to the simulation */
    for(int i = 0; i < LENGTH; i++){
       //std::cout <<"i = " << i << " Krnl Result = " << result_krnl[i] << std::endl;
       fd_bsres_out << result_krnl[i] << endl;
       fd_diff_out  << (result_krnl[i]-result_sim[i]) << endl;
    }
    /* Release memory objects from the host */
    free(result_sim);
    free(result_krnl);
    fd_bsres_out.close();
    fd_diff_out.close();
}
