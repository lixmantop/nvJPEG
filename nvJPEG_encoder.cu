/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// This sample needs at least CUDA 10.1. It demonstrates usages of the nvJPEG
// library nvJPEG encoder supports single and multiple image encode.
#include <cuda_runtime_api.h>
#include "helper_nvJPEG.hxx"
#include <sys/time.h>
#include <math.h>
#include <helper_functions.h>
#include <helper_cuda.h>

// *************
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <pthread.h>

struct jpeg_compress_in {
	int width;
	int height;
	int size;
	int stride;
};
struct jpeg_compress_out {
	int etat; // 0=error 1=ok
	int lenght;
};

struct thread_param{
   int  id_cuda;
   pthread_t id_thread;
};

static const char *socket_path = "/tmp/vnc_jpeg_cuda";
static const unsigned int nIncomingConnections = 5;

// *************

struct encode_params_t {
	int quality;
	int huf;
	int dev;
};





float getTime(struct timeval t0, struct timeval t1) {
	return (t1.tv_sec - t0.tv_sec) * 1000.0f
			+ (t1.tv_usec - t0.tv_usec) / 1000.0f;
}

__global__ void decode_encode(int n, char *in,char *out, int stride, int w) {
	int i = (blockIdx.x * blockDim.x + threadIdx.x) ;
	if (i < n ){
		int nbstride=(i/stride);
		int pad= i- nbstride*stride;
		if(stride==w) {
			char old=in[i*4];
			out[i*3]=in[i*4+2];
			out[i*3+1]=in[i*4+1];
			out[i*3+2]=old;
		} else {
			if(pad<w) {
				int v=(stride-w)*nbstride*3;
				char old=in[(i*4)];
				out[i*3-v]=in[i*4+2];
				out[i*3+1-v]=in[i*4+1];
				out[i*3+2-v]=old;

			}
		}
	}
}

void * start_server(void *param) {
	nvjpegEncoderParams_t encode_params;
	nvjpegHandle_t nvjpeg_handle;
	nvjpegJpegState_t jpeg_state;
	nvjpegEncoderState_t encoder_state;
	jpeg_compress_in *recv_buf = NULL;
	jpeg_compress_out *send_buf = NULL;
	char *bufInOut = NULL;
	char *cin = NULL;
	int size_dd = 0;
	int maxval=0;

	struct timeval t0;
	struct timeval t1;
	unsigned char *pBuffer;
	int size_pBuffer=65535;
	int size=0;

	encode_params_t params;
//	params.dev = 0;
	struct thread_param *myparam = (thread_param *)param;
	params.dev = myparam->id_cuda;

	cudaDeviceProp props;
	checkCudaErrors(cudaGetDeviceProperties(&props, params.dev));

	printf("Using GPU %d (%s, %d SMs, %d th/SM max, CC %d.%d, ECC %s) maxGridSize=%d maxThreadsPerBlock=%d\n",
			params.dev, props.name, props.multiProcessorCount,
			props.maxThreadsPerMultiProcessor, props.major, props.minor,
			props.ECCEnabled ? "on" : "off",props.maxGridSize, props.maxThreadsPerBlock);

	checkCudaErrors(nvjpegCreate(NVJPEG_BACKEND_DEFAULT, NULL,&nvjpeg_handle));
	checkCudaErrors(nvjpegJpegStateCreate(nvjpeg_handle, &jpeg_state));
	checkCudaErrors(nvjpegEncoderStateCreate(nvjpeg_handle, &encoder_state, NULL));
	checkCudaErrors(nvjpegEncoderParamsCreate(nvjpeg_handle, &encode_params, NULL));


	cudaMalloc(&pBuffer, size_pBuffer);

	//create server side
	int s = 0;
	int s2 = 0;
	struct sockaddr_un local, remote;
	int len = 0;
	int data_recv = 0;
	mode_t saved_umask;
	if (s == 0) {
		s = socket(AF_UNIX, SOCK_STREAM, 0);
		if (-1 == s) {
			printf("Error on socket() call \n");
			return NULL;
		}

		local.sun_family = AF_UNIX;
		strcpy(local.sun_path, socket_path);
		unlink(local.sun_path);
		saved_umask = umask(0777);
		len = strlen(local.sun_path) + sizeof(local.sun_family);
		if (bind(s, (struct sockaddr*) &local, len) != 0) {
			printf("Error on binding socket \n");
			return NULL;
		}
		umask(saved_umask);
		if (chmod(socket_path, 0777) < 0) {
			printf(">> erreur chmod !!!");
			return NULL;
		}
		if (listen(s, nIncomingConnections) != 0) {
			printf("Error on listen call \n");
		}
	}

	bool bWaiting = true;
	if (recv_buf == NULL) {
		recv_buf = (jpeg_compress_in*) malloc(sizeof(jpeg_compress_in));
		send_buf = (jpeg_compress_out*) malloc(sizeof(jpeg_compress_out));
	}
	while (bWaiting) {

		if(s2==0) {
			unsigned int sock_len = 0;
			if ((s2 = accept(s, (struct sockaddr*) &remote, &sock_len)) == -1) {
				printf("Error on accept() call \n");
				return NULL;
			}
		}


		//header
		data_recv = recv(s2, recv_buf, sizeof(jpeg_compress_in), MSG_WAITALL);
		if (data_recv <= 0)  {
			close(s2);
			s2=0;
			continue;
		}
		if (size_dd < recv_buf->size) {
			if (bufInOut != NULL) {
				printf("free size=%d => up to size=%d\n",size_dd, recv_buf->size);
				free(bufInOut);
			}
			bufInOut = (char *) malloc(sizeof(char) * recv_buf->size);
			size_dd = recv_buf->size;
		}

		data_recv = recv(s2, bufInOut, recv_buf->size, MSG_WAITALL);
		if (data_recv <= 0) {
			printf("Error on recv datacall \n");
			close(s2);
			s2=0;
			continue;
		}
//*********************************************************
		gettimeofday(&t0, 0);

		size=recv_buf->size;

		if (size_pBuffer<size*2) {
			checkCudaErrors(cudaFree(pBuffer));
			cudaError_t eCopy = cudaMalloc(&pBuffer, size*2);
			size_pBuffer=size*2;
			if (cudaSuccess != eCopy) {
				std::cerr << "cudaMalloc failed for component Y: "
						<< cudaGetErrorString(eCopy) << std::endl;
				return NULL;
			}

		}
		cudaMemcpy(pBuffer+size, bufInOut,size * sizeof(char), cudaMemcpyHostToDevice);
		decode_encode<<<((size/4)+511)/512, 512>>>(size/4, (char *)pBuffer+size,  (char *)pBuffer ,recv_buf->stride,recv_buf->width);
		cudaDeviceSynchronize();

		nvjpegImage_t imgdesc = { { pBuffer,
				pBuffer	+ recv_buf->width * recv_buf->height,
				pBuffer	+ recv_buf->width * recv_buf->height * 2,
				pBuffer	+ recv_buf->width * recv_buf->height * 3 }, {
				(unsigned int) recv_buf->width * 3,
				(unsigned int) recv_buf->width,
				(unsigned int) recv_buf->width,
				(unsigned int) recv_buf->width } };


		size_t length;
		params.huf = 0;//2;
		params.quality = 92;//92;
		checkCudaErrors(nvjpegEncoderParamsSetSamplingFactors(encode_params, NVJPEG_CSS_420, NULL));
		checkCudaErrors(nvjpegEncoderParamsSetQuality(encode_params, params.quality, NULL));
		checkCudaErrors(nvjpegEncoderParamsSetOptimizedHuffman(encode_params, params.huf, NULL));
		nvjpegInputFormat_t iformat = NVJPEG_INPUT_RGBI;

		cudaDeviceSynchronize();
		checkCudaErrors(nvjpegEncodeImage(nvjpeg_handle, encoder_state, encode_params, &imgdesc, iformat, recv_buf->width, recv_buf->height, NULL));
		checkCudaErrors(nvjpegEncodeRetrieveBitstream( nvjpeg_handle, encoder_state, NULL, &length, NULL));
		checkCudaErrors(nvjpegEncodeRetrieveBitstream( nvjpeg_handle, encoder_state, (unsigned char*)bufInOut, &length, NULL));

		send_buf->lenght = length;

		gettimeofday(&t1, 0);
		maxval++;
		if(false) {
			float elapsed = getTime(t0, t1);
			printf(" time=%f ms  size=%d \n\n",elapsed,size	);
			maxval=0;
		}


//*********************************************************

		send_buf->etat = 1;
		data_recv =send(s2, send_buf, sizeof(jpeg_compress_out), 0);
		if(data_recv<0) {
			close(s2);
			s2=0;
			continue;
		}
		data_recv = 0;
		int ret=0;
		do {
			data_recv = send(s2, bufInOut + ret,send_buf->lenght - ret, 0);
			if(data_recv<0) {
				close(s2);
				s2=0;
				break;
			}
			ret+=data_recv;
		} while (ret != send_buf->lenght);



	}


	if(s2 !=0) {
		close(s2);
	}

	if (bufInOut != NULL) {
		free(bufInOut);
	}
	checkCudaErrors(cudaFree(pBuffer));

	checkCudaErrors(cudaFree(cin));


	checkCudaErrors(nvjpegEncoderParamsDestroy(encode_params));
	checkCudaErrors(nvjpegEncoderStateDestroy(encoder_state));
	checkCudaErrors(nvjpegJpegStateDestroy(jpeg_state));
	checkCudaErrors(nvjpegDestroy(nvjpeg_handle));

 return NULL;
}


//int main(int argc, const char *argv[]) {
//	thread_param param;
//	param.id_cuda=findCudaDevice(argc, argv);
//	param.id_thread=1;
//    start_server((void *)&param);
//	return 0;
//}

int main(int argc, const char *argv[]) {
	pthread_t tid;
	thread_param param;
	param.id_cuda=findCudaDevice(argc, argv);
	param.id_thread=1;
    pthread_create(&tid,NULL,start_server,(void *)&param);
    pthread_exit(NULL);
	return 0;
}
