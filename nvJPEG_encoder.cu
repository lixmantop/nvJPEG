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
	bool raz_cache;
	char *socket_path; //max 255

};
struct jpeg_compress_out {
	int x;
	int y;
	int w;
	int h;
	int etat; // 0=error 1=ok -1=last update area
	int lenght;
};

struct thread_param{
   int  id_cuda;
   int cuda_slot;
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
			out[i*3]=in[i*4+2];
			out[i*3+1]=in[i*4+1];
			out[i*3+2]=in[i*4];
		} else {
			if(pad<w) {
				int v=(stride-w)*nbstride*3;
				out[i*3-v]=in[i*4+2];
				out[i*3+1-v]=in[i*4+1];
				out[i*3+2-v]=in[(i*4)];

			}
		}
	}
}
__device__ int  set_block(char *Image, int stride) {

	for(int y=0;y<8;y++) {
		for(int x=0;x<8;x++) {
			Image[(y*stride) + (x*4)] =0;
			Image[(y*stride) + (x*4)+1] =0;
			Image[(y*stride) + (x*4)+2] = 0;
			}
		}

	return 0;
}

__device__ int  diff_block(char *oldImage, char *newImage, int stride) {

	for(int y=0;y<8;y++) {
		for(int x=0;x<8;x++) {
			if(oldImage[(y*stride) + (x*4)] != newImage[(y*stride) + (x*4)] ||
				oldImage[(y*stride) + (x*4)+1] != newImage[(y*stride) + (x*4)+1] ||
				oldImage[(y*stride) + (x*4)+2] != newImage[(y*stride) + (x*4)+2] ){
				return 1;
			}
		}
	}

	return 0;
}

__global__ void diff(int h, char *newImage,char *oldImage,int * rect, int stride, int w) {
	int y = (blockIdx.x * blockDim.x + threadIdx.x) ;
	unsigned int  val=0;
	int pos=0;
	int pad=w*4;
	int trouver=0;
	int jump=0;

	if (y < h ){
		rect[(y*w)+pos]=-1;
		for(int x=0;x<w;x+=8) {
			val=(x*4)+(y*8*w*4);
	//		printf("x=%d y=%d val=%u \n",x,i,val);
			if(trouver==0 && diff_block(oldImage+val,newImage+val,w*4)==1) {
				//save valeur
				rect[(y*w)+pos]=x;
				rect[(y*w)+pos+1]=-1;
				pos++;
				trouver=1;
				} else{
					if(trouver==1 && diff_block(oldImage+val,newImage+val,w*4)==0) {
						if(jump>4) {
						//save valeur
						rect[(y*w)+pos]=x;
						rect[(y*w)+pos+1]=-1;
						pos++;
						trouver=0;
						jump=0;
						} else{
							jump++;
						}
					}
				}
			}
		if(trouver==1) {
			rect[(y*w)+pos]=w;
			rect[(y*w)+pos+1]=-1;

		}
//		printf("y=%d pos=%d trouver=%d \n",y,pos,trouver);
		}
}
__global__ void opac(int h, char *Image,int * rect, int stride, int w) {
	int y = (blockIdx.x * blockDim.x + threadIdx.x) ;
	unsigned int  val=0;
	int pos=0;
	int pad=w*4;
	int trouver=0;
	if (y < h ){
		for(int x=0;x<2400;x+=2) {
		//	printf(" x=%d   \n",x);
			if(rect[(y*w)+x] == -1) {
				break;
			}
			int x1 = rect[(y*w)+x];
			int x2 = rect[(y*w)+x+1];
			//printf(" x1=%d  x2=%d \n",x1,x2);
				for(int deb=x1;deb<=x2;deb+=32) {
					set_block(Image+deb+(y*8*w*4),w*4);
				}
			}
			}
}

int getbuffer(int n, uint8_t *out, int stride, int w) {
	for(int i=0;i<n;i++) {
		int nbstride=(i/stride);
		int pad= i- nbstride*stride;
		if(stride==w) {
			char old=out[i*4];
			out[i*3+1]=out[i*4+1];
			out[i*3]=out[i*4+2];
			out[i*3+2]=old;
		} else {
			if(pad<w) {
				int v=(stride-w)*nbstride*3;
				char old=out[(i*4)];
				out[i*3+1-v]=out[i*4+1];
				out[i*3-v]=out[i*4+2];
				out[i*3+2-v]=old;

			}
		}
	}
return 0;

}

int findNext( int xx, int yy, int x, int *listR,
		jpeg_compress_in *recv_buf) {

	int iter=1;
	int pad=xx + (yy * recv_buf->width);
	int w=listR[iter + xx + pad]-x;
	int x1=0;
	int gain=0;
	while(true) {
		if(listR[1+iter + xx + pad] == -1) {
			break;
		}
		x1 = (listR[iter + xx + pad]);
		if((listR[1+iter + xx + pad] - x1) <128) {
			w=listR[2+iter + pad]-x;
			gain+=listR[1+iter + xx + pad] - x1;
		}
		iter+=2;
	}
//	printf(" Gain =%d\n",gain);
	return w;
}

int diffcpu(char *newbuffer, char *oldbuffer,int *rect ,int w, int h) {
	int i=0;
	int pad=0;
	int pad1=0;


	for(int y=0;y<h/8;y++) {
		pad=w*4;
		pad1=pad*y*8;
		rect[y*w]=-1;
		rect[y*w+1]=-1;
		rect[y*w+2]=-1;
		if(memcmp(newbuffer+pad1,oldbuffer+pad1,w*4)!=0 ||
				memcmp(newbuffer+pad1+pad,oldbuffer+pad+pad1,w*4)!=0 ||
				memcmp(newbuffer+pad1+(pad*2),oldbuffer+pad1+(pad*2),w*4)!=0 ||
				memcmp(newbuffer+pad1+(pad*3),oldbuffer+pad1+(pad*3),w*4)!=0 ||
				memcmp(newbuffer+pad1+(pad*4),oldbuffer+pad1+(pad*4),w*4)!=0 ||
				memcmp(newbuffer+pad1+(pad*5),oldbuffer+pad1+(pad*5),w*4)!=0 ||
				memcmp(newbuffer+pad1+(pad*6),oldbuffer+pad1+(pad*6),w*4)!=0 ||
				memcmp(newbuffer+pad1+(pad*7),oldbuffer+pad1+(pad*7),w*4)!=0 ) {
			rect[y*w]=0;
			rect[y*w+1]=w;
			i++;
		}
	}
	if(i>0) {
		return 1;
	} else {
		return 0;
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
//	int maxval=0;

	struct timeval t0;
	struct timeval t1;
	unsigned char *pBuffer;
	unsigned char *oldBuffer;
	char *oldBuffer1;
	int *listrect;
	int *listR;
	int size_pBuffer=1950*1024*NVJPEG_MAX_COMPONENT;
	unsigned int size=0;
	int w_save=0;
	int h_save=0;

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
	params.huf = 9;//2;
	params.quality = 50;//70;//92;
	checkCudaErrors(nvjpegEncoderParamsSetSamplingFactors(encode_params, NVJPEG_CSS_440, NULL));
//	checkCudaErrors(nvjpegEncoderParamsSetSamplingFactors(encode_params, NVJPEG_CSS_444, NULL));
	checkCudaErrors(nvjpegEncoderParamsSetQuality(encode_params, params.quality, NULL));
	checkCudaErrors(nvjpegEncoderParamsSetOptimizedHuffman(encode_params, params.huf, NULL));
	nvjpegInputFormat_t iformat = NVJPEG_INPUT_RGBI;


	cudaMalloc(&pBuffer, size_pBuffer);
	cudaMalloc(&oldBuffer, size_pBuffer);
	cudaMalloc(&listrect, size_pBuffer*sizeof(int));
	listR = (int *)malloc(size_pBuffer*sizeof(int));
	cudaMallocHost(&oldBuffer1, sizeof(char) *size_pBuffer,cudaHostAllocDefault);
	
	//create server side
	int s = 0;
	int s2 = 0;
	struct sockaddr_un local, remote;
	int len = 0;
	int data_recv = 0;
	mode_t saved_umask;
	nvjpegImage_t imgdesc;
	size_t length;
	nvjpegStatus_t error_cuda;
	int frame=0;
	int framerate=0;
	int mode=0;
	int init=1;

	if (s == 0) {
		s = socket(AF_UNIX, SOCK_STREAM, 0);
		if (-1 == s) {
			printf("Error on socket() call \n");
			return NULL;
		}

		local.sun_family = AF_UNIX;
		char temp[255];
		sprintf(temp,"%s%d",socket_path,myparam->cuda_slot);
		strcpy(local.sun_path, temp);
		unlink(local.sun_path);
		saved_umask = umask(0777);
		len = strlen(local.sun_path) + sizeof(local.sun_family);
		if (bind(s, (struct sockaddr*) &local, len) != 0) {
			printf("Error on binding socket \n");
			return NULL;
		}
		umask(saved_umask);
		if (listen(s, nIncomingConnections) != 0) {
			printf("Error on listen call \n");
		}
	}

	bool bWaiting = true;
	if (recv_buf == NULL) {
		recv_buf = (jpeg_compress_in*) malloc(sizeof(jpeg_compress_in));
		send_buf = (jpeg_compress_out*) malloc(sizeof(jpeg_compress_out));
	}
	gettimeofday(&t0, 0);

	while (bWaiting) {
		send_buf->etat = 1;
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
				printf(">>free size=%d => up to size=%d w=%d h=%d stride=%d \n",size_dd, recv_buf->size , recv_buf->height,recv_buf->height,recv_buf->stride);
				cudaFreeHost(bufInOut);
			}
			cudaHostAlloc(&bufInOut, sizeof(char) * recv_buf->size, cudaHostAllocDefault);
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

		size=recv_buf->size;

		if (size_pBuffer<size*2) {
			checkCudaErrors(cudaFree(pBuffer));
			checkCudaErrors(cudaFree(oldBuffer));
			checkCudaErrors(cudaFree(listrect));
			free(listR);
			cudaFreeHost(oldBuffer1);
			cudaError_t eCopy = cudaMalloc(&pBuffer, size*2 );
			eCopy = cudaMalloc(&oldBuffer, size );
			eCopy = cudaMalloc(&listrect, size *sizeof(int));
			listR = (int *)malloc(size*sizeof(int));
			cudaMallocHost(&oldBuffer1,sizeof(char) *size,cudaHostAllocDefault);
			init=1;

			size_pBuffer=size*2;
			if (cudaSuccess != eCopy) {
				std::cerr << "cudaMalloc failed for component Y: "
						<< cudaGetErrorString(eCopy) << std::endl;
				return NULL;
			}

		}

		gettimeofday(&t1, 0);
		frame++;
		if(getTime(t0, t1)>10000) {
			float elapsed = getTime(t0, t1);
			printf("\rframe = %d   Lan=%d ko  time=%f ms   mode=%d       ",frame/10, framerate/10000,elapsed,mode);
			fflush(stdout);
			gettimeofday(&t0, 0);
			if(mode!= 1 && framerate/10000>1000) {
				checkCudaErrors(nvjpegEncoderParamsSetSamplingFactors(encode_params, NVJPEG_CSS_420, NULL)); //NVJPEG_CSS_420
				checkCudaErrors(nvjpegEncoderParamsSetQuality(encode_params, 50, NULL));
				mode=1;
			}
			if(mode!= 0 && framerate/10000<500) {
				checkCudaErrors(nvjpegEncoderParamsSetSamplingFactors(encode_params, NVJPEG_CSS_422, NULL)); // NVJPEG_CSS_422
				checkCudaErrors(nvjpegEncoderParamsSetQuality(encode_params, 70, NULL)); //70
				mode=0;
			}
			frame=0;
			framerate=0;
		}


		if(recv_buf->raz_cache || h_save!=recv_buf->height || w_save!=recv_buf->width) {
			init=1;
		}

		if(init==1) { // initialisation
			h_save=recv_buf->height;
			w_save=recv_buf->width;
			cudaMemcpy(pBuffer+size, bufInOut,size * sizeof(char), cudaMemcpyHostToDevice);
			char *temp = bufInOut;
			bufInOut=oldBuffer1;
			oldBuffer1=temp;
			decode_encode<<<((size/4)+1023)/1024, 1024>>>(size/4, (char *)pBuffer+size,  (char *)pBuffer ,recv_buf->stride,recv_buf->width);
			init=0;
			memset(listR,-1,size * sizeof(int));
			for(int yy=0;yy<recv_buf->height/8;yy++) {
				listR[(yy*recv_buf->width)]=0;
				listR[1+(yy*recv_buf->width)]=recv_buf->width;
				listR[2+(yy*recv_buf->width)]=-1;
			}
		} else {
			for(int yy=0;;yy++) {
				if(yy*recv_buf->width>size){
					break;
				}
				listR[(yy*recv_buf->width)]=-1;
			}
			if(diffcpu((char *)bufInOut,  (char *)oldBuffer1,(int *) listR ,recv_buf->width,recv_buf->height)==0) {
				/****  no diffrence    ****/
				send_buf->x=0;
				send_buf->y=0;
				send_buf->h=0;
				send_buf->w=0;
				send_buf->lenght = 0;
				send_buf->etat=-1; //last update
				data_recv =send(s2, send_buf, sizeof(jpeg_compress_out), 0);
				if(data_recv<0) {
					close(s2);
					s2=0;
					continue;
				}
				continue;

			}
			cudaMemcpy(pBuffer+size, bufInOut,size * sizeof(char), cudaMemcpyHostToDevice);

			char *temp = bufInOut;
			bufInOut=oldBuffer1;
			oldBuffer1=temp;

			decode_encode<<<((size/4)+1023)/1024, 1024>>>(size/4, (char *)pBuffer+size,  (char *)pBuffer ,recv_buf->stride,recv_buf->width);

			init=0;
		}


		 cudaDeviceSynchronize();
		int pad=0;
		int x=0;
		int y=0;
		int h=0;
		int w=0;
		int add=0;
		int add1=0;
		for(int yy=0;yy<recv_buf->height/8;yy++) {
				x=listR[(yy*recv_buf->width)];
				if(x==-1 ){
					continue;
				}
				h=8;
				w=recv_buf->width;
				y=yy*8;
				while(true) {// recherche les blocs contigus
					if(listR[((yy+1)*recv_buf->width)]>-1) {
						h+=8;
						yy++;
						if((y+h)>=recv_buf->height) {
							break;
						}
					}else{
						break;
					}
				}


				pad=x*3 + (y*3*recv_buf->width);//pad=x+(y*3*recv_buf->width);
				add=recv_buf->width*recv_buf->height;
				add1=recv_buf->width;
				imgdesc = { { pBuffer+pad,
						pBuffer+pad	+ add,
						pBuffer+pad	+ add * 2}, {
						(unsigned int) add1* 3,
						(unsigned int) add1,
						(unsigned int) add1 } };


				error_cuda = nvjpegEncodeImage(nvjpeg_handle, encoder_state, encode_params, &imgdesc, iformat, w, h, NULL);
				if(error_cuda) {
					printf(">>>>> error_cuda  size=%d  x=%d y=%d  w=%d h=%d stride=%d \n\n",size, x,y,w,h,recv_buf->stride);
					init=1;
					continue;
				} else {
					checkCudaErrors(nvjpegEncodeRetrieveBitstream( nvjpeg_handle, encoder_state, NULL, &length, NULL));
					checkCudaErrors(nvjpegEncodeRetrieveBitstream( nvjpeg_handle, encoder_state, (unsigned char*)bufInOut, &length, NULL));
				}

				cudaDeviceSynchronize();
				send_buf->x=x;
				send_buf->y=y;
				send_buf->h=h;
				send_buf->w=w;
				send_buf->lenght = length;
				send_buf->etat = 1;

		//*********************************************************

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

			framerate += length;
		}

		/****  fin maj   ****/
		send_buf->x=0;
		send_buf->y=0;
		send_buf->h=0;
		send_buf->w=0;
		send_buf->lenght = 0;
		send_buf->etat=-1; //last update

		data_recv =send(s2, send_buf, sizeof(jpeg_compress_out), 0);
		if(data_recv<0) {
			close(s2);
			s2=0;
			continue;
		}
		/********/
	}


	if(s2 !=0) {
		close(s2);
	}

	if (bufInOut != NULL) {
		cudaFreeHost(bufInOut);
	}
	checkCudaErrors(cudaFree(pBuffer));

	checkCudaErrors(cudaFree(cin));


	checkCudaErrors(nvjpegEncoderParamsDestroy(encode_params));
	checkCudaErrors(nvjpegEncoderStateDestroy(encoder_state));
	checkCudaErrors(nvjpegJpegStateDestroy(jpeg_state));
	checkCudaErrors(nvjpegDestroy(nvjpeg_handle));

 return NULL;
}


int main(int argc, const char *argv[]) {
	thread_param param;
	param.id_cuda=findCudaDevice(argc, argv);
	param.id_thread=1;
	param.cuda_slot=1;
	if(argc==2) {
		param.cuda_slot=atoi(argv[1]);
	}
    start_server((void *)&param);
	return 0;
}


//
//int main(int argc, const char *argv[]) {
//	pthread_t tid;
////	thread_param param;
//	for(int i=0;i<1;i++) {
//		thread_param param;
//		param.id_cuda=findCudaDevice(argc, argv);
//		param.id_thread=i;
//		param.cuda_slot=(i+1);
//		pthread_create(&tid,NULL,start_server,(void *)param);
//	}
//
//    pthread_exit(NULL);
//	return 0;
//}



//__global__ void diff(int h, char *newImage,char *oldImage,int * rect, int stride, int w) {
//	int i = (blockIdx.x * blockDim.x + threadIdx.x) ;
//	int val=0;
//	int pos=0;
//	int trouver=0;
//	if (i < h ){
//		rect[(i*w)+pos]=-1;
//		for(int x=0;x<w;x++) {
//			val=(x*4)+(i*w*4);
////			printf("i=%d \n",i);
//			if(trouver==0 && (newImage[val]!=oldImage[val] ||
//					   newImage[val+2]!=oldImage[val+2] ||
//					   newImage[val+1]!=oldImage[val+1])) {
//				//save valeur
//				rect[(i*w)+pos]=x;
//				rect[(i*w)+pos+1]=-1;
//		//		printf(">>>>>>>y=%d  x=%d  w=%d trouver=1\n",i,x,w);
//				pos++;
//				trouver=1;
//					} else if(trouver==1 && (newImage[val]==oldImage[val] &&
//								   newImage[val+2]==oldImage[val+2] &&
//								   newImage[val+1]==oldImage[val+1])) {
//							//save valeur
//							rect[(i*w)+pos]=x-1;
//							rect[(i*w)+pos+1]=-1;
//
//			//				printf(" y=%d    x=%d >>>>    <<<<<  x=%d\n", i, rect[(i*w)+pos-1],x);
//							pos++;
//							trouver=0;
//
//						}
//			}
//		}
//}
