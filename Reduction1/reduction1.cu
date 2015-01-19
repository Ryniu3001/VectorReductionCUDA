// Kod do wersji 1 zadania.

#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

// for srand( time( NULL ) )
#include <ctime>
#include <stdlib.h>
#include <math.h>
/**
 * CUDA Kernel Device code
Redukcja wektora.
W pierwsze iteracji dla s=1 w¹tki o thId parzystym dodaj¹ elementy tablicy w miescu np. sdata[thId] += sdata[thId+1]
W drugiej iteracji dla s=2, odtep miedzy dodawanymi elementami rowny jest s, dlatego pracuje co 4 w¹tek dodaj¹c elementy oddalone o 2
itd....

Czy sa konflikty ?? Jesli nie to zmienic typ danych na double (8 bajtowy) zeby dana nie zmiescila sie w jednym slowie pamieci karty
Wtedy bedzie 2-way conflict
 */
__global__ void reduction(int *i_data, int *o_data, int numElements)
{
	extern __shared__ int sdata[];
	// Kazdy watek laduje jeden element z pamieci globalnej to pamieci wspoldzielonej
	unsigned int thId = threadIdx.x;							//ID w obrebie warpu ? 0-31 ?
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;		//globalne id watku ??
	sdata[thId] = i_data[i];
	__syncthreads();

	// Redukcja w pamieci wspoldzielonej
	for (unsigned int s = 1; s < blockDim.x; s *= 2){
		if (thId % (2 * s) == 0){
			sdata[thId] += sdata[thId + s];				//powoduje branch divergence, coraz mniej watow w warpie pracuje
		}												//czy sa konflikty ??
		__syncthreads();
	}

	//zapis wyniku tego bloku do globalnej pamieci
	if (thId == 0)
		o_data[blockIdx.x] = sdata[0];
}

/**
 * Host main routine
 */
int main(void)
{
	srand(time(NULL));
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
    int numElements = 50000;
    size_t size = numElements * sizeof(int);
    printf("[Vector reduction of %d elements]\n", numElements);

	//Determine amount of blocks and threads per block
	int threadsPerBlock = 256;
	int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

	size_t o_size = blocksPerGrid * sizeof(int);

    // Allocate the host input vector
    int *h_input = (int *)malloc(size);

    // Allocate the host output vector
    int *h_output = (int *)malloc(sizeof(int));

    // Verify that allocations succeeded
    if (h_input == NULL || h_output == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host vectors
	int checkSum = 0;
    for (int i = 0; i < numElements; ++i)
    {
        h_input[i] = 1;
		checkSum += h_input[i];
    }

    // Allocate the device input vector
    int *d_input = NULL;
    err = cudaMalloc((void **)&d_input, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device input vector (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector
    int *d_output = NULL;
    err = cudaMalloc((void **)&d_output, o_size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device output vector (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vector host memory to the device input vector in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	//Zerowanie output na karcie 
	/*
    err = cudaMemcpy(d_output, h_output, o_size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy output vector from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	*/
    // Launch the reduction CUDA Kernel
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
   
	/*
	reduction<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, numElements);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch reduction kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	*/


	bool turn = true;

	while (true){

		if (turn){

			reduction << <blocksPerGrid, threadsPerBlock, threadsPerBlock*sizeof(int) >> >(d_input, d_output, numElements);
			turn = false;

		}
		else{

			reduction << <blocksPerGrid, threadsPerBlock, threadsPerBlock*sizeof(int) >> >(d_output, d_input, numElements);
			turn = true;

		}

		err = cudaGetLastError();

		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to launch reduction kernel (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		if (blocksPerGrid == 1) break;

		numElements = blocksPerGrid;
		blocksPerGrid = ceil((double)blocksPerGrid / threadsPerBlock);

	}
	
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernel!\n", err);
		//exit(EXIT_FAILURE);
	}



	if (turn)
		err = cudaMemcpy(h_output, d_input, sizeof(int), cudaMemcpyDeviceToHost);
	else
		err = cudaMemcpy(h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy output vector from device to host (error code %s)!\n", cudaGetErrorString(err));
		printf("turn = %d\n numElem = %d\n",turn, numElements);
		//exit(EXIT_FAILURE);
	}


    // Copy the device result vector in device memory to the host result vector
    // in host memory.
	/*
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_output, d_output, o_size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy output vector from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	*/
    // Verify that the result vector is correct

	if (h_output[0] != checkSum)
	{
		fprintf(stderr, "Result verification failed! host result: %d !=  device result: %d\n",checkSum, h_output[0]);
		//exit(EXIT_FAILURE);
	}

    printf("Test PASSED\n");

    // Free device global memory
    err = cudaFree(d_input);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device input vector (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_output);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device output vector (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(h_input);
    free(h_output);

    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Done\n");
	system("PAUSE");
    return 0;
}

