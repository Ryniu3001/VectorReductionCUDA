// Kod do wersji 2 zadania.

#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

// for srand( time( NULL ) )
#include <ctime>

/**
 * CUDA Kernel Device code
 Brak branch divergence! Warpy ktore policzyly swoja sume nie wykonuja ifa i sie koncza.
 nie ubiegaja sie o procesor!

 Sa konfikty ?

 */
__global__ void reduction(int *i_data, int *o_data, int numElements)
{
	extern __shared__ int sdata[];
	// Kazdy watek laduje jeden element z pamieci globalnej to pamieci wspoldzielonej
	unsigned int thId = threadIdx.x;							//ID w obrebie bloku ?
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;		//globalne id watku ??
	sdata[thId] = 0;
	if (i < numElements)
		sdata[thId] = i_data[i];
	__syncthreads();

	// Redukcja w pamieci wspoldzielonej
	for (unsigned int s = 1; s < blockDim.x; s *= 2){
		int index = 2 * s * thId;

		if (index < blockDim.x){
			sdata[index] += sdata[index + s];				
		}												
		__syncthreads();
	}

	//zapis wyniku tego bloku do globalnej pamieci
	if (thId == 0)
		o_data[blockIdx.x] = sdata[0];
}

int main(void)
{
	cudaError_t err = cudaSuccess;

	int numElements = 50000;
	size_t size = numElements * sizeof(int);
	printf("[Vector reduction of %d elements]\n", numElements);

	//Determine amount of blocks and threads per block
	int threadsPerBlock = 256;
	int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

	size_t o_size = blocksPerGrid * sizeof(int);

	// Allocate the host vectors
	int *h_input = (int *)malloc(size);

	int h_output = 0;

	if (h_input == NULL)
	{
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}

	// Initialize the host vector
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

	printf("Copy input data from the host memory to the CUDA device\n");
	err = cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Launch the reduction CUDA Kernel
	printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
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
		exit(EXIT_FAILURE);
	}

	if (turn)
		err = cudaMemcpy(&h_output, &d_input[0], sizeof(int), cudaMemcpyDeviceToHost);
	else
		err = cudaMemcpy(&h_output, &d_output[0], sizeof(int), cudaMemcpyDeviceToHost);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy output vector from device to host (error code %s)!\n", cudaGetErrorString(err));
		printf("turn = %d\n numElem = %d\n", turn, numElements);
		exit(EXIT_FAILURE);
	}

	// Verify that the result vector is correct

	if (h_output != checkSum)
	{
		fprintf(stderr, "Result verification failed! host result: %d !=  device result: %d\n", checkSum, h_output);
		exit(EXIT_FAILURE);
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

	err = cudaDeviceReset();

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	printf("Done\n");
	return 0;
}