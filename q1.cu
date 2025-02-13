#include<stdio.h>
#include<cuda.h>

__global__ void print(){
	printf("it works\n");
}

int main(){
	print<<<1,1>>>();
	cudaDeviceSynchronize();
	return 0;
}
