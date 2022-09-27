#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <stdlib.h>
#include <time.h>

int Seed = 1;

int Debug = 0;
int Debug2 = 0;

#define MASK 4294967295 //2^32-1
#define PRIME 2305843009213693951 //2^61-1
#define MASKFORTY 1099511627775
#define MASKTS 134217727 //2^27-1

typedef unsigned long long uint64;
typedef unsigned int uint32;

void initurn(int seed)
{
    struct timeval tv;
    struct timezone tz;
    
    if (seed == 0) {
	gettimeofday(&tv,&tz);
	srand48(tv.tv_usec);
    }

    else srand48(seed);
}


double urn()
{
    return(drand48());
}

long long rndl(long long x)
{
    return(floorl(x*drand48()));
}

int rnd(int x)
{
    return(floor(x*drand48()));
}

/*  Generates a random number with Poisson distrubtion,
 *  mean 1
 */

double Prng()
{
    return (-1.0 * log(drand48()));
}

//efficient modular arithmetic function for p=2^61-1. Only works for this value of p.
//This function might
//return a number slightly greater than p (possibly by an additive factor of 8);
//It'd be cleaner to check if the answer is greater than p
//and if so subtract p, but I don't want to pay that
//efficiency hit here, so the user should just be aware of this.
__host__ __device__ uint64 myMod(uint64 x)
{
  return (x >> 61) + (x & PRIME);
}  

//efficient modular multiplication function mod 2^61-1
__host__ __device__ uint64 myModMult(uint64 x, uint64 y)
{
	uint64 hi_x = x >> 32;
	uint64 hi_y = y >> 32;
	uint64 low_x = x & MASK;
	uint64 low_y = y & MASK;
	//since myMod might return something slightly large than 2^61-1,
  
	//we need to multiply by 8 in two pieces to avoid overflow.
	uint64 piece1 = myMod((hi_x * hi_y)<< 3);
	uint64 z = (hi_x * low_y + hi_y * low_x);
	uint64 hi_z = z >> 32;
	uint64 low_z = z & MASK;

	//Note 2^64 mod (2^61-1) is 8
	uint64 piece2 = myMod((hi_z<<3) + myMod((low_z << 32)));
	uint64 piece3 = myMod(low_x * low_y);
	uint64 result = myMod(piece1 + piece2 + piece3);
	return result;
}

__host__ __device__ int myHash(uint64 x, int num_hash, int num_cells)
{
  int answer;
  if(num_hash == 0)
  {
     answer = ((int) (myModMult(x, 129034299023) + 12493290)) % num_cells;
  }  
  else if(num_hash == 1)
  {
    answer = ((int) myModMult(x, 9034903490) + 9023902390) % num_cells;
  }  
  else if(num_hash == 2)
  {
    answer = ((int) myModMult(x, 7834782390) + 9034590123) % num_cells;
  }  
  else if(num_hash == 3)
  {
    answer = ((int) myModMult(x, 2340923923) + 34043903) % num_cells;
  }
  else if(num_hash == 4)
  {
    answer = ((int) myModMult(x, 90239090) + 21300230) % num_cells;
  }
  else if(num_hash == 5)
  {
    answer = ((int) myModMult(x,459054903490) + 2302323490234) % num_cells;
  }
  
  if(answer >= 0)
    return answer;
  else
    return answer + num_cells; 
}


__global__ void testatomicXor(unsigned int* array)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i == 0)
	{
	  atomicXor(array, (uint32) 1);
	}
	if(i == 2)
	{
	  atomicXor(array, (uint32) 2);
	}
	if(i == 2)
	{
	  atomicXor(array, (uint32) 4);
	}
}


__global__ void ZeroOut(uint32* tablekeyhigh, int tablesize)
{ 
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < tablesize)
	  tablekeyhigh[i] = 0;
}

__global__ void ZeroOut64(uint64* tablekeyhigh, int tablesize)
{ 
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < tablesize)
	  tablekeyhigh[i] = 0;
}

__host__ __device__ uint64 checksum(uint64 x)
{
  uint64 answer = myModMult(x, x);
  if(answer < PRIME)
  	return answer;
  else 
  	return answer - PRIME;
}


__global__ void ComputeVals(uint64* keyvalueA_GPU, uint64* keysA_GPU, int elts)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i < elts)
	{
	  keyvalueA_GPU[i] = checksum(keysA_GPU[i]);
	}
}

__global__ void FillTable(uint32* tablekeyhigh, uint32* tablekeylow, uint32* tablevaluehigh, uint32* tablevaluelow, uint64* keysA, uint64* keyvalueA, int tablehashes, int pertablesize, int elts ) 
{ 
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= elts)
		return;
	int cell; 
	
	uint32 xorkeyhigh = (keysA[i]>>32) & MASK;
	uint32 xorkeylow = keysA[i] & MASK;
	uint32 xorvaluehigh = (keyvalueA[i] >> 32) & MASK;
	uint32 xorvaluelow = keyvalueA[i] & MASK;
	
	for (int j = 0; j < tablehashes; j++) 
	{
		cell = j*pertablesize + myHash((uint64) keysA[i], j, pertablesize);
	    atomicXor(&(tablekeyhigh[cell]), xorkeyhigh);
	    atomicXor(&(tablekeylow[cell]), xorkeylow);
	    atomicXor(&(tablevaluehigh[cell]), xorvaluehigh);
	    atomicXor(&(tablevaluelow[cell]), xorvaluelow);
	}
}

__global__ void peel(uint32* tablekeyhigh, uint32* tablekeylow, uint32* tablevaluehigh, uint32* tablevaluelow, uint32* recovered, uint64* recoveredsymbols, int tablehashes, int pertablesize, int elts, int count ) 
{ 
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= pertablesize)
		return;

	uint64 templong;
	uint64 combined_key; 
	uint64 combined_val;
	int cell;
	
	//replace i (local table index) with global index
	int k = count*pertablesize + i;
	
	uint32 xorkeyhigh = tablekeyhigh[k];
	uint32 xorkeylow = tablekeylow[k];
	uint32 xorvaluehigh = tablevaluehigh[k];
	uint32 xorvaluelow= tablevaluelow[k];
		
	combined_key = (((uint64) xorkeyhigh) << 32) + (uint64) xorkeylow;
	combined_val = (((uint64) xorvaluehigh) << 32) + (uint64) xorvaluelow;
	templong = checksum(combined_key);

	if ( (combined_val != 0) && ((combined_val == templong) || (combined_val == templong - PRIME) || (combined_val == templong+PRIME))) 
	{
		recovered[k] = recovered[k]+1;
		recoveredsymbols[k] = combined_key;
		//do the peeling
	  	for (int j = 0; j < tablehashes; j++) 
	  	{
	    	//peel cell l, where cell is another cell that the j'th message symbol hashes to
	    	cell = j*pertablesize + myHash((uint64) combined_key, j, pertablesize);
	    	atomicXor(&(tablekeyhigh[cell]), xorkeyhigh);
	    	atomicXor(&(tablekeylow[cell]), xorkeylow);
	    	atomicXor(&(tablevaluehigh[cell]), xorvaluehigh);
	    	atomicXor(&(tablevaluelow[cell]), xorvaluelow);
	  	}
	}
}

__global__ void myReduce(uint32* table, uint32* result, int tablesize)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	uint32 sum = 0;
	if(i == 0)
	{
	  for(int j = 0; j < tablesize; j++)
	  {
	    sum = sum + table[j];
	  }
	  *result = sum;
	}
}

void cudasafe( cudaError_t error)
{
    if(error!=cudaSuccess) { fprintf(stderr,"ERROR: : %i\n",error); exit(-1); }
}


main (int argc, char * argv[])
{   
    if (Debug) 
    {
    	printf("Step 1\n");
    	fflush(stdout);
    }

	int i,j,k,m;
    int trials,elts,errors,baderrors,tablesize,tablehashes, pertablesize, num_recovered, old_recovered;
        
    trials = atoi(argv[1]);
    elts = atoi(argv[2]);
    errors = atoi(argv[3]);
    baderrors = atoi(argv[4]);
    tablesize = atoi(argv[5]);
    tablehashes = atoi(argv[6]);

    pertablesize = tablesize/tablehashes;
    fprintf(stdout, "trials %d elts %d errors %d baderrors %d tablesize (total) %d, pertablesize %d tablehashes %d\n",trials, elts, errors, baderrors, tablesize, pertablesize, tablehashes);
    
    uint64* keysA = (uint64*) malloc(elts*sizeof(uint64));
    uint64* keysB = (uint64*) malloc(elts*sizeof(uint64));

    int errorpositions[50000];
    
    uint64* keysA_GPU;
    uint64* keysB_GPU; 
    uint64* keyvalueA_GPU; 
    uint64* keyvalueB_GPU;
    
    //unfortunately I'm going to have to break table into high-order 32 bits
    //and low-order 32 bits of each field, because the resonance GPU
    //doesn't support 64-bit atomic AND operations
    uint32* tablekeyhigh_GPU; 
    uint32* tablekeylow_GPU;
    uint32* tablevaluehigh_GPU; 
    uint32* tablevaluelow_GPU; 
    uint32* recovered_GPU;
    uint64* recoveredsymbols_GPU;
      
  	cudasafe(cudaMalloc(&keysA_GPU, elts*sizeof(uint64)));
    cudasafe(cudaMalloc(&keysB_GPU, elts*sizeof(uint64)));
	cudasafe(cudaMalloc(&keyvalueA_GPU, elts*sizeof(uint64)));
	cudasafe(cudaMalloc(&keyvalueB_GPU, elts*sizeof(uint64)));
	cudasafe(cudaMalloc(&tablekeyhigh_GPU, tablesize*sizeof(uint32)));
	cudasafe(cudaMalloc(&tablekeylow_GPU, tablesize*sizeof(uint32)));
	cudasafe(cudaMalloc(&tablevaluehigh_GPU, tablesize*sizeof(uint32)));
	cudasafe(cudaMalloc(&tablevaluelow_GPU, tablesize*sizeof(uint32)));
	cudasafe(cudaMalloc(&recovered_GPU, (tablesize+1)*sizeof(uint32)));
	cudasafe(cudaMalloc(&recoveredsymbols_GPU, tablesize*sizeof(uint64)));
	
	//handle the (non-sensical) case where elts < tablesize
	//this only holds if there are more IBLT cells than message symbols, which is never
	//necessary for recovery but is useful for debugging purposes
	int max=100;
	if(elts < tablesize)
	  max = tablesize;
	else
	  max = elts;
	  
	dim3 threadsPerBlock(512);
	dim3 numBlocks((max/512) + 1);

    int done;
    
    initurn(Seed);
    clock_t t=0;
    clock_t a_time = 0;
    clock_t b_time = 0;
    clock_t p_time = 0;
    clock_t c_time = 0;
    
    for (m = 0; m < trials; m++) 
    {
      printf("Trial %d\n",m);
      fflush(stdout);

      // set up keys 
      // Question:  how should we set up the keys errors?  Here they are random.  
      //   In general we might want worst-case?  We might want to set up as val,pos -- right now
      //   we're ignoring the position issue, and only introducing errors into the message symbol field
      //
            
      //set up keys (20 bit random message symbols, followed by 20-bit position)
      //for both A and B, assuming no message errors right now.
      for (i = 0; i < elts; i++) 
      {
		keysA[i] = (((uint64) (rndl(MASKTS-1)+1)) << 32) + i;
		keysB[i] = keysA[i];
      }

      // set up errors 
      //choose error positions in message. first we will not make sure the errors are at distinct
      //message locations, but then we will post-process for distinctness
      if (errors > 0) 
      {
		i = rnd(elts);
		errorpositions[0] = i;
      }
      for (j = 1; j < errors; j++) 
      {
        //fprintf(stdout, "setting up error %d\n", j);
		done = 0;
		while (!done) 
		{
	  		done = 1;
	  		i = rnd(elts);
	
	  		for (k = 0; k < j; k++) 
	  		{
	    		if (i == errorpositions[k]) 
	    		{
	    			//fprintf(stdout, "i=%d, matches errorpositions[%d], which is %d\n", i, k, errorpositions[k]);
	    			done = 0;
	    		}
	  		}
		}
		errorpositions[j] = i;
      }

	  //now actually introduce the errors into the message
      for (j = 0; j < errors; j++) 
      {
		i = errorpositions[j];
		while (keysB[i] == keysA[i]) 
		{
			keysB[i] = (((uint64) (rndl(MASKTS-1)+1)) << 32)+i;
		}
      }
      
	  ZeroOut<<<numBlocks, threadsPerBlock>>>(tablekeyhigh_GPU, tablesize);
	  ZeroOut<<<numBlocks, threadsPerBlock>>>(tablekeylow_GPU, tablesize);
	  ZeroOut<<<numBlocks, threadsPerBlock>>>(tablevaluehigh_GPU, tablesize);
	  ZeroOut<<<numBlocks, threadsPerBlock>>>(tablevaluelow_GPU, tablesize);
	  ZeroOut<<<numBlocks, threadsPerBlock>>>(recovered_GPU, tablesize+1);
	  ZeroOut64<<<numBlocks, threadsPerBlock>>>(recoveredsymbols_GPU, tablesize);
      cudaThreadSynchronize();
      
	  t = clock();
	  	  
	  cudasafe(cudaMemcpy(keysA_GPU, keysA, elts*sizeof(uint64), cudaMemcpyHostToDevice));
	  cudasafe(cudaMemcpy(keysB_GPU, keysB, elts*sizeof(uint64), cudaMemcpyHostToDevice));
	  
	  c_time += clock()-t;	
	  
	  t = clock();
	  
	  ComputeVals<<<numBlocks, threadsPerBlock>>>(keyvalueA_GPU, keysA_GPU, elts);
	    
	  
	  cudaThreadSynchronize();
	  //printf("done computing vals for Alice\n");
	  //fflush(stdout);
	  
	  FillTable<<<numBlocks, threadsPerBlock>>>(tablekeyhigh_GPU, tablekeylow_GPU, tablevaluehigh_GPU, tablevaluelow_GPU, keysA_GPU, keyvalueA_GPU, tablehashes, pertablesize, elts);
	  cudaThreadSynchronize();
	  //printf("done filling table for Alice\n");
	  //fflush(stdout);
      a_time += clock()-t;
      
      
     
      t=clock();
      
       //have Bob delete his received symbols from the table
      ComputeVals<<<numBlocks, threadsPerBlock>>>(keyvalueB_GPU, keysB_GPU, elts);
      cudaThreadSynchronize();
      //printf("done computing vals for Bob\n");
      
      FillTable<<<numBlocks, threadsPerBlock>>>(tablekeyhigh_GPU, tablekeylow_GPU, tablevaluehigh_GPU, tablevaluelow_GPU, keysB_GPU, keyvalueB_GPU, tablehashes, pertablesize, elts);
      cudaThreadSynchronize();
      b_time += clock()-t;
       
      old_recovered = -1;
      num_recovered = 0;
      //peel in iterations until you go an entire iteration without peeling anyone new
      
      while(old_recovered != num_recovered)
      {
      	 t = clock();
         old_recovered = num_recovered;
         for(j = 0; j < tablehashes; j++)
         {
        	peel<<<numBlocks, threadsPerBlock>>>(tablekeyhigh_GPU, tablekeylow_GPU, tablevaluehigh_GPU, tablevaluelow_GPU, recovered_GPU, recoveredsymbols_GPU, tablehashes, pertablesize, elts, j);  
            cudaThreadSynchronize();   
         }
         p_time += clock()-t;      
         myReduce<<<numBlocks, threadsPerBlock>>>(recovered_GPU, &(recovered_GPU[tablesize]), tablesize);
         cudaThreadSynchronize();
         cudaMemcpy(&num_recovered, recovered_GPU + tablesize, sizeof(uint32), cudaMemcpyDeviceToHost);
      }
      printf("Recovered %d of %d items in the IBLT.\n", num_recovered, 2*errors);
      fflush(stdout);
    }
    printf("Total Alice time is: %lf, Total Bob time is: %lf, Total Peel time is: %lf, Total copy time is: %lf", (double) a_time/CLOCKS_PER_SEC,  (double) b_time/CLOCKS_PER_SEC, (double) p_time/CLOCKS_PER_SEC, (double) c_time/CLOCKS_PER_SEC);
}

