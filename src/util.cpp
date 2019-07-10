#include<string.h>
#include<malloc.h>
#include<stdio.h>
#include<assert.h> 

void sort(int *array, int count)  
{/* sorts the elements in array.  count is the number of elements in the array */	
	int ii, jj, minIndex, temp;	
	for (ii = 0; ii < count; ii++) 
	{		
		minIndex = ii;		

		for (jj = ii+1; jj < count; jj++)			
			if (array[jj] < array[minIndex]) 
				minIndex = jj;		
		/* now swap the elements at indices minIndex and ii */		
		temp = array[minIndex];		
		array[minIndex] = array[ii];		
		array[ii] = temp;	
	}
} 


int printDistinctElements(const int *array, int count)  {/* sorts the array with count elements in it and prints the distinct elements.
   The original array is left untouched by this function. */

	int *tempArray;
	int ii;
	int item_numbers = 0;

	tempArray = (int *) calloc(sizeof(int) , count);	
	assert(tempArray != NULL);

	memcpy(tempArray, array, sizeof(int)* count);	/* void *memcpy(void *dest, const void *src, size_t n);
		Description:
		Copies a block of n bytes.
		memcpy is available on UNIX system V systems.
		memcpy copies a block of n bytes from src to dest.
		If src and dest overlap, the behavior of memcpy is undefined.
	*/
	sort(tempArray, count);

	//printf("The distinct elements you entered are:\n");
	//printf("%d ", tempArray[0]);
	for (ii = 1; ii < count; ii++)  {
		if (tempArray[ii] != tempArray[ii-1])
			 //printf("%d ", tempArray[ii]);
			item_numbers++;

	}

	printf("\n");

	free(tempArray);  /*free the memory allocated. Don't forget! */
	return item_numbers;
} 

/*
int countDistinct(int arr[], int n) 
{ 
    // First sort the array so that all 
    // occurrences become consecutive 
    sort(arr, arr + n); 
  
    // Traverse the sorted array 
    int res = 0; 
    for (int i = 0; i < n; i++) { 
  
        // Move the index ahead while 
        // there are duplicates 
        while (i < n - 1 && arr[i] == arr[i + 1]) 
            i++; 
  
        res++; 
    } 
  
    return res; 
} 
*/