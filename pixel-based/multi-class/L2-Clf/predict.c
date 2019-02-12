#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>


int main(int argc, char* argv[])
{
    int numtasks, taskid;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks); /* store number of tasks in this variable*/
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid); /*get task ID*/
    MPI_Barrier(MPI_COMM_WORLD);
    int rowSplits[378];
    int i = 0, j = 0;
    while(i <= 94321){
	rowSplits[i/250] = i;
	i = i + 250;
    }
    rowSplits[377] = 94321;
    //for(i = 0; i < 378; i++){
    //	printf("%d\n",rowSplits[i]);
    // }
    int currentPointer = 0; 
    int numDead = 0;
    if(taskid == 0){
        while(1){
            int masterReceivedID;
            int masterReceiveCount = 0;
	    int currentPointerFinal = -1;
            MPI_Status masterStatus;
            MPI_Recv(&masterReceivedID, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &masterStatus);
	    if (currentPointer < 377) {
        	//we have not finished all jobs yet, send a value
		MPI_Send(&currentPointer, 1, MPI_INT, masterReceivedID, 1, MPI_COMM_WORLD);
		currentPointer = currentPointer + 1;
	    }
	    else{
		 MPI_Send(&currentPointerFinal, 1, MPI_INT, masterReceivedID, 0, MPI_COMM_WORLD);
		 numDead++;
                 if(numDead == numtasks-1){

			break;
		}
	    }
        }
    }
    else{
        while(1){
		 MPI_Send(&taskid, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
		  MPI_Status status;
                  MPI_Recv(&currentPointer, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		  
                  //printf("Node %d Starting now: %d\n",taskid,currentPointer);
                  if(currentPointer >=0){			  
                      int STARTINDEX = rowSplits[currentPointer];
                      int ENDINDEX = rowSplits[currentPointer+1];
                      char *SHELLSCRIPT = (char*)malloc(100*sizeof(char));
                      snprintf(SHELLSCRIPT, 100, "python classifyImage.py %d %d",STARTINDEX, ENDINDEX);
                      //snprintf(SHELLSCRIPT, 100, "python test.py");
                      printf("Node %d Starting now: %d - %d\n",taskid,STARTINDEX, ENDINDEX);
                      system(SHELLSCRIPT);    //it will run the script inside the c code. 
                     }
		  else{
                     break;
		  } 

         }
     }
MPI_Finalize();
return(0);
}
