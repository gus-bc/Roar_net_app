/*****************************************************************************/
/*                                                                           */
/* Version:  4.0   Date:  2004-2024    File: main.c                          */
/* Author:  Marco Chiarandini                                                */
/* email: marco@imada.sdu.dk                                                 */
/*                                                                           */
/*****************************************************************************/


#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "globals.h"
#include "utilities.h"
#include "bin2asc.h"
#include "checker.h"


int **solution_sets;
int *solution_sizes;
long int **constraints_matrix;
long int NUM_COL;
long int SUM_COL;
long int MIN_COL;
long int MAX_COL;
long int NUM_EDGES;
long int NUM_NODES;
float DENSITY;
long int COLORED_NODES;
long int CONF_NODES;
long int CONF_EDGES;
long int NOT_IN_LIST;
long int SET_SIZE;
const char *instance_name;
const char *solution_name;
int problem;

int main(int argc, char *argv[])
{

  char proper;

  checkOptions("test_sol", argc, argv );

  proper = checkConstraints();
  /*num_of_viol = count_violations();*/
  
  
  printf("\ninstance: %s\n",instance_name);
  printf("solution: %s\n",solution_name);
  printf("nodes: %ld\n",NUM_NODES);
  printf("edges: %ld\n",NUM_EDGES);
  printf("density: %.3f\n",DENSITY);
  
  if (proper)
    {
      printf("+---------------------------------------------------------+\n");
      printf("| The proposed coloring uses %4ld colors                  |\n",NUM_COL);
      printf("| from %2ld to %4ld with a max spread (T-span) of %4ld      |\n",MIN_COL,MAX_COL, MAX_COL - MIN_COL);
      printf("| It has a sum of %4ld.                                   |\n",SUM_COL);
      printf("| It satisfies all the constraints of the instance!       |\n");
      printf("+---------------------------------------------------------+\n");   
    }
  else 
    {
      printf("+---------------------------------------------------------------------+\n");
      printf("| The proposed solution uses %4ld colors                              |\n",NUM_COL);
      printf("| from %2ld to %4ld with a max spread (T-span) of %4ld,                 |\n",MIN_COL,MAX_COL, MAX_COL - MIN_COL);
      printf("| It has a total sum of %4ld.                                         |\n",SUM_COL);
      printf("| However, it DOES NOT satisfy all the constraints of the instance!   |\n");
      printf("| There are:                                                          |\n");
      printf("| %4ld vertices not colored,                                          |\n",NUM_NODES-COLORED_NODES);
      printf("| %4ld constraints (edges) not satisfied,                             |\n",CONF_EDGES);
      printf("| %4ld nodes in conflict, and                                         |\n",CONF_NODES);
      printf("| %4ld nodes which do not receive the right n. of col.                |\n",SET_SIZE);
      printf("| The nodes that receive a colour not                                 |\n");
      printf("| in the preference list are %4ld                                     |\n",NOT_IN_LIST);
      printf("+---------------------------------------------------------------------+\n");
    }

  return EXIT_SUCCESS;
}
