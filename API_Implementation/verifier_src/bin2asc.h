
#ifndef BIN2ASC_H
#define BIN2ASC_H


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "globals.h"

#define ASCII_FORMAT	1
#define BIN_FORMAT	2

#define MAX_PREAMBLE 10000

/* If you change MAX_NR_VERTICES, change MAX_NR_VERTICESdiv8 to be
the 1/8th of it */
//#define NMAX 10000	/* maximum number of vertices handles */
#define MAX_NR_VERTICES		5000	/* = NMAX */
#define MAX_NR_VERTICESdiv8	625 	/* = NMAX/8 */



int get_params(void);
char get_edge(  int i,  int j );
void read_graph_DIMACS_bin(const char *file);
void write_graph_DIMACS_ascii(const char *file);

#endif