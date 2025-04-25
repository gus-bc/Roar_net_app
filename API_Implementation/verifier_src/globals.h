
#ifndef GLOBALS_H
#define GLOBALS_H



#define VERBOSE 1

#define TRUE 1
#define FALSE 0

#define LINE_BUF_LEN     2000



extern int **solution_sets;
extern int *solution_sizes;

extern  long int NUM_COL;
extern  long int MAX_COL;
extern  long int MIN_COL;
extern  long int NUM_EDGES;
extern  long int NUM_NODES;
extern float DENSITY;
extern long int COLORED_NODES;
extern  long int CONF_EDGES;
extern  long int CONF_NODES;
extern  long int NOT_IN_LIST;
extern  long int SET_SIZE;
extern  long int SUM_COL;

extern const char *instance_name;
extern const char *solution_name;
extern int problem;




#endif