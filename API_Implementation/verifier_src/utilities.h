#ifndef UTILITIES_H
#define UTILITIES_H


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "globals.h"



#ifdef NOGETOPT
static char name_buf[LINE_BUF_LEN];
static char sol_buf[LINE_BUF_LEN];
//static int     opt;
#endif



#ifndef STR_ERR_UNKNOWN_LONG_OPT
# define STR_ERR_UNKNOWN_LONG_OPT   "%s: unrecognized option `--%s'\n"
#endif

#ifndef STR_ERR_LONG_OPT_AMBIGUOUS
# define STR_ERR_LONG_OPT_AMBIGUOUS "%s: option `--%s' is ambiguous\n"
#endif

#ifndef STR_ERR_MISSING_ARG_LONG
# define STR_ERR_MISSING_ARG_LONG   "%s: option `--%s' requires an argument\n"
#endif

#ifndef STR_ERR_UNEXPEC_ARG_LONG
# define STR_ERR_UNEXPEC_ARG_LONG   "%s: option `--%s' doesn't allow an argument\n"
#endif

#ifndef STR_ERR_UNKNOWN_SHORT_OPT
# define STR_ERR_UNKNOWN_SHORT_OPT  "%s: unrecognized option `-%c'\n"
#endif

#ifndef STR_ERR_MISSING_ARG_SHORT
# define STR_ERR_MISSING_ARG_SHORT  "%s: option `-%c' requires an argument\n"
#endif

#define STR_HELP_INSTANCE			\
  "  -i, --instance ARG    the instance file\n"

#define STR_HELP_SOLUTION			\
  "  -s, --solution ARG    the solution file\n"

#define STR_HELP_PROBLEM				\
  "  -p, --problem ARG     the problem -- see help\n"

#define STR_HELP_HELP						\
  "      --help            display this help text and exit\n"

#define STR_HELP				\
  STR_HELP_INSTANCE				\
  STR_HELP_SOLUTION				\
  STR_HELP_PROBLEM				\
  STR_HELP_HELP



void usage();
void checkOptions(const char *const program_name, const int argc, char **const argv);
int parse_options (const char *const program_name, const int argc, char **const argv);



long int **allocateMatrix(long int rows, long int cols);
long int *allocateLongVector(long int size);
int *allocateIntVector(int size);
int int_compare(const void * a, const void * b);

#endif