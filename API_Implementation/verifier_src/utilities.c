
#include "utilities.h"


/* Set to 1 if option --instance (-i) has been specified.  */
char opt_instance;

/* Set to 1 if option --solution (-s) has been specified.  */
char opt_solution;

/* Set to 1 if option --problem (-t) has been specified.  */
char opt_problem;

/* Set to 1 if option --help has been specified.  */
char opt_help;

/* Parse command line options.  Return index of first non-option argument,
   or -1 if an error is encountered.  */



void usage() {
  fprintf(stderr, "\nSolution Verifier for Graph Coloring, V3.1\n\n");
  fprintf(stderr,
	  "Usage: test_sol -i [INST_FILE] -s [SOL_FILE] -p [PROB_TYPE]\n");
  fprintf(stderr,
	  "--help            -h  \t help: prints this information\n\n");
  fprintf(stderr,
	  "Instance File Format: DIMACS file format. Nodes start from 1.\n");
  fprintf(stderr,
	  "Solution File Format: colors assigned to nodes wrote in column.\n");
  fprintf(stderr,
	  "After each entry the character \\n (new line) has to be printed.\n");
  fprintf(stderr,
	  "Colors start from 1 and the first color in the column represent\n");
  fprintf(stderr, "the color assigned to node 1. \n\n");
  fprintf(stderr,
	  "The Problem: coloring 1, list coloring 2, set coloring 3,\n");
  fprintf(stderr,
	  "             T-coloring 4, set T-coloring 5, interval-col 6\n");
  fprintf(stderr, "Report bugs to <marco@imada.sdu.dk>\n\n");
}

void checkOptions(const char * const program_name, const int argc,
		  char ** const argv) {
  int r = parse_options(program_name, argc, argv);

  if (r < 0) {
    fprintf(stderr,
	    "test_sol: You must specify input files and type of problem\n");
    fprintf(stderr, "Try `test_sol --help' for more information.\n");
    exit(1);
  }
  if (opt_help) {
    usage();
    exit(1);
  }
  if (!opt_instance || !opt_solution || !opt_problem) {
    fprintf(stderr,
	    "test_sol: You must specify input files and type of problem\n");
    fprintf(stderr, "Try `test_sol --help' for more information.\n");
    exit(1);
  }
}

int parse_options(const char * const program_name, const int argc,
		  char ** const argv) {
  static const char * const optstr__instance = "instance";
  static const char * const optstr__solution = "solution";
  static const char * const optstr__problem = "problem";
  static const char * const optstr__help = "help";
  int i = 0;
  opt_instance = 0;
  opt_solution = 0;
  opt_problem = 0;
  opt_help = 0;
  instance_name = 0;
  solution_name = 0;
  problem = 0;
  while (++i < argc) {
    const char *option = argv[i];
    if (*option != '-')
      return i;
    else if (*++option == '\0')
      return i;
    else if (*option == '-') {
      const char *argument;
      size_t option_len;
      ++option;
      if ((argument = strchr(option, '=')) == option)
	goto error_unknown_long_opt;
      else if (argument == 0)
	option_len = strlen(option);
      else
	option_len = argument++ - option;
      switch (*option) {
      case '\0':
	return i + 1;
      case 'h':
	if (strncmp(option + 1, optstr__help + 1, option_len - 1)
	    == 0) {
	  if (argument != 0) {
	    option = optstr__help;
	    goto error_unexpec_arg_long;
	  }
	  opt_help = 1;
	  return i + 1;
	}
	goto error_unknown_long_opt;
      case 'i':
	if (strncmp(option + 1, optstr__instance + 1, option_len - 1)
	    == 0) {
	  if (argument != 0)
	    instance_name = argument;
	  else if (++i < argc)
	    instance_name = argv[i];
	  else {
	    option = optstr__instance;
	    goto error_missing_arg_long;
	  }
	  opt_instance = 1;
	  break;
	}
	goto error_unknown_long_opt;
      case 'p':
	if (strncmp(option + 1, optstr__problem + 1, option_len - 1)
	    == 0) {
	  if (argument != 0)
	    problem = atoi(argument);
	  else if (++i < argc)
	    problem = atoi(argv[i]);
	  else {
	    option = optstr__problem;
	    goto error_missing_arg_long;
	  }
	  opt_problem = 1;
	  break;
	}
	goto error_unknown_long_opt;
      case 's':
	if (strncmp(option + 1, optstr__solution + 1, option_len - 1)
	    == 0) {
	  if (argument != 0)
	    solution_name = argument;
	  else if (++i < argc)
	    solution_name = argv[i];
	  else {
	    option = optstr__solution;
	    goto error_missing_arg_long;
	  }
	  opt_solution = 1;
	  break;
	}
      default:
      error_unknown_long_opt: fprintf(stderr,
				      STR_ERR_UNKNOWN_LONG_OPT, program_name, option);
	return -1;
      error_missing_arg_long: fprintf(stderr,
				      STR_ERR_MISSING_ARG_LONG, program_name, option);
	return -1;
      error_unexpec_arg_long: fprintf(stderr,
				      STR_ERR_UNEXPEC_ARG_LONG, program_name, option);
	return -1;
      }
    } else
      do {
	switch (*option) {
	case 'i':
	  if (option[1] != '\0')
	    instance_name = option + 1;
	  else if (++i < argc)
	    instance_name = argv[i];
	  else
	    goto error_missing_arg_short;
	  option = "\0";
	  opt_instance = 1;
	  break;
	case 's':
	  if (option[1] != '\0')
	    solution_name = option + 1;
	  else if (++i < argc)
	    solution_name = argv[i];
	  else
	    goto error_missing_arg_short;
	  option = "\0";
	  opt_solution = 1;
	  break;
	case 'p':
	  if (option[1] != '\0')
	    problem = atoi(option + 1);
	  else if (++i < argc)
	    problem = atoi(argv[i]);
	  else
	    goto error_missing_arg_short;
	  option = "\0";
	  opt_problem = 1;
	  break;
	default:
	  fprintf(stderr, STR_ERR_UNKNOWN_SHORT_OPT, program_name,
		  *option);
	  return -1;
	error_missing_arg_short: fprintf(stderr,
					 STR_ERR_MISSING_ARG_SHORT, program_name, *option);
	  return -1;
	}
      } while (*++option != '\0');
  }
  return i;
}

long int **allocateMatrix(long int rows, long int cols) {
  long int i;
  long int **matrix;

  matrix = (long int **) malloc((rows + 1) * sizeof(long int *));
  if (matrix == NULL ) {
    fprintf(stderr, "Problems in memory allocation\n");
  }
  for (i = 0; i < (rows + 1); i++) {
    matrix[i] = (long int *) malloc((cols + 1) * sizeof(long int));
    if (matrix[i] == NULL ) {
      fprintf(stderr, "Problems in memory allocation\n");
    }
    memset(matrix[i], 0, (cols + 1) * sizeof(long int));
  }
  return matrix;
}

long int *allocateLongVector(long int size) {
  long int *vector;

  vector = (long int *) malloc((size + 1) * sizeof(long int));
  if (vector == NULL ) {
    fprintf(stderr, "Problems in memory allocation\n");
  }
  memset(vector, 0, (size + 1) * sizeof(long int));
  return vector;
}

int *allocateIntVector(int size) {
  int *vector;

  vector = (int *) malloc((size + 1) * sizeof(int));
  if (vector == NULL ) {
    fprintf(stderr, "Problems in memory allocation\n");
  }
  memset(vector, 0, (size + 1) * sizeof(int));
  return vector;
}

int int_compare(const void * a, const void * b) {
  return (*(int*) a - *(int*) b);
}
