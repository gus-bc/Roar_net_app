/*****************************************************************************/
/*                                                                           */
/* Version:  3.0    Date:  19/02/2004-06/09/2022   File: test_sol.h          */
/* Author:  Marco Chiarandini                                                */
/* email: machud@intellektik.informatik.tu-darmstadt.de                      */
/*                                                                           */
/*****************************************************************************/

#ifndef TEST_SOL_H
#define TEST_SOL_H


#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include "globals.h"
#include "utilities.h"
#include "bin2asc.h"





char checkConstraints();
void checkConstraints_ascii();
void checkConstraints_bin();
void readSolution();

#endif