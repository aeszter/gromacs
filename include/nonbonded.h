/*
 * $Id$
 * 
 *                This source code is part of
 * 
 *                 G   R   O   M   A   C   S
 * 
 *          GROningen MAchine for Chemical Simulations
 * 
 *                        VERSION 3.2.0
 * Written by David van der Spoel, Erik Lindahl, Berk Hess, and others.
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands.
 * Copyright (c) 2001-2004, The GROMACS development team,
 * check out http://www.gromacs.org for more information.

 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 * 
 * If you want to redistribute modifications, please consider that
 * scientific software is very special. Version control is crucial -
 * bugs must be traceable. We will be happy to consider code for
 * inclusion in the official distribution, but derived work must not
 * be called official GROMACS. Details are found in the README & COPYING
 * files - if they are missing, get the official version at www.gromacs.org.
 * 
 * To help us fund GROMACS development, we humbly ask that you cite
 * the papers on the package - you can find them in the top README file.
 * 
 * For more info, check our website at http://www.gromacs.org
 * 
 * And Hey:
 * Gromacs Runs On Most of All Computer Systems
 */

#ifndef _nonbonded_h
#define _nonbonded_h

#include "typedefs.h"
#include "pbc.h"
#include "network.h"
#include "tgroup.h"
#include "genborn.h"

void
gmx_setup_kernels(FILE *fplog);

void
do_nonbonded(t_commrec *cr,t_forcerec *fr,
             rvec x[],rvec f[],t_mdatoms *md,
             real egnb[],real egcoul[],rvec box_size,
             t_nrnb *nrnb,real lambda,real *dvdlambda,
	     gmx_genborn_t *born,
             bool bLR,int nls,int eNL,bool bDoForces);

/* Calculate VdW/charge pair interactions (usually 1-4 interactions).
 * global_atom_index is only passed for printing error messages.
 */
real
do_listed_vdw_q(int ftype,int nbonds,
		const t_iatom iatoms[],const t_iparams iparams[],
		const rvec x[],rvec f[],rvec fshift[],
		const t_pbc *pbc,const t_graph *g,
		real lambda,real *dvdlambda,
		const t_mdatoms *md,
		const t_forcerec *fr,gmx_grppairener_t *grppener,
		int *global_atom_index);

#endif
