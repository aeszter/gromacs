/*
 * 
 *                This source code is part of
 * 
 *                 G   R   O   M   A   C   S
 * 
 *          GROningen MAchine for Chemical Simulations
 * 
 * Written by David van der Spoel, Erik Lindahl, Berk Hess, and others.
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands.
 * Copyright (c) 2001-2008, The GROMACS development team,
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
 * Gallium Rubidium Oxygen Manganese Argon Carbon Silicon
 */

/*! \file pull_rotation.h
 *
 *  @brief Enforced rotation of protein parts or other groups of particles.
 *
 *  This file contains routines that are used to enforce rotational motion
 *  upon a subgroup of particles.  
 *  
 */

#ifndef _pull_rotation_h
#define _pull_rotation_h

#ifdef HAVE_CONFIG_H
  #include <config.h>
#endif

#include "vec.h"
#include "typedefs.h"


#ifdef __cplusplus
extern "C" {
#endif


/*! \brief Initialize the enforced rotation groups.
 * 
 * This routine does the memory allocation for various helper arrays, opens
 * the output files etc.  
 *
 * \param fplog             General output file, normally md.log.
 * \param ir                Struct containing MD input parameters, among those
 *                          also the enforced rotation parameters.
 * \param nfile             Number of entries in the fnm structure.       
 * \param fnm               The filenames struct containing also the names
 *                          of the rotation output files.
 * \param cr                Pointer to MPI communication data.
 * \param x                 The positions of all MD particles.
 * \param mtop              Molecular topology.
 * \param oenv              Needed to open the rotation output xvgr file.
 * \param Flags             Flags passed over from main, used to determine
 *                          whether or not we are doing a rerun.
 */
extern void init_rot(FILE *fplog,t_inputrec *ir,int nfile,const t_filenm fnm[],
        t_commrec *cr, rvec *x, matrix box, gmx_mtop_t *mtop, const output_env_t oenv,
        gmx_bool bVerbose, unsigned long Flags);


/*! \brief Make a selection of the home atoms for all enforced rotation groups.
 *
 * This routine is similar to dd_make_local_pull_groups, but works only with
 * domain decomposition. It should be called at every domain decomposition.
 *
 * \param dd                Structure containing domain decomposition data.
 * \param rot               Pointer to all the enforced rotation data.
 */
extern void dd_make_local_rotation_groups(gmx_domdec_t *dd,t_rot *rot);


/*! \brief Calculation of the enforced rotation potential.
 * 
 * This is the main enforced rotation module which is called during every time
 * step. Here the rotation potential as well as the resulting forces are 
 * calculated.
 * 
 * \param cr                Pointer to MPI communication data.
 * \param ir                Struct containing MD input parameters, among those
 * \param box               Simulation box, needed to make group whole.
 * \param x                 The positions of all the local particles.
 * \param t                 Time.
 * \param step              The time step.
 * \param wcycle            During the potential calculation the wallcycles are
 *                          counted. Later they enter the dynamic load balancing.
 * \param bNS               After domain decomposition / neighborsearching several
 *                          local arrays have to be updated (masses, shifts)
 */
extern void do_rotation(t_commrec *cr,t_inputrec *ir,matrix box,rvec x[],real t,
        gmx_large_int_t step,gmx_wallcycle_t wcycle,gmx_bool bNS);


/*! \brief Add the enforced rotation forces to the official force array.
 * 
 * Adds the forces from enforced rotation potential to the local forces and
 * sums up the contributions to the rotation potential from all the nodes. Since
 * this needs communication, this routine should be called after the SR forces 
 * have been evaluated (in order not to spoil cycle counts). 
 * This routine also outputs data to the various rotation output files (e.g.
 * the potential, the angle of the group, torques and more).
 * 
 * \param rot               Pointer to all the enforced rotation data.
 * \param f                 The local forces to which the rotational forces have
 *                          to be added.
 * \param cr                Pointer to MPI communication data.
 * \param step              The time step, used for output.
 * \param t                 Time, used for output.
 */
extern real add_rot_forces(t_rot *rot, rvec f[], t_commrec *cr, gmx_large_int_t step, real t);


/*! \brief Close the enforced rotation output files.
 *
 * \param fplog             General output file, normally md.log.
 * \param rot               Pointer to all the enforced rotation data.
 */
extern void finish_rot(FILE *fplog,t_rot *rot);


#ifdef __cplusplus
}
#endif


#endif
