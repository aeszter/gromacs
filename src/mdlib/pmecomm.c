/*
 * $Id$
 * 
 *                This source code is part of
 * 
 *                 G   R   O   M   A   C   S
 * 
 *          GROningen MAchine for Chemical Simulations
 * 
 *                        VERSION 3.1
 * Copyright (c) 1991-2001, University of Groningen, The Netherlands
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
 * GROup of MAchos and Cynical Suckers
 */
static char *SRCID_pmecomm_c = "$Id$";
#include <stdio.h>
#include <math.h>
#include "typedefs.h"
#include "txtdump.h"
#include "vec.h"
#include "gmxcomplex.h"
#include "smalloc.h"
#include "futil.h"
#include "fatal.h"
#include "pme.h"
#include "mdrun.h"
#include "network.h"
#include "nrnb.h"

typedef struct {
  rvec x;
  real q;
  int  index;
} t_pme_comm;

static int pcomm_comp(const void *a,const void *b)
{
  t_pme_comm *pa = (t_pme_comm *) a;
  t_pme_comm *pb = (t_pme_comm *) b;
  real xx;
  
  /* Should be changed for triclinic stuff */
  xx = pa->x[XX] - pb->x[XX];
  
  if (xx < 0)
    return -1;
  else if (xx == 0.0)
    return 0;
  else
    return 1;
}

/* This is on the real space nodes */
void send_to_pme_nodes(t_commrec *cr,int natoms,rvec x[],real q[],
		       int start,int end,matrix box,int desort[])
{
  t_pme_comm *pcomm;
  real       border;
  int        i,j,n,nn,send0,send1;

  /* Start out by putting the coordinates and charges in a special array */
  n = end-start;
  snew(pcomm,n);
  for(i=start,j=0; (i<end); i++,j++) {
    copy_rvec(x[i],pcomm[j].x);
    pcomm[j].q     = q[i];
    pcomm[j].index = i;
  }
  
  /* Do the actual sorting only if you have more than 1 processor */
  if (cr->npme > 1) 
    qsort(pcomm,n,sizeof(pcomm[0]),pcomm_comp);
  for(i=0; (i<n); i++)
    desort[i] = pcomm[i].index;
  
  /* Loop over processors */
  send0 = 0;
  for(i=0; (i<cr->npme); i++) {
    if (i<cr->npme-1) {
      border=((i+1)*box[XX][XX])/cr->npme;
      /* This is not efficient, should be some multinary search */
      for(send1=1; (send1<n); send1++)
	if (pcomm[send1].x[XX] > border) 
	  break;
    }
    else
      send1 = n;
    /* Finally send it */
    nn = send1-send0;
    gmx_tx(i,record(nn));
    gmx_tx(i,arrayp(pcomm[send0],nn));
    send0 = send1;
  }
  
  sfree(pcomm);
}

void recv_from_pme_nodes(t_commrec *cr,int natoms,rvec f[],int start,int end,
			 int desort[])
{
  rvec *fptr;
  int  i,nn,send0;

  /* Make temp array for receiving */  
  snew(fptr,natoms);
  send0 = 0;
  for(i=0; (i<cr->npme); i++) {
    gmx_rx(i,record(nn));
    if (nn > 0) {
      gmx_rx(i,arrayp(fptr[send0],nn));
      send0 += nn;
    }
  }
  /* Desort the forces */
  for(i=0; (i<send0); i++) 
    copy_rvec(fptr[i],f[desort[i]]);
  
  sfree(fptr);
}

/* This is on the pme nodes */
static int recv_from_real_nodes(t_commrec *cr,int natoms,rvec x[],real q[],
				int node_atoms[])
{
  t_pme_comm *pcomm;
  int i,nn,send0;
 
  snew(pcomm,natoms);
  send0=0;
  for(i=0; (i<cr->nreal); i++) {
    gmx_rx(cr->npme+i,record(nn));
    if (nn > 0) {
      gmx_rx(cr->npme+i,arrayp(pcomm[send0],nn));
      send0 += nn;
    }
    node_atoms[i] = nn;
  }
  for(i=0; (i<send0); i++) {
    copy_rvec(pcomm[i].x,x[i]);
    q[i] = pcomm[i].q;
  }
  
  sfree(pcomm);
  
  return send0;
}

static void send_to_real_nodes(t_commrec *cr,int n,rvec f[],int node_atoms[])
{
  int i,nn,send0;
  
  /* Send each of the nreal nodes their forces */ 
  send0 = 0;
  for(i=0; (i<cr->nreal); i++) {
    nn = node_atoms[i];
    gmx_tx(cr->npme+1,record(nn));
    gmx_tx(cr->npme+i,arrayp(f[send0],nn));
    send0 += nn;
  }
}

void do_allpme(FILE *fp,t_commrec *cr,int natoms,
	       t_groups *grps,t_inputrec *ir,t_nrnb nrnb[],
	       t_forcerec *fr,t_nsborder *nsb)
{
  int    i,j,nn,*node_atoms;
  rvec   *x,*f;
  real   *q;
  real   ener[F_NRE],terminate;
  tensor lr_vir,shake_vir;
  t_nrnb mynrnb;
  matrix box;
  real   vcm[4];
  
  fprintf(fp,"Starting pme loop on processor %d\n",cr->nodeid);
  snew(x,natoms);
  snew(f,natoms);
  snew(q,natoms);
  snew(node_atoms,cr->npme);
  init_nrnb(&mynrnb);
  for(i=0; (i<=ir->nsteps); i++) {
    /* Write code here to update the box variable (not set by default) */
    distribute_box(cr,box);
    
    nn = recv_from_real_nodes(cr,natoms,x,q,node_atoms);
  
    clear_mat(lr_vir);
    clear_mat(shake_vir);
    clear_rvecs(nn,f);
    for(i=0; (i<4); i++)
      vcm[i]=0;
    for(i=0; (i<F_NRE); i++)
      ener[i]=0.0;
      
    ener[F_LR] = do_pme(fp,FALSE,ir,x,f,q,box,cr,nsb,nrnb,lr_vir,
			fr->ewaldcoeff,FALSE);
    
    send_to_real_nodes(cr,nn,f,node_atoms);
    
    /* Communicate energies etc. */
    global_stat(fp,cr,ener,lr_vir,shake_vir,
		&(ir->opts),grps,&mynrnb,nrnb,vcm,&terminate);
    
  }
  sfree(node_atoms);
  sfree(q);
  sfree(f);
  sfree(x);
}
