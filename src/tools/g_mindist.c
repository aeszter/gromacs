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
 * Green Red Orange Magenta Azure Cyan Skyblue
 */
#include <math.h>
#include <stdlib.h>
#include "config.h"
#include "sysstuff.h"
#include "string.h"
#include "typedefs.h"
#include "smalloc.h"
#include "macros.h"
#include "vec.h"
#include "xvgr.h"
#include "pbc.h"
#include "copyrite.h"
#include "futil.h"
#include "statutil.h"
#include "index.h"
#include "tpxio.h"
#include "rmpbc.h"
#include "xtcio.h"

static void periodic_dist(matrix box,rvec x[],int n,atom_id index[],
			  real *rmin,real *rmax)
{
#define NSHIFT 26
  int  sx,sy,sz,i,j,s;
  real sqr_box,r2min,r2max,r2;
  rvec shift[NSHIFT],d0,d;

  sqr_box = sqr(min(box[XX][XX],min(box[YY][YY],box[ZZ][ZZ])));

  s = 0;
  for(sz=-1; sz<=1; sz++)
    for(sy=-1; sy<=1; sy++)
      for(sx=-1; sx<=1; sx++)
	if (sx!=0 || sy!=0 || sz!=0) {
	  for(i=0; i<DIM; i++)
	    shift[s][i] = sx*box[XX][i]+sy*box[YY][i]+sz*box[ZZ][i];
	  s++;
	}
  
  r2min = sqr_box;
  r2max = 0;

  for(i=0; i<n; i++)
    for(j=i+1; j<n; j++) {
      rvec_sub(x[index[i]],x[index[j]],d0);
      r2 = norm2(d0);
      if (r2 > r2max)
	r2max = r2;
      for(s=0; s<NSHIFT; s++) {
	rvec_add(d0,shift[s],d);
	r2 = norm2(d);
	if (r2 < r2min)
	  r2min = r2;
      }
    }

  *rmin = sqrt(r2min);
  *rmax = sqrt(r2max);
}

static void periodic_mindist_plot(char *trxfn,char *outfn,
				  t_topology *top,int n,atom_id index[],
				  bool bSplit)
{
  FILE   *out;
  char   *leg[5] = { "min per.","max int.","box1","box2","box3" };
  int    status;
  real   t;
  rvec   *x;
  matrix box;
  int    natoms;
  real   r,rmin,rmax,rmint,tmint;
  bool   bFirst;
  
  natoms=read_first_x(&status,trxfn,&t,&x,box);
  
  check_index(NULL,n,index,NULL,natoms);
  
  out = xvgropen(outfn,"Minimum distance to periodic image",
		 time_label(),"Distance (nm)");
  fprintf(out,"@ subtitle \"and maximum internal distance\"\n");
  xvgr_legend(out,5,leg);
    
  rmint = box[XX][XX];
  tmint = 0;
  
  bFirst=TRUE;  
  do {
    rm_pbc(&(top->idef),natoms,box,x,x);
    periodic_dist(box,x,n,index,&rmin,&rmax);
    if (rmin < rmint) {
      rmint = rmin;
      tmint = t;
    }
    if ( bSplit && !bFirst && abs(t/time_factor())<1e-5 )
      fprintf(out, "&\n");
    fprintf(out,"\t%g\t%6.3f %6.3f %6.3f %6.3f %6.3f\n",
	    convert_time(t),rmin,rmax,norm(box[0]),norm(box[1]),norm(box[2]));
    bFirst=FALSE;
  } while(read_next_x(status,&t,natoms,x,box));
    
  fclose(out);
  
  fprintf(stdout,
	  "\nThe shortest periodic distance is %g (nm) at time %g (%s)\n",
	  rmint,convert_time(tmint),time_unit());
}

static void calc_dist(real rcut, matrix box, rvec x[], 
		      int nx1,int nx2, atom_id index1[], atom_id index2[],
		      real *rmin, real *rmax, int *nmin, int *nmax,
		      int *ixmin, int *jxmin, int *ixmax, int *jxmax)
{
  int     i,j,j0=0,j1;
  int     ix,jx;
  atom_id *index3;
  rvec    dx;
  real    r2,rmin2,rmax2,rcut2;
  
  *ixmin = -1;
  *jxmin = -1;
  *ixmax = -1;
  *jxmax = -1;
  *nmin = 0;
  *nmax = 0;
  
  rcut2=sqr(rcut);
  
  /* Must init pbc every step because of pressure coupling */
  init_pbc(box);
  if (index2) {
    j0=0;
    j1=nx2;
    index3=index2;
  } else {
    j1=nx1;
    index3=index1;
  }
  
  rmin2=1e12;
  rmax2=-1e12;
  
  for(i=0; (i < nx1); i++) {
    ix=index1[i];
    if (!index2)
      j0=i+1;
    for(j=j0; (j < j1); j++) {
      jx=index3[j];
      if (ix != jx) {
	pbc_dx(x[ix],x[jx],dx);
	r2=iprod(dx,dx);
	if (r2 < rmin2) {
	  rmin2=r2;
	  *ixmin=ix;
	  *jxmin=jx;
	}
	if (r2 > rmax2) {
	  rmax2=r2;
	  *ixmax=ix;
	  *jxmax=jx;
	}
	if (r2 < rcut2)
	  *nmin++;
	else if (r2 > rcut2)
	  *nmax++;
      }
    }
  }
  *rmin = sqrt(rmin2);
  *rmax = sqrt(rmax2);
}

void dist_plot(char *fn,char *afile,char *dfile,
	       char *nfile,char *rfile,char *xfile,
	       real rcut,bool bMat,t_atoms *atoms,
	       int ng,atom_id *index[],int gnx[],char *grpn[],bool bSplit,
	       bool bMin, int nres, atom_id *residue)
{
  FILE         *atm,*dist,*num;
  int          trxout;
  char         buf[256];
  char         **leg;
  real         t,dmin,dmax,**mindres=NULL,**maxdres=NULL;
  int          nmin,nmax,status;
  int          i=-1,j,k,natoms;
  int	       min1,min2,max1,max2;
  atom_id      oindex[2];
  rvec         *x0;
  matrix       box;
  t_trxframe   frout;
  bool         bFirst;
  
  if ((natoms=read_first_x(&status,fn,&t,&x0,box))==0)
    fatal_error(0,"Could not read coordinates from statusfile\n");
  
  sprintf(buf,"%simum Distance",bMin ? "Min" : "Max");
  dist= xvgropen(dfile,buf,time_label(),"Distance (nm)");
  sprintf(buf,"Number of Contacts %s %g nm",bMin ? "<" : ">",rcut);
  num = nfile ? xvgropen(nfile,buf,time_label(),"Number") : NULL;
  atm = afile ? ffopen(afile,"w") : NULL;
  trxout = xfile ? open_trx(xfile,"w") : NOTSET;
  
  if (bMat) {
    if (ng == 1) {
      snew(leg,1);
      sprintf(buf,"Internal in %s",grpn[0]);
      leg[0]=strdup(buf);
      xvgr_legend(dist,0,leg);
      if (num) xvgr_legend(num,0,leg);
    } 
    else {
      snew(leg,(ng*(ng-1))/2);
      for(i=j=0; (i<ng-1); i++) {
	for(k=i+1; (k<ng); k++,j++) {
	  sprintf(buf,"%s-%s",grpn[i],grpn[k]);
	  leg[j]=strdup(buf);
	}
      }
      xvgr_legend(dist,j,leg);
      if (num) xvgr_legend(num,j,leg);
    }
  }
  else {  
    snew(leg,ng-1);
    for(i=0; (i<ng-1); i++) {
      sprintf(buf,"%s-%s",grpn[0],grpn[i+1]);
      leg[i]=strdup(buf);
    }
    xvgr_legend(dist,ng-1,leg);
    if (num) xvgr_legend(num,ng-1,leg);
  }
  j=0;
  if (nres) {
    snew(mindres, ng-1);
    snew(maxdres, ng-1);
    for(i=1; i<ng; i++) {
      snew(mindres[i-1], nres);
      snew(maxdres[i-1], nres);
      for(j=0; j<nres; j++)
	mindres[i-1][j]=1e6;
      /* maxdres[*][*] is already 0 */
    }
  }
  bFirst=TRUE;  
  do {
    if ( bSplit && !bFirst && abs(t/time_factor())<1e-5 ) {
      fprintf(dist, "&\n");
      if (num) fprintf(num, "&\n");
      if (atm) fprintf(atm, "&\n");
    }
    fprintf(dist,"%12g",convert_time(t));
    if (num) fprintf(num,"%12g",convert_time(t));
    
    if (bMat) {
      if (ng == 1) {
	calc_dist(rcut,box,x0,gnx[0],gnx[0],index[0],index[0],
		  &dmin,&dmax,&nmin,&nmax,&min1,&min2,&max1,&max2);
	fprintf(dist,"  %12g",bMin?dmin:dmax);
	if (num) fprintf(num,"  %8d",bMin?nmin:nmax);
      }
      else {
	for(i=0; (i<ng-1); i++) {
	  for(k=i+1; (k<ng); k++) {
	    calc_dist(rcut,box,x0,gnx[i],gnx[k],index[i],index[k],
		      &dmin,&dmax,&nmin,&nmax,&min1,&min2,&max1,&max2);
	    fprintf(dist,"  %12g",bMin?dmin:dmax);
	    if (num) fprintf(num,"  %8d",bMin?nmin:nmax);
	  }
	}
      }
    }
    else {    
      for(i=1; (i<ng); i++) {
	calc_dist(rcut,box,x0,gnx[0],gnx[i],index[0],index[i],
		  &dmin,&dmax,&nmin,&nmax,&min1,&min2,&max1,&max2);
	fprintf(dist,"  %12g",bMin?dmin:dmax);
	if (num) fprintf(num,"  %8d",bMin?nmin:nmax);
	if (nres) {
	  for(j=0; j<nres; j++) {
	    calc_dist(rcut,box,x0,residue[j+1]-residue[j],gnx[i],
		      &(index[0][residue[j]]),index[i],
		      &dmin,&dmax,&nmin,&nmax,&min1,&min2,&max1,&max2);
	    mindres[i-1][j] = min(mindres[i-1][j],dmin);
	    maxdres[i-1][j] = max(maxdres[i-1][j],dmax);
	  }
	}
      }
    }
    fprintf(dist,"\n");
    if (num) 
      fprintf(num,"\n");
    if ( bMin?min1:max1 != -1 )
      if (atm)
	fprintf(atm,"%12g  %12d  %12d\n",
		convert_time(t),bMin?min1:max1+1,bMin?min2:max2+1);
    
    if (trxout>=0) {
      oindex[0]=bMin?min1:max1;
      oindex[1]=bMin?min2:max2;
      write_trx(trxout,2,oindex,atoms,i,t,box,x0,NULL);
    }
    bFirst=FALSE;
  } while (read_next_x(status,&t,natoms,x0,box));
  
  close_trj(status);
  fclose(dist);
  if (num) fclose(num);
  if (atm) fclose(atm);
  if (trxout>=0) close_xtc(trxout);
  
  if(nres) {
    FILE *res;
    
    sprintf(buf,"%simum Distance",bMin ? "Min" : "Max");
    res=xvgropen(rfile,buf,"Residue (#)","Distance (nm)");
    xvgr_legend(res,ng-1,leg);
    for(j=0; j<nres; j++) {
      fprintf(res, "%4d", j+1);
      for(i=1; i<ng; i++) {
	fprintf(res, " %7g", bMin ? mindres[i-1][j] : maxdres[i-1][j]);
      }
      fprintf(res, "\n");
    }
  }
  
  sfree(x0);
}

int find_residues(t_atoms *atoms, int n, atom_id index[], atom_id **resindex)
{
  int i;
  int nres=0,resnr, presnr;
  int *residx;
  
  /* build index of first atom numbers for each residue */  
  presnr = NOTSET;
  snew(residx, atoms->nres);
  for(i=0; i<n; i++) {
    resnr = atoms->atom[index[i]].resnr;
    if (resnr != presnr) {
      residx[nres]=i;
      nres++;
      presnr=resnr;
    }
  }
  if (debug) printf("Found %d residues out of %d (%d/%d atoms)\n", 
		    nres, atoms->nres, atoms->nr, n);
  srenew(residx, nres+1);
  /* mark end of last residue */
  residx[nres]=n+1;
  *resindex = residx;
  return nres;
}

void dump_res(FILE *out, int nres, atom_id *resindex, int n, atom_id index[])
{
  int i,j;
  
  for(i=0; i<nres-1; i++) {
    fprintf(out,"Res %d (%d):", i, resindex[i+1]-resindex[i]);
    for(j=resindex[i]; j<resindex[i+1]; j++)
      fprintf(out," %d(%d)", j, index[j]);
    fprintf(out,"\n");
  }
}

int gmx_mindist(int argc,char *argv[])
{
  static char *desc[] = {
    "g_mindist computes the distance between one group and a number of",
    "other groups.",
    "Both the minimum distance and the number of contacts within a given",
    "distance are written to two separate output files.",
    "With [TT]-or[tt], minimum distances to each residue in the first",
    "group are determined and plotted as a function of reisdue number.[PAR]",
    "With option [TT]-pi[tt] the minimum distance of a group to its",
    "periodic image is plotted. This is useful for checking if a protein",
    "has seen its periodic image during a simulation. Only one shift in",
    "each direction is considered, giving a total of 26 shifts.",
    "It also plots the maximum distance within the group and the lengths",
    "of the three box vectors.[PAR]",
    "Other programs that calculate distances are [TT]g_dist[tt]",
    "and [TT]g_bond[tt]."
  };
  static char *bugs[] = {
    "The [TT]-pi[tt] option is very slow."
  };
  
  static bool bMat=FALSE,bPer=FALSE,bSplit=FALSE,bMax=FALSE;
  static real rcutoff=0.6;
  t_pargs pa[] = {
    { "-matrix", FALSE, etBOOL, {&bMat},
      "Calculate half a matrix of group-group distances" },
    { "-max",    FALSE, etBOOL, {&bMax},
      "Calculate *maximum* distance instead of minimum" },
    { "-d",      FALSE, etREAL, {&rcutoff},
      "Distance for contacts" },
    { "-pi",     FALSE, etBOOL, {&bPer},
      "Calculate minimum distance with periodic images" },
    { "-split",  FALSE, etBOOL, {&bSplit},
      "Split graph where time is zero" },
  };
  t_topology top;
  char       title[256];
  real       t;
  rvec       *x;
  matrix     box;
  
  FILE      *atm;
  int       i,j,ng,nres=0;
  char      *trxfnm,*tpsfnm,*ndxfnm,*distfnm,*numfnm,*atmfnm,*oxfnm,*resfnm;
  char      **grpname;
  int       *gnx;
  atom_id   **index, *residues=NULL;
  t_filenm  fnm[] = {
    { efTRX, "-f",  NULL,      ffREAD },
    { efTPS,  NULL, NULL,      ffOPTRD },
    { efNDX,  NULL, NULL,      ffOPTRD },
    { efXVG, "-od","mindist",  ffWRITE },
    { efXVG, "-on","numcont",  ffOPTWR },
    { efOUT, "-o", "atm-pair", ffOPTWR },
    { efTRX, "-ox","mindist",  ffOPTWR },
    { efXVG, "-or","mindistres", ffOPTWR }
  };
#define NFILE asize(fnm)

  CopyRight(stderr,argv[0]);
  parse_common_args(&argc,argv,
		    PCA_CAN_VIEW | PCA_CAN_TIME | PCA_TIME_UNIT | PCA_BE_NICE,
		    NFILE,fnm,asize(pa),pa,asize(desc),desc,0,NULL);

  trxfnm = ftp2fn(efTRX,NFILE,fnm);
  tpsfnm = ftp2fn_null(efTPS,NFILE,fnm);
  ndxfnm = ftp2fn_null(efNDX,NFILE,fnm);
  distfnm= opt2fn("-od",NFILE,fnm);
  numfnm = opt2fn_null("-on",NFILE,fnm);
  atmfnm = ftp2fn_null(efOUT,NFILE,fnm);
  oxfnm  = opt2fn_null("-ox",NFILE,fnm);
  resfnm = opt2fn_null("-or",NFILE,fnm);
  
  if (!tpsfnm && !ndxfnm)
    fatal_error(0,"You have to specify either the index file or a tpr file");
  
  if (bPer) {
    ng = 1;
    fprintf(stderr,"Choose a group for distance calculation\n");
  } 
  else {
    if (bMat)
      fprintf(stderr,"You can compute all distances between a number of groups\n"
	      "How many groups do you want (>= 1) ?\n");
    else
      fprintf(stderr,"You can compute the distances between a first group\n"
	      "and a number of other groups.\n"
	      "How many other groups do you want (>= 1) ?\n");
    ng = 0;
    do {
      scanf("%d",&ng);
      if (!bMat)
	ng++;
    } while (ng < 1);
  }
  snew(gnx,ng);
  snew(index,ng);
  snew(grpname,ng);

  if (tpsfnm || !ndxfnm)
    read_tps_conf(tpsfnm,title,&top,&x,NULL,box,FALSE);
  
  get_index(&top.atoms,ndxfnm,ng,gnx,index,grpname);

  if (bMat && (ng == 1)) {
    ng = gnx[0];
    printf("Special case: making distance matrix between all atoms in group %s\n",
	   grpname[0]);
    srenew(gnx,ng);
    srenew(index,ng);
    srenew(grpname,ng);
    for(i=1; (i<ng); i++) {
      gnx[i]      = 1;
      grpname[i]  = grpname[0];
      snew(index[i],1);
      index[i][0] = index[0][i]; 
    }
    gnx[0] = 1;
  }
  
  if (resfnm) {
    nres=find_residues(&top.atoms, gnx[0], index[0], &residues);
    if (debug) dump_res(debug, nres, residues, gnx[0], index[0]);
  }
    
  if (bPer)
    periodic_mindist_plot(trxfnm,distfnm,&top,gnx[0],index[0],bSplit);
  else
    dist_plot(trxfnm,atmfnm,distfnm,numfnm,resfnm,oxfnm,
	      rcutoff,bMat,&top.atoms,ng,index,gnx,grpname,bSplit,
	      !bMax, nres, residues);

  do_view(distfnm,"-nxy");
  if (!bPer)
    do_view(numfnm,"-nxy");
  
  thanx(stderr);
  
  return 0;
}

