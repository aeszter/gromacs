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
 * GROningen Mixture of Alchemy and Childrens' Stories
 */
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif


#include <gmx_thread.h>

/* This file is completely threadsafe - keep it that way! */

#include <string.h>
#include <ctype.h>
#include "sysstuff.h"
#include "smalloc.h"
#include "string2.h"
#include "macros.h"
#include "time.h"
#include "random.h"
#include "statutil.h"
#include "copyrite.h"
#include "strdb.h"
#include "futil.h"

static void pr_two(FILE *out,int c,int i)
{
  if (i < 10)
    fprintf(out,"%c0%1d",c,i);
  else
    fprintf(out,"%c%2d",c,i);
}

void pr_difftime(FILE *out,double dt)
{
  int    ndays,nhours,nmins,nsecs;
  bool   bPrint,bPrinted;

  ndays = dt/(24*3600);
  dt    = dt-24*3600*ndays;
  nhours= dt/3600;
  dt    = dt-3600*nhours;
  nmins = dt/60;
  dt    = dt-nmins*60;
  nsecs = dt;
  bPrint= (ndays > 0);
  bPrinted=bPrint;
  if (bPrint) 
    fprintf(out,"%d",ndays);
  bPrint=bPrint || (nhours > 0);
  if (bPrint) {
    if (bPrinted)
      pr_two(out,'d',nhours);
    else 
      fprintf(out,"%d",nhours);
  }
  bPrinted=bPrinted || bPrint;
  bPrint=bPrint || (nmins > 0);
  if (bPrint) {
    if (bPrinted)
      pr_two(out,'h',nmins);
    else 
      fprintf(out,"%d",nmins);
  }
  bPrinted=bPrinted || bPrint;
  if (bPrinted)
    pr_two(out,':',nsecs);
  else
    fprintf(out,"%ds",nsecs);
  fprintf(out,"\n");
}


bool be_cool(void)
{
  /* Yes, it is bad to check the environment variable every call,
   * but we dont call this routine often, and it avoids using 
   * a mutex for locking the variable...
   */
  return (getenv("GMX_NO_QUOTES") == NULL);
}

void space(FILE *out, int n)
{
  fprintf(out,"%*s",n,"");
}

void f(char *a){int i;for(i=0;i<(int)strlen(a);i++)a[i]=~a[i]; }

static void sp_print(FILE *out,const char *s)
{
  int slen;
  
  slen=strlen(s);
  space(out,(80-slen)/2);
  fprintf(out,"%s\n",s);
}

static void ster_print(FILE *out,const char *s)
{
  int  slen;
  char buf[128];
  
  sprintf(buf,":-)  %s  (-:",s);
  slen=strlen(buf);
  space(out,(80-slen)/2);
  fprintf(out,"%s\n",buf);
}


static void pukeit(char *db,char *defstring, char *retstring, 
		   int retsize, int *cqnum)
{
  FILE *fp;
  char **help;
  int  i,nhlp;
  int  seed;
 
  if (be_cool() && ((fp = low_libopen(db,FALSE)) != NULL)) {
    nhlp=fget_lines(fp,&help);
    fclose(fp);
    seed=time(NULL);
    *cqnum=nhlp*rando(&seed);
    if (strlen(help[*cqnum]) >= STRLEN)
      help[*cqnum][STRLEN-1] = '\0';
    strncpy(retstring,help[*cqnum],retsize);
    f(retstring);
    for(i=0; (i<nhlp); i++)
      sfree(help[i]);
    sfree(help);
  }
  else 
    strncpy(retstring,defstring,min(retsize,strlen(defstring)+1));
}

void bromacs(char *retstring, int retsize)
{
  int dum;

  pukeit("bromacs.dat",
	 "Groningen Machine for Chemical Simulation",
	 retstring,retsize,&dum);
}

void cool_quote(char *retstring, int retsize, int *cqnum)
{
  char *tmpstr;
  char *s,*ptr;
  int tmpcq,*p;
  
  if (cqnum!=NULL)
    p = cqnum;
  else
    p = &tmpcq;
  
  /* protect audience from explicit lyrics */
  snew(tmpstr,retsize+1);
  pukeit("gurgle.dat","Thanx for Using GROMACS - Have a Nice Day",
	 tmpstr,retsize-2,p);

  if ((ptr = strchr(tmpstr,'_')) != NULL) {
    *ptr='\0';
    ptr++;
    sprintf(retstring,"\"%s\" %s",tmpstr,ptr);
  }
  else {
    strcpy(retstring,tmpstr);
  }
  sfree(tmpstr);
}

void CopyRight(FILE *out,char *szProgram)
{
  /* Dont change szProgram arbitrarily - it must be argv[0], i.e. the 
   * name of a file. Otherwise, we won't be able to find the library dir.
   */
#define NCR (int)asize(CopyrightText)
#define NGPL (int)asize(GPLText)

  char buf[256],tmpstr[1024];
  int i;

  set_program_name(szProgram);

  ster_print(out,"G  R  O  M  A  C  S");
  fprintf(out,"\n");
  
  bromacs(tmpstr,1023);
  sp_print(out,tmpstr); 
  fprintf(out,"\n");

  ster_print(out,GromacsVersion());
  fprintf(out,"\n\n");

  /* fprintf(out,"\n");*/

  /* sp_print(out,"PLEASE NOTE: THIS IS A BETA VERSION\n");
  
  fprintf(out,"\n"); */

  for(i=0; (i<NCR); i++) 
    sp_print(out,CopyrightText[i]);
  for(i=0; (i<NGPL); i++)
    sp_print(out,GPLText[i]);

  fprintf(out,"\n");

  sprintf(buf,"%s",Program());
#ifdef GMX_DOUBLE
  strcat(buf," (double precision)");
#endif
  ster_print(out,buf);
  fprintf(out,"\n");
}


void thanx(FILE *fp)
{
  char cq[1024];
  int  cqnum;

  /* protect the audience from suggestive discussions */
  cool_quote(cq,1023,&cqnum);
  
  if (be_cool()) 
    fprintf(fp,"\ngcq#%d: %s\n\n",cqnum,cq);
  else
    fprintf(fp,"\n%s\n\n",cq);
}

typedef struct {
  char *key;
  char *author;
  char *title;
  char *journal;
  int volume,year,p0,p1;
} t_citerec;

void please_cite(FILE *fp,char *key)
{
  static t_citerec citedb[] = {
    { "Berendsen95a",
      "H. J. C. Berendsen, D. van der Spoel and R. van Drunen",
      "GROMACS: A message-passing parallel molecular dynamics implementation",
      "Comp. Phys. Comm.",
      91, 1995, 43, 56 },
    { "Berendsen84a",
      "H. J. C. Berendsen, J. P. M. Postma, A. DiNola and J. R. Haak",
      "Molecular dynamics with coupling to an external bath",
      "J. Chem. Phys.",
      81, 1984, 3684, 3690 },
    { "Ryckaert77a",
      "J. P. Ryckaert and G. Ciccotti and H. J. C. Berendsen",
      "Numerical Integration of the Cartesian Equations of Motion of a System with Constraints; Molecular Dynamics of n-Alkanes",
      "J. Comp. Phys.",
      23, 1977, 327, 341 },
    { "Miyamoto92a",
      "S. Miyamoto and P. A. Kollman",
      "SETTLE: An Analytical Version of the SHAKE and RATTLE Algorithms for Rigid Water Models",
      "J. Comp. Chem.",
      13, 1992, 952, 962 },
    { "Barth95a",
      "E. Barth and K. Kuczera and B. Leimkuhler and R. D. Skeel",
      "Algorithms for Constrained Molecular Dynamics",
      "J. Comp. Chem.",
      16, 1995, 1192, 1209 },
    { "Essman95a",
      "U. Essman, L. Perela, M. L. Berkowitz, T. Darden, H. Lee and L. G. Pedersen ",
      "A smooth particle mesh Ewald method",
      "J. Chem. Phys.",
      103, 1995, 8577, 8592 },
    { "Torda89a",
      "A. E. Torda and R. M. Scheek and W. F. van Gunsteren",
      "Time-dependent distance restraints in molecular dynamics simulations",
      "Chem. Phys. Lett.",
      157, 1989, 289, 294 },
    { "Tironi95a",
      "I. G. Tironi and R. Sperb and P. E. Smith and W. F. van Gunsteren",
      "Generalized reaction field method for molecular dynamics simulations",
      "J. Chem. Phys",
      102, 1995, 5451, 5459 },
    { "Hess97a",
      "B. Hess and H. Bekker and H. J. C. Berendsen and J. G. E. M. Fraaije",
      "LINCS: A Linear Constraint Solver for molecular simulations",
      "J. Comp. Chem.",
      18, 1997, 1463, 1472 },
    { "In-Chul99a",
      "Y. In-Chul and M. L. Berkowitz",
      "Ewald summation for systems with slab geometry",
      "J. Chem. Phys.",
      111, 1999, 3155, 3162 },
    { "DeGroot97a",
      "B. L. de Groot and D. M. F. van Aalten and R. M. Scheek and A. Amadei and G. Vriend and H. J. C. Berendsen",
      "Prediction of Protein Conformational Freedom From Distance Constrains",
      "Proteins",
      29, 1997, 240, 251 },
    { "Spoel98a",
      "D. van der Spoel and P. J. van Maaren and H. J. C. Berendsen",
      "A systematic study of water models for molecular simulation. Derivation of models optimized for use with a reaction-field.",
      "J. Chem. Phys.",
      108, 1998, 10220, 10230 },
    { "Wishart98a",
      "D. S. Wishart and A. M. Nip",
      "Protein Chemical Shift Analysis: A Practical Guide",
      "Biochem. Cell Biol.",
      76, 1998, 153, 163 },
    { "Maiorov95",
      "V. N. Maiorov and G. M. Crippen",
      "Size-Independent Comparison of Protein Three-Dimensional Structures",
      "PROTEINS: Struct. Funct. Gen.",
      22, 1995, 273, 283 },
    { "Feenstra99",
      "K. A. Feenstra and B. Hess and H. J. C. Berendsen",
      "Improving Efficiency of Large Time-scale Molecular Dynamics Simulations of Hydrogen-rich Systems",
      "J. Comput. Chem.",
      20, 1999, 786, 798 },
    { "Lindahl2001a",
      "E. Lindahl and B. Hess and D. van der Spoel",
      "GROMACS 3.0: A package for molecular simulation and trajectory analysis",
      "J. Mol. Mod.",
      7, 2001, 306, 317 },
    { "Wang2001a",
      "J. Wang and W. Wang and S. Huo and M. Lee and P. A. Kollman",
      "Solvation model based on weighted solvent accessible surface area",
      "J. Phys. Chem. B",
      105, 2001, 5055, 5067 },
    { "Eisenberg86a",
      "D. Eisenberg and A. D. McLachlan",
      "Solvation energy in protein folding and binding",
      "Nature",
      319, 1986, 199, 203 },
    { "Eisenhaber95",
      "Frank Eisenhaber and Philip Lijnzaad and Patrick Argos and Chris Sander and Michael Scharf",
      "The Double Cube Lattice Method: Efficient Approaches to Numerical Integration of Surface Area and Volume and to Dot Surface Contouring of Molecular Assemblies",
      "J. Comp. Chem.",
      16, 1995, 273, 284 },
    { "Hess2002",
      "B. Hess, H. Saint-Martin and H.J.C. Berendsen",
      "Flexible constraints: an adiabatic treatment of quantum degrees of freedom, with application to the flexible and polarizable MCDHO model for water",
      "J. Chem. Phys.",
      116, 2002, 9602, 9610 },
    { "Hetenyi2002b",
      "Csaba Hetenyi and David van der Spoel",
      "Efficient docking of peptides to proteins without prior knowledge of the binding site.",
      "Prot. Sci.",
      11, 2002, 1729, 1737 },
    { "Hess2003",
      "B. Hess and R.M. Scheek",
      "Orientation restraints in molecular dynamics simulations using time and ensemble averaging",
      "J. Magn. Res.",
      164, 2003, 19, 27 },
    { "Mu2005a",
      "Y. Mu, P. H. Nguyen and G. Stock",
      "Energy landscape of a small peptide revelaed by dihedral angle principal component analysis",
      "Prot. Struct. Funct. Bioinf.",
      58, 2005, 45, 52 },
    { "Okabe2001a",
      "T. Okabe and M. Kawata and Y. Okamoto and M. Mikami",
      "Replica-exchange {M}onte {C}arlo method for the isobaric-isothermal ensemble",
      "Chem. Phys. Lett.",
      335, 2001, 435, 439 },
    { "Hukushima96a",
      "K. Hukushima and K. Nemoto",
      "Exchange Monte Carlo Method and Application to Spin Glass Simulations",
      "J. Phys. Soc. Jpn.",
      65, 1996, 1604, 1608 },
    { "Tropp80a",
      "J. Tropp",
      "Dipolar Relaxation and Nuclear Overhauser effects in nonrigid molecules: The effect of fluctuating internuclear distances",
      "J. Chem. Phys.",
      72, 1980, 6035, 6043 },
    { "Spoel2006b",
      "D. van der Spoel, P. J. van Maaren, P. Larsson and N. Timneanu",
      "Thermodynamics of hydrogen bonding in hydrophilic and hydrophobic media",
      "J. Phys. Chem. B",
      110, 2006, 4393, 4398 }
  };
#define NSTR (int)asize(citedb)
  
  int  j,index;
  char *author;
  char *title;
#define LINE_WIDTH 79
  
  for(index=0; (index<NSTR) && (strcmp(citedb[index].key,key) != 0); index++)
    ;
  
  fprintf(fp,"\n++++ PLEASE READ AND CITE THE FOLLOWING REFERENCE ++++\n");
  if (index < NSTR) {
    /* Insert newlines */
    author = wrap_lines(citedb[index].author,LINE_WIDTH,0,FALSE);
    title  = wrap_lines(citedb[index].title,LINE_WIDTH,0,FALSE);
    fprintf(fp,"%s\n%s\n%s %d (%d) pp. %d-%d\n",
	    author,title,citedb[index].journal,
	    citedb[index].volume,citedb[index].year,
	    citedb[index].p0,citedb[index].p1);
    sfree(author);
    sfree(title);
  }
  else {
    fprintf(fp,"Entry %s not found in citation database\n",key);
  }
  fprintf(fp,"-------- -------- --- Thank You --- -------- --------\n\n");
  fflush(fp);
}

/* This routine only returns a static (constant) string, so we use a 
 * mutex to initialize it. Since the string is only written to the
 * first time, there is no risk with multiple calls overwriting the
 * output for each other.
 */
const char *GromacsVersion()
{

  /* Concatenate the version info during preprocessing */
  static const char ver_string[]="VERSION " VERSION;
  
  return ver_string;
}
