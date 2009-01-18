#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <math.h>
#include <string.h>

#include "typedefs.h"
#include "smalloc.h"
#include "genborn.h"
#include "vec.h"
#include "grompp.h"
#include "pdbio.h"
#include "names.h"
#include "physics.h"
#include "partdec.h"
#include "network.h"
#include "gmx_fatal.h"

#ifdef GMX_MPI
#include "mpi.h"
#endif

/* If SSE intrinsics are available */
#if ((defined(__GNUC__) || defined(__INTEL_COMPILER) || defined(__PATHSCALE__) || defined(__PGIC__)) && \
(defined(__i386__) || defined(__x86_64)))
#include <xmmintrin.h>
#include <emmintrin.h>
#endif

int init_gb_plist(t_params *p_list)
{
	p_list->nr    = 0;
    p_list->param = NULL;
	
	return 0;
}


static void assign_gb_param(t_functype ftype,t_iparams *new,
							real old[MAXFORCEPARAM],int comb,real reppow)
{
	int  i,j;
	
  /* Set to zero */
  for(j=0; (j<MAXFORCEPARAM); j++)
    new->generic.buf[j]=0.0;
	
	switch (ftype) {
		case F_GB:
			new->gb.c6A=old[0];
			new->gb.c12A=old[1];
			new->gb.c6B=old[2];
			new->gb.c12B=old[3];
			new->gb.sar=old[4];
			new->gb.st=old[5];
			new->gb.pi=old[6];
			new->gb.gbr=old[7];
			new->gb.bmlt=old[8];
			break;
		default:
			gmx_fatal(FARGS,"unknown function type %d in %s line %d",
								ftype,__FILE__,__LINE__);		
	}
}

static void append_gb_interaction(t_ilist *ilist,
                               int type,int nral,atom_id a[MAXATOMLIST])
{
  int i,where1;
  
  where1     = ilist->nr;
  ilist->nr += nral+1;
	
  ilist->iatoms[where1++]=type;
  for (i=0; (i<nral); i++) 
  {
    ilist->iatoms[where1++]=a[i];
  }
}


static int enter_gb_params(t_idef *idef, t_functype ftype,
												real forceparams[MAXFORCEPARAM],int comb,real reppow,
												int start,bool bAppend)
{
  t_iparams new;
  int       type;
	
	assign_gb_param(ftype,&new,forceparams,comb,reppow);
  if (!bAppend) {
		for (type=start; (type<idef->ntypes); type++) {
      if (idef->functype[type]==ftype) {
					if (memcmp(&new,&idef->iparams[type],(size_t)sizeof(new)) == 0)
					return type;
      }
    }
  }
  else
    type=idef->ntypes;
  if (debug)
    fprintf(debug,"copying new to idef->iparams[%d] (ntypes=%d)\n",
						type,idef->ntypes);
  memcpy(&idef->iparams[type],&new,(size_t)sizeof(new));
  
  idef->ntypes++;
  idef->functype[type]=ftype;
	
  return type;
}

/* Initialize all GB datastructs and compute polarization energies */
int init_gb(t_commrec *cr, t_forcerec *fr, born_t *born, t_topology *top,
			rvec x[], int natoms, real rgbradii, int gb_algorithm)
{
	int i,j,m,ai,aj,jj,nalloc;
	double rai,sk,p;
	
	nalloc=0;
	
	for(i=0;i<natoms;i++)
		nalloc+=i;

	//init_gb_nblist(natoms, &(fr->gblist_sr));
	//init_gb_nblist(natoms, &(fr->gblist_lr));
	init_gb_nblist(natoms, &(fr->gblist));
	
	//snew(fr->gblist_sr.jjnr,nalloc*2);
	//snew(fr->gblist_lr.jjnr,nalloc);
	snew(fr->gblist.jjnr,nalloc*2);
	
	born->n12=0;
	born->n13=0;
	born->n14=0;
	
	/* Do the Vsites exclusions (if any) */
	for(i=0;i<natoms;i++)
	{
		jj = top->atoms.atom[i].type;
		born->vs[i]=1;																							
													
	    if(C6(fr->nbfp,fr->ntype,jj,jj)==0 && C12(fr->nbfp,fr->ntype,jj,jj)==0 && top->atoms.atom[i].q==0)
			born->vs[i]=0;
	}
	
	for(i=0;i<F_NRE;i++)
	{
		if(IS_ANGLE(i))
		{
			born->n13+=top->idef.il[i].nr/(1+NRAL(i));
		}
		
		if(IS_CHEMBOND(i))
		{
			switch(i)
			{
				case F_BONDS:
				case F_CONNBONDS:
				case F_CONSTR:
				
					for(j=0;j<top->idef.il[i].nr;)
					{
						m=top->idef.il[i].iatoms[j++];
						ai=top->idef.il[i].iatoms[j++];
						aj=top->idef.il[i].iatoms[j++];
						
						if(born->vs[ai]==1 && born->vs[aj]==1)
						{
							born->n12++;
						}
					}
				
				break;
			}
		}
	}
	
	born->n14=top->idef.il[F_LJ14].nr/(1+NRAL(F_LJ14));
	
	/* If Still model, initialise the polarisation energies */
	if(gb_algorithm==egbSTILL)	
	   init_gb_still(cr, fr,&(top->atomtypes), &(top->idef), &(top->atoms), born, natoms);	
	   
	/* If HCT/OBC,  precalculate the sk*atype->hct factors */
	if(gb_algorithm==egbHCT || gb_algorithm==egbOBC)
	{
		for(i=0;i<natoms;i++)
		{	
			if(born->vs[i]==1)
			{
				rai            = top->atomtypes.gb_radius[top->atoms.atom[i].type]+DOFFSET; //0.09 * 0.1 for nm, dielectric offset
				sk             = rai * top->atomtypes.shct[top->atoms.atom[i].type];
				born->param[i] = sk;
			}
			else
			{
				born->param[i] = 0;
			}
		}
	}
	
	/* Init the logarithm table */
	p=pow(2,LOG_TABLE_ACCURACY);
	snew(born->log_table, p);
	
	fill_log_table(LOG_TABLE_ACCURACY, born->log_table);

	return 0;
}

int generate_gb_topology(t_params *plist, t_idef *idef, t_atomtype *atype, t_atoms atoms, int natoms, t_params *gb_plist, born_t *born)
{
	int i,j,k,type,m,a1,a2,a3,a4,idx,nral,maxtypes,start,comb;
	int n12,n13,n14;
	double p1,p2,p3,cosine,r2,rab,rbc;
	
	bl_t *bl;
	bonds_t *bonds;
	
	bl=(bl_t *) malloc(sizeof(bl_t)*natoms);
	snew(bonds,natoms);
	
	/* To keep the compiler happy */
	rab=rbc=0;
	
	for(i=0;i<F_NRE;i++)
	{
		if(plist[i].nr>0)
		{
			gb_plist->nr+=plist[i].nr;
		}
	}

	snew(gb_plist->param,gb_plist->nr);

	p1=P1;
	p2=P2;
	p3=P3;
	
	idx=0;
	n12=0;
	n13=0;
	n14=0;
	
	for(i=0;i<F_NRE;i++)
	{
		if(IS_CHEMBOND(i))
		{
			switch(i)
			{
				case F_BONDS:
				case F_CONNBONDS:
				case F_CONSTR:
				
					for(j=0;j<plist[i].nr; j++)
					{
						a1=plist[i].param[j].a[0];
						a2=plist[i].param[j].a[1];
					
						if(atoms.atom[a1].q!=0 && atoms.atom[a2].q!=0)
						{
							bl[a1].length[a2]=plist[i].param[j].c[0];
							bl[a2].length[a1]=plist[i].param[j].c[0];
							
							bonds[a1].bond[bonds[a1].nbonds]=a2;
							bonds[a1].nbonds++;
							bonds[a2].bond[bonds[a2].nbonds]=a1;
							bonds[a2].nbonds++;
		
							gb_plist->param[idx].a[0]=a1;
							gb_plist->param[idx].a[1]=a2;
		
						    // LJ parameters	
							gb_plist->param[idx].c[0]=-1;
							gb_plist->param[idx].c[1]=-1;
							gb_plist->param[idx].c[2]=-1;
							gb_plist->param[idx].c[3]=-1;
		
							// GBSA parameters
							gb_plist->param[idx].c[4]=atype->radius[atoms.atom[a1].type]+atype->radius[atoms.atom[a2].type];	
							//pl->param[idx].c[5]=idef->iparams[m].harmonic.rA;
							gb_plist->param[idx].c[5]=plist[i].param[j].c[0];
							gb_plist->param[idx].c[6]=p2;
							gb_plist->param[idx].c[7]=atype->gb_radius[atoms.atom[a1].type]+atype->gb_radius[atoms.atom[a2].type];
							gb_plist->param[idx].c[8]=0.8875;
							//printf("a1=%d, a2=%d, l=%g\n",a1,a2,plist[i].param[j].c[0]);
							n12++;
							idx++;
						}
					}
					break;
				
				case F_G96BONDS:
				case F_MORSE:
				case F_CUBICBONDS:
				case F_HARMONIC:
				case F_FENEBONDS:
				case F_TABBONDS:
				case F_TABBONDSNC:
				case F_POLARIZATION:
				case F_VSITE2:
				case F_VSITE3:
				case F_VSITE3FD:
				case F_VSITE3FAD:
				case F_VSITE3OUT:
				case F_VSITE4FD:
				case F_VSITE4FDN:
					break;
						
				
				default:
					gmx_fatal(FARGS,"generate_gb_topology, F_BONDS");

			}
		}
	}
	
	for(i=0;i<F_NRE;i++)
	{
		if(IS_ANGLE(i))
		{
			switch(i)
			{
				case F_ANGLES:
				
					for(j=0;j<plist[i].nr; j++)
					{
						//m=idef->il[i].iatoms[j++];
						a1=plist[i].param[j].a[0];
						a2=plist[i].param[j].a[1];
						a3=plist[i].param[j].a[2];	
						
						gb_plist->param[idx].a[0]=a1;
						gb_plist->param[idx].a[1]=a3;
		
						// LJ parameters 	
						gb_plist->param[idx].c[0]=-1;
						gb_plist->param[idx].c[1]=-1;
						gb_plist->param[idx].c[2]=-1;
						gb_plist->param[idx].c[3]=-1;
		
						// GBSA parameters 
						gb_plist->param[idx].c[4]=atype->radius[atoms.atom[a1].type]+atype->radius[atoms.atom[a3].type];	
						//pl->param[idx].c[5]=-1;
						//pl->param[idx].c[5]=RAD2DEG*acos(idef->iparams[m].harmonic.rA);
						
						 for(k=0;k<bonds[a2].nbonds;k++)
						 {
							if(bonds[a2].bond[k]==a1)
							{
								rab=bl[a2].length[a1];
							}
                   
							else if(bonds[a2].bond[k]==a3)
							{
								rbc=bl[a2].length[a3];
							}
                   
						}
               
						cosine=cos(plist[i].param[j].c[0]/RAD2DEG);
						r2=rab*rab+rbc*rbc-(2*rab*rbc*cosine);
						gb_plist->param[idx].c[5]=sqrt(r2);
						gb_plist->param[idx].c[6]=p3;
						gb_plist->param[idx].c[7]=atype->gb_radius[atoms.atom[a1].type]+atype->gb_radius[atoms.atom[a3].type];
						gb_plist->param[idx].c[8]=0.3516;
						
						n13++;
						idx++;
					}
					break;
					
				case F_G96ANGLES:
				case F_CONSTR:
				case F_UREY_BRADLEY:
				case F_QUARTIC_ANGLES:
				case F_TABANGLES:
					break;
				
				default:
					gmx_fatal(FARGS,"generate_gb_topology, F_ANGLES");

			}
		}
	}
	
	for(i=0;i<plist[F_LJ14].nr; i++)
	{
		//m=plist[F_LJ14].[i++];
		a1=plist[F_LJ14].param[i].a[0];
		a2=plist[F_LJ14].param[i].a[1];
				
		gb_plist->param[idx].a[0]=a1;
		gb_plist->param[idx].a[1]=a2;
		
		// LJ parameters 
		//pl->param[idx].c[0]=idef->iparams[m].lj14.c6A;
		//pl->param[idx].c[1]=idef->iparams[m].lj14.c12A;
		//pl->param[idx].c[2]=idef->iparams[m].lj14.c6B;
		//pl->param[idx].c[3]=idef->iparams[m].lj14.c12B;
		
		gb_plist->param[idx].c[0]=-1;
		gb_plist->param[idx].c[1]=-1;
		gb_plist->param[idx].c[2]=-1;
		gb_plist->param[idx].c[3]=-1;
		
		// GBSA parameters
		gb_plist->param[idx].c[4]=atype->radius[atoms.atom[a1].type]+atype->radius[atoms.atom[a2].type];	
		gb_plist->param[idx].c[5]=-1;
		gb_plist->param[idx].c[6]=p3;
		gb_plist->param[idx].c[7]=atype->gb_radius[atoms.atom[a1].type]+atype->gb_radius[atoms.atom[a2].type];
		gb_plist->param[idx].c[8]=0.3516;
		idx++;
		n14++;
	}

	gb_plist->nr=n12+n13+n14;
	born->n12=n12;
	born->n13=n13;
	born->n14=n14;
	
	//for(i=0;i<n12;i++)
	//{

	//}
	
	return 0;
	
}

int convert_gb_params(t_idef *idef, t_functype ftype, int start, t_params *gb_plist, born_t *born)
{
	int k,nral,maxtypes,comb,type;
	real reppow;
	
	nral=NRAL(F_GB);
		
	/* pl->nr is the number of gb interactions, so we need to allocate nr*3 elements in iatoms */
	snew(idef->il[F_GB].iatoms,gb_plist->nr*3);

	maxtypes=idef->ntypes;
	comb=3;
	reppow=12;
	
	for(k=0;k<gb_plist->nr;k++)
	{
			if(maxtypes<=idef->ntypes)
			{
				maxtypes+=1000;
				srenew(idef->functype,maxtypes);
				srenew(idef->iparams,maxtypes);
			}
		
		type=enter_gb_params(idef,F_GB,gb_plist->param[k].c,comb,reppow,start,0);
		append_gb_interaction(&idef->il[F_GB],type,NRAL(F_GB),gb_plist->param[k].a);
	
	}
	
	printf("# %10s:   %d\n","GB-12",born->n12);
	printf("# %10s:   %d\n","GB-13",born->n13);
	printf("# %10s:   %d\n","GB-14",born->n14);
	
	return 0;

}

int init_gb_still(t_commrec *cr, t_forcerec  *fr, t_atomtypes *atype, t_idef *idef, t_atoms *atoms, born_t *born,int natoms)
{
  
  int i,j,i1,i2,k,m,nbond,nang,ia,ib,ic,id,nb,idx,idx2,at;
  int iam,ibm;
  int at0,at1;
  real length,angle;
  real r,ri,rj,ri2,ri3,rj2,r2,r3,r4,rk,ratio,term,h,doffset,electric;
  real p1,p2,p3,factor,cosine,rab,rbc;
  
#ifdef GMX_MPI  
  real vsol[natoms];
  real gp[natoms];
#endif  
  
#ifdef GMX_MPI
  pd_at_range(cr,&at0,&at1);
  
  for(i=0;i<natoms;i++)
	vsol[i]=gp[i]=0;
	
#else
 at0=0;
 at1=natoms;
#endif 
  
  for(i=0;i<natoms;i++)
	born->gpol_gromacs[i]=born->vsolv_gromacs[i]=0; 	
	
 /* Compute atomic solvation volumes for Still method */
  for(i=0;i<natoms;i++)
  {	
	ri=atype->gb_radius[atoms->atom[i].type];
	r3=ri*ri*ri;
	born->vsolv_gromacs[i]=(4*M_PI/3)*r3;
  }
	
  for(j=0;j<born->n12*3;j+=3)
  {
	m=idef->il[F_GB].iatoms[j];
	ia=idef->il[F_GB].iatoms[j+1];
	ib=idef->il[F_GB].iatoms[j+2];

	r=1.01*idef->iparams[m].gb.st;
	
    ri   = atype->gb_radius[atoms->atom[ia].type];
	rj   = atype->gb_radius[atoms->atom[ib].type];
	
	ri2  = ri*ri;
	ri3  = ri2*ri;
	rj2  = rj*rj;
	      
    ratio  = (rj2-ri2-r*r)/(2*ri*r);
    h      = ri*(1+ratio);
    term   = (M_PI3)*h*h*(3*ri-h);
	
#ifndef GMX_MPI	
	born->vsolv_gromacs[ia]=born->vsolv_gromacs[ia]-term;
#else
	vsol[ia]+=term;
#endif
							
	ratio  = (ri2-rj2-r*r)/(2*rj*r);
	h      = rj*(1+ratio);
	term   = (M_PI3)*h*h*(3*rj-h);

#ifndef GMX_MPI
	born->vsolv_gromacs[ib]=born->vsolv_gromacs[ib]-term;
#else
	vsol[ib]+=term;
#endif	
	
}
 
#ifdef GMX_MPI
	// Sum solvation volumes 
#ifdef GMX_DOUBLE
	gmx_sumd(natoms,vsol,cr);
#else
	gmx_sumf(natoms,vsol,cr);
#endif
	
	for(i=0;i<natoms;i++)
		born->vsolv_gromacs[i]=born->vsolv_gromacs[i]-vsol[i];
#endif 

  /* Get the self-, 1-2 and 1-3 polarization for analytical Still method */
  /* Self */
  for(j=0;j<natoms;j++)
  {
	if(born->vs[j]==1)
		born->gpol_gromacs[j]=-0.5*ONE_4PI_EPS0/(atype->gb_radius[atoms->atom[j].type]+DOFFSET+P1);
 }
 
 /* 1-2 */
 for(j=0;j<born->n12*3;j+=3)
  {
	 m=idef->il[F_GB].iatoms[j];
	 ia=idef->il[F_GB].iatoms[j+1];
	 ib=idef->il[F_GB].iatoms[j+2];

	 r=idef->iparams[m].gb.st;
	
	 r4=r*r*r*r;
	 
#ifndef GMX_MPI	 
	 born->gpol_gromacs[ia]=born->gpol_gromacs[ia]+P2*born->vsolv_gromacs[ib]/r4;
	 born->gpol_gromacs[ib]=born->gpol_gromacs[ib]+P2*born->vsolv_gromacs[ia]/r4;
#else
	 gp[ia]+=P2*born->vsolv_gromacs[ib]/r4;
	 gp[ib]+=P2*born->vsolv_gromacs[ia]/r4;
#endif
	 
  }
  
  /* 1-3 */
  for(j=born->n12*3;j<born->n12*3+born->n13*3;j+=3)
  {
	 m=idef->il[F_GB].iatoms[j];
	 ia=idef->il[F_GB].iatoms[j+1];
	 ib=idef->il[F_GB].iatoms[j+2];
	
	 r=idef->iparams[m].gb.st;
	 r4=r*r*r*r;
	
#ifndef GMX_MPI	  
	 born->gpol_gromacs[ia]=born->gpol_gromacs[ia]+P3*born->vsolv_gromacs[ib]/r4;
	 born->gpol_gromacs[ib]=born->gpol_gromacs[ib]+P3*born->vsolv_gromacs[ia]/r4;
#else
	 gp[ia]+=P3*born->vsolv_gromacs[ib]/r4;
	 gp[ib]+=P3*born->vsolv_gromacs[ia]/r4;
#endif

  }
  
#ifdef GMX_MPI
	// Sum polarisation eergies 
#ifdef GMX_DOUBLE
	gmx_sumd(natoms,gp,cr);
#else
	gmx_sumf(natoms,gp,cr);
#endif
	
	for(i=0;i<natoms;i++)
		born->gpol_gromacs[i]=born->gpol_gromacs[i]+gp[i];
#endif 

/*
    real vsum, gsum;
   vsum=0; gsum=0;
   
	for(i=0;i<natoms;i++)
     {
      
	    printf("final_init: id=%d, %s: v=%g, v_t=%g, g=%15.15f, g_t=%15.15f\n",
		cr->nodeid,
	      *(atoms->atomname[i]),
		born->vsolv_gromacs[i],
		born->vsolv_gromacs[i]*1000,	
	      born->gpol_gromacs[i],
		born->gpol_gromacs[i]/CAL2JOULE);
		
		vsum=vsum+(born->vsolv_gromacs[i]*1000);
		gsum=gsum+(born->gpol_gromacs[i]/CAL2JOULE);
     }
   
   printf("SUM: Vtot=%15.15f, Gtot=%15.15f\n",vsum,gsum);
    */
	//exit(1);

   return 0;
}


int calc_gb_rad(t_commrec *cr, t_inputrec *ir,int natoms, int nrfa, const t_iatom forceatoms[], const t_iparams forceparams[], const t_atoms *atoms,
				const t_atomtypes *atype, rvec x[], rvec f[],t_nblist *nl, born_t *born,t_mdatoms *md)
{

#if ((defined(__GNUC__) || defined(__INTEL_COMPILER) || defined(__PATHSCALE__) || defined(__PGIC__)) && \
(defined(__i386__) || defined(__x86_64__)))	

	// x86 or x86-64 with GCC inline assembly and/or SSE intrinsics 
	switch(ir->gb_algorithm)
	{
		case egbSTILL:
			calc_gb_rad_still_sse(cr,md->nr,atoms, atype, x[0], nl, born, md); //gblist_sr
			break;
		case egbHCT:
			gmx_fatal(FARGS, "HCT algorithm not supported with sse");
			//calc_gb_rad_hct_sse(cr,md->nr, forceatoms, forceparams,atoms,atype,x,nl,born,md); //gblist_sr
			break;
		case egbOBC:
			//gmx_fatal(FARGS, "OBC algorithm not supported with sse");
			calc_gb_rad_obc_sse(cr,md->nr,atoms,atype,x[0],nl,born,md); //gblist_sr
			break;

		default:
			gmx_fatal(FARGS, "Unknown sse-enabled algorithm for Born radii calculation: %d",ir->gb_algorithm);
	}

#else

	/* Switch for determining which algorithm to use for Born radii calculation */
	switch(ir->gb_algorithm)
	{
		case egbSTILL:
			calc_gb_rad_still(cr,md->nr, forceatoms, forceparams,atoms,atype,x,nl,born,md); //gblist_sr
			break;
		case egbHCT:
			calc_gb_rad_hct(cr,md->nr, forceatoms, forceparams,atoms,atype,x,nl,born,md); //gblist_sr
			break;
		case egbOBC:
			calc_gb_rad_obc(cr,md->nr, forceatoms, forceparams,atoms,atype,x,nl,born,md); //gblist_sr
			break;

		default:
			gmx_fatal(FARGS, "Unknown algorithm for Born radii calculation: %d",ir->gb_algorithm);
	}
	
#endif

	return 0;		
}

int calc_gb_rad_still(t_commrec *cr, int natoms, const t_iatom forceatoms[], const t_iparams forceparams[], const t_atoms *atoms,
								const t_atomtypes *atype, rvec x[], t_nblist *nl, born_t *born,t_mdatoms *md)
{	
	int i,k,n,nj0,nj1,ai,aj,type;
	real gpi,dr,dr2,dr4,idr4,rvdw,ratio,ccf,theta,term,rai,raj;
	real ix1,iy1,iz1,jx1,jy1,jz1,dx11,dy11,dz11;
	real rinv,idr2,idr6,vaj,dccf,cosq,sinq,prod,gpi2;
	real factor;
	
	factor=0.5*ONE_4PI_EPS0;
	
	n=0;

	for(i=0;i<natoms;i++)
		born->bRad_gromacs[i]=md->invsqrta[i]=1;
	
	for(i=0;i<nl->nri;i++ )
	{
		ai  = i;
		
		nj0 = nl->jindex[ai];			
		nj1 = nl->jindex[ai+1];
		
		gpi = born->gpol_gromacs[ai];
		rai = atype->gb_radius[atoms->atom[ai].type];
		
		ix1 = x[ai][0];
		iy1 = x[ai][1];
		iz1 = x[ai][2];
			
		for(k=nj0;k<nj1;k++)
		{
			aj    = nl->jjnr[k];
			
			jx1   = x[aj][0];
			jy1   = x[aj][1];
			jz1   = x[aj][2];
			
			dx11  = ix1-jx1;
			dy11  = iy1-jy1;
			dz11  = iz1-jz1;
			
			dr2   = dx11*dx11+dy11*dy11+dz11*dz11; 
			rinv  = invsqrt(dr2);
			idr2  = rinv*rinv;
			idr4  = idr2*idr2;
			idr6  = idr4*idr2;
			
			raj   = atype->gb_radius[atoms->atom[aj].type];
			rvdw  = rai + raj;
			
			ratio = dr2 / (rvdw * rvdw);
			vaj   = born->vsolv_gromacs[aj];
	
			if(ratio>P5INV) {
				ccf=1.0;
				dccf=0.0;
			}
			else
			{
				theta = ratio*PIP5;
				cosq  = cos(theta);
				term  = 0.5*(1.0-cosq);
				ccf   = term*term;
				sinq  = 1.0 - cosq*cosq;
				dccf  = 2.0*term*sinq*invsqrt(sinq)*PIP5*ratio;
			}
		
			prod          = P4*vaj;
			gpi           = gpi+prod*ccf*idr4;
			md->dadx[n++] = prod*(4*ccf-dccf)*idr6;
			
			//printf("xi=%g, xk=%g, vaj=%g, r6=%g, ccf=%g, dccf=%g\n",ix1,jx1,vaj,1.0/idr6,ccf,dccf); 
		}
		
		gpi2 = gpi * gpi;
		//born->bRad_gromacs[ai] = -0.5*ONE_4PI_EPS0/gpi;
		born->bRad_gromacs[ai] = factor*invsqrt(gpi2);
		md->invsqrta[ai]=invsqrt(born->bRad_gromacs[ai]);

	}
	
	return 0;
}

int calc_gb_rad_still_sse(t_commrec *cr, int natoms, const t_atoms *atoms, 
						  const t_atomtypes *atype, real *x, t_nblist *nl, born_t *born, t_mdatoms *md)
{

	int i,k,n,ai,ai3,aj1,aj2,aj3,aj4,nj0,nj1,offset;
	int aj13,aj23,aj33,aj43;
	int at0,at1;
	
	real gpi_ai,gpi2,gpi_tmp;
	real factor;
	
#ifdef GMX_MPI
    real sum_gpi[natoms];
	pd_at_range(cr,&at0,&at1);
#else
	at0=0;
	at1=natoms;
#endif	
	
	__m128 ix,iy,iz;
	__m128 jx,jy,jz;
	__m128 dx,dy,dz;
	__m128 t1,t2,t3;
	__m128 rsq11,rinv,rinv2,rinv4,rinv6;
	__m128 ratio,gpi,rai,raj,vaj,rvdw,mask_cmp;
	__m128 ccf,dccf,theta,cosq,term,sinq,res,prod;
	__m128 xmm1,xmm2,xmm3,xmm4,xmm5,xmm6,xmm7,xmm8; 
	
	__m128i mask, maski;
	
	const __m128 half  = {0.5f , 0.5f , 0.5f , 0.5f };
	const __m128 three = {3.0f , 3.0f , 3.0f , 3.0f };
	const __m128 one   = {1.0f,  1.0f , 1.0f , 1.0f };
	const __m128 two   = {2.0f , 2.0f , 2.0f,  2.0f };
	const __m128 zero  = {0.0f , 0.0f , 0.0f , 0.0f };
	const __m128 four  = {4.0f , 4.0f , 4.0f , 4.0f };
	
	const __m128 p5inv  = {P5INV, P5INV, P5INV, P5INV};
	const __m128 pip5   = {PIP5,  PIP5,  PIP5,  PIP5};
	const __m128 p4     = {P4,    P4,    P4,    P4};
		
	//float *tmp;
	//tmp=(float *) malloc(sizeof(float)*4);
	
	factor = 0.5 * ONE_4PI_EPS0;
	
	// keep the compiler happy
	t1 = _mm_setzero_ps();
	t2 = _mm_setzero_ps();
	t3 = _mm_setzero_ps();
	
	aj1  = aj2  = aj3  = aj4  = 0;
	aj13 = aj23 = aj33 = aj43 = 0;
	n = 0;
	
	for(i=0;i<natoms;i++)
		born->bRad_gromacs[i]=md->invsqrta[i]=1;
	
	for(i=0;i<nl->nri;i++)
	{
		ai     = i;
		ai3	   = ai*3;
		
		nj0    = nl->jindex[ai];
		nj1    = nl->jindex[ai+1];
		
		offset = (nj1-nj0)%4;
		
		// Polarization energy for atom ai
		gpi_ai = born->gpol_gromacs[ai];
		gpi    = _mm_set1_ps(0.0);
		
		// Load particle ai coordinates 
		ix     = _mm_set1_ps(x[ai3]);
		iy     = _mm_set1_ps(x[ai3+1]);
		iz     = _mm_set1_ps(x[ai3+2]);
		
		// Load particle ai gb_radius */
		rai    = _mm_set1_ps(atype->gb_radius[atoms->atom[ai].type]);
		
#ifdef GMX_MPI
		sum_gpi[ai] = 0;
#endif		
		
		for(k=nj0;k<nj1-offset;k+=4)
		{
			// do this with sse also?
			aj1 = nl->jjnr[k];	 // jnr1-4
			aj2 = nl->jjnr[k+1];
			aj3 = nl->jjnr[k+2];
			aj4 = nl->jjnr[k+3];
			
			aj13 = aj1 * 3; //Replace jnr with j3
			aj23 = aj2 * 3;
			aj33 = aj3 * 3;
			aj43 = aj4 * 3;
			
			// Load particle aj1-4 and transpose
			xmm1 = _mm_loadu_ps(x+aj13);
			xmm2 = _mm_loadu_ps(x+aj23);
			xmm3 = _mm_loadu_ps(x+aj33);
			xmm4 = _mm_loadu_ps(x+aj43);
			
			xmm5 = _mm_unpacklo_ps(xmm1, xmm2); 
			xmm6 = _mm_unpacklo_ps(xmm3, xmm4);
			xmm7 = _mm_unpackhi_ps(xmm1, xmm2);
			xmm8 = _mm_unpackhi_ps(xmm3, xmm4);
			
			jx = _mm_movelh_ps(xmm5, xmm6);
			jy = _mm_unpackhi_ps(xmm5, xmm6);
			jy = _mm_shuffle_ps(jy,jy,_MM_SHUFFLE(3,1,2,0));
			jz = _mm_movelh_ps(xmm7, xmm8);
			
			dx    = _mm_sub_ps(ix, jx);
			dy    = _mm_sub_ps(iy, jy);
			dz    = _mm_sub_ps(iz, jz);
			
			t1    = _mm_mul_ps(dx,dx);
			t2    = _mm_mul_ps(dy,dy);
			t3    = _mm_mul_ps(dz,dz);
			
			rsq11 = _mm_add_ps(t1,t2);
			rsq11 = _mm_add_ps(rsq11,t3); //rsq11=rsquare
			
			/* Perform reciprocal square root lookup, 12 bits accuracy */
			t1        = _mm_rsqrt_ps(rsq11);   /* t1=lookup, r2=x */
			/* Newton-Rhapson iteration */
			t2        = _mm_mul_ps(t1,t1); /* lu*lu */
			t3        = _mm_mul_ps(rsq11,t2);  /* x*lu*lu */
			t3        = _mm_sub_ps(three,t3); /* 3.0-x*lu*lu */
			t3        = _mm_mul_ps(t1,t3); /* lu*(3-x*lu*lu) */
			rinv      = _mm_mul_ps(half,t3); /* result for all four particles */
			
			rinv2     = _mm_mul_ps(rinv,rinv);
			rinv4     = _mm_mul_ps(rinv2,rinv2);
			rinv6     = _mm_mul_ps(rinv2,rinv4);
			 
			// is there is smarter way to do this (without four memory operations) ?????
			/*
			vaj       = _mm_set_ps(born->vsolv_gromacs[aj4],
								   born->vsolv_gromacs[aj3],
								   born->vsolv_gromacs[aj2],
								   born->vsolv_gromacs[aj1]);
			*/
			xmm1 = _mm_load_ss(born->vsolv_gromacs+aj1); //see comment at invsqrta
			xmm2 = _mm_load_ss(born->vsolv_gromacs+aj2); 
			xmm3 = _mm_load_ss(born->vsolv_gromacs+aj3); 
			xmm4 = _mm_load_ss(born->vsolv_gromacs+aj4);
			
			xmm1 = _mm_shuffle_ps(xmm1,xmm2,_MM_SHUFFLE(0,0,0,0)); //j1 j1 j2 j2
			xmm3 = _mm_shuffle_ps(xmm3,xmm4,_MM_SHUFFLE(0,0,0,0)); //j3 j3 j4 j4
			vaj  = _mm_shuffle_ps(xmm1,xmm3,_MM_SHUFFLE(2,0,2,0));
			
			
			raj       = _mm_set_ps(atype->gb_radius[atoms->atom[aj4].type],
								   atype->gb_radius[atoms->atom[aj3].type],
								   atype->gb_radius[atoms->atom[aj2].type],
								   atype->gb_radius[atoms->atom[aj1].type]);  
			
											   
			rvdw      = _mm_add_ps(rai,raj); 
			rvdw      = _mm_mul_ps(rvdw,rvdw);
			ratio     = _mm_div_ps(rsq11,rvdw); //ratio = dr2/(rvdw*rvdw)
						
			mask_cmp  = _mm_cmpgt_ps(ratio,p5inv); //if ratio>p5inv
			
			switch(_mm_movemask_ps(mask_cmp))
			{
				case 0xF:
					ccf  = one;
					dccf = zero;
					break;
				default:
			
					theta	  = _mm_mul_ps(ratio,pip5);
					sincos_ps(theta,&sinq,&cosq); // sine and cosine				
					term      = _mm_sub_ps(one,cosq); //1-cosq
					term      = _mm_mul_ps(half,term); //0.5*(1.0-cosq)
					ccf	      = _mm_mul_ps(term,term); // term*term
					dccf      = _mm_mul_ps(two,term); // 2 * term
					dccf      = _mm_mul_ps(dccf,sinq); // 2*term*sinq
					dccf      = _mm_mul_ps(dccf,pip5); //2*term*sinq*pip5 
					dccf      = _mm_mul_ps(dccf,ratio); //dccf = 2*term*sinq*PIP5*ratio
			
					ccf	      = (mask_cmp & one)  | _mm_andnot_ps(mask_cmp,ccf); //conditional as a mask
					dccf      = (mask_cmp & zero) | _mm_andnot_ps(mask_cmp,dccf);
					
			}
			
			prod      = _mm_mul_ps(p4,vaj);	
			xmm2      = _mm_mul_ps(ccf,rinv4);
			xmm2      = _mm_mul_ps(xmm2,prod); //prod*ccf*idr4
			gpi		  = _mm_add_ps(gpi,xmm2); // gpi = gpi + prod*ccf*idr4	
			
			/* Chain rule terms */
			ccf       = _mm_mul_ps(four,ccf);
			xmm3      = _mm_sub_ps(ccf,dccf);
			xmm3      = _mm_mul_ps(xmm3,rinv6);
			xmm1      = _mm_mul_ps(xmm3,prod);
			
			_mm_storeu_ps(md->dadx+n, xmm1);
			
			n = n + 4;
		}
		
		// deal with odd elements
		if(offset!=0)
		{
			aj1=aj2=aj3=aj4=0;
			
			if(offset==1)
			{
				aj1 = nl->jjnr[k];	 //jnr1-4
				aj13 = aj1 * 3; //Replace jnr with j3
				
				xmm1 = _mm_loadu_ps(x+aj13);
				
				xmm6 = xmm1; // x1 - - - 
				xmm4 = _mm_shuffle_ps(xmm1, xmm1, _MM_SHUFFLE(2,3,0,1)); //y1 - - -
				xmm5 = _mm_shuffle_ps(xmm1, xmm1, _MM_SHUFFLE(3,0,1,2)); //z1 - - - 
				
				raj       = _mm_set_ps(0.0f, 0.0f, 0.0f, atype->gb_radius[atoms->atom[aj1].type]); 
				vaj       = _mm_set_ps(0.0f, 0.0f, 0.0f, born->vsolv_gromacs[aj1]);				   
								   
				mask = _mm_set_epi32(0,0,0,0xffffffff);

			}
			else if(offset==2)
			{
				aj1 = nl->jjnr[k];	 // jnr1-4
				aj2 = nl->jjnr[k+1];
				
				aj13 = aj1 * 3; 
				aj23 = aj2 * 3;
				
				xmm1 = _mm_loadu_ps(x+aj13);
				xmm2 = _mm_loadu_ps(x+aj23);
							
				xmm6 = _mm_unpacklo_ps(xmm1, xmm2); //x1,x2,y1,y2 , done x
				xmm4 = _mm_shuffle_ps( xmm6, xmm6, _MM_SHUFFLE(3,2,3,2)); //y1, y2, y1, y2, done y
				xmm5 = _mm_unpackhi_ps(xmm1, xmm2); //z1, z2, z1, z2 , done z
		
				raj       = _mm_set_ps(0.0f, 0.0f, atype->gb_radius[atoms->atom[aj2].type], atype->gb_radius[atoms->atom[aj1].type]); 
				vaj       = _mm_set_ps(0.0f, 0.0f, born->vsolv_gromacs[aj2], born->vsolv_gromacs[aj1]);		
				
				mask = _mm_set_epi32(0,0,0xffffffff,0xffffffff);

			}
			else
			{
				aj1 = nl->jjnr[k];	 // jnr1-4
				aj2 = nl->jjnr[k+1];
				aj3 = nl->jjnr[k+2];
						
				aj13 = aj1 * 3; 
				aj23 = aj2 * 3;
				aj33 = aj3 * 3;
				
				xmm1 = _mm_loadu_ps(x+aj13);
				xmm2 = _mm_loadu_ps(x+aj23);
				xmm3 = _mm_loadu_ps(x+aj33);
											
				xmm5 = _mm_unpacklo_ps(xmm1, xmm2); // x1, x2, y1, y2
				xmm6 = _mm_movelh_ps(xmm5, xmm3); // x1, x2, x3, y3, done x

				xmm4 = _mm_shuffle_ps(xmm5, xmm5, _MM_SHUFFLE(3,2,3,2)); //y1, y2, x2, x1
				xmm4 = _mm_shuffle_ps(xmm4, xmm3, _MM_SHUFFLE(0,1,1,0)); //y1, y2, y3, x3, done y
				
				xmm5 = _mm_unpackhi_ps(xmm1, xmm2); // z1, z2, -, -
				xmm5 = _mm_shuffle_ps(xmm5, xmm3, _MM_SHUFFLE(3,2,1,0)); //z1, z2, z3, x4, done z
				
				raj       = _mm_set_ps(0.0f, atype->gb_radius[atoms->atom[aj3].type], 
										     atype->gb_radius[atoms->atom[aj2].type], 
											 atype->gb_radius[atoms->atom[aj1].type]); 
											 
				vaj       = _mm_set_ps(0.0f, born->vsolv_gromacs[aj3], 
										     born->vsolv_gromacs[aj2], 
											 born->vsolv_gromacs[aj1]);	
														
				mask = _mm_set_epi32(0,0xffffffff,0xffffffff,0xffffffff);
			}
			
			jx = _mm_and_ps( (__m128) mask, xmm6);
			jy = _mm_and_ps( (__m128) mask, xmm4);
			jz = _mm_and_ps( (__m128) mask, xmm5);
		
			dx    = _mm_sub_ps(ix, jx);
			dy    = _mm_sub_ps(iy, jy);
			dz    = _mm_sub_ps(iz, jz);
			
			t1    = _mm_mul_ps(dx,dx);
			t2    = _mm_mul_ps(dy,dy);
			t3    = _mm_mul_ps(dz,dz);
			
			rsq11 = _mm_add_ps(t1,t2);
			rsq11 = _mm_add_ps(rsq11,t3); //rsq11=rsquare
			
			/* Perform reciprocal square root lookup, 12 bits accuracy */
			t1        = _mm_rsqrt_ps(rsq11);   /* t1=lookup, r2=x */
			/* Newton-Rhapson iteration */
			t2        = _mm_mul_ps(t1,t1); /* lu*lu */
			t3        = _mm_mul_ps(rsq11,t2);  /* x*lu*lu */
			t3        = _mm_sub_ps(three,t3); /* 3.0-x*lu*lu */
			t3        = _mm_mul_ps(t1,t3); /* lu*(3-x*lu*lu) */
			rinv      = _mm_mul_ps(half,t3); /* result for all four particles */
			
			rinv2     = _mm_mul_ps(rinv,rinv);
			rinv4     = _mm_mul_ps(rinv2,rinv2);
			rinv6     = _mm_mul_ps(rinv2,rinv4);
			
			rvdw      = _mm_add_ps(rai,raj); 
			rvdw      = _mm_mul_ps(rvdw,rvdw);
			ratio     = _mm_div_ps(rsq11,rvdw); //ratio = dr2/(rvdw*rvdw)
						
			mask_cmp  = _mm_cmpgt_ps(ratio,p5inv); //if ratio>p5inv
			
			switch(_mm_movemask_ps(mask_cmp))
			{
				case 0xF:
					ccf  = one;
					dccf = zero;
					break;
				default:
			
					theta	  = _mm_mul_ps(ratio,pip5);
					sincos_ps(theta,&sinq,&cosq); // cosine and sine				
					term      = _mm_sub_ps(one,cosq);
					term      = _mm_mul_ps(half,term);
					ccf	      = _mm_mul_ps(term,term);
					dccf      = _mm_mul_ps(two,term);
					dccf      = _mm_mul_ps(dccf,sinq);
					dccf      = _mm_mul_ps(dccf,pip5);
					dccf      = _mm_mul_ps(dccf,theta); //dccf = 2*term*sinq*PIP5*ratio
			
					ccf	      = (mask_cmp & one)  | _mm_andnot_ps(mask_cmp,ccf); //conditional as a mask
					dccf      = (mask_cmp & zero) | _mm_andnot_ps(mask_cmp,dccf);
			}
		
			prod      = _mm_mul_ps(p4,vaj);	
			xmm2      = _mm_mul_ps(ccf,rinv4);
			xmm2      = _mm_mul_ps(xmm2,prod); // prod*ccf*idr4
			xmm2      = _mm_and_ps( (__m128) mask, xmm2);
			gpi       = _mm_add_ps(gpi,xmm2);  //gpi = gpi + prod*ccf*idr4
			
			/* Chain rule terms */
			ccf       = _mm_mul_ps(four,ccf);
			xmm3      = _mm_sub_ps(ccf,dccf);
			xmm3      = _mm_mul_ps(xmm3,rinv6);
			xmm1      = _mm_mul_ps(xmm3,prod);
			xmm1      = _mm_and_ps( (__m128) mask, xmm1);
			
			_mm_storeu_ps(md->dadx+n, xmm1); 
		
			n = n + offset;
			

		} // end offset
		
		// gpi now contains four partial terms that need to be added to particle ai gpi
		xmm2  = _mm_movehl_ps(xmm2,gpi);
		gpi   = _mm_add_ps(gpi,xmm2);
		xmm2  = _mm_shuffle_ps(gpi,gpi,_MM_SHUFFLE(1,1,1,1));
		gpi   = _mm_add_ss(gpi,xmm2);
		
		_mm_store_ss(&gpi_tmp,gpi);

#ifdef GMX_MPI
		sum_gpi[ai]=gpi_tmp;
#else		
		gpi_ai = gpi_ai + gpi_tmp; // add gpi to the initial pol energy gpi_ai
		gpi2   = gpi_ai * gpi_ai;
	
		born->bRad_gromacs[ai]=factor*invsqrt(gpi2);
		md->invsqrta[ai]=invsqrt(born->bRad_gromacs[ai]);
		
		//printf("i=%d, brad=%15.15f\n",i,born->bRad_gromacs[i]);
#endif		
		
	}
//exit(1);

#ifdef GMX_MPI

#ifdef GMX_DOUBLE
	gmx_sumd(natoms,sum_gpi,cr);
#else
	gmx_sumf(natoms,sum_gpi,cr);
#endif	

  for(i=at0;i<at1;i++)
  {
	ai = i;
	gpi_ai = born->gpol_gromacs[ai];
	gpi_ai = gpi_ai + sum_gpi[ai];
	gpi2   = gpi_ai*gpi_ai;
	
	born->bRad_gromacs[ai]=factor*invsqrt(gpi2);
	md->invsqrta[ai]=invsqrt(born->bRad_gromacs[ai]);
  }
  
  // Communicate Born radii
  gb_pd_send(cr,born->bRad_gromacs,md->nr);
  gb_pd_send(cr,md->invsqrta,md->nr);
  
//  for(i=0;i<natoms;i++)
//	printf("nodeid=%d, i=%d, brad=%g, inv=%g\n",cr->nodeid,i,born->bRad_gromacs[i],md->invsqrta[i]);


//	exit(1);
#endif


	return 0;
	

}

int calc_gb_rad_hct_sse()
{
	

	return 0;
}

int calc_gb_rad_obc_sse(t_commrec *cr, int natoms, const t_atoms *atoms,
					const t_atomtypes *atype, real *x, t_nblist *nl, born_t *born,t_mdatoms *md)
{
	int i,k,n,ai,ai3,aj1,aj2,aj3,aj4,nj0,nj1,at0,at1;
	int aj13,aj23,aj33,aj43;
	int offset;
	real ri;
	
	real rr,rr_inv,sum_tmp,sum_ai,gbr;
	real sum_ai2, sum_ai3,tsum,tchain;
	real z = 0;
	
#ifdef GMX_MPI
	real sum_mpi[natoms];
	pd_at_range(cr,&at0,&at1);
#else
	at0=0;
	at1=natoms;
#endif
	
	__m128 ix,iy,iz,jx,jy,jz;
	__m128 dx,dy,dz,t1,t2,t3;
	__m128 rsq11,rinv,r;
	__m128 rai,rai_inv,rai_inv2,sk,sk2,lij,dlij,duij;
	__m128 uij,lij2,uij2,lij3,uij3,diff2;
	__m128 lij_inv,sk2_inv,prod,log_term,tmp,tmp_sum;
	__m128 xmm1,xmm2,xmm3,xmm4,xmm5,xmm6,xmm7,xmm8; 
	__m128 mask_cmp,mask_cmp2,mask_cmp3;
	
	__m128i maski;
	
	const __m128 neg   = {-1.0f , -1.0f , -1.0f , -1.0f };
	const __m128 zero  = {0.0f , 0.0f , 0.0f , 0.0f };
	const __m128 eigth = {0.125f , 0.125f , 0.125f , 0.125f };
	const __m128 qrtr  = {0.25f , 0.25f , 0.25f , 0.25f };
	const __m128 half  = {0.5f , 0.5f , 0.5f , 0.5f };
	const __m128 one   = {1.0f , 1.0f , 1.0f , 1.0f };
	const __m128 two   = {2.0f , 2.0f , 2.0f , 2.0f };
	const __m128 three = {3.0f , 3.0f , 3.0f , 3.0f };
	
	//float apa[4];
	//printf("APA\n");
	//exit(1);	
	for(i=0;i<natoms;i++) {
		born->bRad_gromacs[i]=md->invsqrta[i]=1;
		born->drobc[i]=0;
	}
	
	// keep the compiler happy
	t1 = _mm_setzero_ps();
	t2 = _mm_setzero_ps();
	t3 = _mm_setzero_ps();
	
	aj1=aj2=aj3=aj4=0;
	aj13=aj23=aj33=aj43=0;
	n=0;
	
	for(i=0;i<nl->nri;i++)
	{
		ai       = nl->iinr[i];
		ai3      = ai*3;
		
		nj0      = nl->jindex[ai];
		nj1      = nl->jindex[ai+1];
		
		offset   = (nj1-nj0)%4;
		
		rr       = atype->gb_radius[atoms->atom[ai].type];
		ri       = rr+DOFFSET;
		rai      = _mm_load1_ps(&ri);
		
		ri       = 1.0/ri;
		rai_inv  = _mm_load1_ps(&ri);
		
		ix		 = _mm_load1_ps(x+ai3);
		iy		 = _mm_load1_ps(x+ai3+1);
		iz	     = _mm_load1_ps(x+ai3+2);
		
		sum_ai   = 0;
		tmp_sum = _mm_load1_ps(&z);
		
#ifdef GMX_MPI
		sum_mpi[ai] = 0;
#endif
		
		for(k=nj0;k<nj1-offset;k+=4)
		{
			aj1 = nl->jjnr[k];	 // jnr1-4
			aj2 = nl->jjnr[k+1];
			aj3 = nl->jjnr[k+2];
			aj4 = nl->jjnr[k+3];
			
			aj13 = aj1 * 3; //Replace jnr with j3
			aj23 = aj2 * 3;
			aj33 = aj3 * 3;
			aj43 = aj4 * 3;
			
			// Load particle aj1-4 and transpose
			xmm1 = __builtin_ia32_loadhps(xmm1,(__v2si*) (x+aj13));
			xmm2 = __builtin_ia32_loadhps(xmm2,(__v2si*) (x+aj23));
			xmm3 = __builtin_ia32_loadhps(xmm3,(__v2si*) (x+aj33));
			xmm4 = __builtin_ia32_loadhps(xmm4,(__v2si*) (x+aj43));
			
			xmm5    = _mm_load1_ps(x+aj13+2);  
			xmm6    = _mm_load1_ps(x+aj23+2); 
			xmm7    = _mm_load1_ps(x+aj33+2); 
			xmm8    = _mm_load1_ps(x+aj43+2);
						
			xmm5    = _mm_shuffle_ps(xmm5,xmm6,_MM_SHUFFLE(0,0,0,0));
			xmm6    = _mm_shuffle_ps(xmm7,xmm8,_MM_SHUFFLE(0,0,0,0));
			jz      = _mm_shuffle_ps(xmm5,xmm6,_MM_SHUFFLE(2,0,2,0));
			
			xmm1    = _mm_shuffle_ps(xmm1,xmm2,_MM_SHUFFLE(3,2,3,2));
			xmm2    = _mm_shuffle_ps(xmm3,xmm4,_MM_SHUFFLE(3,2,3,2));
			jx      = _mm_shuffle_ps(xmm1,xmm2,_MM_SHUFFLE(2,0,2,0));
			jy      = _mm_shuffle_ps(xmm1,xmm2,_MM_SHUFFLE(3,1,3,1));
						
			dx    = _mm_sub_ps(ix, jx);
			dy    = _mm_sub_ps(iy, jy);
			dz    = _mm_sub_ps(iz, jz);
			
			t1    = _mm_mul_ps(dx,dx);
			t2    = _mm_mul_ps(dy,dy);
			t3    = _mm_mul_ps(dz,dz);
			
			rsq11 = _mm_add_ps(t1,t2);
			rsq11 = _mm_add_ps(rsq11,t3); //rsq11=rsquare
			
			/* Perform reciprocal square root lookup, 12 bits accuracy */
			t1        = _mm_rsqrt_ps(rsq11);   /* t1=lookup, r2=x */
			/* Newton-Rhapson iteration */
			t2        = _mm_mul_ps(t1,t1); /* lu*lu */
			t3        = _mm_mul_ps(rsq11,t2);  /* x*lu*lu */
			t3        = _mm_sub_ps(three,t3); /* 3.0-x*lu*lu */
			t3        = _mm_mul_ps(t1,t3); /* lu*(3-x*lu*lu) */
			rinv      = _mm_mul_ps(half,t3); /* result for all four particles */
			
			r         = _mm_mul_ps(rinv,rsq11);
			
			xmm1 = _mm_load_ss(born->param+aj1); //load using load_ss and shuffle, since _mm_set_ps is
			xmm2 = _mm_load_ss(born->param+aj2); // buggy and generates buggy code with higher 
			xmm3 = _mm_load_ss(born->param+aj3); // optimization levels
			xmm4 = _mm_load_ss(born->param+aj4);
			
			xmm1 = _mm_shuffle_ps(xmm1,xmm2,_MM_SHUFFLE(0,0,0,0)); //j1 j1 j2 j2
			xmm3 = _mm_shuffle_ps(xmm3,xmm4,_MM_SHUFFLE(0,0,0,0)); //j3 j3 j4 j4
			sk = _mm_shuffle_ps(xmm1,xmm3,_MM_SHUFFLE(2,0,2,0));
						   
			xmm1      = _mm_add_ps(r,sk); //dr+sk					   
			
			// conditional mask for rai<dr+sk
			mask_cmp  = _mm_cmplt_ps(rai,xmm1);
			
			// conditional for rai>dr-sk, ends with mask_cmp2
			xmm2      = _mm_sub_ps(r,sk); //xmm2 = dr-sk
			
			xmm3      = _mm_rcp_ps(xmm2); //1.0/(dr-sk), 12 bits accuracy
			t1        = _mm_mul_ps(xmm3,xmm2);
			t1        = _mm_sub_ps(two,t1);
			xmm3      = _mm_mul_ps(t1,xmm3);
			
			xmm4      = rai_inv;
			xmm5      = zero; 
							
			mask_cmp2 = _mm_cmpgt_ps(rai,xmm2); //rai>dr-sk 
			lij     = (mask_cmp2 & xmm4)  | _mm_andnot_ps(mask_cmp2, xmm3);
			dlij    = (mask_cmp2 & xmm5)  | _mm_andnot_ps(mask_cmp2, one);
			
			uij		= _mm_rcp_ps(xmm1); // better approximation than just _mm_rcp_ps, which is just 12 bits
			t1      = _mm_mul_ps(uij,xmm1);
			t1      = _mm_sub_ps(two,t1);
			uij     = _mm_mul_ps(t1,uij);
			
			lij2    = _mm_mul_ps(lij,lij); 
			lij3    = _mm_mul_ps(lij2,lij);
			uij2    = _mm_mul_ps(uij,uij);
			uij3    = _mm_mul_ps(uij2,uij);		
					
			diff2   = _mm_sub_ps(uij2,lij2);
			
			//lij_inv = _mm_rsqrt_ps(lij2);
			/* Perform reciprocal square root lookup, 12 bits accuracy */
			t1        = _mm_rsqrt_ps(lij2);   /* t1=lookup, r2=x */
			/* Newton-Rhapson iteration */
			t2        = _mm_mul_ps(t1,t1); /* lu*lu */
			t3        = _mm_mul_ps(lij2,t2);  /* x*lu*lu */
			t3        = _mm_sub_ps(three,t3); /* 3.0-x*lu*lu */
			t3        = _mm_mul_ps(t1,t3); /* lu*(3-x*lu*lu) */
			lij_inv   = _mm_mul_ps(half,t3); /* result for all four particles */
			
			sk2     = _mm_mul_ps(sk,sk);
			sk2_inv = _mm_mul_ps(sk2,rinv);
			prod    = _mm_mul_ps(qrtr,sk2_inv);
				
			log_term = _mm_mul_ps(uij,lij_inv);
			log_term = log_ps(log_term);
			
			xmm1    = _mm_sub_ps(lij,uij);
			xmm2    = _mm_mul_ps(qrtr,r); // 0.25*dr
			xmm2    = _mm_mul_ps(xmm2,diff2); //0.25*dr*prod
			xmm1    = _mm_add_ps(xmm1,xmm2); //lij-uij + 0.25*dr*diff2
			xmm2    = _mm_mul_ps(half,rinv); // 0.5*rinv
			xmm2    = _mm_mul_ps(xmm2,log_term); //0.5*rinv*log_term
			xmm1    = _mm_add_ps(xmm1,xmm2); //lij-uij+0.25*dr*diff2+0.5*rinv*log_term
			xmm8    = _mm_mul_ps(neg,diff2); //(-1)*diff2
			xmm2    = _mm_mul_ps(xmm8,prod); //(-1)*diff2*prod
			tmp     = _mm_add_ps(xmm1,xmm2); // done tmp-term
			
			// contitional for rai<sk-dr
			xmm3    = _mm_sub_ps(sk,r);
			mask_cmp3 = _mm_cmplt_ps(rai,xmm3); //rai<sk-dr
			
			xmm4    = _mm_sub_ps(rai_inv,lij);
			xmm4    = _mm_mul_ps(two,xmm4);
			xmm4    = _mm_add_ps(tmp,xmm4);
					
			tmp   = (mask_cmp3 & xmm4) | _mm_andnot_ps(mask_cmp3,tmp); // xmm1 will now contain four tmp values
					
			// the tmp will now contain four partial values, that not all are to be used. Which
			// ones are governed by the mask_cmp mask. 
			tmp     = _mm_mul_ps(half,tmp); //0.5*tmp
			tmp     = (mask_cmp & tmp) | _mm_andnot_ps(mask_cmp, zero);
			tmp_sum = _mm_add_ps(tmp_sum,tmp);
		
			duij   = one;
					
			// start t1
			xmm2   = _mm_mul_ps(half,lij2); //0.5*lij2
			xmm3   = _mm_mul_ps(prod,lij3); //prod*lij3;
			xmm2   = _mm_add_ps(xmm2,xmm3); //0.5*lij2+prod*lij3
			xmm3   = _mm_mul_ps(lij,rinv); //lij*rinv
			xmm4   = _mm_mul_ps(lij3,r); //lij3*dr;
			xmm3   = _mm_add_ps(xmm3,xmm4); //lij*rinv+lij3*dr
			xmm3   = _mm_mul_ps(qrtr,xmm3); //0.25*(lij*rinv+lij3*dr)
			t1     = _mm_sub_ps(xmm2,xmm3); // done t1
					
			// start t2
			xmm2   = _mm_mul_ps(half,uij2); //0.5*uij2
			xmm2   = _mm_mul_ps(neg,xmm2); //(-1)*0.5*uij2
			xmm3   = _mm_mul_ps(qrtr,sk2_inv); //0.25*sk2_rinv
			xmm3   = _mm_mul_ps(xmm3,uij3); //0.25*sk2_rinv*uij3
			xmm2   = _mm_sub_ps(xmm2,xmm3); //(-1)*0.5*lij2-0.25*sk2_rinv*uij3
			xmm3   = _mm_mul_ps(uij,rinv); //uij*rinv
			xmm4   = _mm_mul_ps(uij3,r); //uij3*dr;
			xmm3   = _mm_add_ps(xmm3,xmm4); //uij*rinv+uij*dr
			xmm3   = _mm_mul_ps(qrtr,xmm3); //0.25*(uij*rinv+uij*dr)
			t2     = _mm_add_ps(xmm2,xmm3); // done t2
					
			// start t3
			xmm2   = _mm_mul_ps(sk2_inv,rinv);
			xmm2   = _mm_add_ps(one,xmm2); //1+sk2_rinv*rinv;
			xmm2   = _mm_mul_ps(eigth,xmm2); //0.125*(1+sk2_rinv*rinv)
			xmm2   = _mm_mul_ps(xmm2,xmm8); //0.125*(1+sk2_rinv*rinv)*(-diff2)
			xmm3   = _mm_mul_ps(log_term, rinv); //log_term*rinv
			xmm3   = _mm_mul_ps(xmm3,rinv); //log_term*rinv*rinv
			xmm3   = _mm_mul_ps(qrtr,xmm3); //0.25*log_term*rinv*rinv
			t3     = _mm_add_ps(xmm2,xmm3); // done t3
			
			// chain rule terms 
			xmm2   = _mm_mul_ps(dlij,t1); //dlij*t1
			xmm3   = _mm_mul_ps(duij,t2); //duij*t2
			xmm2   = _mm_add_ps(xmm2,xmm3);//dlij*t1+duij*t2
			xmm2   = _mm_add_ps(xmm2,t3); //
			xmm2   = _mm_mul_ps(xmm2,rinv);
					
			_mm_storeu_ps(md->dadx+n,xmm2);
			
			n      =  n + 4;
					
		} // end normal inner loop
		
		// deal with offset elements
		if(offset!=0)
		{
			aj1=aj2=aj3=aj4=0;
			
			if(offset==1)
			{
				aj1   = nl->jjnr[k];
				aj13  = aj1 * 3;
				
				xmm1  = __builtin_ia32_loadlps(xmm1,(__v2si*) (x+aj13));
				xmm5  = _mm_load1_ps(x+aj13+2);
				
				xmm6  = _mm_shuffle_ps(xmm1,xmm1,_MM_SHUFFLE(0,0,0,0));
				xmm4  = _mm_shuffle_ps(xmm1,xmm1,_MM_SHUFFLE(1,1,1,1));
				
				sk    = _mm_load1_ps(born->param+aj1);
						
				maski = _mm_set_epi32(0,0,0,0xffffffff);
			}
			else if(offset==2)
			{
				aj1   = nl->jjnr[k];
				aj2   = nl->jjnr[k+1];
				
				aj13  = aj1 * 3;
				aj23  = aj2 * 3;

				xmm1  = __builtin_ia32_loadhps(xmm1, (__v2si*) (x+aj13));
				xmm2  = __builtin_ia32_loadhps(xmm2, (__v2si*) (x+aj23));
				
				xmm5  = _mm_load1_ps(x+aj13+2);
				xmm6  = _mm_load1_ps(x+aj23+2);
				
				xmm5  = _mm_shuffle_ps(xmm5,xmm6,_MM_SHUFFLE(0,0,0,0));
				xmm5  = _mm_shuffle_ps(xmm5,xmm5,_MM_SHUFFLE(2,0,2,0));
				
				xmm1  = _mm_shuffle_ps(xmm1,xmm2,_MM_SHUFFLE(3,2,3,2));
				xmm6  = _mm_shuffle_ps(xmm1,xmm1,_MM_SHUFFLE(2,0,2,0));
				xmm4  = _mm_shuffle_ps(xmm1,xmm1,_MM_SHUFFLE(3,1,3,1));
				
				xmm1 = _mm_load1_ps(born->param+aj1);
				xmm2 = _mm_load1_ps(born->param+aj2);
				xmm1 = _mm_shuffle_ps(xmm1,xmm2,_MM_SHUFFLE(0,0,0,0));
				sk   = _mm_shuffle_ps(xmm1,xmm1,_MM_SHUFFLE(2,0,2,0));
				
				maski = _mm_set_epi32(0,0,0xffffffff,0xffffffff);
			}
			else
			{
				aj1   = nl->jjnr[k];
				aj2   = nl->jjnr[k+1];
				aj3   = nl->jjnr[k+2];
				
				aj13  = aj1 * 3;
				aj23  = aj2 * 3;
				aj33  = aj3 * 3;
				
				xmm1 = __builtin_ia32_loadhps(xmm1,(__v2si*) (x+aj13)); 
				xmm2 = __builtin_ia32_loadhps(xmm2,(__v2si*) (x+aj23)); 
				xmm3 = __builtin_ia32_loadhps(xmm3,(__v2si*) (x+aj33)); 
				
				xmm5 = _mm_load1_ps(x+aj13+2); 
				xmm6 = _mm_load1_ps(x+aj23+2); 
				xmm7 = _mm_load1_ps(x+aj33+2); 
											
				xmm5 = _mm_shuffle_ps(xmm5,xmm6, _MM_SHUFFLE(0,0,0,0));
				xmm5 = _mm_shuffle_ps(xmm5,xmm7, _MM_SHUFFLE(3,1,3,1));	// 0 1 0 1						
				
				xmm1 = _mm_shuffle_ps(xmm1,xmm2, _MM_SHUFFLE(3,2,3,2));
				xmm2 = _mm_shuffle_ps(xmm3,xmm3, _MM_SHUFFLE(3,2,3,2));
				
				xmm6 = _mm_shuffle_ps(xmm1,xmm2, _MM_SHUFFLE(2,0,2,0)); //0 1 0 1
				xmm4 = _mm_shuffle_ps(xmm1,xmm2, _MM_SHUFFLE(3,1,3,1));
			
				xmm1 = _mm_load1_ps(born->param+aj1);
				xmm2 = _mm_load1_ps(born->param+aj2);
				xmm3 = _mm_load1_ps(born->param+aj3);
				
				xmm1 = _mm_shuffle_ps(xmm1,xmm2,_MM_SHUFFLE(0,0,0,0)); //j1 j1 j2 j2
				xmm3 = _mm_shuffle_ps(xmm3,xmm3,_MM_SHUFFLE(0,0,0,0)); //j3 j3 j3 j3
				sk   = _mm_shuffle_ps(xmm1,xmm3,_MM_SHUFFLE(2,0,2,0));
				
				
				maski = _mm_set_epi32(0,0xffffffff,0xffffffff,0xffffffff);
			}
			
			jx = _mm_and_ps( (__m128) maski, xmm6);
			jy = _mm_and_ps( (__m128) maski, xmm4);
			jz = _mm_and_ps( (__m128) maski, xmm5);
			
			sk = _mm_and_ps ( (__m128) maski, sk);
			
			dx    = _mm_sub_ps(ix, jx);
			dy    = _mm_sub_ps(iy, jy);
			dz    = _mm_sub_ps(iz, jz);
			
			t1    = _mm_mul_ps(dx,dx);
			t2    = _mm_mul_ps(dy,dy);
			t3    = _mm_mul_ps(dz,dz);
			
			rsq11 = _mm_add_ps(t1,t2);
			rsq11 = _mm_add_ps(rsq11,t3); //rsq11=rsquare
			
			/* Perform reciprocal square root lookup, 12 bits accuracy */
			t1        = _mm_rsqrt_ps(rsq11);   /* t1=lookup, r2=x */
			/* Newton-Rhapson iteration */
			t2        = _mm_mul_ps(t1,t1); /* lu*lu */
			t3        = _mm_mul_ps(rsq11,t2);  /* x*lu*lu */
			t3        = _mm_sub_ps(three,t3); /* 3.0-x*lu*lu */
			t3        = _mm_mul_ps(t1,t3); /* lu*(3-x*lu*lu) */
			rinv      = _mm_mul_ps(half,t3); /* result for all four particles */
			
			r         = _mm_mul_ps(rinv,rsq11);
			
			xmm1      = _mm_add_ps(r,sk); //dr+sk					   
			
			// conditional mask for rai<dr+sk
			mask_cmp  = _mm_cmplt_ps(rai,xmm1);
			
			// conditional for rai>dr-sk, ends with mask_cmp2
			xmm2      = _mm_sub_ps(r,sk); //xmm2 = dr-sk
			
			xmm3      = _mm_rcp_ps(xmm2); //1.0/(dr-sk)
			t1        = _mm_mul_ps(xmm3,xmm2);
			t1        = _mm_sub_ps(two,t1);
			xmm3      = _mm_mul_ps(t1,xmm3);
						
			xmm4      = rai_inv;
			xmm5      = zero; 
							
			mask_cmp2 = _mm_cmpgt_ps(rai,xmm2); //rai>dr-sk 
			lij     = (mask_cmp2 & xmm4)  | _mm_andnot_ps(mask_cmp2, xmm3);
			dlij    = (mask_cmp2 & xmm5)  | _mm_andnot_ps(mask_cmp2, one);
	
			uij		= _mm_rcp_ps(xmm1);
			t1      = _mm_mul_ps(uij,xmm1);
			t1      = _mm_sub_ps(two,t1);
			uij     = _mm_mul_ps(t1,uij);
			
			lij2    = _mm_mul_ps(lij,lij); 
			lij3    = _mm_mul_ps(lij2,lij);
			uij2    = _mm_mul_ps(uij,uij);
			uij3    = _mm_mul_ps(uij2,uij);		
					
			diff2   = _mm_sub_ps(uij2,lij2);
			
			//lij_inv = _mm_rsqrt_ps(lij2);
			t1        = _mm_rsqrt_ps(lij2);   /* t1=lookup, r2=x */
			/* Newton-Rhapson iteration */
			t2        = _mm_mul_ps(t1,t1); /* lu*lu */
			t3        = _mm_mul_ps(lij2,t2);  /* x*lu*lu */
			t3        = _mm_sub_ps(three,t3); /* 3.0-x*lu*lu */
			t3        = _mm_mul_ps(t1,t3); /* lu*(3-x*lu*lu) */
			lij_inv   = _mm_mul_ps(half,t3); /* result for all four particles */

			sk2     = _mm_mul_ps(sk,sk);
			sk2_inv = _mm_mul_ps(sk2,rinv);
			prod    = _mm_mul_ps(qrtr,sk2_inv);
				
			log_term = _mm_mul_ps(uij,lij_inv);
			log_term = log_ps(log_term);
							
			xmm1    = _mm_sub_ps(lij,uij);
			xmm2    = _mm_mul_ps(qrtr,r); // 0.25*dr
			xmm2    = _mm_mul_ps(xmm2,diff2); //0.25*dr*prod
			xmm1    = _mm_add_ps(xmm1,xmm2); //lij-uij + 0.25*dr*diff2
			xmm2    = _mm_mul_ps(half,rinv); // 0.5*rinv
			xmm2    = _mm_mul_ps(xmm2,log_term); //0.5*rinv*log_term
			xmm1    = _mm_add_ps(xmm1,xmm2); //lij-uij+0.25*dr*diff2+0.5*rinv*log_term
			xmm8    = _mm_mul_ps(neg,diff2); //(-1)*diff2
			xmm2    = _mm_mul_ps(xmm8,prod); //(-1)*diff2*prod
			tmp     = _mm_add_ps(xmm1,xmm2); // done tmp-term
			
			// contitional for rai<sk-dr					
			xmm3    = _mm_sub_ps(sk,r);
			mask_cmp3 = _mm_cmplt_ps(rai,xmm3); //rai<sk-dr
			
			xmm4    = _mm_sub_ps(rai_inv,lij);
			xmm4    = _mm_mul_ps(two,xmm4);
			xmm4    = _mm_add_ps(xmm1,xmm4);
					
			tmp    = (mask_cmp3 & xmm4) | _mm_andnot_ps(mask_cmp3,tmp); // xmm1 will now contain four tmp values
				
			// tmp will now contain four partial values, that not all are to be used. Which
			// ones are governed by the mask_cmp mask. 
			tmp     = _mm_mul_ps(half,tmp); //0.5*tmp
			tmp     = (mask_cmp & tmp) | _mm_andnot_ps(mask_cmp, zero);
			tmp_sum = _mm_add_ps(tmp_sum,tmp);
			
			duij   = one;
					
			// start t1
			xmm2   = _mm_mul_ps(half,lij2); //0.5*lij2
			xmm3   = _mm_mul_ps(prod,lij3); //prod*lij3;
			xmm2   = _mm_add_ps(xmm2,xmm3); //0.5*lij2+prod*lij3
			xmm3   = _mm_mul_ps(lij,rinv); //lij*rinv
			xmm4   = _mm_mul_ps(lij3,r); //lij3*dr;
			xmm3   = _mm_add_ps(xmm3,xmm4); //lij*rinv+lij3*dr
			xmm3   = _mm_mul_ps(qrtr,xmm3); //0.25*(lij*rinv+lij3*dr)
			t1     = _mm_sub_ps(xmm2,xmm3); // done t1
		
			// start t2
			xmm2   = _mm_mul_ps(half,uij2); //0.5*uij2
			xmm2   = _mm_mul_ps(neg,xmm2); //(-1)*0.5*lij2
			xmm3   = _mm_mul_ps(qrtr,sk2_inv); //0.25*sk2_rinv
			xmm3   = _mm_mul_ps(xmm3,uij3); //0.25*sk2_rinv*uij3
			xmm2   = _mm_sub_ps(xmm2,xmm3); //(-1)*0.5*lij2-0.25*sk2_rinv*uij3
			xmm3   = _mm_mul_ps(uij,rinv); //uij*rinv
			xmm4   = _mm_mul_ps(uij3,r); //uij3*dr;
			xmm3   = _mm_add_ps(xmm3,xmm4); //uij*rinv+uij*dr
			xmm3   = _mm_mul_ps(qrtr,xmm3); //0.25*(uij*rinv+uij*dr)
			t2     = _mm_add_ps(xmm2,xmm3); // done t2
					
			// start t3
			xmm2   = _mm_mul_ps(sk2_inv,rinv);
			xmm2   = _mm_add_ps(one,xmm2); //1+sk2_rinv*rinv;
			xmm2   = _mm_mul_ps(eigth,xmm2); //0.125*(1+sk2_rinv*rinv)
			xmm2   = _mm_mul_ps(xmm2,xmm8); //0.125*(1+sk2_rinv*rinv)*(-diff2)
			xmm3   = _mm_mul_ps(log_term, rinv); //log_term*rinv
			xmm3   = _mm_mul_ps(xmm3,rinv); //log_term*rinv*rinv
			xmm3   = _mm_mul_ps(qrtr,xmm3); //0.25*log_term*rinv*rinv
			t3     = _mm_add_ps(xmm2,xmm3); // done t3
						 
			// chain rule terms 
			xmm2   = _mm_mul_ps(dlij,t1); //dlij*t1
			xmm3   = _mm_mul_ps(duij,t2); //duij*t2
			xmm2   = _mm_add_ps(xmm2,xmm3);//dlij*t1+duij*t2
			xmm2   = _mm_add_ps(xmm2,t3); //everyhting * t3
			xmm2   = _mm_mul_ps(xmm2,rinv); //everything * t3 *rinv
					
			_mm_storeu_ps(md->dadx+n,xmm2); // store excess elements, but since we are only advancing n by
											// offset, this will be corrected by the "main" loop
			
			n      =  n + offset;
			
		} // end offset
		
	
		// the tmp array will contain partial values that need to be added together
		tmp     = _mm_movehl_ps(tmp,tmp_sum);
		tmp_sum = _mm_add_ps(tmp_sum,tmp);
		tmp     = _mm_shuffle_ps(tmp_sum,tmp_sum,_MM_SHUFFLE(1,1,1,1));
		tmp_sum = _mm_add_ss(tmp_sum,tmp);
		
		_mm_store_ss(&sum_tmp,tmp_sum);

#ifdef GMX_MPI
		sum_mpi[ai]=sum_tmp;
#else
		sum_ai=sum_tmp;

		// calculate the born radii
		sum_ai  = (rr+DOFFSET) * sum_ai;
		sum_ai2 = sum_ai       * sum_ai;
		sum_ai3 = sum_ai2      * sum_ai;

		tsum    = tanh(AOBC*sum_ai-BOBC*sum_ai2+COBC*sum_ai3);
		born->bRad_gromacs[ai] = ri - tsum/rr;
		born->bRad_gromacs[ai] = 1.0/(born->bRad_gromacs[ai]);
		
		md->invsqrta[ai] = invsqrt(born->bRad_gromacs[ai]);
		tchain = (rr+DOFFSET)*(AOBC-2*BOBC*sum_ai+3*COBC*sum_ai2);
		born->drobc[ai] = (1.0 - tsum*tsum)*tchain*(1.0/rr);
		
		//if(born->vs[ai]==0)
		//	born->drobc[ai]=0;
				
		//printf("i=%d, brad=%15.15f,sum=%15.15f\n",i,born->bRad_gromacs[i],sum_ai);		
#endif

	}
	
//exit(1);

#ifdef GMX_MPI
	
#ifdef GMX_DOUBLE	
	gmx_sumd(natoms,sum_mpi,cr);
#else
	gmx_sumf(natoms,sum_mpi,cr);
#endif
	
	for(i=at0;i<at1;i++)
	{
		ai      = i;
		rr      = atype->gb_radius[atoms->atom[ai].type];
		rr_inv  = 1.0/rr;
		
		sum_ai  = sum_mpi[ai];
		sum_ai  = (rr+DOFFSET) * sum_ai;
		sum_ai2 = sum_ai       * sum_ai;
		sum_ai3 = sum_ai2      * sum_ai;
			
		tsum    = tanh(AOBC*sum_ai-BOBC*sum_ai2+COBC*sum_ai3);
		//born->bRad_gromacs[ai] = 1.0/(rr+DOFFSET) - tsum*rr_inv;
		gbr     = 1.0/(rr+DOFFSET) - tsum*rr_inv;
		
		//born->bRad_gromacs[ai] = 1.0 / born->bRad_gromacs[ai];
		born->bRad_gromacs[ai] = 1.0 / gbr;
		md->invsqrta[ai]=invsqrt(born->bRad_gromacs[ai]);
					
		tchain  = (rr+DOFFSET) * (AOBC-2*BOBC*sum_ai+3*COBC*sum_ai2);
		born->drobc[ai] = (1.0-tsum*tsum)*tchain*rr_inv;
	}
		
	gb_pd_send(cr,born->bRad_gromacs,md->nr);
	gb_pd_send(cr,md->invsqrta,md->nr);
	gb_pd_send(cr,born->drobc,md->nr);
	
#endif
	
	return 0;
}

int calc_gb_rad_hct(t_commrec *cr,int natoms, const t_iatom forceatoms[], const t_iparams forceparams[], const t_atoms *atoms,
					const t_atomtypes *atype, rvec x[], t_nblist *nl, born_t *born,t_mdatoms *md)
{
	int i,k,n,ai,aj,nj0,nj1,dum;
	real rai,raj,gpi,dr2,dr,sk,sk2,lij,uij,diff2,tmp,sum_ai;
	real rad,min_rad,rinv,rai_inv;
	real ix1,iy1,iz1,jx1,jy1,jz1,dx11,dy11,dz11;
	real lij2, uij2, lij3, uij3, t1,t2,t3;
	real lij_inv,dlij,duij,sk2_rinv,prod,log_term;
	rvec dx;


	/* Keep the compiler happy */
	n=0;
	prod=0;
	
#ifdef GMX_MPI
	real sum_tmp[natoms];
#endif
	
	for(i=0;i<natoms;i++)
		born->bRad_gromacs[i]=md->invsqrta[i]=1;
	
	for(i=0;i<nl->nri;i++)
	{
		ai = nl->iinr[i];
			
		nj0 = nl->jindex[ai];			
		nj1 = nl->jindex[ai+1];
		
		rai     = atype->gb_radius[atoms->atom[ai].type]+DOFFSET; //0.09 * 0.1 for nm, dielectric offset
		sum_ai  = 1.0/rai;
		rai_inv = sum_ai;
		
		ix1 = x[ai][0];
		iy1 = x[ai][1];
		iz1 = x[ai][2];
		
#ifdef GMX_MPI
		/* Only have the master node do this, since we only want one value at one time */
		if(MASTER(cr))
			sum_tmp[ai]=sum_ai;
		else
			sum_tmp[ai]=0;
#endif
		
		for(k=nj0;k<nj1;k++)
		{
			aj    = nl->jjnr[k];
				
			jx1   = x[aj][0];
			jy1   = x[aj][1];
			jz1   = x[aj][2];
			
			dx11  = ix1 - jx1;
			dy11  = iy1 - jy1;
			dz11  = iz1 - jz1;
			
			dr2   = dx11*dx11+dy11*dy11+dz11*dz11;
			rinv  = invsqrt(dr2);
			dr    = rinv*dr2;
			
			sk    = born->param[aj];
					
			if(rai < dr+sk)
			{
				//lij     = 1.0/(rai > dr-sk ? rai : dr-sk);
				lij     = 1.0/(dr-sk);
				dlij    = 1.0;
				
				if(rai>dr-sk) {
					lij  = rai_inv;
					dlij = 0.0;
				}
				
				lij2     = lij*lij;
				lij3     = lij2*lij;
				
				uij      = 1.0/(dr+sk);
				uij2     = uij*uij;
				uij3     = uij2*uij;
				
				diff2    = uij2-lij2;
				
				lij_inv  = invsqrt(lij2);
				sk2      = sk*sk;
				sk2_rinv = sk2*rinv;
				prod     = 0.25*sk2_rinv;
				
				//log_term = table_log(uij*lij_inv,born->log_table,LOG_TABLE_ACCURACY);
				log_term = log(uij*lij_inv);
				
				tmp      = lij-uij + 0.25*dr*diff2 + (0.5*rinv)*log_term + prod*(-diff2);
								
				if(rai<sk-dr)
					tmp = tmp + 2.0 * (rai_inv-lij);
					
				duij    = 1.0;
				t1      = 0.5*lij2 + prod*lij3 - 0.25*(lij*rinv+lij3*dr);
				t2      = -0.5*uij2 - 0.25*sk2_rinv*uij3 + 0.25*(uij*rinv+uij3*dr);
				t3      = 0.125*(1.0+sk2_rinv*rinv)*(-diff2)+0.25*log_term*rinv*rinv;
	
				md->dadx[n++] = (dlij*t1+duij*t2+t3)*rinv; //rb2 is moved to chainrule	
							
#ifdef GMX_MPI
				sum_tmp[ai] -= 0.5*tmp;
#else
				sum_ai -= 0.5*tmp;
#endif
			}
		}
	
#ifndef GMX_MPI			
		min_rad = rai - DOFFSET;
		rad=1.0/sum_ai; 
		
		born->bRad_gromacs[ai]=rad > min_rad ? rad : min_rad;
		md->invsqrta[ai]=invsqrt(born->bRad_gromacs[ai]);

#endif

	}

#ifdef GMX_MPI

#ifdef GMX_DOUBLE
	/* Do a global summation of all the parital sum_ai:s */
	gmx_sumd(natoms,sum_tmp,cr);
#else
	gmx_sumf(natoms,sum_tmp,cr);
#endif	
	
#endif


#ifdef GMX_MPI
	/* Calculate the Born radii so that they are available on all nodes */
	for(i=0;i<natoms;i++)
	{
		ai      = i;
		min_rad = atype->gb_radius[atoms->atom[ai].type]; 
		rad     = 1.0/sum_tmp[ai];
		
		born->bRad_gromacs[ai]=rad > min_rad ? rad : min_rad;
		//md->invsqrta[ai]=1.0/sqrt(born->bRad_gromacs[ai]);
		md->invsqrta[ai]=invsqrt(born->bRad_gromacs[ai]);
		
		//if(cr->nodeid==0)
		//	printf("MPI: brad=%15.15f\n",born->bRad_gromacs[i]);
	}
#endif

	return 0;
}

int calc_gb_rad_obc(t_commrec *cr, int natoms, const t_iatom forceatoms[], const t_iparams forceparams[], const t_atoms *atoms,
					const t_atomtypes *atype, rvec x[], t_nblist *nl, born_t *born,t_mdatoms *md)
{
	int i,k,ai,aj,nj0,nj1,n;
	real rai,raj,gpi,dr2,dr,sk,sk2,lij,uij,diff2,tmp,sum_ai;
	real rad, min_rad,sum_ai2,sum_ai3,tsum,tchain,rinv,rai_inv,lij_inv,rai_inv2;
	real log_term,prod,sk2_rinv;
	real ix1,iy1,iz1,jx1,jy1,jz1,dx11,dy11,dz11;
	real lij2,uij2,lij3,uij3,dlij,duij,t1,t2,t3,tmp2;

	//rvec dx;
	
	/* Keep the compiler happy */
	n=0;
	prod=0;
	
#ifdef GMX_MPI
		real sum_tmp[natoms];
#endif
	
	for(i=0;i<natoms;i++) {
		born->bRad_gromacs[i]=md->invsqrta[i]=1;
	}
		
	for(i=0;i<nl->nri;i++)
	{
		ai  = nl->iinr[i];
		
		nj0 = nl->jindex[ai];
		nj1 = nl->jindex[ai+1];
		
		rai      = atype->gb_radius[atoms->atom[ai].type]+DOFFSET;
		sum_ai   = 0;
		rai_inv  = 1.0/rai;
		rai_inv2 = 1.0/atype->gb_radius[atoms->atom[ai].type];
		
		ix1 = x[ai][0];
		iy1 = x[ai][1];
		iz1 = x[ai][2];
		
#ifdef GMX_MPI
		sum_tmp[ai]=0;
#endif	

		for(k=nj0;k<nj1;k++)
		{
			aj    = nl->jjnr[k];
	
			jx1   = x[aj][0];
			jy1   = x[aj][1];
			jz1   = x[aj][2];
			
			dx11  = ix1 - jx1;
			dy11  = iy1 - jy1;
			dz11  = iz1 - jz1;
			
			dr2   = dx11*dx11+dy11*dy11+dz11*dz11;
			rinv  = invsqrt(dr2);
			dr    = dr2*rinv;
		
			/* sk is precalculated in init_gb() */
			sk    = born->param[aj];
			//printf("ai=%d, aj=%d, sk=%g\n",ai,aj,sk);
			if(rai < dr+sk)
			{
				//lij      = 1.0/(rai > dr-sk ? rai : dr-sk);
				lij       = 1.0/(dr-sk);
				dlij      = 1.0;
								
				if(rai>dr-sk) {
					lij  = rai_inv;
					dlij = 0.0;
				}
				
				uij      = 1.0/(dr+sk);
				lij2     = lij  * lij;
				lij3     = lij2 * lij;
				uij2     = uij  * uij;
				uij3     = uij2 * uij;
				
				diff2    = uij2-lij2;
				
				lij_inv  = invsqrt(lij2);
				sk2      = sk*sk;
				sk2_rinv = sk2*rinv;	
				prod     = 0.25*sk2_rinv;
				
				/* Try to take the log away */
				log_term = log(uij*lij_inv);
				//log_term = table_log(uij*lij_inv,born->log_table,LOG_TABLE_ACCURACY);
				tmp      = lij-uij + 0.25*dr*diff2 + (0.5*rinv)*log_term + prod*(-diff2);
				
				if(rai < sk-dr)
					tmp = tmp + 2.0 * (rai_inv-lij);
					
				duij    = 1.0;
				t1      = 0.5*lij2 + prod*lij3 - 0.25*(lij*rinv+lij3*dr);
				t2      = -0.5*uij2 - 0.25*sk2_rinv*uij3 + 0.25*(uij*rinv+uij3*dr);
				t3      = 0.125*(1.0+sk2_rinv*rinv)*(-diff2)+0.25*log_term*rinv*rinv;
	
				md->dadx[n++] = (dlij*t1+duij*t2+t3)*rinv; //rb2 is moved to chainrule	
				
				sum_ai += 0.5*tmp;
				
#ifdef GMX_MPI	
				sum_tmp[ai] += 0.5*tmp;
#endif
				
			}
		}	

#ifndef GMX_MPI
	
	//if(born->vs[i]==1)
	//{
		sum_ai  = rai     * sum_ai;
		sum_ai2 = sum_ai  * sum_ai;
		sum_ai3 = sum_ai2 * sum_ai;
			
		tsum    = tanh(AOBC*sum_ai-BOBC*sum_ai2+COBC*sum_ai3);
		born->bRad_gromacs[ai] = rai_inv - tsum*rai_inv2;
		born->bRad_gromacs[ai] = 1.0 / born->bRad_gromacs[ai];
				
		md->invsqrta[ai]=invsqrt(born->bRad_gromacs[ai]);
			
		tchain  = rai * (AOBC-2*BOBC*sum_ai+3*COBC*sum_ai2);
		born->drobc[ai] = (1.0-tsum*tsum)*tchain*rai_inv2;
		/*
			}
		else
		{
			born->bRad_gromacs[i]=md->invsqrta[i]=1;
			born->drobc[i]=0;
			printf("APA2\n");
		}*/
		//printf("i=%d, brad=%15.15f,sum=%15.15f\n",i,born->bRad_gromacs[i],sum_ai);
		
#endif		
	}


#ifdef GMX_MPI

#ifdef GMX_DOUBLE
	gmx_sumd(natoms,sum_tmp,cr);
#else
	gmx_sumf(natoms,sum_tmp,cr);
#endif

	for(i=0;i<natoms;i++)
	{
		ai      = i;
		rai = atype->gb_radius[atoms->atom[ai].type];
		rai_inv = 1.0/rai;
		
		sum_ai  = sum_tmp[ai];
		sum_ai  = rai     * sum_ai;
		sum_ai2 = sum_ai  * sum_ai;
		sum_ai3 = sum_ai2 * sum_ai;
			
		tsum    = tanh(AOBC*sum_ai-BOBC*sum_ai2+COBC*sum_ai3);
		born->bRad_gromacs[ai] = rai_inv - tsum/atype->gb_radius[atoms->atom[ai].type];
		
		born->bRad_gromacs[ai] = 1.0 / born->bRad_gromacs[ai];
		md->invsqrta[ai]=invsqrt(born->bRad_gromacs[ai]);
			
		tchain  = rai * (AOBC-2*BOBC*sum_ai+3*COBC*sum_ai2);
		born->drobc[ai] = (1.0-tsum*tsum)*tchain/atype->gb_radius[atoms->atom[ai].type];
	}
	
#endif
	
	return 0;
}



real gb_bonds_tab(int nbonds, real *x, real *f, real *charge, real *p_gbtabscale,
				  real *invsqrta, real *dvda, real *GBtab, const t_iatom forceatoms[],
				  real epsilon_r, real facel)
{
	int i, n0,nnn,type,ai,aj,ai3,aj3;
	real isai,isaj;
	real r,rsq11,ix1,iy1,iz1,jx1,jy1,jz1;
	real dx11,dy11,dz11,rinv11,iq,facel2;
	real isaprod,qq,gbscale,gbtabscale,Y,F,Geps,Heps2,Fp,VV,FF,rt,eps,eps2;
	real vgb,fgb,vcoul,fijC,dvdatmp,fscal,tx,ty,tz,dvdaj;
	real vctot;	

	gbtabscale=*p_gbtabscale;
	vctot = 0.0;
	
	for(i=0;i<nbonds; )
	{
		type          = forceatoms[i++];
		ai            = forceatoms[i++];
		aj            = forceatoms[i++];
		ai3           = ai*3;
		aj3           = aj*3; 
	    isai          = invsqrta[ai];
		ix1           = x[ai3+0];
		iy1           = x[ai3+1];
		iz1           = x[ai3+2];
		iq            = (-1)*facel*charge[ai];
		jx1           = x[aj3+0];
		jy1           = x[aj3+1];
		jz1           = x[aj3+2];
		dx11          = ix1 - jx1;
		dy11          = iy1 - jy1;
		dz11          = iz1 - jz1;
		rsq11         = dx11*dx11+dy11*dy11+dz11*dz11;
		rinv11        = invsqrt(rsq11);
		isaj          = invsqrta[aj];
		isaprod       = isai*isaj;
		qq            = isaprod*iq*charge[aj];
		gbscale       = isaprod*gbtabscale;
		r             = rsq11*rinv11;
		rt            = r*gbscale;
		n0            = rt;
		eps           = rt-n0;
		eps2          = eps*eps;
		nnn           = 4*n0;
		Y             = GBtab[nnn];
		F             = GBtab[nnn+1];
		Geps          = eps*GBtab[nnn+2];
		Heps2         = eps2*GBtab[nnn+3];
		Fp            = F+Geps+Heps2;
		VV            = Y+eps*Fp;
		FF            = Fp+Geps+2.0*Heps2;
		vgb           = qq*VV;
		fijC          = qq*FF*gbscale;
		dvdatmp       = -(vgb+fijC*r)*0.5;
		dvda[aj]      = dvda[aj] + dvdatmp*isaj*isaj;
		dvda[ai]      = dvda[ai] + dvdatmp*isai*isai;
		vctot         = vctot + vgb;
		fgb           = -(fijC)*rinv11;
		tx            = fgb*dx11;
		ty            = fgb*dy11;
		tz            = fgb*dz11;
	
		f[aj3+0]      = f[aj3+0] - tx;
		f[aj3+1]      = f[aj3+1] - ty;
		f[aj3+2]      = f[aj3+2] - tz;
		
		f[ai3+0]      = f[ai3+0] + tx;
		f[ai3+1]      = f[ai3+1] + ty;
		f[ai3+2]      = f[ai3+2] + tz;
	}
	
	return vctot;
}


real calc_gb_selfcorrections(t_commrec *cr, int natoms, const t_atoms atoms[],
							 const t_atomtypes *atype, born_t *born, real *dvda, t_mdatoms *md, double facel)
{	
	int i,ai,at0,at1;
	real rai,e,derb,charge,charge2,fi,rai_inv,vtot;

#ifdef GMX_MPI
	pd_at_range(cr,&at0,&at1);
#else
	at0=0;
	at1=natoms;
#endif
			
	vtot=0.0;
	
	/* Apply self corrections */	
	for(i=at0;i<at1;i++)
	{
		if(born->vs[i]==1)
		{
			ai       = i;
			rai      = born->bRad_gromacs[ai];
			rai_inv  = 1.0/rai;
			charge   = atoms->atom[ai].q;
			charge2  = charge*charge;
			fi       = facel*charge2;
			e        = fi*rai_inv;
			derb     = 0.5*e*rai_inv*rai_inv;
			dvda[ai] += derb*rai;
			vtot     -= 0.5*e;
		}
	}
	
   return vtot;	
	
}

real calc_gb_nonpolar(t_commrec *cr, int natoms,born_t *born, const t_atoms atoms[], 
					  const t_atomtypes *atype, real *dvda,int gb_algorithm, t_mdatoms *md)
{
	int ai,i,at0,at1;
	real e,es,rai,rbi,term,probe,tmp,factor;
	real rbi_inv,rbi_inv2;
	
	/* To keep the compiler happy */
	factor=0;
	
#ifdef GMX_MPI
	pd_at_range(cr,&at0,&at1);
#else
	at0=0;
	at1=natoms;
#endif	
	
	/* The surface area factor is 0.0049 for Still model, 0.0054 for HCT/OBC */
	if(gb_algorithm==egbSTILL)
		factor=0.0049*100*CAL2JOULE;
		
	if(gb_algorithm==egbHCT || gb_algorithm==egbOBC)
		factor=0.0054*100*CAL2JOULE;	
	
	es    = 0;
	probe = 0.14;
	term  = M_PI*4;

	for(i=at0;i<at1;i++)
	{
		if(born->vs[i]==1)
		{
			ai        = i;
			rai		  = atype->gb_radius[atoms->atom[ai].type];
			rbi_inv   = md->invsqrta[ai];
			rbi_inv2  = rbi_inv * rbi_inv;
			tmp       = (rai*rbi_inv2)*(rai*rbi_inv2);
			tmp       = tmp*tmp*tmp;
			e         = factor*term*(rai+probe)*(rai+probe)*tmp;
			dvda[ai]  = dvda[ai] - 6*e*rbi_inv2;	
			es        = es + e;
		}
	}	

	return es;
}

real calc_gb_forces(t_commrec *cr, t_mdatoms *md, born_t *born, const t_atoms *atoms, const t_atomtypes *atype, int nr, 
                    rvec x[], rvec f[], t_forcerec *fr, const t_iatom forceatoms[], int gb_algorithm, bool bRad)
{
	real v=0;
	
	/* Do a simple ACE type approximation for the non-polar solvation */
	v += calc_gb_nonpolar(cr, md->nr, born, atoms, atype, md->dvda, gb_algorithm,md);
	
	//printf("np=%g\n",v);
	//v=0;

	/* Calculate the bonded GB-interactions */
	v += gb_bonds_tab(nr,x[0],f[0],md->chargeA,&(fr->gbtabscale),
					  md->invsqrta,md->dvda,fr->gbtab.tab,forceatoms,fr->epsilon_r, fr->epsfac);
					  
					  //printf("bonds=%g\n",v);
					  //v=0;
					  
	/* Calculate self corrections to the GB energies */
	v += calc_gb_selfcorrections(cr,md->nr,atoms, atype, born, md->dvda, md, fr->epsfac); 		
	
		//printf("self=%g\n",v);	
	
#ifdef GMX_MPI
	 /* Sum dvda */
#ifdef GMX_DOUBLE
	gmx_sumd(md->nr,md->dvda, cr);
#else 
	gmx_sumf(md->nr,md->dvda,cr);
#endif

#endif	
/*
#if ((defined(__GNUC__) || defined(__INTEL_COMPILER) || defined(__PATHSCALE__) || defined(__PGIC__)) && \
(defined(__i386__) || defined(__x86_64__)))	

	// x86 or x86-64 with GCC inline assembly and/or SSE intrinsics 
	calc_gb_chainrule_sse(md->nr, &(fr->gblist), md->dadx, md->dvda, x[0], f[0], gb_algorithm, born);	
		
#else
*/
	/* Calculate the forces due to chain rule terms with non sse code */
	calc_gb_chainrule(md->nr, &(fr->gblist), x, f, md->dvda, md->dadx, gb_algorithm, born);	

//#endif	
	/*
	for(i=0;i<md->nr;i++)
	{
		printf("i=%d, fx=%15.15f, fy=%15.15f, fz=%15.15f\n",i,f[i][0]/(CAL2JOULE*10),f[i][1]/(CAL2JOULE*10),f[i][2]/(CAL2JOULE*10));
	}
	exit(1);
	*/
	//exit(1);
	return v;

}


real calc_gb_chainrule(int natoms, t_nblist *nl, rvec x[], rvec t[], real *dvda, real *dadx, 
					   int gb_algorithm, born_t *born)
{	
	int i,k,n,ai,aj,nj0,nj1;
	real fgb,fij,rb2,rbi,fix1,fiy1,fiz1;
	real ix1,iy1,iz1,jx1,jy1,jz1,dx11,dy11,dz11,rsq11;
	real rinv11,tx,ty,tz,rbai;
	real rb[natoms];
	rvec dx;

	n=0;		
	
	/* Loop to get the proper form for the Born radius term */
	if(gb_algorithm==egbSTILL) {
		for(i=0;i<natoms;i++)
		{
			rbi   = born->bRad_gromacs[i];
			rb[i] = (2 * rbi * rbi * dvda[i])/ONE_4PI_EPS0;
		}
	}
		
	if(gb_algorithm==egbHCT) {
		for(i=0;i<natoms;i++)
		{
			rbi   = born->bRad_gromacs[i];
			rb[i] = rbi * rbi * dvda[i];
		}
	}
	 
	if(gb_algorithm==egbOBC) {
		for(i=0;i<natoms;i++)
		{
			rbi   = born->bRad_gromacs[i];
			rb[i] = rbi * rbi * born->drobc[i] * dvda[i];
			
			printf("i=%d, dvda=%g, drob=%g,rb=%g\n",i,dvda[i],born->drobc[i],rb[i]);
		}
	}
		exit(1);
	for(i=0;i<nl->nri;i++)
	{
		ai   = nl->iinr[i];
		nj0	 = nl->jindex[ai];
		nj1  = nl->jindex[ai+1];
		
		ix1  = x[ai][0];
		iy1  = x[ai][1];
		iz1  = x[ai][2];
		
		fix1 = 0;
		fiy1 = 0;
		fiz1 = 0;
		
		rbai = rb[ai];
				
		for(k=nj0;k<nj1;k++)
		{
			aj = nl->jjnr[k];
		
			jx1     = x[aj][0];
			jy1     = x[aj][1];
			jz1     = x[aj][2];
				
			dx11    = ix1 - jx1;
			dy11    = iy1 - jy1;
			dz11    = iz1 - jz1;
				
			fgb     = rbai*dadx[n++]; 
		
			//printf("xi=%g, xk=%g, dadx=%15.15f\n",ix1,jx1,dadx[n-1]/(CAL2JOULE*100));
		
			tx      = fgb * dx11;
			ty      = fgb * dy11;
			tz      = fgb * dz11;
				
			fix1    = fix1 + tx;
			fiy1    = fiy1 + ty;
			fiz1    = fiz1 + tz;
				
			/* Update force on atom aj */
			t[aj][0] = t[aj][0] - tx;
			t[aj][1] = t[aj][1] - ty;
			t[aj][2] = t[aj][2] - tz;
				
		}
		
		/* Update force on atom ai */
		t[ai][0] = t[ai][0] + fix1;
		t[ai][1] = t[ai][1] + fiy1;
		t[ai][2] = t[ai][2] + fiz1;
		
	}


	return 0;	
}

real calc_gb_chainrule_sse(int natoms, t_nblist *nl, real *dadx, real *dvda, real *xd, real *f, int gb_algorithm, born_t *born)						
{
	int    i,k,n,ai,aj,ai3,nj0,nj1,offset;
	int	   aj1,aj2,aj3,aj4; 
	
	real   rbi;
	real   rb[natoms+4];
	
	__m128 ix,iy,iz;
	__m128 jx,jy,jz;
	__m128 fix,fiy,fiz;
	__m128 dx,dy,dz;
	__m128 t1,t2,t3;
	__m128 dva,dax,fgb;
	__m128 xmm1,xmm2,xmm3,xmm4,xmm5,xmm6,xmm7,xmm8;
	
	__m128i mask   = _mm_set_epi32(0, 0xffffffff,0xffffffff,0xffffffff);
	__m128i maski  = _mm_set_epi32(0, 0xffffffff,0xffffffff,0xffffffff);
	
	const __m128 two = {2.0f , 2.0f , 2.0f , 2.0f };
	real z = 0;
			
	float apa[4];	
		
	/* Loop to get the proper form for the Born radius term, sse style */
	offset=natoms%4;
	
	if(offset!=0)
	{
		if(offset==1)
			mask = _mm_set_epi32(0,0,0,0xffffffff);
		else if(offset==2)
			mask = _mm_set_epi32(0,0,0xffffffff,0xffffffff);
		else
			mask = _mm_set_epi32(0,0xffffffff,0xffffffff,0xffffffff);
	}
	
	if(gb_algorithm==egbSTILL) {
		
		xmm3 = _mm_set1_ps(ONE_4PI_EPS0);
	
		for(i=0;i<natoms-offset;i+=4)
		{
			xmm1 = _mm_loadu_ps(born->bRad_gromacs+i);
			xmm1 = _mm_mul_ps(xmm1,xmm1);
			xmm1 = _mm_mul_ps(xmm1,two); //2 * rbi * rbi
			
			xmm2 = _mm_loadu_ps(dvda+i);
			
			xmm1 = _mm_mul_ps(xmm1,xmm2); // 2 * rbi * rbi * dvda[i]
			xmm1 = _mm_div_ps(xmm1,xmm3); // (2 * rbi * rbi * dvda[i]) / ONE_4PI_EPS0
		
			_mm_storeu_ps(rb+i, xmm1); // store to memory
		}
		
		// with the offset element, the mask stores excess elements to zero. This could cause problems
		// when something gets allocated right after rb (solution: allocate three positions bigger)
		if(offset!=0)
		{
			xmm1 = _mm_loadu_ps(born->bRad_gromacs+i);
			xmm1 = _mm_mul_ps(xmm1,xmm1);
			xmm1 = _mm_mul_ps(xmm1,two); 
			
			xmm2 = _mm_loadu_ps(dvda+i);
			
			xmm1 = _mm_mul_ps(xmm1,xmm2); 
			xmm1 = _mm_div_ps(xmm1,xmm3); 
			xmm1 = _mm_and_ps( (__m128) mask, xmm1);
		
			_mm_storeu_ps(rb+i, xmm1); 
		}
	}
		
	if(gb_algorithm==egbHCT) {
		for(i=0;i<natoms-offset;i+=4)
		{
			xmm1 = _mm_loadu_ps(born->bRad_gromacs+i);
			xmm1 = _mm_mul_ps(xmm1, xmm1); // rbi*rbi
			 
			xmm2 = _mm_loadu_ps(dvda+i);
			
			xmm1 = _mm_mul_ps(xmm1,xmm2); // rbi*rbi*dvda[i]
			
			_mm_storeu_ps(rb+i, xmm1); 
		}
		
		if(offset!=0)
		{
			xmm1 = _mm_loadu_ps(born->bRad_gromacs+i);
			xmm1 = _mm_mul_ps(xmm1, xmm1); // rbi*rbi
			 
			xmm2 = _mm_loadu_ps(dvda+i);
			
			xmm1 = _mm_mul_ps(xmm1,xmm2); // rbi*rbi*dvda[i]
			xmm1 = _mm_and_ps( (__m128) mask, xmm1);
			
			_mm_storeu_ps(rb+i, xmm1);
		}
	}
	 
	if(gb_algorithm==egbOBC) {
		for(i=0;i<natoms-offset;i+=4)
		{
			xmm1 = _mm_loadu_ps(born->bRad_gromacs+i);
			xmm1 = _mm_mul_ps(xmm1, xmm1); // rbi*rbi
			 
			xmm2 = _mm_loadu_ps(dvda+i);
			xmm1 = _mm_mul_ps(xmm1,xmm2); // rbi*rbi*dvda[i]
			xmm2 = _mm_loadu_ps(born->drobc+i);
			xmm1 = _mm_mul_ps(xmm1, xmm2); //rbi*rbi*dvda[i]*born->drobc[i]
						
			_mm_storeu_ps(rb+i, xmm1);
			_mm_storeu_ps(apa,xmm1);
			printf("rb: %g %g %g %g\n",apa[0],apa[1],apa[2],apa[3]);
		}
		
		
		if(offset!=0)
		{
			xmm1 = _mm_loadu_ps(born->bRad_gromacs+i);
			xmm1 = _mm_mul_ps(xmm1, xmm1); // rbi*rbi
			 
			xmm2 = _mm_loadu_ps(dvda+i);
			xmm1 = _mm_mul_ps(xmm1,xmm2); // rbi*rbi*dvda[i]
			xmm2 = _mm_loadu_ps(born->drobc+i);
			xmm1 = _mm_mul_ps(xmm1, xmm2); //rbi*rbi*dvda[i]*born->drobc[i]
			xmm1 = _mm_and_ps( (__m128) mask, xmm1);				
												
			_mm_storeu_ps(rb+i, xmm1);
			_mm_storeu_ps(apa,xmm1);
			printf("rb_offset: %g %g %g %g\n",apa[0],apa[1],apa[2],apa[3]);
		}
		exit(1);
	}

	// Keep the compiler happy 
	//t1 = _mm_setzero_ps();
	//t2 = _mm_setzero_ps();
	//t3 = _mm_setzero_ps();
	t1 = _mm_load1_ps(&z);
	t2 = _mm_load1_ps(&z);
	t3 = _mm_load1_ps(&z);
	
	aj1 = aj2 = aj3 = aj4 = 0;

	for(i=0;i<nl->nri;i++)
	{
		ai     = nl->iinr[i];
		ai3	   = ai*3;
		
		nj0    = nl->jindex[ai];
		nj1    = nl->jindex[ai+1];
		
		offset = (nj1-nj0)%4;
		
		// Load particle ai coordinates 
		ix  = _mm_load1_ps(xd+ai3);
		iy  = _mm_load1_ps(xd+ai3+1);
		iz  = _mm_load1_ps(xd+ai3+2);
		
		// Load particle ai dvda 
		dva = _mm_load1_ps(rb+ai);
		
		fix = _mm_load1_ps(&z);
		fiy = _mm_load1_ps(&z);
		fiz = _mm_load1_ps(&z);
				
		// Inner loop for all particles where n%4==0
		for(k=nj0;k<nj1-offset;k+=4)
		{
			// do this with sse also?
			aj1 = nl->jjnr[k];	 // jnr1-4
			aj2 = nl->jjnr[k+1];
			aj3 = nl->jjnr[k+2];
			aj4 = nl->jjnr[k+3];
		
			aj1 = aj1 * 3; //Replace jnr with j3
			aj2 = aj2 * 3;
			aj3 = aj3 * 3;
			aj4 = aj4 * 3;
			
			// Load j1-4 coordinates, first x and y
			xmm1 = __builtin_ia32_loadhps(xmm1,(__v2si*) (xd+aj1)); //x1 y1 - -
			xmm2 = __builtin_ia32_loadhps(xmm2,(__v2si*) (xd+aj2)); //x2 y2 - -
			xmm3 = __builtin_ia32_loadhps(xmm3,(__v2si*) (xd+aj3)); //x3 y3 - -
			xmm4 = __builtin_ia32_loadhps(xmm4,(__v2si*) (xd+aj4)); //x4 y4 - -
			
			// ... then z
			xmm5 = _mm_load1_ps(xd+aj1+2); // z1 z1 z1 z1 
			xmm6 = _mm_load1_ps(xd+aj2+2); // z2 z2 z2 z2
			xmm7 = _mm_load1_ps(xd+aj3+2); // z3 z3 z3 z3
			xmm8 = _mm_load1_ps(xd+aj4+2); // z4 z4 z4 z4
			
			//transpose
			xmm5 = _mm_shuffle_ps(xmm5,xmm6, _MM_SHUFFLE(0,0,0,0)); // z1 z1 z2 z2
			xmm6 = _mm_shuffle_ps(xmm7,xmm8, _MM_SHUFFLE(0,0,0,0)); // z3 z3 z4 z4
			jz   = _mm_shuffle_ps(xmm5,xmm6, _MM_SHUFFLE(2,0,2,0)); // z1 z2 z3 z4
			
			xmm1 = _mm_shuffle_ps(xmm1,xmm2, _MM_SHUFFLE(3,2,3,2)); // x1 y1 x2 y2
			xmm2 = _mm_shuffle_ps(xmm3,xmm4, _MM_SHUFFLE(3,2,3,2)); // x3 y3 x4 y4
			
			jx   = _mm_shuffle_ps(xmm1,xmm2, _MM_SHUFFLE(2,0,2,0)); // x1 x2 x3 x4
			jy   = _mm_shuffle_ps(xmm1,xmm2, _MM_SHUFFLE(3,1,3,1)); // y1 y2 y3 y4
			
			// load chain rule terms for j1-4
			dax = _mm_loadu_ps(dadx+k);
			
			// distances i -> j1-4
			dx   = _mm_sub_ps(ix, jx);
			dy   = _mm_sub_ps(iy, jy);
			dz   = _mm_sub_ps(iz, jz);
			
			// calculate scalar force
			fgb  = _mm_mul_ps(dva,dax); 
		
			// calculate partial force terms
			t1   = _mm_mul_ps(fgb,dx); // fx1, fx2, fx3, fx4
			t2   = _mm_mul_ps(fgb,dy); // fy1, fy2, fy3, fy4 
			t3   = _mm_mul_ps(fgb,dz); // fz1, fz2, fz3, fz4
		
			// update the i force
			fix       = _mm_add_ps(fix,t1);
			fiy       = _mm_add_ps(fiy,t2);
			fiz       = _mm_add_ps(fiz,t3);	
			
			// accumulate the aj1-4 fx and fy forces from memory
			xmm1 = __builtin_ia32_loadhps(xmm1, (__v2si*) (f+aj1)); //fx1 fy1 - - 
			xmm2 = __builtin_ia32_loadhps(xmm2, (__v2si*) (f+aj2)); //fx2 fy2 - - 
			xmm3 = __builtin_ia32_loadhps(xmm3, (__v2si*) (f+aj3)); //fx3 fy3 - -
			xmm4 = __builtin_ia32_loadhps(xmm4, (__v2si*) (f+aj4)); //fx4 fy4 - - 
			
			xmm5 = _mm_load1_ps(f+aj1+2); // fz1 fz1 fz1 fz1 
			xmm6 = _mm_load1_ps(f+aj2+2); // fz2 fz2 fz2 fz2
			xmm7 = _mm_load1_ps(f+aj3+2); // fz3 fz3 fz3 fz3
			xmm8 = _mm_load1_ps(f+aj4+2); // fz4 fz4 fz4 fz4
			
			// transpose forces
			xmm5 = _mm_shuffle_ps(xmm5,xmm6, _MM_SHUFFLE(0,0,0,0)); // fz1 fz1 fz2 fz2
			xmm6 = _mm_shuffle_ps(xmm7,xmm8, _MM_SHUFFLE(0,0,0,0)); // fz3 fz3 fz4 fz4
			xmm7 = _mm_shuffle_ps(xmm5,xmm6, _MM_SHUFFLE(2,0,2,0)); // fz1 fz2 fz3 fz4
			
			xmm1 = _mm_shuffle_ps(xmm1,xmm2,_MM_SHUFFLE(3,2,3,2)); //fx1 fy1 fx2 fy2
			xmm2 = _mm_shuffle_ps(xmm3,xmm4,_MM_SHUFFLE(3,2,3,2)); //fx2 fy3 fx4 fy4
			
			xmm5 = _mm_shuffle_ps(xmm1,xmm2,_MM_SHUFFLE(2,0,2,0)); //fx1 fx2 fx3 fx4
			xmm6 = _mm_shuffle_ps(xmm1,xmm2,_MM_SHUFFLE(3,1,3,1)); //fy1 fy2 fy3 fy4 
			
			// subtract partial forces
			xmm5 = _mm_sub_ps(xmm5, t1); //fx1 fx2 fx3 fx4
			xmm6 = _mm_sub_ps(xmm6, t2); //fy1 fy2 fy3 fy4
			xmm7 = _mm_sub_ps(xmm7, t3); //fz1 fz2 fz3 fz4
	
			// transpose back fx's and fy's
			xmm1 = _mm_shuffle_ps(xmm5,xmm6,_MM_SHUFFLE(1,0,1,0)); //fx1 fx2 fy1 fy2 
			xmm2 = _mm_shuffle_ps(xmm5,xmm6,_MM_SHUFFLE(3,2,3,2)); //fx3 fx4 fy3 fy4
			xmm1 = _mm_shuffle_ps(xmm1,xmm1,_MM_SHUFFLE(3,1,2,0)); //fx1 fy1 fx2 fy2
			xmm2 = _mm_shuffle_ps(xmm2,xmm2,_MM_SHUFFLE(3,1,2,0)); //fx3 fy3 fx4 fy4
			
			// store the force, first fx's and fy's
			__builtin_ia32_storelps( (__v2si*) (f+aj1), xmm1);
			__builtin_ia32_storehps( (__v2si*) (f+aj2), xmm1);
			__builtin_ia32_storelps( (__v2si*) (f+aj3), xmm2);
			__builtin_ia32_storehps( (__v2si*) (f+aj4), xmm2);
			
			// now do fz«s, why j4 before j3?
			_mm_store_ss(f+aj1+2,xmm7); //fz1
			xmm7 = _mm_shuffle_ps(xmm7,xmm7,_MM_SHUFFLE(0,3,2,1));
			_mm_store_ss(f+aj2+2,xmm7); //fz2
			xmm7 = _mm_shuffle_ps(xmm7,xmm7,_MM_SHUFFLE(1,0,3,2));
			_mm_store_ss(f+aj4+2,xmm7); //fz4
			xmm7 = _mm_shuffle_ps(xmm7,xmm7,_MM_SHUFFLE(2,1,0,3)); // 1 0 3 2
			_mm_store_ss(f+aj3+2,xmm7); //fz3
		}
		
		//deal with odd elements
		if(offset!=0) {
		
			aj1 = aj2 = aj3 = aj4 = 0;
			
			if(offset==1)
			{
				aj1 = nl->jjnr[k];	 //jnr1-4
				aj1 = aj1 * 3; //Replace jnr with j3
				
				xmm1 = __builtin_ia32_loadlps(xmm1,(__v2si*) (xd+aj1)); //x1 y1
				xmm5 = _mm_load1_ps(xd+aj1+2); //z1 z1 z1 z1
								
				xmm6 = _mm_shuffle_ps(xmm1, xmm1, _MM_SHUFFLE(0,0,0,0)); //x1 - - -  
				xmm4 = _mm_shuffle_ps(xmm1, xmm1, _MM_SHUFFLE(1,1,1,1)); //y1 - - -
				
				mask = _mm_set_epi32(0,0,0,0xffffffff);
			}
			else if(offset==2)
			{
				aj1 = nl->jjnr[k];	 // jnr1-4
				aj2 = nl->jjnr[k+1];
				
				aj1 = aj1 * 3; 
				aj2 = aj2 * 3;
				
				xmm1 = __builtin_ia32_loadhps(xmm1,(__v2si*) (xd+aj1)); // x1 y1 - -
				xmm2 = __builtin_ia32_loadhps(xmm2,(__v2si*) (xd+aj2)); // x2 y2 - -
				
				xmm5 = _mm_load1_ps(xd+aj1+2); //z1 z1 z1 z1
				xmm6 = _mm_load1_ps(xd+aj2+2); //z2 z2 z2 z2
				
				xmm5 = _mm_shuffle_ps(xmm5,xmm6,_MM_SHUFFLE(0,0,0,0)); //z1 z1 z2 z2
				xmm5 = _mm_shuffle_ps(xmm5,xmm5,_MM_SHUFFLE(2,0,2,0)); //z1 z2 z1 z2
				
				xmm1 = _mm_shuffle_ps(xmm1,xmm2,_MM_SHUFFLE(3,2,3,2)); //x1 y1 x2 y2
				xmm6 = _mm_shuffle_ps(xmm1,xmm1,_MM_SHUFFLE(2,0,2,0)); //x1 x2 x1 x2
				xmm4 = _mm_shuffle_ps(xmm1,xmm1,_MM_SHUFFLE(3,1,3,1)); //y1 y2 y1 y2	
										
				mask = _mm_set_epi32(0,0,0xffffffff,0xffffffff);
			}
			else
			{
				aj1 = nl->jjnr[k];	 // jnr1-4
				aj2 = nl->jjnr[k+1];
				aj3 = nl->jjnr[k+2];
						
				aj1 = aj1 * 3; 
				aj2 = aj2 * 3;
				aj3 = aj3 * 3;
				
				xmm1 = __builtin_ia32_loadhps(xmm1,(__v2si*) (xd+aj1)); //x1 y1 - -
				xmm2 = __builtin_ia32_loadhps(xmm2,(__v2si*) (xd+aj2)); //x2 y2 - -
				xmm3 = __builtin_ia32_loadhps(xmm3,(__v2si*) (xd+aj3)); //x3 y3 - -
				
				xmm5 = _mm_load1_ps(xd+aj1+2); // z1 z1 z1 z1 
				xmm6 = _mm_load1_ps(xd+aj2+2); // z2 z2 z2 z2
				xmm7 = _mm_load1_ps(xd+aj3+2); // z3 z3 z3 z3
											
				xmm5 = _mm_shuffle_ps(xmm5,xmm6, _MM_SHUFFLE(0,0,0,0)); // z1 z1 z2 z2
				xmm5 = _mm_shuffle_ps(xmm5,xmm7, _MM_SHUFFLE(3,1,3,1)); // z1 z2 z3 z3							
				
				xmm1 = _mm_shuffle_ps(xmm1,xmm2, _MM_SHUFFLE(3,2,3,2)); // x1 y1 x2 y2
				xmm2 = _mm_shuffle_ps(xmm3,xmm3, _MM_SHUFFLE(3,2,3,2)); // x3 y3 x3 y3
				
				xmm6 = _mm_shuffle_ps(xmm1,xmm2, _MM_SHUFFLE(2,0,2,0)); // x1 x2 x3 x3
				xmm4 = _mm_shuffle_ps(xmm1,xmm2, _MM_SHUFFLE(3,1,3,1)); // y1 y2 y3 y3
				
				mask = _mm_set_epi32(0,0xffffffff,0xffffffff,0xffffffff);
			}
						
			jx = _mm_and_ps( (__m128) mask, xmm6);
			jy = _mm_and_ps( (__m128) mask, xmm4);
			jz = _mm_and_ps( (__m128) mask, xmm5);
			
			dax = _mm_loadu_ps(dadx+k);
			dax = _mm_and_ps( (__m128) mask, dax); //possible reason?
			
			dx   = _mm_sub_ps(ix, jx);
			dy   = _mm_sub_ps(iy, jy);
			dz   = _mm_sub_ps(iz, jz);
			
			fgb  = _mm_mul_ps(dva,dax); //scalar force
		
			t1   = _mm_mul_ps(fgb,dx); // fx1, fx2, fx3, fx4
			t2   = _mm_mul_ps(fgb,dy); // fy1, fy2, fy3, fy4 
			t3   = _mm_mul_ps(fgb,dz); // fz1, fz2, fz3, fz4
				
			if(offset==1) {
				xmm1 = __builtin_ia32_loadlps(xmm1, (__v2si*) (f+aj1)); // fx1 fy1
				xmm7 = _mm_load1_ps(f+aj1+2); // fz1 fz1 fz1 fz1
				
				xmm5 = _mm_shuffle_ps(xmm1,xmm1, _MM_SHUFFLE(0,0,0,0)); // fx1 - - - 
				xmm6 = _mm_shuffle_ps(xmm1,xmm1, _MM_SHUFFLE(1,1,1,1)); // fy1 - - - 
				
				xmm5 = _mm_sub_ps(xmm5,t1);
				xmm6 = _mm_sub_ps(xmm6,t2);
				xmm7 = _mm_sub_ps(xmm7,t3);
								
				_mm_store_ss(f+aj1 , xmm5);
				_mm_store_ss(f+aj1+1,xmm6);
				_mm_store_ss(f+aj1+2,xmm7);
			}
			else if(offset==2) {
				xmm1 = __builtin_ia32_loadhps(xmm1, (__v2si*) (f+aj1)); //fx1 fy1 - - 
				xmm2 = __builtin_ia32_loadhps(xmm2, (__v2si*) (f+aj2)); //fx2 fy2 - - 
				
				xmm5 = _mm_load1_ps(f+aj1+2); // fz1 fz1 fz1 fz1 
				xmm6 = _mm_load1_ps(f+aj2+2); // fz2 fz2 fz2 fz2
				
				xmm5 = _mm_shuffle_ps(xmm5,xmm6,_MM_SHUFFLE(0,0,0,0)); //fz1 fz1 fz2 fz2
				xmm7 = _mm_shuffle_ps(xmm5,xmm5,_MM_SHUFFLE(2,0,2,0)); //fz1 fz2 fz1 fz2
				
				xmm1 = _mm_shuffle_ps(xmm1,xmm2,_MM_SHUFFLE(3,2,3,2)); //x1 y1 x2 y2
				xmm5 = _mm_shuffle_ps(xmm1,xmm1,_MM_SHUFFLE(2,0,2,0)); //x1 x2 x1 x2
				xmm6 = _mm_shuffle_ps(xmm1,xmm1,_MM_SHUFFLE(3,1,3,1)); //y1 y2 y1 y2
				
				xmm5 = _mm_sub_ps(xmm5, t1);
				xmm6 = _mm_sub_ps(xmm6, t2);
				xmm7 = _mm_sub_ps(xmm7, t3);
				
				xmm1 = _mm_shuffle_ps(xmm5,xmm6,_MM_SHUFFLE(1,0,1,0)); //fx1 fx2 fy1 fy2
				xmm5 = _mm_shuffle_ps(xmm1,xmm1,_MM_SHUFFLE(3,1,2,0)); //fx1 fy1 fx2 fy2
				
				__builtin_ia32_storelps( (__v2si*) (f+aj1), xmm5);
				__builtin_ia32_storehps( (__v2si*) (f+aj2), xmm5);
				
				_mm_store_ss(f+aj1+2,xmm7);
				xmm7 = _mm_shuffle_ps(xmm7,xmm7,_MM_SHUFFLE(0,3,2,1));
				_mm_store_ss(f+aj2+2,xmm7);
			}
			else {
				xmm1 = __builtin_ia32_loadhps(xmm1, (__v2si*) (f+aj1)); //fx1 fy1 - - 
				xmm2 = __builtin_ia32_loadhps(xmm2, (__v2si*) (f+aj2)); //fx2 fy2 - - 
				xmm3 = __builtin_ia32_loadhps(xmm3, (__v2si*) (f+aj3)); //fx3 fy3 - -
				
				xmm5 = _mm_load1_ps(f+aj1+2); // fz1 fz1 fz1 fz1 
				xmm6 = _mm_load1_ps(f+aj2+2); // fz2 fz2 fz2 fz2
				xmm7 = _mm_load1_ps(f+aj3+2); // fz3 fz3 fz3 fz3
				
				xmm5 = _mm_shuffle_ps(xmm5,xmm6, _MM_SHUFFLE(0,0,0,0)); // fz1 fz1 fz2 fz2
				xmm6 = _mm_shuffle_ps(xmm7,xmm7, _MM_SHUFFLE(0,0,0,0)); // fz3 fz3 fz3 fz3
				xmm7 = _mm_shuffle_ps(xmm5,xmm6, _MM_SHUFFLE(2,0,2,0)); // fz1 fz2 fz3 fz4
				
				xmm1 = _mm_shuffle_ps(xmm1,xmm2,_MM_SHUFFLE(3,2,3,2)); //fx1 fy1 fx2 fy2
				xmm2 = _mm_shuffle_ps(xmm3,xmm3,_MM_SHUFFLE(3,2,3,2)); //fx2 fy3 fx3 fy3
			
				xmm5 = _mm_shuffle_ps(xmm1,xmm2,_MM_SHUFFLE(2,0,2,0)); //fx1 fx2 fx3 fx3
				xmm6 = _mm_shuffle_ps(xmm1,xmm2,_MM_SHUFFLE(3,1,3,1)); //fy1 fy2 fy3 fy3 
				
				xmm5 = _mm_sub_ps(xmm5, t1);
				xmm6 = _mm_sub_ps(xmm6, t2);
				xmm7 = _mm_sub_ps(xmm7, t3);
				
				xmm1 = _mm_shuffle_ps(xmm5,xmm6,_MM_SHUFFLE(1,0,1,0)); //fx1 fx2 fy1 fy2 
				xmm2 = _mm_shuffle_ps(xmm5,xmm6,_MM_SHUFFLE(3,2,3,2)); //fx3 fx3 fy3 fy3
				xmm1 = _mm_shuffle_ps(xmm1,xmm1,_MM_SHUFFLE(3,1,2,0)); //fx1 fy1 fx2 fy2
				xmm2 = _mm_shuffle_ps(xmm2,xmm2,_MM_SHUFFLE(3,1,2,0)); //fx3 fy3 fx3 fy3
			
				__builtin_ia32_storelps( (__v2si*) (f+aj1), xmm1);
				__builtin_ia32_storehps( (__v2si*) (f+aj2), xmm1);
				__builtin_ia32_storelps( (__v2si*) (f+aj3), xmm2);
				
				_mm_store_ss(f+aj1+2,xmm7); //fz1
				xmm7 = _mm_shuffle_ps(xmm7,xmm7,_MM_SHUFFLE(0,3,2,1));
				_mm_store_ss(f+aj2+2,xmm7); //fz2
				xmm7 = _mm_shuffle_ps(xmm7,xmm7,_MM_SHUFFLE(1,0,3,2));
				_mm_store_ss(f+aj3+2,xmm7); //fz3

			}
			
			t1 = _mm_and_ps( (__m128) mask, t1);
			t2 = _mm_and_ps( (__m128) mask, t2);
			t3 = _mm_and_ps( (__m128) mask, t3);

			fix = _mm_add_ps(fix,t1);
			fiy = _mm_add_ps(fiy,t2);
			fiz = _mm_add_ps(fiz,t3);	
			
		} //end offset!=0
	 
		// fix/fiy/fiz now contain four partial force terms, that all should be
		// added to the i particle forces. 
		t1 = _mm_movehl_ps(t1,fix);
		t2 = _mm_movehl_ps(t2,fiy);
		t3 = _mm_movehl_ps(t3,fiz);
	
		fix = _mm_add_ps(fix,t1);
		fiy = _mm_add_ps(fiy,t2);
		fiz = _mm_add_ps(fiz,t3);
		
		t1 = _mm_shuffle_ps( fix, fix, _MM_SHUFFLE(1,1,1,1) );
		t2 = _mm_shuffle_ps( fiy, fiy, _MM_SHUFFLE(1,1,1,1) );
		t3 = _mm_shuffle_ps( fiz, fiz, _MM_SHUFFLE(1,1,1,1) );
		
		fix = _mm_add_ss(fix,t1); //fx - - -
		fiy = _mm_add_ss(fiy,t2); //fy - - - 
		fiz = _mm_add_ss(fiz,t3); //fz - - - 
		
		xmm2 = _mm_unpacklo_ps(fix,fiy); //fx, fy, - -
		xmm2 = _mm_movelh_ps(xmm2,fiz);
		xmm2 = _mm_and_ps( (__m128) maski, xmm2);
		
		// load i force from memory
		xmm4 = __builtin_ia32_loadlps(xmm4,(__v2si*) (f+ai3)); //fx fy - - 
		xmm5 = _mm_load1_ps(f+ai3+2); // fz fz fz fz
		xmm4 = _mm_shuffle_ps(xmm4,xmm5,_MM_SHUFFLE(3,2,1,0)); //fx fy fz fz
		
		// add to i force
		xmm4 = _mm_add_ps(xmm4,xmm2);
		
		// store i force to memory
		__builtin_ia32_storelps( (__v2si*) (f+ai3),xmm4); //fx fy - -
		xmm4 = _mm_shuffle_ps(xmm4,xmm4,_MM_SHUFFLE(2,2,2,2)); // only the third term is correct for fz
		_mm_store_ss(f+ai3+2,xmm4); //fz
	}	
		 	
	
	return 0;	
}

/* Skriv om den här rutinen så att den får samma format som Charmm SASA
 * Antagligen kan man då beräkna derivatan i samma loop som ytan,
 * vilket är snabbare (görs inte i Charmm, men det borde gå :-) )
 */
int calc_surfStill(t_inputrec *ir,
		  t_idef     *idef,
		  t_atoms    *atoms,
		  rvec       x[],
			rvec       f[],						 
		  born_t     *born,
		  t_atomtypes *atype,
			double     *faction,
			int        natoms,
			t_nblist   *nl,
			t_iparams  forceparams[],
			t_iatom    forceatoms[],
			int        nbonds)
{
  int i,j,n,ia,ib,ic;
  
  real pc[3],radp,probd,radnp,s,prob,bmlt;
  real dx,dy,dz,d,rni,dist,bee,ri,rn,asurf,t1ij,t2ij,bij,bji,dbijp;
  real dbjip,tip,tmp,tij,tji,t3ij,t4ij,t5ij,t3ji,t4ji,t5ji,dbij,dbji;
  real dpf,dpx,dpy,dpz,pi,pn,si,sn;
  real aprob[natoms];

  int factor=1;
  

  //bonds_t *bonds,*bonds13;
  
  //snew(bonds,natoms);
  //snew(bonds13,natoms);
   
  /* Zero out the forces to compare only surface area contribution */
  /*
  printf("Zeroing out forces before surface area..\n");
 
   for(i=0;i<natoms;i++)
    {
      
		if(!(is_hydrogen(*atoms->atomname[i])))
		{
			printf("x=%g, s=%g, p=%g, r=%g\n",
				x[i][0],
				atype->vol[atoms->atom[i].type],
				atype->surftens[atoms->atom[i].type],			
				factor*atype->radius[atoms->atom[i].type]);
	
		}
 
      //printf("faction[i]=%g\n",faction[i*3]);
      //faction[i*3]=0;
      //faction[i*3+1]=0;
      //faction[i*3+2]=0;
			
			f[i][0]=f[i][1]=f[i][2]=0;
    }
//exit(1);
*/
  /* Radius of probe */
  radp=0.14;
  probd=2*radp;
  	
	/*********************************************************
	 *********************************************************
	 *********************************************************
	 *********************************************************
	 *********************************************************
	 Begin SA calculation Gromacs-style
	 *********************************************************
	 *********************************************************
	 *********************************************************
	 *********************************************************
	 *********************************************************/
	
	int k,type,ai,aj,nj0,nj1;
	real dr2,sar,rai,raj,fij;
	rvec dxx;
	
	/* First set up the individual areas */
	for(n=0;n<natoms;n++)
	{	
		rn=atype->radius[atoms->atom[n].type];
		born->asurf_gromacs[n]=(4*M_PI*(rn+radp)*(rn+radp));
	}
	
	/* Then loop over the bonded interactions */
	for(i=0;i<nbonds; )
	{
		type = forceatoms[i++];
		ai   = forceatoms[i++];
		aj   = forceatoms[i++];
		
		if(!is_hydrogen(*(atoms->atomname[ai])) &&
			 !is_hydrogen(*(atoms->atomname[aj])))
		{
		
			//printf("genborn.c: xi=%g, xj=%g\n",factor*x[ai][0],factor*x[aj][0]);	
		
			rvec_sub(x[ai],x[aj],dxx);	
			dr2  = iprod(dxx,dxx);	
			
			sar  = forceparams[type].gb.sar;
			bmlt = forceparams[type].gb.bmlt;
			rni  = sar+probd;
			
			rn   = atype->radius[atoms->atom[ai].type];
			ri   = atype->radius[atoms->atom[aj].type];
			pn   = atype->surftens[atoms->atom[ai].type];
			pi   = atype->surftens[atoms->atom[aj].type];			
			//sn   = atype->vol[atoms->atom[ai].type];
			//si   = atype->vol[atoms->atom[aj].type];
			
			//rni  = rn + ri +probd;
			//printf("genborn.c: rn=%g, ri=%g\n",ri,rn);
			if(dr2<rni*rni)
			{
				//printf("d=%g, s=%g\n",dr2,rni*rni);
				dist = dr2*invsqrt(dr2);
				t1ij = M_PI*(rni-dist);
				t2ij = (rn-ri)/dist;
				bij  = (rn+radp)*t1ij*(1-t2ij);
				bji  = (ri+radp)*t1ij*(1+t2ij);
				tij  = pn*bmlt*bij/(4*M_PI*(rn+radp)*(rn+radp));
				tji  = pi*bmlt*bji/(4*M_PI*(ri+radp)*(ri+radp));
				
				born->asurf_gromacs[ai] = born->asurf_gromacs[ai]*(1-tij);
				born->asurf_gromacs[aj] = born->asurf_gromacs[aj]*(1-tji);
			}
		}
	}
	
	/* Now loop over interactions >= 1-4 */
	bmlt=0.3516;
	
	//printf("NONBONDED INTERACTIONS\n");
	
	for(i=0;i<natoms;i++)
	{
		ai    = i;
		nj0   = nl->jindex[ai];
		nj1   = nl->jindex[ai+1];
		
		rai   = factor*atype->radius[atoms->atom[ai].type];
		pn    = atype->surftens[atoms->atom[ai].type];
		//sn    = atype->vol[atoms->atom[ai].type];
		
		for(k=nj0;k<nj1;k++)
		{
			aj  = nl->jjnr[k];
			
			if(!is_hydrogen(*(atoms->atomname[ai])) &&
					!is_hydrogen(*(atoms->atomname[aj])))
			{
				raj = factor*atype->radius[atoms->atom[aj].type];
				pi  = atype->surftens[atoms->atom[aj].type];
				//si  = atype->vol[atoms->atom[aj].type];
				rvec_sub(x[ai],x[aj],dxx);
			
				dr2 = factor*factor*iprod(dxx,dxx);
				rni = rai + raj + probd;
				//printf("genborn.c: rn=%g, ri=%g, sar=%g, dr2=%g\n",rai,raj,sar,dr2);	
				//printf("genborn.c: xi=%g, xj=%g\n",factor*x[ai][0],factor*x[aj][0]);
				
				if(dr2<rni*rni)
				{
					//printf("d=%g, s=%g\n",dr2,rni*rni);	
					dist = dr2*invsqrt(dr2);
					t1ij = M_PI*(rni-dist);
					t2ij = (rai-raj)/dist;
					bij  = (rai+radp)*t1ij*(1-t2ij);
					bji  = (raj+radp)*t1ij*(1+t2ij);
					tij  = pn*bmlt*bij/(4*M_PI*(rai+radp)*(rai+radp));
					tji  = pi*bmlt*bji/(4*M_PI*(raj+radp)*(raj+radp));
									
					born->asurf_gromacs[ai]=born->asurf_gromacs[ai]*(1-tij);
					born->asurf_gromacs[aj]=born->asurf_gromacs[aj]*(1-tji);
				}
			}
		}
	}
	/*
	printf("AFTER BOTH AREA CALCULATIONS\n");
	n=0;
	for(i=0;i<natoms;i++)
	{
      if(!is_hydrogen(*atoms->atomname[i]))
			{	
				printf("%d, Still=%g, gromacs=%g\n",n,born->asurf[i], born->asurf_gromacs[i]);
				
				//born->as=born->as+born->asurf[i];
				born->as_gromacs=born->as_gromacs+born->asurf_gromacs[i];
			}
		n++;
	}
	*/
	//printf("Total Still area=%g, Total new area=%g\n",born->as, born->as_gromacs);
	 //printf("nbonds=%d\n",nbonds);
	/* Start to calculate the forces */
	for(i=0;i<nbonds; )
	{
		type = forceatoms[i++];
		ai   = forceatoms[i++];
		aj   = forceatoms[i++];
		
		if(!is_hydrogen(*(atoms->atomname[ai])) &&
			 !is_hydrogen(*(atoms->atomname[aj])))
		{
			rvec_sub(x[ai],x[aj],dxx);	
			
			dr2  = factor*factor*iprod(dxx,dxx);	
		
			sar  = factor*forceparams[type].gb.sar;
			bmlt = forceparams[type].gb.bmlt;
			rni  = sar+probd;
			
			rn   = factor*atype->radius[atoms->atom[ai].type];
			ri   = factor*atype->radius[atoms->atom[aj].type];
			pn   = atype->surftens[atoms->atom[ai].type];
			pi   = atype->surftens[atoms->atom[aj].type];			
			sn   = atype->vol[atoms->atom[ai].type];
			si   = atype->vol[atoms->atom[aj].type];
			
			if(dr2<rni*rni)
			{
				dist = dr2*invsqrt(dr2);
				t1ij = M_PI*(rni-dist);
				t2ij = (rn-ri)/dist;
				bij  = (rn+radp)*t1ij*(1-t2ij);
				bji  = (ri+radp)*t1ij*(1+t2ij);
				
				dbij = M_PI*(rn+radp)*(dr2-(rni*(rn-ri)));
				dbji = M_PI*(ri+radp)*(dr2+(rni*(rn-ri)));
				
				t3ij = sn*born->asurf_gromacs[ai]*dbij;
				t4ij = (4*M_PI*(rn+radp)*(rn+radp))/(pn*bmlt)-bij;
				t5ij = t3ij/t4ij;
				
				t3ji = si*born->asurf_gromacs[aj]*dbji;
				t4ji = (4*M_PI*(ri+radp)*(ri+radp))/(pi*bmlt)-bji;
				t5ji = t3ji/t4ji;
				
				dpf  = (t5ij+t5ji)/(dr2*dist);
				//printf("deriv_cut: ai=%d, xi=%g aj=%d, xj=%g\n",ai,x[ai][0], aj,x[aj][0]);
				for(k=0;k<DIM;k++)
				{
					fij = factor*(-dpf)*dxx[k];
					f[ai][k]+=fij;
					f[aj][k]-=fij;
				}
			}
		}
	}
	
	/* Now calculate forces for all interactions >= 1-4 */
	bmlt = 0.3516;
	
	for(i=0;i<natoms;i++)
	{
		ai  = i;
		nj0 = nl->jindex[ai];
		nj1 = nl->jindex[ai+1];
		
		rai   = factor*atype->radius[atoms->atom[ai].type];
		pn    = atype->surftens[atoms->atom[ai].type];
		sn    = atype->vol[atoms->atom[ai].type];
		
		for(k=nj0;k<nj1;k++)
		{
			aj = nl->jjnr[k];
			
			if(!is_hydrogen(*(atoms->atomname[ai])) &&
				 !is_hydrogen(*(atoms->atomname[aj])))
			{
				raj = factor*atype->radius[atoms->atom[aj].type];
				pi  = atype->surftens[atoms->atom[aj].type];
				si  = atype->vol[atoms->atom[aj].type];
				
				rvec_sub(x[ai],x[aj],dxx);
				
				dr2 = factor*factor*iprod(dxx,dxx);
				rni = rai + raj + probd;
				
				if(dr2<rni*rni)
				{
					dist = dr2*invsqrt(dr2);
					t1ij = M_PI*(rni-dist);
					t2ij = (rai-raj)/dist;
					bij  = (rai+radp)*t1ij*(1-t2ij);
					bji  = (raj+radp)*t1ij*(1+t2ij);
					
					dbij = M_PI*(rai+radp)*(dr2-(rni*(rai-raj)));
					dbji = M_PI*(raj+radp)*(dr2+(rni*(rai-raj)));
					
					t3ij = sn*born->asurf_gromacs[ai]*dbij;
					t4ij = (4*M_PI*(rai+radp)*(rai+radp))/(pn*bmlt)-bij;
					t5ij = t3ij/t4ij;
					
					t3ji = si*born->asurf_gromacs[aj]*dbji;
					t4ji = (4*M_PI*(raj+radp)*(raj+radp))/(pi*bmlt)-bji;
					t5ji = t3ji/t4ji;
					
					dpf  = (t5ij+t5ji)/(dr2*dist);
					//printf("deriv_cut: ai=%d, xi=%g aj=%d, xj=%g\n",ai,x[ai][0], aj,x[aj][0]);
					for(n=0;n<DIM;n++)
					{
						fij = factor*(-dpf)*dxx[n];
						f[ai][n]+=fij;
						f[aj][n]-=fij;
					}
				}
			}
		}
	}
	/*
	printf("AFTER BOTH FORCES CALCULATIONS\n");
	n=0;
	for(i=0;i<natoms;i++)
	{
		if(!is_hydrogen(*atoms->atomname[i]))
		{	
			printf("%d, gx=%g, gy=%g, gz=%g\n",
						 n,
						 //faction[i*3], 
						 //faction[i*3+1],
						 //faction[i*3+2],
						 f[i][0],
						 f[i][1],
						 f[i][2]);
		}
		n++;
	}
	*/
  return 0;
}

int calc_surfBrooks(t_inputrec *ir,
		    t_idef     *idef,
		    t_atoms    *atoms,
		    rvec       x[],
		    born_t     *born,
		    t_atomtypes *atype,
		    double      *faction,
		    int natoms)
{
  int i,j,k;
  real xi,yi,zi,dx,dy,dz,ri,rj;
  real rho,rho2,rho6,r2,r,aij,aijsum,daij;
  real kappa,sassum,tx,ty,tz,fix1,fiy1,fiz1;

  real ck[5];
  real Aij[natoms];
  real sasi[natoms];

  /* Brooks parameter for cutoff between atom pairs
   * Increasing kappa will increase the number of atom pairs
   * included in the calculation, which will also slow the calculation
   */
  kappa=0;

  sassum=0;

  /* Hydrogen atoms are included in via the ck parameters for the
   * heavy atoms
   */
  for(i=0;i<natoms;i++)
    {
      //if(strncmp(*atoms->atomname[i],"H",1)!=0)
      //{
	  xi=x[i][0];
	  yi=x[i][1];
	  zi=x[i][2];
	  
	  fix1=0;
	  fiy1=0;
	  fiz1=0;

	  ri=atype->radius[atoms->atom[i].type];
	  aijsum=0;
	  
	  for(j=0;j<natoms;j++)
	    {
	      //if(strncmp(*atoms->atomname[j],"H",1)!=0 && i!=j)
	      if(i!=j)
	      {
		  dx=xi-x[j][0];
		  dy=yi-x[j][1];
		  dz=zi-x[j][2];
		  
		  r2=dx*dx+dy*dy+dz*dz;
		  r=sqrt(r2);
		  rj=atype->radius[atoms->atom[j].type];
		  
		  rho=ri+rj+kappa;
		  rho2=rho*rho;
		  rho6=rho2*rho2*rho2;

		    /* Cutoff test */
		    if(r<=rho)
		      {
			aij=pow((rho2-r2)*(rho2+2*r2),2)/rho6;
			daij=((4*r*(rho2-r2)*(rho2-r2))/rho6)-((4*r*(rho2-r2)*(2*r2+rho2))/rho6);
			tx=daij*dx;
			ty=daij*dy;
			tz=daij*dz;
			
			fix1=fix1+tx;
			fiy1=fiy1+ty;
			fiz1=fiz1+tz;

			faction[j*3]=faction[j*3]-tx;
			faction[j*3+1]=faction[j*3+1]-ty;
			faction[j*3+2]=faction[j*3+2]-tz;
			
			aijsum=aijsum+aij;
			printf("xi=%g, xj=%g, fscal=%g\n",xi,x[j][0],daij);
		      }
		}
	    }
	  
	  faction[i*3]=faction[i*3]+fix1;
	  faction[i*3+1]=faction[i*3+1]+fiy1;
	  faction[i*3+2]=faction[i*3+2]+fiz1;
	  
	  /* Calculate surface area coefficient */
	  Aij[i]=pow(aijsum,1/4);
	  
	  for(k=0;k<5;k++)
	    {
	      sasi[i]=sasi[i]+ck[k]*(pow(Aij[i],k));
	    }
	  
	  /* Increase total surface area */
	  sassum=sassum+sasi[i];
	  
	  //}
    }

  printf("Brooks total surface area is: %g\n", sassum);


  return 0;
}


/* This will set up a really simple neighborlist for GB calculations
 * so that each atom will have enervy other atom in its list.
 * We don't worry about things liks load balancing etc ...
 */
 int do_gb_neighborlist(t_forcerec *fr, int natoms,t_atoms *atoms, t_ilist *il, int nbonds, int n12n13)
 {
   int i,j,k,m,ai,aj,ak,an,idx=0,idx_sr,vs_idx;
   int nalloc=0;
   int skip[natoms];
   bonds_t *bonds,*bonds13;
	 
   /* Calculate the number of elements in the jjnr array 
	* For the gblist_sr, this will be an exact allocation, but for
	* gblist_lr, this will be a few elements to much
	*/
   fr->gblist_sr.count=0;
   fr->gblist_sr.nri=natoms;
   fr->gblist_sr.maxnri=natoms;
	 
   fr->gblist_lr.count=0;
   fr->gblist_lr.nri=natoms;
   fr->gblist_lr.maxnri=natoms;
	 
   for(i=0;i<natoms;i++)
     {
		fr->gblist_sr.iinr[i]=i;
		fr->gblist_sr.gid[i]=0;
		fr->gblist_sr.shift[i]=0;
			 
		fr->gblist_lr.iinr[i]=i;
		fr->gblist_lr.gid[i]=0;
		fr->gblist_lr.shift[i]=0;
     }

   fr->gblist_sr.nltype=0;
   fr->gblist_sr.maxlen=natoms;
   fr->gblist_lr.nltype=0;
   fr->gblist_lr.maxlen=natoms;
	 
   /* Start the lr list */
	idx=0;
	idx_sr=0;
			 
	for(i=0;i<natoms;i++)
	 skip[i]=-1;
		 
	snew(bonds,natoms); 
	snew(bonds13,natoms);
	 
	for(i=0;i<nbonds; )
	{
		m=il->iatoms[i++];
		ai=il->iatoms[i++];
		aj=il->iatoms[i++];
				 
		bonds[ai].bond[bonds[ai].nbonds]=aj;
		bonds[ai].nbonds++;
		bonds[aj].bond[bonds[aj].nbonds]=ai;
		bonds[aj].nbonds++;
	 }
		 
	for(i=nbonds;i<n12n13; )
	 {
		 m=il->iatoms[i++];
		 ai=il->iatoms[i++];
		 aj=il->iatoms[i++];
		 
		 bonds13[ai].bond[bonds13[ai].nbonds]=aj;
		 bonds13[ai].nbonds++;
		 bonds13[aj].bond[bonds13[aj].nbonds]=ai;
		 bonds13[aj].nbonds++;
	 }
	 	 
	for(i=0;i<natoms;i++)
	 {
		 skip[i]=i;
					
		 for(k=0;k<bonds[i].nbonds;k++)
			skip[bonds[i].bond[k]]=i;
		 
		 for(k=0;k<bonds13[i].nbonds;k++)
			skip[bonds13[i].bond[k]]=i;
			
		 fr->gblist_lr.jindex[i]=idx;	
		 fr->gblist_sr.jindex[i]=idx_sr;
		 
		 for(k=i+1;k<natoms;k++)
		 {
			if(skip[k]!=i)
			{
				fr->gblist_lr.jjnr[idx++]=k;
			}
		 }
		 
		for(k=0;k<natoms;k++)
		 {
			if(skip[k]!=i)
			{
				fr->gblist_sr.jjnr[idx_sr++]=k;
			}
		 }
		 
	 }
	
	 fr->gblist_lr.jindex[i]=idx;
	 fr->gblist_sr.jindex[i]=idx_sr;
   
	 sfree(bonds);
	 sfree(bonds13);
	   
	 return 0;
 }

int gb_nblist_siev(t_commrec *cr, int natoms, int gb_algorithm, real gbcut, rvec x[], t_forcerec *fr, t_ilist *il, int n14)
{
	int i,l,ii,j,k,n,nj0,nj1,ai,aj,idx,ii_idx,nalloc,at0,at1;
	double dr2,gbcut2;
	rvec  dxx;
	t_nblist *nblist;

	int count[natoms];
	int **atoms;
	
	memset(count,0,sizeof(int)*natoms);
	atoms=(int **) malloc(sizeof(int *)*natoms);
	
#ifdef GMX_MPI
	pd_at_range(cr,&at0,&at1); 
#else
	at0=0;
	at1=natoms;
#endif
	
	for(i=0;i<natoms;i++)
		atoms[i]=(int *) malloc(sizeof(int)*natoms);

	if(gb_algorithm==egbHCT || gb_algorithm==egbOBC)
	{
		/* Loop over 1-2, 1-3 and 1-4 interactions */
		for(k=0;k<il->nr;k+=3)
		{
			ai=il->iatoms[k+1];
			aj=il->iatoms[k+2];
		
			/* When doing HCT or OBC, we need to add all interactions to the nb-list twice 
			 * since the loop for calculating the Born-radii runs over all vs all atoms */
			atoms[ai][count[ai]]=aj;
			count[ai]++;
			
			atoms[aj][count[aj]]=ai;
			count[aj]++;
		}
	}
		
	if(gb_algorithm==egbSTILL)
	{
		/* Loop over 1-4 interactions */
		for(k=n14;k<il->nr;k+=3)
		{
			ai=il->iatoms[k+1];
			aj=il->iatoms[k+2];
			
			/* Also for Still, we need to add (1-4) interactions twice */
			atoms[ai][count[ai]]=aj;
			count[ai]++;
			
			atoms[aj][count[aj]]=ai;
			count[aj]++;
			
		}
	}
			
	/* Loop over the VDWQQ and VDW nblists to set up the nonbonded part of the GB list */
	for(n=0; (n<fr->nnblists); n++)
	{
		for(i=0; (i<eNL_NR); i++)
		{
			nblist=&(fr->nblists[n].nlist_sr[i]);
			
			if(nblist->nri>0 && (i==eNL_VDWQQ || i==eNL_QQ))
			{
				for(j=0;j<nblist->nri;j++)
				{
					ai = nblist->iinr[j];
			
					nj0=nblist->jindex[j];
					nj1=nblist->jindex[j+1];
				
					for(k=nj0;k<nj1;k++)
					{
						aj=nblist->jjnr[k];
																										
						if(ai>aj)
						{
							atoms[aj][count[aj]]=ai;
							count[aj]++;
							
							/* We need to add all interactions to the nb-list twice 
							 * since the loop for calculating the Born-radii runs over all vs all atoms 
							 */
							atoms[ai][count[ai]]=aj;
							count[ai]++;
						}
						else
						{
							atoms[ai][count[ai]]=aj;
							count[ai]++;
							
							atoms[aj][count[aj]]=ai;
							count[aj]++;
						}
					}
				}
			}
		}
	}
		
	idx=0;
	ii_idx=0;
	
	for(i=0;i<natoms;i++)
	{
		fr->gblist.iinr[ii_idx]=i;
	
		for(k=0;k<count[i];k++)
		{
			fr->gblist.jjnr[idx++]=atoms[i][k];
		}
		
		fr->gblist.jindex[ii_idx+1]=idx;
		ii_idx++;
	}
	
	fr->gblist.nrj=idx;
	
	for(i=0;i<natoms;i++)
		free(atoms[i]);
	
	free(atoms);
	
	return 0;
}


int init_gb_nblist(int natoms, t_nblist *nl)
{
	 nl->maxnri      = natoms*4;
	 nl->maxnrj      = 0;
     nl->maxlen      = 0;
     nl->nri         = natoms;
     nl->nrj         = 0;
     nl->iinr        = NULL;
     nl->gid         = NULL;
     nl->shift       = NULL;
     nl->jindex      = NULL;
     //nl->nltype      = nltype;
	
	 srenew(nl->iinr,   nl->maxnri);
     srenew(nl->gid,    nl->maxnri);
     srenew(nl->shift,  nl->maxnri);
     srenew(nl->jindex, nl->maxnri+1);
       
	 nl->jindex[0] = 0;
	
	return 0;
}

int print_nblist(int natoms, t_nblist *nl)
{
	int i,k,ai,aj,nj0,nj1;
	
	 printf("genborn.c: print_nblist, natoms=%d\n",natoms); 
     for(i=0;i<natoms;i++)
	  {
		ai=nl->iinr[i];
		nj0=nl->jindex[i];
		nj1=nl->jindex[i+1];
		//printf("ai=%d, nj0=%d, nj1=%d\n",ai,nj0,nj1);
		for(k=nj0;k<nj1;k++)
		{	
			aj=nl->jjnr[k];
			printf("ai=%d, aj=%d\n",ai,aj);
		}
	  }

	return 0;	
}

void fill_log_table(const int n, real *table)
{
	float numlog;
	int i;
	int *const exp_ptr=((int*)&numlog);
	int x = *exp_ptr;
	
	x=0x3F800000;
	*exp_ptr = x;
	
	int incr = 1 << (23-n);
	int p=pow(2,n);
	
	for(i=0;i<p;++i)
	{
		table[i]=log2(numlog);
		x+=incr;
		*exp_ptr=x;
	}
}


real table_log(float val, const real *table, const int n)
{
	int *const exp_ptr = ((int*)&val);
	int x              = *exp_ptr;
	const int log_2    = ((x>>23) & 255) - 127;
	x &= 0x7FFFFF;
	x = x >> (23-n);
	val = table[x];
	return ((val+log_2)*0.69314718);  
}

void gb_pd_send(t_commrec *cr, real *send_data, int nr)
{
#ifndef GMX_MPI
	gmx_call("gb_pd_send");
#else

#ifdef GMX_DOUBLE
#define mpi_type MPI_DOUBLE
#else
#define mpi_type MPI_FLOAT
#endif

	int i,cur;
	int *index,*sendc,*disp;
	
	snew(sendc,cr->nnodes);
	snew(disp,cr->nnodes);
	
	index = pd_index(cr);
	cur   = cr->nodeid;
	
	/* Setup count/index arrays */
	for(i=0;i<cr->nnodes;i++)
	{
		sendc[i]  = index[i+1]-index[i];
		disp[i]   = index[i];	
	}
	
	/* Do communication */
	MPI_Gatherv(send_data+index[cur],sendc[cur],mpi_type,send_data,sendc,disp,mpi_type,0,cr->mpi_comm_mygroup);
	MPI_Bcast(send_data,nr,mpi_type,0,cr->mpi_comm_mygroup);
		

#endif

}



#if ((defined(__GNUC__) || defined(__INTEL_COMPILER) || defined(__PATHSCALE__) || defined(__PGIC__)) && \
(defined(__i386__) || defined(__x86_64)))

void sincos_ps(__m128 x, __m128 *s, __m128 *c) {
  __m128 xmm1, xmm2, xmm3 = _mm_setzero_ps(), sign_bit_sin, y;
  __m64 mm0, mm1, mm2, mm3, mm4, mm5;
  sign_bit_sin = x;
  /* take the absolute value */
  x = _mm_and_ps(x, *(__m128*)_ps_inv_sign_mask);
  /* extract the sign bit (upper one) */
  sign_bit_sin = _mm_and_ps(sign_bit_sin, *(__m128*)_ps_sign_mask);
  
  /* scale by 4/Pi */
  y = _mm_mul_ps(x, *(__m128*)_ps_cephes_FOPI);
    
  /* store the integer part of y in mm0:mm1 */
  xmm3 = _mm_movehl_ps(xmm3, y);
  mm2 = _mm_cvttps_pi32(y);
  mm3 = _mm_cvttps_pi32(xmm3);

  /* j=(j+1) & (~1) (see the cephes sources) */
  mm2 = _mm_add_pi32(mm2, *(__m64*)_pi32_1);
  mm3 = _mm_add_pi32(mm3, *(__m64*)_pi32_1);
  mm2 = _mm_and_si64(mm2, *(__m64*)_pi32_inv1);
  mm3 = _mm_and_si64(mm3, *(__m64*)_pi32_inv1);

  y = _mm_cvtpi32x2_ps(mm2, mm3);

  mm4 = mm2;
  mm5 = mm3;

  /* get the swap sign flag for the sine */
  mm0 = _mm_and_si64(mm2, *(__m64*)_pi32_4);
  mm1 = _mm_and_si64(mm3, *(__m64*)_pi32_4);
  mm0 = _mm_slli_pi32(mm0, 29);
  mm1 = _mm_slli_pi32(mm1, 29);
  __m128 swap_sign_bit_sin;
  COPY_MM_TO_XMM(mm0, mm1, swap_sign_bit_sin);

  /* get the polynom selection mask for the sine */

  mm2 = _mm_and_si64(mm2, *(__m64*)_pi32_2);
  mm3 = _mm_and_si64(mm3, *(__m64*)_pi32_2);
  mm2 = _mm_cmpeq_pi32(mm2, _mm_setzero_si64());
  mm3 = _mm_cmpeq_pi32(mm3, _mm_setzero_si64());
  __m128 poly_mask;
  COPY_MM_TO_XMM(mm2, mm3, poly_mask);

  /* The magic pass: "Extended precision modular arithmetic" 
     x = ((x - y * DP1) - y * DP2) - y * DP3; */
  xmm1 = *(__m128*)_ps_minus_cephes_DP1;
  xmm2 = *(__m128*)_ps_minus_cephes_DP2;
  xmm3 = *(__m128*)_ps_minus_cephes_DP3;
  xmm1 = _mm_mul_ps(y, xmm1);
  xmm2 = _mm_mul_ps(y, xmm2);
  xmm3 = _mm_mul_ps(y, xmm3);
  x = _mm_add_ps(x, xmm1);
  x = _mm_add_ps(x, xmm2);
  x = _mm_add_ps(x, xmm3);


  /* get the sign flag for the cosine */

  mm4 = _mm_sub_pi32(mm4, *(__m64*)_pi32_2);
  mm5 = _mm_sub_pi32(mm5, *(__m64*)_pi32_2);
  mm4 = _mm_andnot_si64(mm4, *(__m64*)_pi32_4);
  mm5 = _mm_andnot_si64(mm5, *(__m64*)_pi32_4);
  mm4 = _mm_slli_pi32(mm4, 29);
  mm5 = _mm_slli_pi32(mm5, 29);
  __m128 sign_bit_cos;
  COPY_MM_TO_XMM(mm4, mm5, sign_bit_cos);

  sign_bit_sin = _mm_xor_ps(sign_bit_sin, swap_sign_bit_sin);

  
  /* Evaluate the first polynom  (0 <= x <= Pi/4) */
  __m128 z = _mm_mul_ps(x,x);
  y = *(__m128*)_ps_coscof_p0;

  y = _mm_mul_ps(y, z);
  y = _mm_add_ps(y, *(__m128*)_ps_coscof_p1);
  y = _mm_mul_ps(y, z);
  y = _mm_add_ps(y, *(__m128*)_ps_coscof_p2);
  y = _mm_mul_ps(y, z);
  y = _mm_mul_ps(y, z);
  __m128 tmp = _mm_mul_ps(z, *(__m128*)_ps_0p5);
  y = _mm_sub_ps(y, tmp);
  y = _mm_add_ps(y, *(__m128*)_ps_1);
  
  /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

  __m128 y2 = *(__m128*)_ps_sincof_p0;
  y2 = _mm_mul_ps(y2, z);
  y2 = _mm_add_ps(y2, *(__m128*)_ps_sincof_p1);
  y2 = _mm_mul_ps(y2, z);
  y2 = _mm_add_ps(y2, *(__m128*)_ps_sincof_p2);
  y2 = _mm_mul_ps(y2, z);
  y2 = _mm_mul_ps(y2, x);
  y2 = _mm_add_ps(y2, x);

  /* select the correct result from the two polynoms */  
  xmm3 = poly_mask;
  __m128 ysin2 = _mm_and_ps(xmm3, y2);
  __m128 ysin1 = _mm_andnot_ps(xmm3, y);
  y2 = _mm_sub_ps(y2,ysin2);
  y = _mm_sub_ps(y, ysin1);

  xmm1 = _mm_add_ps(ysin1,ysin2);
  xmm2 = _mm_add_ps(y,y2);
 
  /* update the sign */
  *s = _mm_xor_ps(xmm1, sign_bit_sin);
  *c = _mm_xor_ps(xmm2, sign_bit_cos);
  _mm_empty(); /* good-bye mmx */
}


__m128 log_ps(__m128 x) {
  __m64 mm0, mm1;
  __m128 one = *(__m128*)_ps_1;

  __m128 invalid_mask = _mm_cmple_ps(x, _mm_setzero_ps());

  x = _mm_max_ps(x, *(__m128*)_ps_min_norm_pos);  /* cut off denormalized stuff */

  
  /* part 1: x = frexpf(x, &e); */
  COPY_XMM_TO_MM(x, mm0, mm1);
  mm0 = _mm_srli_pi32(mm0, 23);
  mm1 = _mm_srli_pi32(mm1, 23);
  /* keep only the fractional part */
  x = _mm_and_ps(x, *(__m128*)_ps_inv_mant_mask);
  x = _mm_or_ps(x, *(__m128*)_ps_0p5);

  /* now e=mm0:mm1 contain the really base-2 exponent */
  mm0 = _mm_sub_pi32(mm0, *(__m64*)_pi32_0x7f);

 
  mm1 = _mm_sub_pi32(mm1, *(__m64*)_pi32_0x7f);
  //printf("log_ps: e="); print2i(mm0); print2i(mm1); printf("\n"); 

  __m128 e = _mm_cvtpi32x2_ps(mm0, mm1);
  e = _mm_add_ps(e, one);

  /* part2: 
     if( x < SQRTHF ) {
       e -= 1;
       x = x + x - 1.0;
     } else { x = x - 1.0; }
  */
  //xmm_mm_union mmask;
  //mmask.xmm = x;
  __m128 mask = _mm_cmplt_ps(x, *(__m128*)_ps_cephes_SQRTHF);
  //printf("log_ps: mask=");print2i(mmask.mm[0]); print2i(mmask.mm[1]); printf("\n");

  __m128 tmp = _mm_and_ps(x, mask);
  x = _mm_sub_ps(x, one);
  e = _mm_sub_ps(e, _mm_and_ps(one, mask));
  x = _mm_add_ps(x, tmp);


  __m128 z = _mm_mul_ps(x,x);

  __m128 y = *(__m128*)_ps_cephes_log_p0;
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, *(__m128*)_ps_cephes_log_p1);
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, *(__m128*)_ps_cephes_log_p2);
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, *(__m128*)_ps_cephes_log_p3);
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, *(__m128*)_ps_cephes_log_p4);
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, *(__m128*)_ps_cephes_log_p5);
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, *(__m128*)_ps_cephes_log_p6);
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, *(__m128*)_ps_cephes_log_p7);
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, *(__m128*)_ps_cephes_log_p8);
  y = _mm_mul_ps(y, x);

  y = _mm_mul_ps(y, z);
  

  tmp = _mm_mul_ps(e, *(__m128*)_ps_cephes_log_q1);
  y = _mm_add_ps(y, tmp);


  tmp = _mm_mul_ps(z, *(__m128*)_ps_0p5);
  y = _mm_sub_ps(y, tmp);

  tmp = _mm_mul_ps(e, *(__m128*)_ps_cephes_log_q2);
  x = _mm_add_ps(x, y);
  x = _mm_add_ps(x, tmp);
  x = _mm_or_ps(x, invalid_mask); // negative arg will be NAN
  _mm_empty();
  return x;
}

__m128 exp_ps(__m128 x) {
  __m128 tmp = _mm_setzero_ps(), fx;
  __m64 mm0, mm1;
  __m128 one = *(__m128*)_ps_1;

  x = _mm_min_ps(x, *(__m128*)_ps_exp_hi);
  x = _mm_max_ps(x, *(__m128*)_ps_exp_lo);

  /* express exp(x) as exp(g + n*log(2)) */
  fx = _mm_mul_ps(x, *(__m128*)_ps_cephes_LOG2EF);
  fx = _mm_add_ps(fx, *(__m128*)_ps_0p5);

  /* how to perform a floorf with SSE: just below */
  /* step 1 : cast to int */
  tmp = _mm_movehl_ps(tmp, fx);
  mm0 = _mm_cvttps_pi32(fx);
  mm1 = _mm_cvttps_pi32(tmp);
  /* step 2 : cast back to float */
  tmp = _mm_cvtpi32x2_ps(mm0, mm1);
  /* if greater, substract 1 */
  __m128 mask = _mm_cmpgt_ps(tmp, fx);    
  mask = _mm_and_ps(mask, one);
  fx = _mm_sub_ps(tmp, mask);

  tmp = _mm_mul_ps(fx, *(__m128*)_ps_cephes_exp_C1);
  __m128 z = _mm_mul_ps(fx, *(__m128*)_ps_cephes_exp_C2);
  x = _mm_sub_ps(x, tmp);
  x = _mm_sub_ps(x, z);

  z = _mm_mul_ps(x,x);
  
  __m128 y = *(__m128*)_ps_cephes_exp_p0;
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, *(__m128*)_ps_cephes_exp_p1);
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, *(__m128*)_ps_cephes_exp_p2);
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, *(__m128*)_ps_cephes_exp_p3);
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, *(__m128*)_ps_cephes_exp_p4);
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, *(__m128*)_ps_cephes_exp_p5);
  y = _mm_mul_ps(y, z);
  y = _mm_add_ps(y, x);
  y = _mm_add_ps(y, one);

  /* build 2^n */
  z = _mm_movehl_ps(z, fx);
  mm0 = _mm_cvttps_pi32(fx);
  mm1 = _mm_cvttps_pi32(z);
  mm0 = _mm_add_pi32(mm0, *(__m64*)_pi32_0x7f);
  mm1 = _mm_add_pi32(mm1, *(__m64*)_pi32_0x7f);
  mm0 = _mm_slli_pi32(mm0, 23); 
  mm1 = _mm_slli_pi32(mm1, 23);
  
  __m128 pow2n; 
  COPY_MM_TO_XMM(mm0, mm1, pow2n);
  
  y = _mm_mul_ps(y, pow2n);
  _mm_empty();
  return y;
}

#endif
