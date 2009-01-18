#ifndef _genborn_h
#define _genborn_h

#include "typedefs.h"
#include "grompp.h"


#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

/* If SSE intrinsics are available */
#if ((defined(__GNUC__) || defined(__INTEL_COMPILER) || defined(__PATHSCALE__) || defined(__PGIC__)) && \
(defined(__i386__) || defined(__x86_64)))
#include <xmmintrin.h>
#endif

#define M_PI3 M_PI/3
#define DOFFSET -0.09*0.1 /* length */

/* Tinker value */
//#define ELECTRIC 332.05382

/* Still parameters */
#define P1 0.073*0.1 /* length */
#define P2 0.921*0.1*CAL2JOULE    /* energy*length */
#define P3 6.211*0.1*CAL2JOULE    /* energy*length */
#define P4 15.236*0.1*CAL2JOULE
#define P5 1.254 

#define P5INV 1.0/P5
#define PIP5 M_PI*P5
#define DWATER 78.3

/* OBC (I) parameters */
//#define AOBC 0.8
//#define BOBC 0
//#define COBC 2.91 

/* OBC (II) parameters */
#define AOBC 1.00
#define BOBC 0.80
#define COBC 4.85

#define LOG_TABLE_ACCURACY 15 /* Accuracy of the table logarithm */


typedef struct
{
  int nbonds;
  int bond[10];
} bonds_t;

typedef struct
{
  real length[20000];
  real angle[20000];
} bl_t;

/* Struct to hold all the information for GB 
 * All these things are currently allocated in md.c
 */
typedef struct
{
  int nr;
  int n12;
  int n13;
  int n14;
  
  /* Atomic polarisation energies */
  real  *gpol;
  real  *gpol_gromacs;
  /* Atomic Born radii */
  real  *bRad;
  real  *bRad_gromacs;
  /* Atomic solvation volumes */
  real  *vsolv;
  real  *vsolv_gromacs;
  /* Array for inverse square root of the Born radii */
  real    *bRadInvSq;
  
  /* Array for vsites-exclusions */
  int *vs;
  int nvs;
  
  /* Solvation energy and derivatives */
  real es;
  real es_gromacs;
	
  real *drb;
  real *aes;
  rvec *des;

  /* Atomic surface area */
  real *asurf;
  real *asurf_gromacs;
  
  /* Surface area derivatives */
  rvec *dasurf;
  rvec *dasurf_gromacs;
 
   /* Total surface area */
  real as;
  real as_gromacs;
  
  /* Overlap factors for HCT method */
  real *shct;
  
  /* Parameters for OBC chain rule calculation */
  real *drobc;
  
  /* Used to precompute the factor raj*atype->shct for HCT/OBC */
  real *param;
  
  /* Table for logarithm lookup */
  real *log_table;
    
} 
gmx_genborn_t;


/* Initialise GB stuff */
int init_gb_still(t_commrec *cr, t_forcerec  *fr, t_atomtypes *atype, t_idef *idef, t_atoms *atoms, gmx_genborn_t *born, int natoms);
				 				 
int init_gb(t_commrec *cr, t_forcerec *fr, born_t *born, t_topology *top, rvec x[], int natoms, real rgbradii, int gb_algorithm);




/* Born radii calculations, both with and without SSE acceleration */
int calc_gb_rad(t_commrec *cr, t_inputrec *ir, int natoms, int nrfa, const t_iatom forceatoms[], const t_iparams forceparams[], const t_atoms *atoms,
				const t_atomtypes *atype, rvec x[], rvec f[], t_nblist *nl, gmx_genborn_t *born,t_mdatoms *md);
								
int calc_gb_rad_still(t_commrec *cr, int natoms, const t_iatom forceatoms[], const t_iparams forceparams[], const t_atoms *atoms,
								const t_atomtypes *atype, rvec x[], t_nblist *nl, gmx_genborn_t *born,t_mdatoms *md);
								
int calc_gb_rad_hct(t_commrec *cr, int natoms, const t_iatom forceatoms[], const t_iparams forceparams[], const t_atoms *atoms,
					const t_atomtypes *atype, rvec x[], t_nblist *nl, gmx_genborn_t *born,t_mdatoms *md);
					
int calc_gb_rad_obc(t_commrec *cr, int natoms, const t_iatom forceatoms[], const t_iparams forceparams[], const t_atoms *atoms,
					const t_atomtypes *atype, rvec x[], t_nblist *nl, gmx_genborn_t *born,t_mdatoms *md);	
					
int calc_gb_rad_still_sse(t_commrec *cr, int natoms, const t_atoms *atoms, const t_atomtypes *atype, real *x, t_nblist *nl, gmx_genborn_t *born, t_mdatoms *md );

int calc_gb_rad_hct_sse();

int calc_gb_rad_obc_sse(t_commrec *cr, int natoms, const t_atoms *atoms,
					const t_atomtypes *atype, real *x, t_nblist *nl, gmx_genborn_t *born,t_mdatoms *md);													




/* Bonded GB interactions */								
real gb_bonds_tab(int nbonds, real *x, real *f, real *charge, real *p_gbtabscale,
				  real *invsqrta, real *dvda, real *GBtab, const t_iatom forceatoms[],
				  real epsilon_r, real facel);





/* Functions for setting up the F_GB list in grompp */
int init_gb_plist(t_params *p_list);

int convert_gb_params(t_idef *idef, t_functype ftype, int start, t_params *gb_plist, gmx_genborn_t *born);

int generate_gb_topology(t_params *plist, t_idef *idef, t_atomtype *atype, t_atoms atoms, int natoms, 
						 t_params *gb_plist, gmx_genborn_t *born);
						 
static void assign_gb_param(t_functype ftype,t_iparams *new, real old[MAXFORCEPARAM],int comb,real reppow);

static void append_gb_interaction(t_ilist *ilist, int type,int nral,atom_id a[MAXATOMLIST]);

static int enter_gb_params(t_idef *idef, t_functype ftype, real forceparams[MAXFORCEPARAM],int comb,real reppow,
						   int start,bool bAppend);





/* Functions for calculating adjustments due to ie chain rule terms */
real calc_gb_forces(t_commrec *cr, t_mdatoms *md, gmx_genborn_t *born, const t_atoms *atoms, const t_atomtypes *atype, int nr, 
                    rvec x[], rvec f[], t_forcerec *fr,const t_iatom forceatoms[],int gb_algorithm, bool bRad);
													 												 
real calc_gb_nonpolar(t_commrec *cr, int natoms,gmx_genborn_t *born, const t_atoms atoms[], const t_atomtypes *atype, real *dvda,
					  int gb_algorithm, t_mdatoms *md);
													 
real calc_gb_selfcorrections(t_commrec *cr, int natoms, const t_atoms atoms[],const t_atomtypes *atype,gmx_genborn_t *born, real *dvda, t_mdatoms *md, double facel);														

real calc_gb_chainrule(int natoms, t_nblist *nl, rvec x[], rvec t[], real *dvda, real *dadx, int gb_algorithm, gmx_genborn_t *born);

real calc_gb_chainrule_sse(int natoms, t_nblist *nl, real *dadx, real *dvda, real *xd, real *f, int gb_algorithm, gmx_genborn_t *born);						

/* Surface areas functions */								
int calc_surfStill(t_inputrec *ir,
		   t_idef     *idef,
		   t_atoms    *atoms,
		   rvec       x[],
			 rvec       f[], 						 
		   gmx_genborn_t     *born,
		   t_atomtypes *atype,
		   double     *faction,
		   int        natoms,
       t_nblist   *nl,
       t_iparams  forceparams[],
       t_iatom    forceatoms[],
       int        nbonds);

int calc_surfBrooks(t_inputrec *ir,
		    t_idef     *idef,
		    t_atoms    *atoms,
		    rvec       x[],
		    gmx_genborn_t     *born,
		    t_atomtypes *atype,
		    double      *faction,
		    int natoms);

int calc_gmx_surface(t_inputrec *ir,
		     t_idef     *idef,
		     t_atoms    *atoms,
		     rvec       x[],
		     gmx_genborn_t     *born,
		     t_atomtypes *atype,
		     int        natoms);

int do_gb_neighborlist(t_forcerec *fr, int natoms, t_atoms *atoms, t_ilist *il, int nbonds, int n12n13);

int init_gb_nblist(int natoms, t_nblist *nl);

int gb_nblist_siev(t_commrec *cr, int natoms, int gb_algorithm, real gbcut, rvec x[], t_forcerec *fr, t_ilist *il, int n14);

int print_nblist(int natoms, t_nblist *nl);

void fill_log_table(const int n, real *table);

real table_log(float val, const real *table, const int n);


/* MPI routines for sending Born-radii stuff with particle decomposition */
void gb_pd_send(t_commrec *cr, real *send_data, int nr);



#if ((defined(__GNUC__) || defined(__INTEL_COMPILER) || defined(__PATHSCALE__) || defined(__PGIC__)) && \
(defined(__i386__) || defined(__x86_64)))

/* SIMD (SSE1+MMX indeed) implementation of sin, cos, exp and log

   Inspired by Intel Approximate Math library, and based on the
   corresponding algorithms of the cephes math library
*/

/* Copyright (C) 2007  Julien Pommier

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.

  (this is the zlib license)
*/

#ifdef _MSC_VER /* visual c++ */
# define ALIGN16_BEG __declspec(align(16))
# define ALIGN16_END 
#else /* gcc or icc */
# define ALIGN16_BEG
# define ALIGN16_END __attribute__((aligned(16)))
#endif

#define _PS_CONST(Name, Val)                                            \
  static const ALIGN16_BEG float _ps_##Name[4] ALIGN16_END = { Val, Val, Val, Val }
#define _PI32_CONST(Name, Val)                                            \
  static const ALIGN16_BEG int _pi32_##Name[4] ALIGN16_END = { Val, Val, Val, Val }
#define _PS_CONST_TYPE(Name, Type, Val)                                 \
  static const ALIGN16_BEG Type _ps_##Name[4] ALIGN16_END = { Val, Val, Val, Val }


_PS_CONST(1  , 1.0f);
_PS_CONST(0p5, 0.5f);
/* the smallest non denormalized float number */
_PS_CONST_TYPE(min_norm_pos, int, 0x00800000);
_PS_CONST_TYPE(mant_mask, int, 0x7f800000);
_PS_CONST_TYPE(inv_mant_mask, int, ~0x7f800000);

_PS_CONST_TYPE(sign_mask, int, 0x80000000);
_PS_CONST_TYPE(inv_sign_mask, int, ~0x80000000);

_PI32_CONST(1, 1);
_PI32_CONST(inv1, ~1);
_PI32_CONST(2, 2);
_PI32_CONST(4, 4);
_PI32_CONST(0x7f, 0x7f);

_PS_CONST(cephes_SQRTHF, 0.707106781186547524);
_PS_CONST(cephes_log_p0, 7.0376836292E-2);
_PS_CONST(cephes_log_p1, - 1.1514610310E-1);
_PS_CONST(cephes_log_p2, 1.1676998740E-1);
_PS_CONST(cephes_log_p3, - 1.2420140846E-1);
_PS_CONST(cephes_log_p4, + 1.4249322787E-1);
_PS_CONST(cephes_log_p5, - 1.6668057665E-1);
_PS_CONST(cephes_log_p6, + 2.0000714765E-1);
_PS_CONST(cephes_log_p7, - 2.4999993993E-1);
_PS_CONST(cephes_log_p8, + 3.3333331174E-1);
_PS_CONST(cephes_log_q1, -2.12194440e-4);
_PS_CONST(cephes_log_q2, 0.693359375);

_PS_CONST(minus_cephes_DP1, -0.78515625);
_PS_CONST(minus_cephes_DP2, -2.4187564849853515625e-4);
_PS_CONST(minus_cephes_DP3, -3.77489497744594108e-8);
_PS_CONST(sincof_p0, -1.9515295891E-4);
_PS_CONST(sincof_p1,  8.3321608736E-3);
_PS_CONST(sincof_p2, -1.6666654611E-1);
_PS_CONST(coscof_p0,  2.443315711809948E-005);
_PS_CONST(coscof_p1, -1.388731625493765E-003);
_PS_CONST(coscof_p2,  4.166664568298827E-002);
_PS_CONST(cephes_FOPI, 1.27323954473516); // 4 / M_PI

_PS_CONST(exp_hi,	88.3762626647949f);
_PS_CONST(exp_lo,	-88.3762626647949f);

_PS_CONST(cephes_LOG2EF, 1.44269504088896341);
_PS_CONST(cephes_exp_C1, 0.693359375);
_PS_CONST(cephes_exp_C2, -2.12194440e-4);

_PS_CONST(cephes_exp_p0, 1.9875691500E-4);
_PS_CONST(cephes_exp_p1, 1.3981999507E-3);
_PS_CONST(cephes_exp_p2, 8.3334519073E-3);
_PS_CONST(cephes_exp_p3, 4.1665795894E-2);
_PS_CONST(cephes_exp_p4, 1.6666665459E-1);
_PS_CONST(cephes_exp_p5, 5.0000001201E-1);

typedef union xmm_mm_union {
  __m128 xmm;
  __m64 mm[2];
} xmm_mm_union;


#define COPY_XMM_TO_MM(xmm_, mm0_, mm1_) {          \
    xmm_mm_union u; u.xmm = xmm_;                   \
    mm0_ = u.mm[0];                                 \
    mm1_ = u.mm[1];                                 \
}

#define COPY_MM_TO_XMM(mm0_, mm1_, xmm_) {                         \
    xmm_mm_union u; u.mm[0]=mm0_; u.mm[1]=mm1_; xmm_ = u.xmm;      \
  }

__m128 log_ps(__m128 x);

void sincos_ps(__m128 x, __m128 *s, __m128 *c);

__m128 exp_ps(__m128 x);

#endif


#endif /* _genborn_h */
