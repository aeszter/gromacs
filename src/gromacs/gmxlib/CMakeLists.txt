include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# The nonbonded directory contains subdirectories that are only
# conditionally built, so we cannot use a GLOB_RECURSE here.
file(GLOB GMXLIB_SOURCES *.c 
     statistics/*.c nonbonded/*.c nonbonded/nb_kernel_c/*.c)

if(GMX_DOUBLE)
  set(SSETYPE sse2)
else()
  set(SSETYPE sse)
endif()

if(GMX_IA32_ASM)
  file(GLOB GMX_SSEKERNEL_C_SRC   nonbonded/nb_kernel_ia32_${SSETYPE}/*.c)
  if(GMX_ASM_USEASM-NASM)
    file(GLOB GMX_SSEKERNEL_ASM_SRC nonbonded/nb_kernel_ia32_${SSETYPE}/*intel_syntax*.s)    
  else()
    file(GLOB GMX_SSEKERNEL_ASM_SRC nonbonded/nb_kernel_ia32_${SSETYPE}/*${SSETYPE}.s nonbonded/nb_kernel_ia32_${SSETYPE}/*asm.s)
  endif()
endif(GMX_IA32_ASM)

if(GMX_X86_64_ASM)
  file(GLOB GMX_SSEKERNEL_C_SRC   nonbonded/nb_kernel_x86_64_${SSETYPE}/*.c)
  if(GMX_ASM_USEASM-NASM)
    file(GLOB GMX_SSEKERNEL_ASM_SRC nonbonded/nb_kernel_x86_64_${SSETYPE}/*intel_syntax*.s)
  else()
    file(GLOB GMX_SSEKERNEL_ASM_SRC nonbonded/nb_kernel_x86_64_${SSETYPE}/*${SSETYPE}.s nonbonded/nb_kernel_x86_64_${SSETYPE}/*asm.s)
  endif()
endif(GMX_X86_64_ASM)

if(GMX_FORTRAN)
  if (GMX_DOUBLE)
    file(GLOB FORTRAN_SOURCES nonbonded/nb_kernel_f77_double/*.[cf])
  else(GMX_DOUBLE)
    file(GLOB FORTRAN_SOURCES nonbonded/nb_kernel_f77_single/*.[cf])
  endif(GMX_DOUBLE)
endif(GMX_FORTRAN)

if(GMX_POWER6)
  file(GLOB FORTRAN_SOURCES nonbonded/nb_kernel_power6/*.[cF])
endif(GMX_POWER6)

if(GMX_BLUEGENE)
  file(GLOB GMX_BLUEGENE_C_SRC nonbonded/nb_kernel_bluegene/*.c)
endif(GMX_BLUEGENE)

if(GMX_PPC_ALTIVEC)
  file(GLOB GMX_PPC_ALTIVEC_SRC nonbonded/nb_kernel_ppc_altivec/*.c)
endif(GMX_PPC_ALTIVEC)

if(NOT GMX_EXTERNAL_BLAS)
  file(GLOB BLAS_SOURCES gmx_blas/*.c)
endif(NOT GMX_EXTERNAL_BLAS)

if(NOT GMX_EXTERNAL_LAPACK)
  file(GLOB LAPACK_SOURCES gmx_lapack/*.c)
endif(NOT GMX_EXTERNAL_LAPACK)

# This would be the standard way to include thread_mpi, but we want libgmx
# to link the functions directly
#if(GMX_THREADS)
#    add_subdirectory(thread_mpi)
#endif(GMX_THREADS)
#target_link_libraries(gmx ${GMX_EXTRA_LIBRARIES} ${THREAD_MPI_LIB})

# Files called xxx_test.c are test drivers with a main() function for module xxx.c,
# so they should not be included in the library
file(GLOB_RECURSE NOT_GMXLIB_SOURCES *_test.c *\#*)
list(REMOVE_ITEM GMXLIB_SOURCES ${NOT_GMXLIB_SOURCES})  

# An ugly hack to get absolute paths...
file(GLOB THREAD_MPI_SOURCES ${THREAD_MPI_SRC})

set(GMX_SSEKERNEL_ASM_SRC ${GMX_SSEKERNEL_ASM_SRC} PARENT_SCOPE)
set(GMXLIB_SOURCES ${GMXLIB_SOURCES} ${BLAS_SOURCES} ${LAPACK_SOURCES}
    ${GMX_SSEKERNEL_C_SRC} ${FORTRAN_SOURCES}
    ${GMX_BLUEGENE_C_SRC} ${GMX_PPC_ALTIVEC_SRC} ${THREAD_MPI_SOURCES}
    PARENT_SCOPE)
