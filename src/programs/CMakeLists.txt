add_subdirectory(mdrun)
if (NOT GMX_FAHCORE)
    add_subdirectory(g_ana)
    add_subdirectory(g_protonate)
    add_subdirectory(g_x2top)
    add_subdirectory(gmxcheck)
    add_subdirectory(gmxdump)
    add_subdirectory(grompp)
    add_subdirectory(pdb2gmx)
    add_subdirectory(tpbconv)
endif (NOT GMX_FAHCORE)
