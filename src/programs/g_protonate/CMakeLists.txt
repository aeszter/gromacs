include_directories(${CMAKE_SOURCE_DIR}/src/gromacs/gmxpreprocess)

file(GLOB PROTONATE_SOURCES g_protonate.c)

add_executable(g_protonate ${PROTONATE_SOURCES})
target_link_libraries(g_protonate ${GMX_EXTRA_LIBRARIES} libgromacs)
set_target_properties(g_protonate PROPERTIES OUTPUT_NAME "g_protonate${GMX_BINARY_SUFFIX}")
install(TARGETS g_protonate DESTINATION ${BIN_INSTALL_DIR} COMPONENT runtime)
