include_directories(${CMAKE_SOURCE_DIR}/src/gromacs/gmxpreprocess)

file(GLOB TPBCONV_SOURCES tpbconv.c)

add_executable(tpbconv ${TPBCONV_SOURCES})
target_link_libraries(tpbconv ${GMX_EXTRA_LIBRARIES} libgromacs)
set_target_properties(tpbconv PROPERTIES OUTPUT_NAME "tpbconv${GMX_BINARY_SUFFIX}")
install(TARGETS tpbconv DESTINATION ${BIN_INSTALL_DIR} COMPONENT runtime)
