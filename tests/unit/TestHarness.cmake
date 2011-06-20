# Build Google Test
find_package(Threads)
if (CMAKE_USE_PTHREADS_INIT)  # The pthreads library is available.
  set(cxx_base_flags "${cxx_base_flags} -DGTEST_HAS_PTHREAD=1")
endif()

set(TEST_HARNESS_TEMP_DIRECTORY ${gmxTesting_BINARY_DIR}/Temporary)
set(TEST_HARNESS_DATA_DIRECTORY ${GMX_DATA_ROOT})

# Build Google Testing
set ( HarnessSource
  harness/googletest/src/gtest-all.cc
  harness/gmxTestHarnessMain.cxx
)
include_directories ( harness harness/googletest harness/googletest/include )

add_library(gmxTestHarness ${HarnessSource})
link_libraries(gmxTestHarness)
if (CMAKE_USE_PTHREADS_INIT)
  target_link_libraries(gmxTestHarness ${CMAKE_THREAD_LIBS_INIT})
endif()


# Add all the tests by parsing the source code
# This macro searches for GoogleTest macros and adds them as test automatically
macro(ADD_GOOGLE_TESTS executable)
  foreach ( source ${ARGN} )
    file(READ "${source}" contents)

    # Find all test and long test lists
    string(REGEX MATCHALL "TEST_?F?\\(([A-Za-z_0-9 ,]+)\\) /\\* Long \\*/" LongTests ${contents})
    string(REGEX MATCHALL "TEST_?F?\\(([A-Za-z_0-9 ,]+)\\)" AllTests ${contents})

    # Convert the C++ code into a short test name
    set ( AllTestsHits "" )
    foreach(hit ${AllTests})
      string(REGEX REPLACE ".*\\(([A-Za-z_0-9]+)[, ]*([A-Za-z_0-9]+)\\).*" "\\1.\\2" test_name ${hit})
      set ( AllTestsHits ${AllTestsHits} ${test_name} )
    endforeach()
    set ( LongTestsHits "" )
    foreach(hit ${LongTests})
      string(REGEX REPLACE ".*\\(([A-Za-z_0-9]+)[, ]*([A-Za-z_0-9]+)\\).*" "\\1.\\2" test_name ${hit})
      set ( LongTestsHits ${LongTestsHits} ${test_name} )
    endforeach()

    list ( SORT AllTestsHits )
    foreach(hit ${AllTestsHits})
      add_test(${hit} ${executable} --gtest_filter=${hit} ${TEST_HARNESS_DATA_DIRECTORY} ${TEST_HARNESS_TEMP_DIRECTORY})
    endforeach(hit)
  endforeach()
endmacro()
