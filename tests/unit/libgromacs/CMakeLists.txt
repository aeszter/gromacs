set(gmxTestSource gmxUnitTest.cxx)

add_executable(gmxUnitTests ${gmxTestSource})
target_link_libraries(gmxUnitTests gmxTestHarness libgromacs)
add_google_tests(${EXECUTABLE_OUTPUT_PATH}/gmxUnitTests
  ${gmxTestSource})
