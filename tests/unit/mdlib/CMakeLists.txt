set(mdTestSource mdUnitTest.cxx)

add_executable(mdUnitTests ${mdTestSource})
target_link_libraries(mdUnitTests gmxTestHarness mdlib)
add_google_tests(${EXECUTABLE_OUTPUT_PATH}/mdUnitTests
  ${mdTestSource})
