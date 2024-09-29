cmake_minimum_required(VERSION 3.21)
project(nekrsTests NONE)
enable_testing()

set(EXAMPLES_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(LAUNCHER "nrsmpi" CACHE STRING "Path to the launcher executable")
set(NRANKS "0" CACHE STRING "Number of MPI ranks to run a test")
set(TESTS "")

if(NOT DEFINED ENV{NEKRS_HOME})
    message(FATAL_ERROR "NEKRS_HOME is not defined. Please set the environment variable.")
else()
    message(STATUS "NEKRS_HOME is set to: $ENV{NEKRS_HOME}")
endif()

if(NRANKS GREATER 0)
  message(STATUS "numer of MPI ranks to run tests: ${NRANKS}")
endif()

function(get_tests filter_string result_var)
  get_property(all_tests GLOBAL PROPERTY CTEST_TESTS)
  set(matching_tests "")
  foreach(test_name ${all_tests})
    string(FIND "${test_name}" "${filter_string}" found_index)
    if(found_index GREATER -1)
      list(APPEND matching_tests ${test_name})
    endif()
  endforeach()
  set(${result_var} "${matching_tests}" PARENT_SCOPE)
endfunction()

function(add_nekrs_test target num_processes par_file cimodes fp32)

set(working_directory "${EXAMPLES_DIR}/${target}")
  if (ARGC GREATER 5)
    set(working_directory "${ARGV5}")
  endif()

 foreach(cimode ${cimodes})
    set (name "${target}_${cimode}")
    list(APPEND TESTS ${test_name}) 
    message(STATUS "adding test ${name}")
 
    if(NOT TARGET ${target})
      add_custom_target(${target})
    endif()
 
    if(NRANKS GREATER 0)
      set(num_processes ${NRANKS})
    endif()
 
    if(fp32)
      set(fp32_value 1)
    else()
      set(fp32_value 0)
    endif()
 
    add_test(NAME ${name} WORKING_DIRECTORY ${working_directory}
             COMMAND ${LAUNCHER} ${par_file} ${num_processes} --cimode ${cimode}
             DEPENDS ${target})
    set_tests_properties(${name} PROPERTIES ENVIRONMENT "FP32=${fp32_value}")
  endforeach()
endfunction()

add_nekrs_test("ethier" 2 "ethier.par" "1;2;3;4;7;8;9;10;11;12;14;15;19" FALSE)
add_nekrs_test("ethier" 2 "mv_ethier.par" "5;6" FALSE)
add_nekrs_test("ethier" 2 "ethierScalar.par" "13" FALSE)
add_nekrs_test("ethier" 1 "ethier.par" "22" TRUE)

add_nekrs_test("lowMach" 2 "lowMach.par" "1" FALSE)

add_nekrs_test("mv_cyl" 2 "mv_cyl.par" "1;2;3;5;6" FALSE)

add_nekrs_test("mv_cyl_derived_bc" 2 "mv_cyl_derived_bc.par" "1;2;3;5;6" FALSE "${EXAMPLES_DIR}/mv_cyl")

add_nekrs_test("conj_ht" 2 "conj_ht.par" "1;2" FALSE)

add_nekrs_test("channel" 2 "channel.par" "1;2" FALSE)

add_nekrs_test("shlChannel" 2 "channel.par" "1;2" FALSE)

add_nekrs_test("ktauChannel" 2 "channel.par" "1;2" FALSE)

function(add_eddy_neknek)
  set(name "eddyNekNek") 

  if(NRANKS GREATER 0 AND NOT NRANKS EQUAL 2)
    message (STATUS "disabling test ${name}")
    return()
  endif()

  add_nekrs_test(${name} 2 "${name}.sess" "1;3;4" FALSE)
  add_nekrs_test(${name} 2 "${name}Mv.sess" "2" FALSE)
endfunction()

function(add_eddy_nekneknek)
  set(name "eddyNekNekNek") 
  set(working_directory "${EXAMPLES_DIR}/eddyNekNek")

  if(NRANKS GREATER 0 AND NOT NRANKS EQUAL 3)
    message (STATUS "disabling test ${name}")
    return()
  endif()

  add_nekrs_test(${name} 3 "${name}.sess" "1" FALSE ${working_directory})
  add_nekrs_test(${name} 3 "${name}Mv.sess" "2" FALSE ${working_directory})
endfunction()

add_eddy_neknek()
add_eddy_nekneknek()

foreach(test ${TESTS})
  set_tests_properties(${test} PROPERTIES EXCLUDE_FROM_ALL TRUE)
endforeach()
get_property(all_targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
add_custom_target(run_all_targets ALL DEPENDS ${all_targets})
