# Additional clean files
cmake_minimum_required(VERSION 3.16)

if("${CONFIG}" STREQUAL "" OR "${CONFIG}" STREQUAL "Debug")
  file(REMOVE_RECURSE
  "_deps/nlopt-build/src/swig/CMakeFiles/nlopt_python.dir/nloptPYTHON_wrap.cxx"
  "_deps/nlopt-build/src/swig/nlopt.py"
  )
endif()
