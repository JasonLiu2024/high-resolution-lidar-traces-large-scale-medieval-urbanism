# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/Users/kronos.di.vlad/Desktop/SEA_folder/cmake-build-debug/_deps/cli11-src"
  "/Users/kronos.di.vlad/Desktop/SEA_folder/cmake-build-debug/_deps/cli11-build"
  "/Users/kronos.di.vlad/Desktop/SEA_folder/cmake-build-debug/_deps/cli11-subbuild/cli11-populate-prefix"
  "/Users/kronos.di.vlad/Desktop/SEA_folder/cmake-build-debug/_deps/cli11-subbuild/cli11-populate-prefix/tmp"
  "/Users/kronos.di.vlad/Desktop/SEA_folder/cmake-build-debug/_deps/cli11-subbuild/cli11-populate-prefix/src/cli11-populate-stamp"
  "/Users/kronos.di.vlad/Desktop/SEA_folder/cmake-build-debug/_deps/cli11-subbuild/cli11-populate-prefix/src"
  "/Users/kronos.di.vlad/Desktop/SEA_folder/cmake-build-debug/_deps/cli11-subbuild/cli11-populate-prefix/src/cli11-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/Users/kronos.di.vlad/Desktop/SEA_folder/cmake-build-debug/_deps/cli11-subbuild/cli11-populate-prefix/src/cli11-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/Users/kronos.di.vlad/Desktop/SEA_folder/cmake-build-debug/_deps/cli11-subbuild/cli11-populate-prefix/src/cli11-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
