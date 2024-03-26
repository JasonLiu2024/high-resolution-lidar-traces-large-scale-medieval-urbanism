# Install script for directory: /Users/kronos.di.vlad/Desktop/SEA_folder/cmake-build-debug/_deps/nlopt-src

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/Users/kronos.di.vlad/Desktop/SEA_folder/cmake-build-debug/_deps/nlopt-build/nlopt.pc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES
    "/Users/kronos.di.vlad/Desktop/SEA_folder/cmake-build-debug/_deps/nlopt-src/src/api/nlopt.h"
    "/Users/kronos.di.vlad/Desktop/SEA_folder/cmake-build-debug/_deps/nlopt-build/nlopt.hpp"
    "/Users/kronos.di.vlad/Desktop/SEA_folder/cmake-build-debug/_deps/nlopt-build/nlopt.f"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/Users/kronos.di.vlad/Desktop/SEA_folder/cmake-build-debug/_deps/nlopt-build/libnlopt.a")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libnlopt.a" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libnlopt.a")
    execute_process(COMMAND "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/ranlib" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libnlopt.a")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Development" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/nlopt/NLoptLibraryDepends.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/nlopt/NLoptLibraryDepends.cmake"
         "/Users/kronos.di.vlad/Desktop/SEA_folder/cmake-build-debug/_deps/nlopt-build/CMakeFiles/Export/f1922ff44cc4dc88f95ca1075600a5b3/NLoptLibraryDepends.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/nlopt/NLoptLibraryDepends-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/nlopt/NLoptLibraryDepends.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/nlopt" TYPE FILE FILES "/Users/kronos.di.vlad/Desktop/SEA_folder/cmake-build-debug/_deps/nlopt-build/CMakeFiles/Export/f1922ff44cc4dc88f95ca1075600a5b3/NLoptLibraryDepends.cmake")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/nlopt" TYPE FILE FILES "/Users/kronos.di.vlad/Desktop/SEA_folder/cmake-build-debug/_deps/nlopt-build/CMakeFiles/Export/f1922ff44cc4dc88f95ca1075600a5b3/NLoptLibraryDepends-debug.cmake")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Development" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/nlopt" TYPE FILE FILES
    "/Users/kronos.di.vlad/Desktop/SEA_folder/cmake-build-debug/_deps/nlopt-build/NLoptConfig.cmake"
    "/Users/kronos.di.vlad/Desktop/SEA_folder/cmake-build-debug/_deps/nlopt-build/NLoptConfigVersion.cmake"
    )
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/Users/kronos.di.vlad/Desktop/SEA_folder/cmake-build-debug/_deps/nlopt-build/src/api/cmake_install.cmake")
  include("/Users/kronos.di.vlad/Desktop/SEA_folder/cmake-build-debug/_deps/nlopt-build/src/swig/cmake_install.cmake")

endif()
