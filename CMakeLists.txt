cmake_minimum_required(VERSION 3.11)
project(EDStudy)

find_package(PNG)
if(NOT PNG_FOUND)
  find_package(ZLIB) # zlib is needed by libPNG
  if(NOT ZLIB_FOUND)
        set(ZLIB_BUILD_EXAMPLES OFF)
        add_subdirectory(third_party/zlib-1.3.1)
        set(ZLIB_FOUND TRUE)
        set(ZLIB_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/zlib-1.3.1")
        set_target_properties(${ZLIB_LIBRARIES} PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${ZLIB_INCLUDE_DIRS}")
        add_library(ZLIB::ZLIB ALIAS "${ZLIB_LIBRARIES}")
  endif()

  set(PNG_SHARED OFF)
  set(PNG_TESTS OFF)
  set(PNG_TOOLS OFF)
  set(SKIP_INSTALL_ALL 1)
  add_subdirectory(third_party/libpng-1.6.53)
  set_target_properties(${PNG_LIBRARIES} PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${PNG_INCLUDE_DIRS}")
  add_library(PNG::PNG ALIAS "${PNG_LIBRARIES}")
endif()

set(CMAKE_CXX_STANDARD 11)

add_executable(edgeDrawing
    main.cpp
    ED-perso.cpp ED-perso.h
    Chain.cpp Chain.h
    image.cpp image.h
    io_png.c io_png.h
    cmdLine.h)
target_link_libraries(edgeDrawing PNG::PNG)

if(CMAKE_CXX_COMPILER_ID MATCHES "(GNU)|(CLANG)")
  set_target_properties(edgeDrawing PROPERTIES COMPILE_FLAGS "-Wall -Wextra")
endif()
