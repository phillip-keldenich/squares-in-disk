option(IVARP_NO_CUDA "Build without CUDA even if CUDA is found and supported." OFF)

# only set the standard etc. if we are building ivarp + tests, not
# if we are just using it.
if(IVARP_BUILDING_IVARP)
	# unfortunately, more is not supported by CUDA/nvcc.
	set(CMAKE_CXX_STANDARD 14)
	set(CMAKE_CXX_EXTENSIONS Off)

	# compile with coverage information
	option(IVARP_ENABLE_COVERAGE "Compile and link with coverage information generation for debugging/testing." Off)
endif()

if("${CMAKE_CXX_COMPILER_ID}" MATCHES "[Gg][Nn][Uu]")
	if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 10.0)
		string(REPLACE "-O3" "-O2" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
	endif()
endif()

# include cmake utilities; cuda_support must be first for make_flags_cuda_compatible.
include("${CMAKE_CURRENT_LIST_DIR}/cmake/cuda_support.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/cmake/workarounds.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/cmake/imported_target_utils.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/cmake/enable_warnings.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/cmake/rounding_flags.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/cmake/debug_use_asan.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/cmake/enable_coverage.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/cmake/enable_lto.cmake")

# add find package scripts to the cmake module path and find packages
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake;${CMAKE_MODULE_PATH}")
find_package(Boost REQUIRED)
find_package(GMPXX REQUIRED)
find_package(Threads REQUIRED)

add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/src")
add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/include")

add_library(ivarp SHARED)
set_target_properties(ivarp PROPERTIES
	C_VISIBILITY_PRESET hidden
	CXX_VISIBILITY_PRESET hidden
	VISIBILITY_INLINES_HIDDEN On
)
target_include_directories(ivarp PUBLIC "${CMAKE_CURRENT_LIST_DIR}/include")
target_link_libraries(ivarp
		PUBLIC util::enable_rounding util::use_sse
		       Boost::boost ivarp::GMPXX ivarp::MPFR Threads::Threads
		       ivarp::cuda_support
		PRIVATE util::enable_warnings util::debug_use_asan  __ivarp_sources __ivarp_headers __ivarp_workarounds
)
target_enable_lto(ivarp)

if(IVARP_BUILDING_IVARP)
	add_library(doctest INTERFACE)
	target_include_directories(doctest INTERFACE "${CMAKE_CURRENT_LIST_DIR}/include")
	target_sources(doctest INTERFACE "${CMAKE_CURRENT_LIST_DIR}/include/doctest/doctest.h"
	                                 "${CMAKE_CURRENT_LIST_DIR}/include/doctest/doctest_fixed.hpp")

	if(IVARP_ENABLE_COVERAGE)
		target_link_libraries(ivarp PUBLIC util::enable_coverage)
	endif()

	add_subdirectory(test)
endif()

