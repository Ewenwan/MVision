#Edit if using Windows!
FIND_PATH(LIBCONFIG_INCLUDE_DIR libconfig.h++ /usr/include /usr/local/include C:/Frameworks/libconfig/build/include)

SET(CMAKE_FIND_LIBRARY_PREFIXES "" "lib")
SET(CMAKE_FIND_LIBRARY_SUFFIXES ".lib" ".dll" ".so" ".a")
#Edit if using Windows!
FIND_LIBRARY(LIBCONFIG_LIBRARIES NAMES config++ PATHS /usr/lib /usr/local/lib C:/Frameworks/libconfig/build/lib)

IF (LIBCONFIG_INCLUDE_DIR AND LIBCONFIG_LIBRARIES)
    SET(CONFIG++_FOUND TRUE)
ENDIF ( LIBCONFIG_INCLUDE_DIR AND LIBCONFIG_LIBRARIES)

IF (CONFIG++_FOUND)
    IF (NOT CONFIG++_FIND_QUIETLY)
	MESSAGE(STATUS "Found Config++: ${LIBCONFIG_LIBRARIES}")
    ENDIF (NOT  CONFIG++_FIND_QUIETLY)
ELSE(CONFIG++_FOUND)
    IF (Config++_FIND_REQUIRED)
	IF(NOT LIBCONFIG_INCLUDE_DIR)
	    MESSAGE(FATAL_ERROR "Could not find LibConfig++ header file! Check path in file!")
	ENDIF(NOT LIBCONFIG_INCLUDE_DIR)

	IF(NOT LIBCONFIG_LIBRARIES)
	    MESSAGE(FATAL_ERROR "Could not find LibConfig++ library file! Check path in file!")
	ENDIF(NOT LIBCONFIG_LIBRARIES)
    ENDIF (Config++_FIND_REQUIRED)
ENDIF (CONFIG++_FOUND)
