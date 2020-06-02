# sets the rpath for all HeFFTe targets
macro(heffte_set_rpath Heffte_target)
    get_target_property(Heffte_list ${Heffte_target} LINK_LIBRARIES)
    foreach(_heffte_lib ${Heffte_list})
        get_filename_component(_heffte_libpath ${_heffte_lib} DIRECTORY)
        list(APPEND _heffte_rpath ${_heffte_libpath})
    endforeach()
    if (_heffte_rpath)
        list(REMOVE_DUPLICATES _heffte_rpath)
        set_target_properties(${Heffte_target} PROPERTIES INSTALL_RPATH "${_heffte_rpath}")
    endif()
    unset(Heffte_list)
    unset(_heffte_lib)
    unset(_heffte_rpath)
    unset(_heffte_libpath)
endmacro()

if (TARGET Heffte)
    heffte_set_rpath(Heffte)
endif()

if (TARGET heffte_gpu)
    heffte_set_rpath(heffte_gpu)
endif()
