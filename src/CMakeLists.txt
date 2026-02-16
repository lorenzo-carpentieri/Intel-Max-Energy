# Find all .cpp files in this folder
file(GLOB SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

# Loop over each .cpp file and create an executable
foreach(SRC_FILE ${SRC_FILES})
    # Extract the filename without path and extension
    get_filename_component(EXE_NAME ${SRC_FILE} NAME_WE)

    # Create an executable with the same name
    add_executable(${EXE_NAME} ${SRC_FILE})

    # Include the top-level include directory
    target_include_directories(${EXE_NAME} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
    )
    target_link_libraries(${EXE_NAME} PRIVATE synergy)

    # Set C++17
    target_compile_features(${EXE_NAME} PRIVATE cxx_std_17)
   install(TARGETS ${EXE_NAME}
            RUNTIME DESTINATION bin)
endforeach()

