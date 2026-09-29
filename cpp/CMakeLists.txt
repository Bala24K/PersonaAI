cmake_minimum_required(VERSION 3.10)
project(PersonaCppAnalyzer VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(persona_analyzer_lib STATIC
    src/analyzer.cpp
)

add_executable(persona_cpp_analyzer
    src/main.cpp
)
target_link_libraries(persona_cpp_analyzer PRIVATE persona_analyzer_lib)

add_executable(test_analyzer
    src/test_analyzer.cpp
)
target_link_libraries(test_analyzer PRIVATE persona_analyzer_lib)
