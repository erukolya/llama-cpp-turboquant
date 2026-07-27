set(TARGET llama-moe-plan-check)

add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE llama-common)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()

if(BUILD_TESTING)
    set(TEST_TARGET test-moe-expert-plan)
    add_executable(${TEST_TARGET} ../../tests/test-moe-expert-plan.cpp)
    target_link_libraries(${TEST_TARGET} PRIVATE llama-common)
    target_compile_features(${TEST_TARGET} PRIVATE cxx_std_17)
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
    set_property(TEST ${TEST_TARGET} PROPERTY LABELS main)
endif()
