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

    set(LOAD_SCOPE_TEST_TARGET test-moe-load-scope)
    add_executable(
        ${LOAD_SCOPE_TEST_TARGET}
        ../../tests/test-moe-load-scope.cpp
        ../../src/llama-moe-placement.cpp)
    target_include_directories(${LOAD_SCOPE_TEST_TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/src)
    target_compile_features(${LOAD_SCOPE_TEST_TARGET} PRIVATE cxx_std_17)
    add_test(NAME ${LOAD_SCOPE_TEST_TARGET} COMMAND $<TARGET_FILE:${LOAD_SCOPE_TEST_TARGET}>)
    set_property(TEST ${LOAD_SCOPE_TEST_TARGET} PROPERTY LABELS main)

    set(POOL_PLAN_TEST_TARGET test-moe-pool-plan)
    add_executable(
        ${POOL_PLAN_TEST_TARGET}
        ../../tests/test-moe-pool-plan.cpp
        ../../src/llama-moe-placement.cpp
        ../../src/llama-moe-pool-plan.cpp)
    target_include_directories(${POOL_PLAN_TEST_TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/src)
    target_compile_features(${POOL_PLAN_TEST_TARGET} PRIVATE cxx_std_17)
    add_test(NAME ${POOL_PLAN_TEST_TARGET} COMMAND $<TARGET_FILE:${POOL_PLAN_TEST_TARGET}>)
    set_property(TEST ${POOL_PLAN_TEST_TARGET} PROPERTY LABELS main)
endif()
