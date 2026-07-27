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
    target_include_directories(
        ${LOAD_SCOPE_TEST_TARGET}
        PRIVATE
            ${PROJECT_SOURCE_DIR}/src
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/ggml/include)
    target_compile_features(${LOAD_SCOPE_TEST_TARGET} PRIVATE cxx_std_17)
    add_test(NAME ${LOAD_SCOPE_TEST_TARGET} COMMAND $<TARGET_FILE:${LOAD_SCOPE_TEST_TARGET}>)
    set_property(TEST ${LOAD_SCOPE_TEST_TARGET} PROPERTY LABELS main)

    # A tiny DLL/import test catches missing LLAMA_API annotations on Windows
    # without compiling the complete llama shared library and llama-server.
    set(LOAD_SHARED_LIB_TARGET test-moe-placement-shared)
    add_library(
        ${LOAD_SHARED_LIB_TARGET}
        SHARED
        ../../src/llama-moe-placement.cpp)
    target_include_directories(
        ${LOAD_SHARED_LIB_TARGET}
        PRIVATE
            ${PROJECT_SOURCE_DIR}/src
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/ggml/include)
    target_compile_definitions(${LOAD_SHARED_LIB_TARGET} PRIVATE LLAMA_SHARED LLAMA_BUILD)
    target_compile_features(${LOAD_SHARED_LIB_TARGET} PRIVATE cxx_std_17)

    set(LOAD_SHARED_TEST_TARGET test-moe-load-shared)
    add_executable(${LOAD_SHARED_TEST_TARGET} ../../tests/test-moe-load-scope.cpp)
    target_include_directories(
        ${LOAD_SHARED_TEST_TARGET}
        PRIVATE
            ${PROJECT_SOURCE_DIR}/src
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/ggml/include)
    target_compile_definitions(
        ${LOAD_SHARED_TEST_TARGET}
        PRIVATE
            LLAMA_SHARED
            LLAMA_MOE_SHARED_API_TEST)
    target_link_libraries(${LOAD_SHARED_TEST_TARGET} PRIVATE ${LOAD_SHARED_LIB_TARGET})
    target_compile_features(${LOAD_SHARED_TEST_TARGET} PRIVATE cxx_std_17)
    add_test(NAME ${LOAD_SHARED_TEST_TARGET} COMMAND $<TARGET_FILE:${LOAD_SHARED_TEST_TARGET}>)
    set_property(TEST ${LOAD_SHARED_TEST_TARGET} PROPERTY LABELS main)

    set(POOL_PLAN_TEST_TARGET test-moe-pool-plan)
    add_executable(
        ${POOL_PLAN_TEST_TARGET}
        ../../tests/test-moe-pool-plan.cpp
        ../../src/llama-moe-placement.cpp
        ../../src/llama-moe-pool-layout.cpp
        ../../src/llama-moe-pool-plan.cpp)
    target_include_directories(
        ${POOL_PLAN_TEST_TARGET}
        PRIVATE
            ${PROJECT_SOURCE_DIR}/src
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/ggml/include)
    target_link_libraries(${POOL_PLAN_TEST_TARGET} PRIVATE ggml)
    target_compile_features(${POOL_PLAN_TEST_TARGET} PRIVATE cxx_std_17)
    add_test(NAME ${POOL_PLAN_TEST_TARGET} COMMAND $<TARGET_FILE:${POOL_PLAN_TEST_TARGET}>)
    set_property(TEST ${POOL_PLAN_TEST_TARGET} PROPERTY LABELS main)

    set(STORAGE_TEST_TARGET test-moe-storage)
    add_executable(
        ${STORAGE_TEST_TARGET}
        ../../tests/test-moe-storage.cpp
        ../../src/llama-moe-storage.cpp)
    target_include_directories(
        ${STORAGE_TEST_TARGET}
        PRIVATE
            ${PROJECT_SOURCE_DIR}/src
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/ggml/include)
    target_link_libraries(${STORAGE_TEST_TARGET} PRIVATE ggml)
    target_compile_features(${STORAGE_TEST_TARGET} PRIVATE cxx_std_17)
    add_test(NAME ${STORAGE_TEST_TARGET} COMMAND $<TARGET_FILE:${STORAGE_TEST_TARGET}>)
    set_property(TEST ${STORAGE_TEST_TARGET} PROPERTY LABELS main)

    set(REGISTRY_TEST_TARGET test-moe-registry)
    add_executable(
        ${REGISTRY_TEST_TARGET}
        ../../tests/test-moe-registry.cpp
        ../../src/llama-moe-placement.cpp
        ../../src/llama-moe-pool-layout.cpp
        ../../src/llama-moe-pool-plan.cpp
        ../../src/llama-moe-storage.cpp
        ../../src/llama-moe-registry-meta.cpp
        ../../src/llama-moe-registry.cpp)
    target_include_directories(
        ${REGISTRY_TEST_TARGET}
        PRIVATE
            ${PROJECT_SOURCE_DIR}/src
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/ggml/include)
    target_link_libraries(${REGISTRY_TEST_TARGET} PRIVATE ggml)
    target_compile_features(${REGISTRY_TEST_TARGET} PRIVATE cxx_std_17)
    add_test(NAME ${REGISTRY_TEST_TARGET} COMMAND $<TARGET_FILE:${REGISTRY_TEST_TARGET}>)
    set_property(TEST ${REGISTRY_TEST_TARGET} PROPERTY LABELS main)
endif()
