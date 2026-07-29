set(TARGET llama-moe-plan-check)

add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE llama-common)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()

# The U2 loader intentionally inspects internal model-owned compact storages.
# Keep it in static builds so no internal C++ ABI has to become part of the
# shared llama library contract.
if(NOT BUILD_SHARED_LIBS)
    set(U2_LOAD_TARGET llama-moe-load-check)
    add_executable(${U2_LOAD_TARGET} u2-main.cpp)
    target_include_directories(
        ${U2_LOAD_TARGET}
        PRIVATE
            ${PROJECT_SOURCE_DIR}/src
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/ggml/include)
    target_link_libraries(${U2_LOAD_TARGET} PRIVATE llama-common)
    target_compile_features(${U2_LOAD_TARGET} PRIVATE cxx_std_17)

    if(LLAMA_TOOLS_INSTALL)
        install(TARGETS ${U2_LOAD_TARGET} RUNTIME)
    endif()
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

    set(ROUTE_PARTITION_TEST_TARGET test-moe-route-partition)
    add_executable(
        ${ROUTE_PARTITION_TEST_TARGET}
        ../../tests/test-moe-route-partition.cpp
        ../../src/llama-moe-route-partition.cpp)
    target_include_directories(
        ${ROUTE_PARTITION_TEST_TARGET}
        PRIVATE
            ${PROJECT_SOURCE_DIR}/src
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/ggml/include)
    target_compile_features(${ROUTE_PARTITION_TEST_TARGET} PRIVATE cxx_std_17)
    add_test(NAME ${ROUTE_PARTITION_TEST_TARGET} COMMAND $<TARGET_FILE:${ROUTE_PARTITION_TEST_TARGET}>)
    set_property(TEST ${ROUTE_PARTITION_TEST_TARGET} PROPERTY LABELS main)

    set(ROUTE_REMAP_TEST_TARGET test-moe-route-remap)
    add_executable(
        ${ROUTE_REMAP_TEST_TARGET}
        ../../tests/test-moe-route-remap.cpp)
    target_include_directories(
        ${ROUTE_REMAP_TEST_TARGET}
        PRIVATE
            ${PROJECT_SOURCE_DIR}/ggml/include)
    target_link_libraries(${ROUTE_REMAP_TEST_TARGET} PRIVATE ggml)
    target_compile_features(${ROUTE_REMAP_TEST_TARGET} PRIVATE cxx_std_17)
    add_test(NAME ${ROUTE_REMAP_TEST_TARGET} COMMAND $<TARGET_FILE:${ROUTE_REMAP_TEST_TARGET}>)
    set_property(TEST ${ROUTE_REMAP_TEST_TARGET} PROPERTY LABELS main)
endif()
