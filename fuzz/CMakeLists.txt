# libFuzzer harnesses over the untrusted-input surfaces (M7):
# wav reader, image loader, meta.json parser, CLI arg path.
# Weights (GGUF) are TRUSTED (privacy-filter threat model).
#
#   cmake --preset fuzz && cmake --build --preset fuzz
#   ./build/fuzz/fuzz/fuzz_wav corpus/wav -max_len=1048576
foreach(target fuzz_wav fuzz_image fuzz_json)
    add_executable(${target} ${target}.cpp)
    target_link_libraries(${target} PRIVATE tp)
    target_include_directories(${target} PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_compile_options(${target} PRIVATE -fsanitize=fuzzer)
    target_link_options(${target} PRIVATE -fsanitize=fuzzer)
endforeach()
