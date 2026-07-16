# ctest suite. Asset-dependent tests read env:
#   TP_GGUF_DIR (default: <harness>/ggufs)  TP_REF_DUMPS (harness dumps/)
# and SKIP (exit 77) when assets are missing.
function(tp_add_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE tp)
    target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/src)
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
endfunction()

tp_add_test(test_json)
tp_add_test(test_wav)
tp_add_test(test_gguf)
tp_add_test(test_ops)
tp_add_test(test_parity)
set_tests_properties(test_parity PROPERTIES LABELS "model")
