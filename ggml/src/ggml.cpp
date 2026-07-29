#include "ggml-impl.h"

#include <cstdlib>
#include <exception>

static std::terminate_handler previous_terminate_handler;

GGML_NORETURN static void ggml_uncaught_exception() {
    ggml_print_backtrace();
    if (previous_terminate_handler) {
        previous_terminate_handler();
    }
    abort(); // unreachable unless previous_terminate_handler was nullptr
}

static bool ggml_uncaught_exception_init = []{
    const char * GGML_NO_BACKTRACE = getenv("GGML_NO_BACKTRACE");
    if (GGML_NO_BACKTRACE) {
        return false;
    }
    const auto prev{std::get_terminate()};
    // Static Windows builds may link this initialization through more than
    // one dependency path. Re-registering the same handler must be a no-op;
    // assigning it as its own predecessor would recurse during termination.
    if (prev == ggml_uncaught_exception) {
        return true;
    }
    previous_terminate_handler = prev;
    std::set_terminate(ggml_uncaught_exception);
    return true;
}();
