# Сборка нативного модуля покерной математики (C++/nanobind).
# Вызывается через scikit-build-core при `uv sync` / `uv build`.
cmake_minimum_required(VERSION 3.15...3.30)
project(poker_equity LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)  # эквити — горячий цикл, нужен -O
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # compile_commands.json для IntelliSense

find_package(Python 3.13 COMPONENTS Interpreter Development.Module REQUIRED)

# nanobind поставлен как build-зависимость — находим его CMake-конфиг через сам пакет.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_equity native/equity.cpp)
install(TARGETS _equity LIBRARY DESTINATION poker_analyzer)

# Зеркалим compile_commands.json в корень репо, чтобы IntelliSense VS Code знал пути
# к заголовкам nanobind/Python (build-dir с тегом колеса — нестабильный путь). В .gitignore.
add_custom_target(
  mirror_compile_commands ALL
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_BINARY_DIR}/compile_commands.json"
          "${CMAKE_SOURCE_DIR}/compile_commands.json"
  VERBATIM)
