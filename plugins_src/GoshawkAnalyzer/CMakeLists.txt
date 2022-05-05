set(LLVM_LINK_COMPONENTS
  Support
  )

set(LLVM_EXPORTED_SYMBOL_FILE ${CMAKE_CURRENT_SOURCE_DIR}/GoshawkChecker.exports)
add_llvm_library(GoshawkAnalyzer MODULE BUILDTREE_ONLY GoshawkChecker.cpp PLUGIN_TOOL clang)

clang_target_link_libraries(GoshawkAnalyzer PRIVATE
  clangAnalysis
  clangAST
  clangStaticAnalyzerCore
  clangStaticAnalyzerFrontend
  )
