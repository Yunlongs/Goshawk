add_llvm_library(MemoryDataFlow MODULE MemoryDataFlow.cpp PLUGIN_TOOL clang)

if(LLVM_ENABLE_PLUGINS AND (WIN32 OR CYGWIN))
  set(LLVM_LINK_COMPONENTS
    Support
  )
  clang_target_link_libraries(MemoryDataFlow PRIVATE
    clangAST
    clangBasic
    clangFrontend
    )
endif()
