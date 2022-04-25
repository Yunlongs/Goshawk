# Clang plugins and analyzers

## Building Clang plugins
The four plugins: ExtractFunctionPrototypes, FreeNullCheck, MemoryDataFlow and MemoryDataFree are used to 
    parse the source codes. 

Here are three steps to build them:    
- Place them in the directory of `clang/examples/`.
- Add the following in the file of `clang/examples/CMakeLists.txt`:

```buildoutcfg
add_subdirectory(ExtractFunctionPrototypes)
add_subdirectory(FreeNullCheck)
add_subdirectory(MemoryDataFlow)
add_subdirectory(MemoryDataFree)
```
- Change into the build directory of clang, and run the build command.


Finally, there are four corresponding .so files in the `lib` directory which in the build directory of clang. 
    
## Building Clang analyzers
The analyzer, GoshawkAnalyzer, is build upon the engine of CSA. Same to clang plugins, here are three steps to build it:

- Place them in the directory of `clang/lib/Analysis/plugins/`.
- Add the following in the file of `clang/lib/Analysis/plugins/CMakeLists.txt`:
```buildoutcfg
add_subdirectory(GoshawkAnalyzer)
```
- Change into the build directory of clang, and run the build command.

Finally, the GoshawkAnalyzer.so will be generated in the `lib` directory which in the build directory of clang.
