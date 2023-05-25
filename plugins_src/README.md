# Clang plugins and analyzers

## Manually building Clang plugins

Firstly，to check out the source code and build the project, follow steps 1-4 of the [Clang Getting Started page](https://clang.llvm.org/get_started.html).

The four plugins: ExtractFunctionPrototypes, FreeNullCheck, MemoryDataFlow and MemoryDataFree are used to 
    parse the source codes. 

Here are three steps to build them:    
- Place them in the directory of `clang/examples/`.
- Add the following in the file of `clang/examples/CMakeLists.txt`:

```buildoutcfg
add_subdirectory(ExtractFunctionPrototypes)
add_subdirectory(FreeNullCheck)
add_subdirectory(MemoryDataFlow)
add_subdirectory(MemoryDataFlowFree)
```
- Change into the build directory of clang, and run the build command:
- - `ninja ExtractFunctionPrototypes`
- - `ninja FreeNullCheck`
- - `ninja MemoryDataFlow`
- - `ninja MemoryDataFlowFree`


Finally, there are four corresponding .so files in the `lib` directory which in the build directory of clang. 
    
## Building Clang analyzers
The analyzer, GoshawkAnalyzer, is build upon the engine of CSA. Same to clang plugins, here are three steps to build it:

- Place them in the directory of `clang/lib/Analysis/plugins/`.
- Add the following in the file of `clang/lib/Analysis/plugins/CMakeLists.txt`:
```buildoutcfg
add_subdirectory(GoshawkAnalyzer)
```
- Change into the build directory of clang, and run the build command:
- - `ninja GoshawkAnalyzer`

Finally, the GoshawkAnalyzer.so will be generated in the `lib` directory which in the build directory of clang.

## Automatically building these plugins
Run the follow scripts will automatically build and replace these plugins.
`python3 auto_build.py clang_dir clang_build_dir`
