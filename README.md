# Sobel Filter + SIMD + OpenCV

Using SIMD instructions in image processing using OpenCV

![image](https://user-images.githubusercontent.com/1549028/216737449-f7c5dcb6-563e-45bd-98a7-9eb0e9e5575b.png)

## Build and Install OpenCV4

I've explained the build and installation steps in this gist: [Build OpenCV 4 in Ubuntu](https://gist.github.com/m3y54m/4f9b960e52f0da4e62e5a36f71d04fd7)

## Configure project

The recommended method for configuring OpenCV-based projects
is using CMake.

```console
sudo apt install cmake
```

If you are on Windows, download and execute CMake installer:

https://cmake.org/download/

The basic project configurations are written in
[`CMakeLists.txt`](https://github.com/opencv/opencv/blob/master/samples/cpp/example_cmake/CMakeLists.txt)
file which will be recognized by CMake tool. All source code of this project is in `src` directory.
In order to generate the Makefile and other files used to
build this project in a directory called `build` run these commands:

```console
cmake -S src -B build
```
## Build

```console
cmake --build build --config Release
```

## Run

Supposed that you are in `build` directory:

```console
./sobel_simd_opencv
```

## Resources

- [Intel® Intrinsics Guide ](https://software.intel.com/sites/landingpage/IntrinsicsGuide/)
- [Sobel operator](https://en.wikipedia.org/wiki/Sobel_operator)
- [Set the OpenCV environment variable and add it to the systems path (in Windows)](https://docs.opencv.org/4.x/d3/d52/tutorial_windows_install.html#tutorial_windows_install_path)
- [How do I check OS with a preprocessor directive?](https://stackoverflow.com/questions/142508/how-do-i-check-os-with-a-preprocessor-directive)
- [Header files for x86 SIMD intrinsics](https://stackoverflow.com/questions/11228855/header-files-for-x86-simd-intrinsics)
- [Measure execution time in C (on Windows)](https://stackoverflow.com/questions/15720542/measure-execution-time-in-c-on-windows)
- [Acquiring high-resolution time stamps](https://learn.microsoft.com/en-us/windows/win32/sysinfo/acquiring-high-resolution-time-stamps)
- [clock_gettime: identifier not found in Visual Studio in Windows 10](https://stackoverflow.com/questions/57668563/clock-gettime-identifier-not-found-in-visual-studio-in-windows-10)
- [CMAKE_BUILD_TYPE is not being used in CMakeLists.txt](https://stackoverflow.com/questions/24460486/cmake-build-type-is-not-being-used-in-cmakelists-txt)
- [Why aren't binaries placed in CMAKE_CURRENT_BINARY_DIR?](https://stackoverflow.com/questions/46371176/why-arent-binaries-placed-in-cmake-current-binary-dir)
- [SSE2 option in Visual C++ (x64)](https://stackoverflow.com/questions/1067630/sse2-option-in-visual-c-x64)