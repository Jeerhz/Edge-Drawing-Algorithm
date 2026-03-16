# edgeDrawing (An implementation of the ED algorithm)

Version 0.9, 03/16/2026.

Future versions: <https://github.com/pmonasse/Edge-Drawing>

The ED algorithm[^1] is a vectorized edge detector. It extends the Canny edge detector by replacing the hysteresis thresholding with edge tracking along level lines. 

## Build
*Requirements:*

  - CMake >. 3,11 <https://cmake.org/download/>
  - C++ compiler
  
*Build instructions:*
- Unix, MacOS:
  ```
  $ cd /path_to_this_file/
  $ cmake -DCMAKE_BUILD_TYPE:bool=Release -S . -B Build
  $ cmake --build Build
  ```
- Windows with MinGW:
  ```
  $ cd /path_to_this_file/
  $ cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE:bool=Release S . -B Build
  $ cmake --build Build
  ```

## Usage
```
Build/edgeDrawing [options] in.png out.png
-g, --grad-min=ARG Min gradient (6)
-a, --angchor-gap=ARG Min gap of gradient for anchor (2)
-l, --length-min=ARG Min length of edge segment (10)
-s, --sigma=ARG Sigma of Gaussian blur (1)
-e, --epsNFA=ARG 1 or lower, <=0 means no NFA validation (0)
```

Typical settings for a contrario validation of edges:

- a smaller value of -g could be used (2).
- a smaller value of -a could be used (0).
- the NFA treshold -e could be 1.

Example:
```
Build/edgeDrawing data/shapes.png shapes_out.png
```
Compare `shapes_out.png` and reference `data/shapes_out_png`.

[^1] Original implementation by the authors of ED, Cihan Topal and Cuneyt Akinlar (https://github.com/CihanTopal/ED_Lib)
