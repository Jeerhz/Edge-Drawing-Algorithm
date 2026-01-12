# EDLines Study Project

This project compares two **EDLines implementations**:

- Custom version in the root folder (`edlines-implementation`)
- Original EDLib as a **git submodule** in `ED_Lib/`

We refactored the Edge Detector and Lines Detector scripts in Cihan Topal and al repository (https://github.com/CihanTopal/ED_Lib) for explanatory and clarity purposes.

---

## Structure

```
.
├── ED_Lib/         # Original EDLines library
├── images/         # Test images
├── results/        # Result of algorithms
├── CMakeLists.txt  # Build config
├── Makefile        # Init, build, run
├── ...             # New cpp project (.cpp and .h files)
├── test_ED.cpp     # Main test program
└── README.md
```

---

## Requirements

- **CMake ≥ 3.11**
- **g++** or **clang** with C++11
- **OpenCV**

---

## Setup

```bash
git clone https://github.com/<username>/EDLINES-IMPLEMENTATION.git
cd EDGE-DRAWING-ALGORITHM
make init
```

---

## Build

```bash
make build
```

- Builds custom project in `build/`
- Builds EDLib submodule in `ED_Lib/build/`

---

## Run

Run the original and new versions with:

```bash
make run IMAGE=<image_filename>
```

example:

```bash
make run IMAGE=billiard.jpg
```

Results are saved in `results/`.

---

## Clean

```bash
make clean     # Remove builds and results
make rebuild   # Clean and build again
```
