# ....................
# @license: %MIT License%:~ http://www.opensource.org/licenses/MIT
# @project: src
# @file: /build.sh
# @created: Wednesday, 8th April 2020
# @author: brentian (chuwzhang@126.com)
# @modified: brentian (chuwzhang@126.com>)
#    Wednesday, 8th April 2020 8:34:53 pm
# ....................
# @description:
JULIA_PATH=/Applications/Julia-1.4.app/Contents/Resources/julia
# create trace compile
julia --startup-file=no --trace-compile=app_precompile.jl main.jl

# create sys image
julia --startup-file=no \
  -J/Applications/Julia-1.4.app/Contents/Resources/julia/lib/julia/sys.dylib \
  --output-o main.o create_sysimage.jl

# to dylib
clang -shared \
  -o main.dylib -Wl,-all_load main.o \
  -L${JULIA_PATH}/lib \
  -ljulia

# build binary
gcc -DJULIAC_PROGRAM_LIBNAME=\"main.dylib\" -o main main.c main.dylib -O2 -fPIE \
  -I${JULIA_PATH}/include/julia \
  -L${JULIA_PATH}/lib \
  -ljulia \
  -Wl,-rpath,${JULIA_PATH}/lib
