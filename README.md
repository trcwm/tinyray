# Tinyray

Adapted from https://github.com/ssloy/tinyraytracer and https://github.com/BrunoLevy/learn-fpga/tree/master/FemtoRV/FIRMWARE/CPP_EXAMPLES

Additions:
* thin lens model with stochastic sampling to simulate depth of field.
* gamma correction.
* write result to PPM file.
* changed std::vector to std::array.


## Building
* run ./bootstrap.sh
* cd build
* ninja

## Dependencies
* cmake
* ninja-build
* gcc or clang

## Example

![Tinyray example image](https://github.com/trcwm/tinyray/blob/master/tinyray.png)
