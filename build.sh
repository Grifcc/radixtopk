mdkir -p build  
cd build
cmake ..
make -j$(nproc)
cd ..