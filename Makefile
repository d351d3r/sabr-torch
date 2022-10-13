all:
	mkdir build && \
	cd build && \
	lima cmake -DCMAKE_PREFIX_PATH=./libtorch .. && \
	lima cmake --build . --config Release
clean:
	rm -rf build