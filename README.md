# NanoPCT4catdnncamera
g++ -ggdb -O3 catdnncamera.cpp -o camera `pkg-config --cflags --libs /usr/local/lib/pkgconfig/opencv.pc`

. setqt5env

./camera
