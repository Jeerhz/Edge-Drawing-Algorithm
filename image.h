// SPDX-License-Identifier: MPL-2.0
/**
 * @file image.h
 * @brief Elementary image class
 * @author Robin Gay
 *         Pascal Monasse <pascal.monasse@enpc.fr>
 * @date 2021-2023, 2026
 */

#ifndef IMAGE_H
#define IMAGE_H

struct Point { unsigned short x, y; };

/// A simple (simplistic?) image class
template <typename T>
struct Image {
    int w,h;
    T* data;
public:
    Image();
    Image(int w0, int h0);
    Image(const Image&);
    ~Image();

    void reset(int w0, int h0);

    bool operator!() const { return (w==0 || h==0); }
    void fill(T value);
    void read(T* rawdata, int dx=1);

    T operator()(int i, int j) const;
    T& operator()(int i, int j);
    T operator()(Point p) const { return operator()(p.x, p.y); }
    T& operator()(Point p) { return operator()(p.x, p.y); }

    T* begin() { return data; }
    T* end() { return data+w*h; }
    const T* begin() const { return data; }
    const T* end() const { return data+w*h; }
    
    bool inside(int x, int y) const {
        return (0<=x && x<w && 0<=y && y<h);
    }

private:
    void operator=(const Image&);
};

void blurGaussian(Image<float>& im, float sigma);

/// Constructor
template <typename T>
Image<T>::Image()
: w(0), h(0), data(0) {}

/// Constructor
template <typename T>
Image<T>::Image(int w0, int h0)
: w(w0), h(h0) {
    data = new T[w*h];
}

/// Copy constructor
template <typename T>
Image<T>::Image(const Image<T>& im)
: w(im.w), h(im.h), data(new T[im.w*im.h]) {
    read(im.data);
}

/// Destructor
template <typename T>
Image<T>::~Image() {
    delete [] data;
}

/// Reset dimensions of image
template <typename T>
void Image<T>::reset(int w0, int h0) {
    delete [] data;
    w = w0;
    h = h0;
    data = new T[w*h];
}

/// Fill with constant value
template <typename T>
void Image<T>::fill(T value) {
    read(&value, 0);
}

/// Copying data
template <typename T>
void Image<T>::read(T* rawData, int dx) {
    T* out=data;
    for(int i=w*h; i>0; i--) {
        *out++ = *rawData;
        rawData += dx;
    }
}

/// Pixel access (read-only)
template <typename T>
T Image<T>::operator()(int i, int j) const {
    return data[i+j*w];
}

/// Pixel access (read/write)
template <typename T>
T& Image<T>::operator()(int i, int j) {
    return data[i+j*w];
}

#endif
