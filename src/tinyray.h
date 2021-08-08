/*

    TinyRay - adapted from https://github.com/ssloy/tinyraytracer
              and https://github.com/BrunoLevy/learn-fpga/tree/master/FemtoRV/FIRMWARE/CPP_EXAMPLES
              
    N.A. Moseley, 8 Aug 2021
    
*/

#pragma once

#include <cstdlib>
#include <cassert>

template <size_t DIM, typename T>
struct vec
{
    vec()
    {
        for (size_t i = DIM; i--; data_[i] = T())
            ;
    }

    T &operator[](const size_t i)
    {
        assert(i < DIM);
        return data_[i];
    }

    const T &operator[](const size_t i) const
    {
        assert(i < DIM);
        return data_[i];
    }

private:
    T data_[DIM];
};

typedef vec<2, float> Vec2f;
typedef vec<3, float> Vec3f;
typedef vec<3, int> Vec3i;
typedef vec<4, float> Vec4f;

template <typename T>
struct vec<2, T>
{
    vec() : x(T()), y(T()) {}
    vec(T X, T Y) : x(X), y(Y) {}
    
    template <class U>
    vec<2, T>(const vec<2, U> &v);
    T &operator[](const size_t i)
    {
        assert(i < 2);
        return i <= 0 ? x : y;
    }

    const T &operator[](const size_t i) const
    {
        assert(i < 2);
        return i <= 0 ? x : y;
    }

    T x, y;
};

template <typename T>
struct 
vec<3, T>
{
    constexpr vec() : x(T()), y(T()), z(T()) {}
    constexpr vec(T X, T Y, T Z) : x(X), y(Y), z(Z) {}

    constexpr void operator+=(const vec<3, T> &other) noexcept
    {
        x += other.x;
        y += other.y;
        z += other.z;
    }

    constexpr vec<3,T> operator/(const float &v) noexcept
    {
        return {x/v, y/v, z/v};
    }

    constexpr vec<3,T> operator*(const float &v) noexcept
    {
        return {x*v, y*v, z*v};
    }

    constexpr T &operator[](const size_t i) noexcept
    {
        assert(i < 3);
        return i <= 0 ? x : (1 == i ? y : z);
    }

    constexpr const T &operator[](const size_t i) const noexcept
    {
        assert(i < 3);
        return i <= 0 ? x : (1 == i ? y : z);
    }

    float norm() const noexcept
    { 
        return std::sqrt(x * x + y * y + z * z); 
    }

    void normalize(T l = 1)
    {
        auto n = norm();
        x /= n;
        y /= n;
        z /= n;
    }

    vec<3, T> normalized() const
    {
        auto n = norm();
        return {x/n,y/n,z/n};
    }

    T x, y, z;
};

Vec3f operator*(const float &f, const Vec3f &v)
{
    return {v.x*f, v.y*f, v.z*f};
}

template <typename T>
struct vec<4, T>
{
    vec() : x(T()), y(T()), z(T()), w(T()) {}
    vec(T X, T Y, T Z, T W) : x(X), y(Y), z(Z), w(W) {}
    
    T &operator[](const size_t i)
    {
        assert(i < 4);
        return i <= 0 ? x : (1 == i ? y : (2 == i ? z : w));
    }
    
    const T &operator[](const size_t i) const
    {
        assert(i < 4);
        return i <= 0 ? x : (1 == i ? y : (2 == i ? z : w));
    }
    
    T x, y, z, w;
};

template <size_t DIM, typename T>
T operator*(const vec<DIM, T> &lhs, const vec<DIM, T> &rhs)
{
    T ret = T();
    for (size_t i = DIM; i--; ret += lhs[i] * rhs[i])
        ;
    return ret;
}

template <size_t DIM, typename T>
vec<DIM, T> operator+(vec<DIM, T> lhs, const vec<DIM, T> &rhs)
{
    for (size_t i = DIM; i--; lhs[i] += rhs[i])
        ;
    return lhs;
}

template <size_t DIM, typename T>
vec<DIM, T> operator-(vec<DIM, T> lhs, const vec<DIM, T> &rhs)
{
    for (size_t i = DIM; i--; lhs[i] -= rhs[i])
        ;
    return lhs;
}

template <size_t DIM, typename T, typename U>
vec<DIM, T> operator*(const vec<DIM, T> &lhs, const U &rhs)
{
    vec<DIM, T> ret;
    for (size_t i = DIM; i--; ret[i] = lhs[i] * rhs)
        ;
    return ret;
}

template <size_t DIM, typename T>
vec<DIM, T> operator-(const vec<DIM, T> &lhs)
{
    return lhs * T(-1);
}

template <typename T>
vec<3, T> cross(vec<3, T> v1, vec<3, T> v2)
{
    return vec<3, T>(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
}

struct Ray
{
    Vec3f m_org;
    Vec3f m_dir;
};
