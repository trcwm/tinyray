/*

    TinyRay - adapted from https://github.com/ssloy/tinyraytracer
              and https://github.com/BrunoLevy/learn-fpga/tree/master/FemtoRV/FIRMWARE/CPP_EXAMPLES
              
    N.A. Moseley, 8 Aug 2021
    
*/

#include <cmath>
#include <limits>
#include <array>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <random>

#include "tinyray.h"

#define preview

constexpr const float aperature = 0.25f;

#ifndef preview
constexpr const int32_t RaysPerPixel = 128;
constexpr const int32_t GL_width  = 2560;
constexpr const int32_t GL_height = 1440;
#else
constexpr const int32_t RaysPerPixel = 128;
constexpr const int32_t GL_width  = 2560/4;
constexpr const int32_t GL_height = 1440/4;
#endif

#pragma pack(1)
struct Pixel
{
    uint8_t r,g,b;
};

#pragma pack()

std::array<Pixel, GL_width*GL_height> g_pixelBuffer;

struct Light
{
    Light(const Vec3f &p, const float i) : position(p), intensity(i) {}
    Vec3f position;
    float intensity;
};

struct Material
{
    Material(const float r, const Vec4f &a, const Vec3f &color, const float spec) : refractive_index(r), albedo(a), diffuse_color(color), specular_exponent(spec) {}
    Material() : refractive_index(1), albedo(1, 0, 0, 0), diffuse_color(), specular_exponent() {}
    float refractive_index;
    Vec4f albedo;
    Vec3f diffuse_color;
    float specular_exponent;
};

struct Sphere
{
    Vec3f m_center;
    float m_radius2;
    Material m_material;

    constexpr Sphere(const Vec3f &c, const float r, const Material &m) : m_center(c), m_radius2(r*r), m_material(m) {}

    bool ray_intersect(const Vec3f &orig, const Vec3f &dir, float &t0) const
    {
        Vec3f L = m_center - orig;
        float tca = L * dir;
        float d2 = L * L - tca * tca;
        
        if (d2 > m_radius2)
            return false;

        float thc = sqrtf(m_radius2 - d2);

        t0 = tca - thc;
        float t1 = tca + thc;
        if (t0 < 0.f)
            t0 = t1;

        if (t0 < 0.f)
            return false;

        return true;
    }
};

Vec3f reflect(const Vec3f &I, const Vec3f &N)
{
    return I - N * 2.f * (I * N);
}

Vec3f refract(const Vec3f &I, const Vec3f &N, const float eta_t, const float eta_i = 1.f)
{ // Snell's law
    float cosi = -std::max(-1.f, std::min(1.f, I * N));
    if (cosi < 0)
        return refract(I, -N, eta_i, eta_t); // if the ray comes from the inside the object, swap the air and the media
    float eta = eta_i / eta_t;
    float k = 1 - eta * eta * (1 - cosi * cosi);
    return k < 0 ? Vec3f(1, 0, 0) : I * eta + N * (eta * cosi - sqrtf(k)); // k<0 = total reflection, no ray to refract. I refract it anyways, this has no physical meaning
}

template<size_t size1>
bool scene_intersect(const Vec3f &orig, const Vec3f &dir, const std::array<Sphere, size1> &spheres, Vec3f &hit, Vec3f &N, Material &material)
{
    float spheres_dist = std::numeric_limits<float>::max();
    for(auto const& sphere : spheres)
    {
        float dist_i;
        if (sphere.ray_intersect(orig, dir, dist_i) && dist_i < spheres_dist)
        {
            spheres_dist = dist_i;
            hit = orig + dir * dist_i;
            N = (hit - sphere.m_center).normalized();
            material = sphere.m_material;
        }
    }

    float checkerboard_dist = std::numeric_limits<float>::max();
    if (fabs(dir.y) > 1e-3)
    {
        float d = -(orig.y + 4) / dir.y; // the checkerboard plane has equation y = -4
        Vec3f pt = orig + dir * d;
        if (d > 0 && fabs(pt.x) < 10 && pt.z < -10 && pt.z > -30 && d < spheres_dist)
        {
            checkerboard_dist = d;
            hit = pt;
            N = Vec3f(0, 1, 0);
            material.diffuse_color = (int(.5 * hit.x + 1000) + int(.5 * hit.z)) & 1 ? Vec3f(.3, .3, .3) : Vec3f(.3, .2, .1);
        }
    }
    return std::min(spheres_dist, checkerboard_dist) < 1000;
}

template<size_t size1, size_t size2>
Vec3f cast_ray(const Vec3f &orig, const Vec3f &dir, const std::array<Sphere, size1> &spheres, const std::array<Light, size2> &lights, size_t depth = 0)
{
    Vec3f point, N;
    Material material;

    if (depth > 4 || !scene_intersect(orig, dir, spheres, point, N, material))
    {
        return Vec3f(0.2, 0.7, 0.8); // background color
    }

    Vec3f reflect_dir = reflect(dir, N).normalized();
    Vec3f refract_dir = refract(dir, N, material.refractive_index).normalized();
    Vec3f reflect_orig = reflect_dir * N < 0 ? point - N * 1e-3 : point + N * 1e-3; // offset the original point to avoid occlusion by the object itself
    Vec3f refract_orig = refract_dir * N < 0 ? point - N * 1e-3 : point + N * 1e-3;
    Vec3f reflect_color = cast_ray(reflect_orig, reflect_dir, spheres, lights, depth + 1);
    Vec3f refract_color = cast_ray(refract_orig, refract_dir, spheres, lights, depth + 1);

    float diffuse_light_intensity = 0, specular_light_intensity = 0;
    for(auto const& light : lights)
    {
        Vec3f light_dir = (light.position - point).normalized();
        float light_distance = (light.position - point).norm();

        Vec3f shadow_orig = light_dir * N < 0 ? point - N * 1e-3 : point + N * 1e-3; // checking if the point lies in the shadow of the lights[i]
        Vec3f shadow_pt, shadow_N;
        Material tmpmaterial;
        
        if (scene_intersect(shadow_orig, light_dir, spheres, shadow_pt, shadow_N, tmpmaterial) && (shadow_pt - shadow_orig).norm() < light_distance)
            continue;

        diffuse_light_intensity += light.intensity * std::max(0.f, light_dir * N);
        specular_light_intensity += powf(std::max(0.f, -reflect(-light_dir, N) * dir), material.specular_exponent) * light.intensity;
    }

    return material.diffuse_color * diffuse_light_intensity * material.albedo[0] + Vec3f(1., 1., 1.) * specular_light_intensity * material.albedo[1] + reflect_color * material.albedo[2] + refract_color * material.albedo[3];
}

void set_pixel(int x, int y, float r, float g, float b)
{
    auto Pixel = &g_pixelBuffer[y*GL_width + x];

    Pixel->r = static_cast<uint8_t>(std::clamp(255.0f*r,0.0f,255.0f));
    Pixel->g = static_cast<uint8_t>(std::clamp(255.0f*g,0.0f,255.0f));
    Pixel->b = static_cast<uint8_t>(std::clamp(255.0f*b,0.0f,255.0f));
}

static std::normal_distribution<float> gs_distribution(0.0f,0.707f);
static std::default_random_engine gs_generator;

const float jitter()
{
    return gs_distribution(gs_generator);

    //return -0.5f + std::rand() / static_cast<float>(RAND_MAX);
}

// gamma correction, https://mitchellkember.com/blog/post/ray-tracer/
const float gammaCorrection(float c)
{
    if (c < 0.0031308f)
    {
        return 12.92f*c;
    }
    return 1.055f*powf(c,1.0f/2.4f) - 0.055f;
}

constexpr Vec3f elementMultiply(const Vec3f &o1, const Vec3f &o2)
{
    return {o1.x * o2.x, o1.y * o2.y, o1.z * o2.z};
}

Ray genCameraRay(const Vec2f &pixel)
{
    Ray r;
    constexpr float fov = M_PI / 3.f;
    
    r.m_org = {0.f,0.f,0.f};
    r.m_dir.z = -GL_height / (2. * tan(fov / 2.f));
    r.m_dir.x = (pixel.x + 0.5f) - GL_width / 2.f;
    r.m_dir.y = -(pixel.y + 0.5f) + GL_height / 2.f; // this flips the image at the same time
    r.m_dir.normalize();
    return r;
}

// https://raytracing.github.io/books/RayTracingInOneWeekend.html#defocusblur/athinlensapproximation
class ThinLensCamera
{
public:

    ThinLensCamera(const Vec3f &cameraPos, const Vec3f &lookAt, const Vec3f &upVector,
        float verticalFieldOfView /* indegrees */,
        float aspectRatio,
        float aperature,
        float focusDistance)
    {
        const float theta = verticalFieldOfView / 180.0f * M_PI;
        auto const h = tan(theta/2.f);
        auto const viewportHeight = 2.0f * h;
        auto const viewportWidth  = aspectRatio * viewportHeight;

        m_w = (cameraPos - lookAt).normalized();
        m_u = cross(upVector, m_w).normalized();
        m_v = cross(m_w,m_u);

        m_origin = cameraPos;
        m_horizontal = focusDistance * viewportWidth * m_u;
        m_vertical = focusDistance * viewportHeight * m_v;
        m_lowerLeft = m_origin - m_horizontal/2 - m_vertical/2 - focusDistance*m_w;

        m_lensRadius = aperature / 2.f;
    }

    Ray generateRay(float x, float y) const
    {
        const Vec3f rd = m_lensRadius * Vec3f(jitter(), jitter(), 0.0f);
        const Vec3f offset = m_u * rd.x + m_v * rd.y;

        const Vec3f dir = (m_lowerLeft + x*m_horizontal + y*m_vertical   
            - m_origin - offset).normalized();

        return Ray{m_origin + offset, dir};
    }

protected:
    Vec3f m_lowerLeft;
    Vec3f m_horizontal;
    Vec3f m_vertical;
    Vec3f m_origin;
    Vec3f m_w, m_u, m_v;
    float m_lensRadius;
};

template<std::size_t size1, std::size_t size2>
void render(const std::array<Sphere, size1> &spheres, const std::array<Light, size2> &lights)
{   
    ThinLensCamera camera({0.f, 3.0f, -2.f}, {0.f, 0.0f, -18.f}, {0.f, -1.f, 0.f},
        75.f, GL_width / static_cast<float>(GL_height), aperature, 10.f +3.0f);

    for (int j = 0; j < GL_height; j++)
    { 
        for (int i = 0; i < GL_width; i++)
        {
            Vec3f RGB{0.f, 0.f, 0.f};
            
            for(uint32_t r=0; r<RaysPerPixel; r++)
            {
                auto ray = camera.generateRay((GL_width - i)/static_cast<float>(GL_width),
                    j/static_cast<float>(GL_height));

                RGB += cast_ray(ray.m_org, ray.m_dir, spheres, lights);
            }

            set_pixel(i, j, 
                gammaCorrection(RGB.x / static_cast<float>(RaysPerPixel)), 
                gammaCorrection(RGB.y / static_cast<float>(RaysPerPixel)), 
                gammaCorrection(RGB.z / static_cast<float>(RaysPerPixel)));
        }
        std::cout << "Row " << j << "\r" << std::flush;
    }
    std::cout << "\n";
}

int main()
{

    Material ivory(1.0, Vec4f(0.6, 0.3, 0.1, 0.0), Vec3f(0.4, 0.4, 0.3), 50.);
    Material glass(1.5, Vec4f(0.0, 0.5, 0.1, 0.8), Vec3f(0.6, 0.7, 0.8), 125.);
    Material red_rubber(1.0, Vec4f(0.9, 0.1, 0.0, 0.0), Vec3f(0.3, 0.1, 0.1), 10.);
    Material mirror(1.0, Vec4f(0.0, 10.0, 0.8, 0.0), Vec3f(1.0, 1.0, 1.0), 1425.);

    const std::array<Sphere, 4> spheres =
    {
        Sphere{Vec3f(-3, 0, -16), 2, ivory},
        Sphere{Vec3f(-1.0, -1.5, -12), 2, glass},
        Sphere{Vec3f(1.5, -0.5, -18), 3, red_rubber},
        Sphere{Vec3f(7, 5, -18), 4, mirror}
    };

    const std::array<Light, 3> lights =
    {
        Light(Vec3f(-20, 20, 20), 1.5),
        Light(Vec3f(30, 50, -25), 1.8),
        Light(Vec3f(30, 20, 30), 1.7)
    };
    
    render(spheres, lights);

    // write buffer to ppm file
    std::ofstream ofile("tinyray.ppm", std::ios::binary);
    if (!ofile.good())
    {
        std::cerr << "Cannot open tinytrace.ppm for writing!\n";
        return EXIT_FAILURE;
    }

    ofile << "P6 " << GL_width << " " << GL_height << " 255\n";
    ofile.write((char*)&g_pixelBuffer[0], g_pixelBuffer.size()*sizeof(Pixel));
    ofile.close();

    std::cout << "Image written to tinyray.ppm\n";

    return EXIT_SUCCESS;
}
