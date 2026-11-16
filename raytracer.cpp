// =============================================================================
//  raytracer.cpp: a physically based path tracer in ONE file (C++17, no dependencies)
//
//  Features: global illumination, diffuse / metal / glass / emissive materials,
//  spheres, quads, boxes and instancing, BVH acceleration, depth of field,
//  multithreading (output is identical for any thread count), a built-in PNG
//  encoder and a built-in self-test suite.
//
//  Build:   g++ -std=c++17 -O2 -pthread raytracer.cpp -o raytracer
//           (MSVC: cl /std:c++17 /O2 /EHsc raytracer.cpp)
//  Run:     ./raytracer --scene showcase --width 800 --spp 100 --out showcase.png
//           ./raytracer --scene cornell --width 400 --height 400 --spp 400 -o cornell.png
//  Test:    ./raytracer --selftest
//  Help:    ./raytracer --help        Scenes: showcase, cornell, night
//
//  MIT License, Copyright (c) 2026
// =============================================================================
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------
// vec3
// ---------------------------------------------------------------------------
// Small 3D vector type used for points, directions and RGB colors.

namespace rt {

struct Vec3 {
    double x = 0, y = 0, z = 0;

    constexpr Vec3() = default;
    constexpr Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

    constexpr Vec3 operator-() const { return {-x, -y, -z}; }
    constexpr double operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }
    double& operator[](int i) { return i == 0 ? x : (i == 1 ? y : z); }

    Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }
    Vec3& operator*=(double t) { x *= t; y *= t; z *= t; return *this; }

    constexpr double length_squared() const { return x * x + y * y + z * z; }
    double length() const { return std::sqrt(length_squared()); }

    // True when the vector is almost zero in every component.
    bool near_zero() const {
        constexpr double s = 1e-8;
        return std::fabs(x) < s && std::fabs(y) < s && std::fabs(z) < s;
    }
};

using Point3 = Vec3;
using Color = Vec3;

constexpr Vec3 operator+(const Vec3& a, const Vec3& b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
constexpr Vec3 operator-(const Vec3& a, const Vec3& b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
constexpr Vec3 operator*(const Vec3& a, const Vec3& b) { return {a.x * b.x, a.y * b.y, a.z * b.z}; }
constexpr Vec3 operator*(double t, const Vec3& v) { return {t * v.x, t * v.y, t * v.z}; }
constexpr Vec3 operator*(const Vec3& v, double t) { return t * v; }
constexpr Vec3 operator/(const Vec3& v, double t) { return (1.0 / t) * v; }

constexpr double dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
constexpr Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline Vec3 unit(const Vec3& v) { return v / v.length(); }

// Mirror reflection of v about the unit normal n.
constexpr Vec3 reflect(const Vec3& v, const Vec3& n) { return v - 2 * dot(v, n) * n; }

// Snell's law. `uv` must be a unit vector, `n` a unit normal.
inline Vec3 refract(const Vec3& uv, const Vec3& n, double etai_over_etat) {
    const double cos_theta = std::fmin(dot(-uv, n), 1.0);
    const Vec3 perp = etai_over_etat * (uv + cos_theta * n);
    const Vec3 parallel = -std::sqrt(std::fabs(1.0 - perp.length_squared())) * n;
    return perp + parallel;
}

}  // namespace rt

// ---------------------------------------------------------------------------
// rng
// ---------------------------------------------------------------------------
// Fast, seedable random numbers (xoshiro256** seeded through splitmix64).
// Every image row gets its own generator, so renders are identical for any thread count.

namespace rt {

class Rng {
public:
    explicit Rng(uint64_t seed = 1) {
        uint64_t x = seed;
        for (auto& s : s_) s = splitmix(x);
    }

    uint64_t next() {
        const uint64_t result = rotl(s_[1] * 5, 7) * 9;
        const uint64_t t = s_[1] << 17;
        s_[2] ^= s_[0]; s_[3] ^= s_[1]; s_[1] ^= s_[2]; s_[0] ^= s_[3];
        s_[2] ^= t;
        s_[3] = rotl(s_[3], 45);
        return result;
    }

    // Uniform double in [0, 1).
    double uniform() { return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0); }
    double uniform(double a, double b) { return a + (b - a) * uniform(); }

    Vec3 vec(double a, double b) { return {uniform(a, b), uniform(a, b), uniform(a, b)}; }

    Vec3 in_unit_sphere() {
        for (;;) {
            const Vec3 p = vec(-1, 1);
            if (p.length_squared() < 1) return p;
        }
    }
    Vec3 unit_vector() {
        for (;;) {
            const Vec3 p = in_unit_sphere();
            const double l2 = p.length_squared();
            if (l2 > 1e-160) return p / std::sqrt(l2);
        }
    }
    Vec3 in_unit_disk() {
        for (;;) {
            const Vec3 p{uniform(-1, 1), uniform(-1, 1), 0};
            if (p.length_squared() < 1) return p;
        }
    }

private:
    static uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    static uint64_t splitmix(uint64_t& x) {
        uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    uint64_t s_[4];
};

}  // namespace rt

// ---------------------------------------------------------------------------
// core
// ---------------------------------------------------------------------------
// Rays, hit records and axis-aligned bounding boxes.

namespace rt {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kPi = 3.14159265358979323846;
constexpr double radians(double deg) { return deg * kPi / 180.0; }

struct Ray {
    Point3 origin;
    Vec3 dir;
    Point3 at(double t) const { return origin + t * dir; }
};

class Material;

struct HitRecord {
    Point3 p;
    Vec3 normal;  // always faces against the incoming ray
    double t = 0;
    bool front_face = true;
    const Material* mat = nullptr;

    void set_face_normal(const Ray& r, const Vec3& outward) {
        front_face = dot(r.dir, outward) < 0;
        normal = front_face ? outward : -outward;
    }
};

// Axis-aligned bounding box. A default-constructed box is empty.
struct AABB {
    Vec3 lo{kInf, kInf, kInf};
    Vec3 hi{-kInf, -kInf, -kInf};

    AABB() = default;
    AABB(const Vec3& a, const Vec3& b)
        : lo{std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)},
          hi{std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)} {}

    static AABB surround(const AABB& a, const AABB& b) {
        AABB r;
        r.lo = {std::min(a.lo.x, b.lo.x), std::min(a.lo.y, b.lo.y), std::min(a.lo.z, b.lo.z)};
        r.hi = {std::max(a.hi.x, b.hi.x), std::max(a.hi.y, b.hi.y), std::max(a.hi.z, b.hi.z)};
        return r;
    }

    // Flat shapes (quads) would otherwise have a zero-thickness box.
    AABB padded(double d = 1e-4) const {
        AABB r = *this;
        for (int a = 0; a < 3; ++a)
            if (r.hi[a] - r.lo[a] < d) { r.lo[a] -= d / 2; r.hi[a] += d / 2; }
        return r;
    }

    int longest_axis() const {
        const Vec3 e = hi - lo;
        return e.x > e.y ? (e.x > e.z ? 0 : 2) : (e.y > e.z ? 1 : 2);
    }
    Vec3 centroid() const { return 0.5 * (lo + hi); }

    // Slab test. Written so NaNs (0 * inf) never poison the interval.
    bool hit(const Ray& r, double tmin, double tmax) const {
        for (int a = 0; a < 3; ++a) {
            const double inv = 1.0 / r.dir[a];
            double t0 = (lo[a] - r.origin[a]) * inv;
            double t1 = (hi[a] - r.origin[a]) * inv;
            if (inv < 0) std::swap(t0, t1);
            if (t0 > tmin) tmin = t0;
            if (t1 < tmax) tmax = t1;
            if (tmax <= tmin) return false;
        }
        return true;
    }
};

}  // namespace rt

// ---------------------------------------------------------------------------
// material
// ---------------------------------------------------------------------------
// Surface materials: diffuse (optionally checkered), metal, glass and area lights.

namespace rt {

class Material {
public:
    virtual ~Material() = default;
    // Returns true when the ray scatters; fills the attenuation and the new ray.
    virtual bool scatter(const Ray&, const HitRecord&, Rng&, Color&, Ray&) const { return false; }
    virtual Color emitted(const HitRecord&) const { return {0, 0, 0}; }
};

class Lambertian : public Material {
public:
    explicit Lambertian(const Color& albedo) : a_(albedo), b_(albedo), scale_(0) {}
    // Checkerboard on the ground plane (x/z) with squares of side `scale`.
    Lambertian(const Color& a, const Color& b, double scale) : a_(a), b_(b), scale_(scale) {}

    bool scatter(const Ray&, const HitRecord& h, Rng& rng, Color& att, Ray& out) const override {
        Vec3 dir = h.normal + rng.unit_vector();
        if (dir.near_zero()) dir = h.normal;
        out = {h.p, dir};
        att = color_at(h.p);
        return true;
    }

private:
    Color color_at(const Point3& p) const {
        if (scale_ <= 0) return a_;
        const long s = static_cast<long>(std::floor(p.x / scale_)) + static_cast<long>(std::floor(p.z / scale_));
        return (s % 2 == 0) ? a_ : b_;
    }
    Color a_, b_;
    double scale_;
};

class Metal : public Material {
public:
    Metal(const Color& albedo, double fuzz) : albedo_(albedo), fuzz_(fuzz < 1 ? fuzz : 1) {}
    bool scatter(const Ray& in, const HitRecord& h, Rng& rng, Color& att, Ray& out) const override {
        const Vec3 r = reflect(unit(in.dir), h.normal);
        out = {h.p, r + fuzz_ * rng.unit_vector()};
        att = albedo_;
        return dot(out.dir, h.normal) > 0;
    }

private:
    Color albedo_;
    double fuzz_;
};

class Dielectric : public Material {
public:
    explicit Dielectric(double ior) : ior_(ior) {}
    bool scatter(const Ray& in, const HitRecord& h, Rng& rng, Color& att, Ray& out) const override {
        att = {1, 1, 1};
        const double ratio = h.front_face ? 1.0 / ior_ : ior_;
        const Vec3 u = unit(in.dir);
        const double cos_t = std::fmin(dot(-u, h.normal), 1.0);
        const double sin_t = std::sqrt(1.0 - cos_t * cos_t);
        const bool total_internal = ratio * sin_t > 1.0;
        const Vec3 dir = (total_internal || schlick(cos_t, ratio) > rng.uniform())
                             ? reflect(u, h.normal)
                             : refract(u, h.normal, ratio);
        out = {h.p, dir};
        return true;
    }

private:
    static double schlick(double cosine, double ratio) {
        double r0 = (1 - ratio) / (1 + ratio);
        r0 *= r0;
        return r0 + (1 - r0) * std::pow(1 - cosine, 5);
    }
    double ior_;
};

class DiffuseLight : public Material {
public:
    explicit DiffuseLight(const Color& c) : c_(c) {}
    Color emitted(const HitRecord& h) const override { return h.front_face ? c_ : Color{0, 0, 0}; }

private:
    Color c_;
};

}  // namespace rt

// ---------------------------------------------------------------------------
// shapes
// ---------------------------------------------------------------------------
// Geometry: spheres, parallelograms, boxes, instancing (rotate + translate) and a BVH.

namespace rt {

class Hittable {
public:
    virtual ~Hittable() = default;
    virtual bool hit(const Ray& r, double tmin, double tmax, HitRecord& rec) const = 0;
    virtual AABB bbox() const = 0;
};

using HittablePtr = std::shared_ptr<Hittable>;
using MaterialPtr = std::shared_ptr<Material>;

class Sphere : public Hittable {
public:
    // A negative radius flips the normals inward: handy for hollow glass shells.
    Sphere(const Point3& c, double r, MaterialPtr m) : c_(c), r_(r), mat_(std::move(m)) {
        const double a = std::fabs(r);
        box_ = AABB(c - Vec3(a, a, a), c + Vec3(a, a, a));
    }

    bool hit(const Ray& r, double tmin, double tmax, HitRecord& rec) const override {
        const Vec3 oc = c_ - r.origin;
        const double a = r.dir.length_squared();
        const double h = dot(r.dir, oc);
        const double c = oc.length_squared() - r_ * r_;
        const double disc = h * h - a * c;
        if (disc < 0) return false;
        const double sq = std::sqrt(disc);
        double root = (h - sq) / a;
        if (root <= tmin || root >= tmax) {
            root = (h + sq) / a;
            if (root <= tmin || root >= tmax) return false;
        }
        rec.t = root;
        rec.p = r.at(root);
        rec.set_face_normal(r, (rec.p - c_) / r_);
        rec.mat = mat_.get();
        return true;
    }
    AABB bbox() const override { return box_; }

private:
    Point3 c_;
    double r_;
    MaterialPtr mat_;
    AABB box_;
};

// Parallelogram spanned by u and v starting at corner q.
class Quad : public Hittable {
public:
    Quad(const Point3& q, const Vec3& u, const Vec3& v, MaterialPtr m) : q_(q), u_(u), v_(v), mat_(std::move(m)) {
        const Vec3 n = cross(u, v);
        normal_ = unit(n);
        d_ = dot(normal_, q);
        w_ = n / dot(n, n);
        box_ = AABB::surround(AABB(q, q + u + v), AABB(q + u, q + v)).padded();
    }

    bool hit(const Ray& r, double tmin, double tmax, HitRecord& rec) const override {
        const double denom = dot(normal_, r.dir);
        if (std::fabs(denom) < 1e-12) return false;
        const double t = (d_ - dot(normal_, r.origin)) / denom;
        if (t <= tmin || t >= tmax) return false;
        const Vec3 planar = r.at(t) - q_;
        const double alpha = dot(w_, cross(planar, v_));
        const double beta = dot(w_, cross(u_, planar));
        if (alpha < 0 || alpha > 1 || beta < 0 || beta > 1) return false;
        rec.t = t;
        rec.p = r.at(t);
        rec.set_face_normal(r, normal_);
        rec.mat = mat_.get();
        return true;
    }
    AABB bbox() const override { return box_; }

private:
    Point3 q_;
    Vec3 u_, v_, normal_, w_;
    double d_;
    MaterialPtr mat_;
    AABB box_;
};

class HittableList : public Hittable {
public:
    void add(HittablePtr o) { box_ = AABB::surround(box_, o->bbox()); objects.push_back(std::move(o)); }

    bool hit(const Ray& r, double tmin, double tmax, HitRecord& rec) const override {
        bool any = false;
        double closest = tmax;
        for (const auto& o : objects)
            if (o->hit(r, tmin, closest, rec)) { any = true; closest = rec.t; }
        return any;
    }
    AABB bbox() const override { return box_; }

    std::vector<HittablePtr> objects;

private:
    AABB box_;
};

// Axis-aligned box made of six quads.
inline std::shared_ptr<HittableList> make_box(const Point3& a, const Point3& b, const MaterialPtr& m) {
    auto s = std::make_shared<HittableList>();
    const Point3 lo(std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z));
    const Point3 hi(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z));
    const Vec3 dx(hi.x - lo.x, 0, 0), dy(0, hi.y - lo.y, 0), dz(0, 0, hi.z - lo.z);
    s->add(std::make_shared<Quad>(Point3(lo.x, lo.y, hi.z), dx, dy, m));
    s->add(std::make_shared<Quad>(Point3(hi.x, lo.y, hi.z), -dz, dy, m));
    s->add(std::make_shared<Quad>(Point3(hi.x, lo.y, lo.z), -dx, dy, m));
    s->add(std::make_shared<Quad>(Point3(lo.x, lo.y, lo.z), dz, dy, m));
    s->add(std::make_shared<Quad>(Point3(lo.x, hi.y, hi.z), dx, -dz, m));
    s->add(std::make_shared<Quad>(Point3(lo.x, lo.y, lo.z), dx, dz, m));
    return s;
}

// Rotates an object about the Y axis, then moves it. Rays are transformed into object space.
class Instance : public Hittable {
public:
    Instance(HittablePtr obj, double angle_deg, const Vec3& offset)
        : obj_(std::move(obj)), offset_(offset), sin_(std::sin(radians(angle_deg))), cos_(std::cos(radians(angle_deg))) {
        const AABB b = obj_->bbox();
        Vec3 lo(kInf, kInf, kInf), hi(-kInf, -kInf, -kInf);
        for (int i = 0; i < 8; ++i) {
            const Vec3 c((i & 1) ? b.hi.x : b.lo.x, (i & 2) ? b.hi.y : b.lo.y, (i & 4) ? b.hi.z : b.lo.z);
            const Vec3 w = to_world(c);
            for (int a = 0; a < 3; ++a) { lo[a] = std::min(lo[a], w[a]); hi[a] = std::max(hi[a], w[a]); }
        }
        box_ = AABB(lo, hi);
    }

    bool hit(const Ray& r, double tmin, double tmax, HitRecord& rec) const override {
        const Ray local{to_object(r.origin - offset_), to_object_dir(r.dir)};
        if (!obj_->hit(local, tmin, tmax, rec)) return false;
        rec.p = to_world(rec.p);
        rec.normal = rotate(rec.normal);
        return true;
    }
    AABB bbox() const override { return box_; }

private:
    Vec3 rotate(const Vec3& v) const { return {cos_ * v.x + sin_ * v.z, v.y, -sin_ * v.x + cos_ * v.z}; }
    Vec3 to_object_dir(const Vec3& v) const { return {cos_ * v.x - sin_ * v.z, v.y, sin_ * v.x + cos_ * v.z}; }
    Vec3 to_object(const Vec3& v) const { return to_object_dir(v); }
    Vec3 to_world(const Vec3& v) const { return rotate(v) + offset_; }

    HittablePtr obj_;
    Vec3 offset_;
    double sin_, cos_;
    AABB box_;
};

// Bounding volume hierarchy: splits along the longest axis at the median.
// Turns an O(n) scene scan into roughly O(log n) per ray.
class BVH : public Hittable {
public:
    explicit BVH(std::vector<HittablePtr> objs) : BVH(objs, 0, objs.size()) {}

    bool hit(const Ray& r, double tmin, double tmax, HitRecord& rec) const override {
        if (!box_.hit(r, tmin, tmax)) return false;
        const bool hl = left_->hit(r, tmin, tmax, rec);
        const bool hr = right_->hit(r, tmin, hl ? rec.t : tmax, rec);
        return hl || hr;
    }
    AABB bbox() const override { return box_; }

private:
    BVH(std::vector<HittablePtr>& objs, size_t lo, size_t hi) {
        AABB all;
        for (size_t i = lo; i < hi; ++i) all = AABB::surround(all, objs[i]->bbox());
        const int axis = all.longest_axis();
        const size_t n = hi - lo;
        if (n == 1) {
            left_ = right_ = objs[lo];
        } else if (n == 2) {
            left_ = objs[lo];
            right_ = objs[lo + 1];
        } else {
            std::sort(objs.begin() + static_cast<long>(lo), objs.begin() + static_cast<long>(hi),
                      [axis](const HittablePtr& a, const HittablePtr& b) {
                          return a->bbox().centroid()[axis] < b->bbox().centroid()[axis];
                      });
            const size_t mid = lo + n / 2;
            left_ = std::shared_ptr<BVH>(new BVH(objs, lo, mid));
            right_ = std::shared_ptr<BVH>(new BVH(objs, mid, hi));
        }
        box_ = AABB::surround(left_->bbox(), right_->bbox());
    }

    HittablePtr left_, right_;
    AABB box_;
};

}  // namespace rt

// ---------------------------------------------------------------------------
// camera
// ---------------------------------------------------------------------------
// Thin-lens camera with depth of field.

namespace rt {

class Camera {
public:
    Camera(const Point3& from, const Point3& at, const Vec3& up, double vfov_deg, double aspect,
           double aperture, double focus_dist)
        : origin_(from), lens_radius_(aperture / 2) {
        const double h = std::tan(radians(vfov_deg) / 2);
        const double vh = 2 * h, vw = aspect * vh;
        w_ = unit(from - at);
        u_ = unit(cross(up, w_));
        v_ = cross(w_, u_);
        horizontal_ = focus_dist * vw * u_;
        vertical_ = focus_dist * vh * v_;
        lower_left_ = origin_ - horizontal_ / 2 - vertical_ / 2 - focus_dist * w_;
    }

    // s, t in [0, 1]: (0, 0) is the bottom-left corner of the image.
    Ray get_ray(double s, double t, Rng& rng) const {
        const Vec3 rd = lens_radius_ * rng.in_unit_disk();
        const Vec3 off = u_ * rd.x + v_ * rd.y;
        return {origin_ + off, lower_left_ + s * horizontal_ + t * vertical_ - origin_ - off};
    }

private:
    Point3 origin_, lower_left_;
    Vec3 horizontal_, vertical_, u_, v_, w_;
    double lens_radius_;
};

}  // namespace rt

// ---------------------------------------------------------------------------
// image
// ---------------------------------------------------------------------------
// 8-bit RGB image plus PPM and PNG writers. The PNG encoder is self-contained
// (stored deflate blocks, CRC-32, Adler-32), so there is no zlib dependency.

namespace rt {

struct Image {
    int width = 0, height = 0;
    std::vector<uint8_t> rgb;  // row-major, top row first
    Image() = default;
    Image(int w, int h) : width(w), height(h), rgb(static_cast<size_t>(w) * static_cast<size_t>(h) * 3, 0) {}
    uint8_t* pixel(int x, int y) { return &rgb[(static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)) * 3]; }
    const uint8_t* pixel(int x, int y) const { return &rgb[(static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)) * 3]; }
};

namespace png {

inline const std::array<uint32_t, 256>& crc_table() {
    static const std::array<uint32_t, 256> t = [] {
        std::array<uint32_t, 256> a{};
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            a[i] = c;
        }
        return a;
    }();
    return t;
}

inline uint32_t crc32(const uint8_t* d, size_t n, uint32_t crc = 0) {
    crc = ~crc;
    const auto& t = crc_table();
    for (size_t i = 0; i < n; ++i) crc = t[(crc ^ d[i]) & 0xFFu] ^ (crc >> 8);
    return ~crc;
}

inline uint32_t adler32(const uint8_t* d, size_t n) {
    uint32_t a = 1, b = 0;
    for (size_t i = 0; i < n; ++i) { a = (a + d[i]) % 65521u; b = (b + a) % 65521u; }
    return (b << 16) | a;
}

inline void put32(std::vector<uint8_t>& o, uint32_t v) {
    for (int s = 24; s >= 0; s -= 8) o.push_back(static_cast<uint8_t>(v >> s));
}

inline void chunk(std::vector<uint8_t>& out, const char type[4], const std::vector<uint8_t>& data) {
    put32(out, static_cast<uint32_t>(data.size()));
    std::vector<uint8_t> body(type, type + 4);
    body.insert(body.end(), data.begin(), data.end());
    out.insert(out.end(), body.begin(), body.end());
    put32(out, crc32(body.data(), body.size()));
}

// Wraps raw bytes in a zlib stream made of uncompressed ("stored") deflate blocks.
inline std::vector<uint8_t> zlib_store(const std::vector<uint8_t>& raw) {
    std::vector<uint8_t> z{0x78, 0x01};
    size_t pos = 0;
    do {
        const size_t n = std::min<size_t>(65535, raw.size() - pos);
        const bool last = pos + n == raw.size();
        z.push_back(last ? 1 : 0);
        z.push_back(static_cast<uint8_t>(n & 0xFF));
        z.push_back(static_cast<uint8_t>(n >> 8));
        z.push_back(static_cast<uint8_t>(~n & 0xFF));
        z.push_back(static_cast<uint8_t>((~n >> 8) & 0xFF));
        z.insert(z.end(), raw.begin() + static_cast<long>(pos), raw.begin() + static_cast<long>(pos + n));
        pos += n;
    } while (pos < raw.size());
    put32(z, adler32(raw.data(), raw.size()));
    return z;
}

}  // namespace png

inline std::vector<uint8_t> encode_png(const Image& img) {
    std::vector<uint8_t> out{0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    std::vector<uint8_t> ihdr;
    png::put32(ihdr, static_cast<uint32_t>(img.width));
    png::put32(ihdr, static_cast<uint32_t>(img.height));
    ihdr.insert(ihdr.end(), {8, 2, 0, 0, 0});  // 8-bit, truecolor, no interlace
    png::chunk(out, "IHDR", ihdr);

    std::vector<uint8_t> raw;
    raw.reserve(img.rgb.size() + static_cast<size_t>(img.height));
    for (int y = 0; y < img.height; ++y) {
        raw.push_back(0);  // filter type: none
        const uint8_t* row = img.pixel(0, y);
        raw.insert(raw.end(), row, row + static_cast<size_t>(img.width) * 3);
    }
    png::chunk(out, "IDAT", png::zlib_store(raw));
    png::chunk(out, "IEND", {});
    return out;
}

inline bool write_png(const std::string& path, const Image& img) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    const auto bytes = encode_png(img);
    f.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    return static_cast<bool>(f);
}

inline bool write_ppm(const std::string& path, const Image& img) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f << "P6\n" << img.width << ' ' << img.height << "\n255\n";
    f.write(reinterpret_cast<const char*>(img.rgb.data()), static_cast<std::streamsize>(img.rgb.size()));
    return static_cast<bool>(f);
}

}  // namespace rt

// ---------------------------------------------------------------------------
// render
// ---------------------------------------------------------------------------
// The path tracer: scene description, recursive radiance estimation and a threaded render loop.

namespace rt {

struct Scene {
    HittablePtr world;
    Camera camera;
    bool use_sky = true;           // gradient sky, or a flat background color
    Color sky_top{0.5, 0.7, 1.0};
    Color sky_bottom{1, 1, 1};
    Color background{0, 0, 0};
};

struct RenderSettings {
    int width = 800;
    int height = 450;
    int spp = 50;         // samples per pixel
    int max_depth = 50;   // maximum bounces
    int threads = 0;      // 0 = use all hardware threads
    uint64_t seed = 1;
};

inline Color ray_color(const Ray& r, const Scene& sc, int depth, Rng& rng) {
    if (depth <= 0) return {0, 0, 0};
    HitRecord rec;
    if (!sc.world->hit(r, 1e-3, kInf, rec)) {
        if (!sc.use_sky) return sc.background;
        const double t = 0.5 * (unit(r.dir).y + 1.0);
        return (1.0 - t) * sc.sky_bottom + t * sc.sky_top;
    }
    const Color emitted = rec.mat->emitted(rec);
    Ray scattered;
    Color attenuation;
    if (!rec.mat->scatter(r, rec, rng, attenuation, scattered)) return emitted;
    return emitted + attenuation * ray_color(scattered, sc, depth - 1, rng);
}

inline uint8_t to_byte(double linear) {
    const double g = std::sqrt(std::fmax(0.0, linear));  // gamma 2.0
    return static_cast<uint8_t>(256 * std::fmin(g, 0.999));
}

using ProgressFn = std::function<void(int rows_done, int rows_total)>;

inline Image render(const Scene& sc, const RenderSettings& s, const ProgressFn& progress = nullptr) {
    Image img(s.width, s.height);
    const double aspect = static_cast<double>(s.width) / s.height;
    (void)aspect;
    std::atomic<int> next_row{0}, done{0};
    const unsigned hw = std::thread::hardware_concurrency();
    const unsigned n_threads = s.threads > 0 ? static_cast<unsigned>(s.threads) : (hw ? hw : 1);

    auto worker = [&] {
        for (;;) {
            const int j = next_row.fetch_add(1);
            if (j >= s.height) return;
            // One generator per row => identical output no matter how rows are scheduled.
            Rng rng(s.seed * 0x9E3779B97F4A7C15ULL + static_cast<uint64_t>(j) + 1);
            for (int i = 0; i < s.width; ++i) {
                Color sum{0, 0, 0};
                for (int k = 0; k < s.spp; ++k) {
                    const double u = (i + rng.uniform()) / s.width;
                    const double v = 1.0 - (j + rng.uniform()) / s.height;
                    Color c = ray_color(sc.camera.get_ray(u, v, rng), sc, s.max_depth, rng);
                    if (!(std::isfinite(c.x) && std::isfinite(c.y) && std::isfinite(c.z))) continue;  // drop NaN/inf samples
                    c = {std::fmin(c.x, 20.0), std::fmin(c.y, 20.0), std::fmin(c.z, 20.0)};          // tame fireflies
                    sum += c;
                }
                sum *= 1.0 / s.spp;
                uint8_t* p = img.pixel(i, j);
                p[0] = to_byte(sum.x); p[1] = to_byte(sum.y); p[2] = to_byte(sum.z);
            }
            const int d = done.fetch_add(1) + 1;
            if (progress) progress(d, s.height);
        }
    };

    std::vector<std::thread> pool;
    for (unsigned t = 1; t < n_threads; ++t) pool.emplace_back(worker);
    worker();
    for (auto& t : pool) t.join();
    return img;
}

}  // namespace rt

// ---------------------------------------------------------------------------
// scenes
// ---------------------------------------------------------------------------
// Built-in demo scenes.

namespace rt {

inline const std::vector<std::string>& scene_names() {
    static const std::vector<std::string> n{"showcase", "cornell", "night"};
    return n;
}

// Hundreds of small random spheres around three hero spheres, with depth of field.
inline Scene scene_showcase(double aspect, uint64_t seed) {
    Rng rng(seed + 1234);
    std::vector<HittablePtr> objs;
    objs.push_back(std::make_shared<Sphere>(Point3(0, -1000, 0), 1000,
                                            std::make_shared<Lambertian>(Color(0.16, 0.18, 0.26), Color(0.85, 0.86, 0.9), 1.0)));
    const Point3 heroes[] = {{0, 1, 0}, {-4, 1, 0}, {4, 1, 0}};
    for (int a = -11; a < 11; ++a) {
        for (int b = -11; b < 11; ++b) {
            const Point3 c(a + 0.9 * rng.uniform(), 0.2, b + 0.9 * rng.uniform());
            const double pick = rng.uniform();
            bool clear = true;
            for (const auto& h : heroes) if ((c - Point3(h.x, 0.2, h.z)).length() < 1.3) clear = false;
            if (!clear) continue;
            MaterialPtr m;
            if (pick < 0.65) m = std::make_shared<Lambertian>(rng.vec(0, 1) * rng.vec(0, 1));
            else if (pick < 0.9) m = std::make_shared<Metal>(rng.vec(0.5, 1), rng.uniform(0, 0.4));
            else m = std::make_shared<Dielectric>(1.5);
            objs.push_back(std::make_shared<Sphere>(c, 0.2, m));
        }
    }
    auto glass = std::make_shared<Dielectric>(1.5);
    objs.push_back(std::make_shared<Sphere>(Point3(0, 1, 0), 1.0, glass));
    objs.push_back(std::make_shared<Sphere>(Point3(0, 1, 0), -0.9, glass));  // hollow shell
    objs.push_back(std::make_shared<Sphere>(Point3(-4, 1, 0), 1.0, std::make_shared<Lambertian>(Color(0.4, 0.2, 0.1))));
    objs.push_back(std::make_shared<Sphere>(Point3(4, 1, 0), 1.0, std::make_shared<Metal>(Color(0.7, 0.6, 0.5), 0.0)));

    return Scene{std::make_shared<BVH>(std::move(objs)),
                 Camera(Point3(13, 2, 3), Point3(0, 0.6, 0), Vec3(0, 1, 0), 20, aspect, 0.1, 10.5)};
}

// The classic Cornell box: global illumination, color bleeding and soft shadows from one small light.
inline Scene scene_cornell(double aspect, uint64_t) {
    auto red = std::make_shared<Lambertian>(Color(0.65, 0.05, 0.05));
    auto white = std::make_shared<Lambertian>(Color(0.73, 0.73, 0.73));
    auto green = std::make_shared<Lambertian>(Color(0.12, 0.45, 0.15));
    auto light = std::make_shared<DiffuseLight>(Color(17, 17, 17));

    std::vector<HittablePtr> objs;
    objs.push_back(std::make_shared<Quad>(Point3(555, 0, 0), Vec3(0, 555, 0), Vec3(0, 0, 555), red));
    objs.push_back(std::make_shared<Quad>(Point3(0, 0, 0), Vec3(0, 555, 0), Vec3(0, 0, 555), green));
    objs.push_back(std::make_shared<Quad>(Point3(343, 554, 332), Vec3(-130, 0, 0), Vec3(0, 0, -105), light));
    objs.push_back(std::make_shared<Quad>(Point3(0, 0, 0), Vec3(555, 0, 0), Vec3(0, 0, 555), white));
    objs.push_back(std::make_shared<Quad>(Point3(555, 555, 555), Vec3(-555, 0, 0), Vec3(0, 0, -555), white));
    objs.push_back(std::make_shared<Quad>(Point3(0, 0, 555), Vec3(555, 0, 0), Vec3(0, 555, 0), white));
    objs.push_back(std::make_shared<Instance>(make_box(Point3(0, 0, 0), Point3(165, 330, 165), white), 15, Vec3(265, 0, 295)));
    objs.push_back(std::make_shared<Instance>(make_box(Point3(0, 0, 0), Point3(165, 165, 165), white), -18, Vec3(130, 0, 65)));
    objs.push_back(std::make_shared<Sphere>(Point3(190, 235, 120), 70, std::make_shared<Dielectric>(1.5)));

    Scene s{std::make_shared<BVH>(std::move(objs)),
            Camera(Point3(278, 278, -800), Point3(278, 278, 0), Vec3(0, 1, 0), 40, aspect, 0.0, 10)};
    s.use_sky = false;
    return s;
}

// Glowing spheres at night: emission, glass and mirrors with no sky light at all.
inline Scene scene_night(double aspect, uint64_t) {
    std::vector<HittablePtr> objs;
    objs.push_back(std::make_shared<Sphere>(Point3(0, -1000, 0), 1000, std::make_shared<Metal>(Color(0.08, 0.09, 0.12), 0.25)));
    objs.push_back(std::make_shared<Sphere>(Point3(-2.6, 1, 0), 1.0, std::make_shared<DiffuseLight>(Color(1.7, 0.6, 0.12))));
    objs.push_back(std::make_shared<Sphere>(Point3(2.6, 1, 0.3), 1.0, std::make_shared<DiffuseLight>(Color(0.15, 0.7, 1.9))));
    objs.push_back(std::make_shared<Sphere>(Point3(0, 0.8, -1.2), 0.8, std::make_shared<Dielectric>(1.5)));
    objs.push_back(std::make_shared<Sphere>(Point3(0, 0.8, -1.2), 0.45, std::make_shared<DiffuseLight>(Color(0.5, 2.2, 0.7))));
    objs.push_back(std::make_shared<Sphere>(Point3(0.3, 0.45, 2.6), 0.45, std::make_shared<Metal>(Color(0.95, 0.95, 0.98), 0.0)));
    objs.push_back(std::make_shared<Sphere>(Point3(-1.4, 0.3, 2.0), 0.3, std::make_shared<Lambertian>(Color(0.8, 0.8, 0.85))));
    objs.push_back(std::make_shared<Sphere>(Point3(0, 6, -6), 1.2, std::make_shared<DiffuseLight>(Color(8, 8, 9))));

    Scene s{std::make_shared<BVH>(std::move(objs)),
            Camera(Point3(0, 1.7, 8), Point3(0, 0.9, 0), Vec3(0, 1, 0), 32, aspect, 0.05, 8)};
    s.use_sky = false;
    s.background = {0.006, 0.008, 0.02};
    return s;
}

inline Scene make_scene(const std::string& name, double aspect, uint64_t seed) {
    if (name == "showcase") return scene_showcase(aspect, seed);
    if (name == "cornell") return scene_cornell(aspect, seed);
    if (name == "night") return scene_night(aspect, seed);
    throw std::invalid_argument("unknown scene '" + name + "'");
}

}  // namespace rt

// ---------------------------------------------------------------------------
// self-tests (run with --selftest)
// ---------------------------------------------------------------------------

namespace selftest {
using namespace rt;

static int g_checks = 0, g_failed = 0;
#define CHECK(cond)                                                                   \
    do {                                                                              \
        ++g_checks;                                                                   \
        if (!(cond)) { ++g_failed; std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << "  " #cond "\n"; } \
    } while (0)
static bool approx(double a, double b, double eps = 1e-9) { return std::fabs(a - b) < eps; }

static void test_vec3() {
    const Vec3 a(1, 2, 3), b(4, 5, 6);
    CHECK(approx(dot(a, b), 32));
    const Vec3 c = cross(a, b);
    CHECK(approx(c.x, -3) && approx(c.y, 6) && approx(c.z, -3));
    CHECK(approx(unit(Vec3(3, 0, 4)).length(), 1));
    const Vec3 r = reflect(Vec3(1, -1, 0), Vec3(0, 1, 0));
    CHECK(approx(r.x, 1) && approx(r.y, 1) && approx(r.z, 0));
    // Straight-on refraction does not bend.
    const Vec3 t = refract(Vec3(0, -1, 0), Vec3(0, 1, 0), 1.0 / 1.5);
    CHECK(approx(t.x, 0) && approx(t.y, -1));
}

static void test_rng() {
    Rng a(42), b(42), c(43);
    CHECK(a.next() == b.next());
    CHECK(Rng(42).next() != c.next());
    Rng r(7);
    double sum = 0;
    bool in_range = true;
    for (int i = 0; i < 100000; ++i) { const double u = r.uniform(); if (u < 0 || u >= 1) in_range = false; sum += u; }
    CHECK(in_range);
    CHECK(std::fabs(sum / 100000 - 0.5) < 0.01);
    for (int i = 0; i < 1000; ++i) CHECK(approx(r.unit_vector().length(), 1, 1e-9));
}

static void test_sphere_and_quad() {
    auto m = std::make_shared<Lambertian>(Color(1, 1, 1));
    Sphere s(Point3(0, 0, -5), 1, m);
    HitRecord h;
    CHECK(s.hit({{0, 0, 0}, {0, 0, -1}}, 1e-3, kInf, h));
    CHECK(approx(h.t, 4) && approx(h.normal.z, 1) && h.front_face);
    CHECK(!s.hit({{0, 0, 0}, {0, 1, 0}}, 1e-3, kInf, h));
    CHECK(!s.hit({{0, 0, 0}, {0, 0, -1}}, 1e-3, 3.0, h));  // beyond tmax
    CHECK(s.hit({{0, 0, -5}, {0, 0, -1}}, 1e-3, kInf, h) && !h.front_face);  // from inside

    Quad q(Point3(-1, -1, -3), Vec3(2, 0, 0), Vec3(0, 2, 0), m);
    CHECK(q.hit({{0, 0, 0}, {0, 0, -1}}, 1e-3, kInf, h) && approx(h.t, 3));
    CHECK(!q.hit({{1.5, 0, 0}, {0, 0, -1}}, 1e-3, kInf, h));  // misses the edge
    CHECK(!q.hit({{0, 0, 0}, {1, 0, 0}}, 1e-3, kInf, h));     // parallel
}

static void test_instance() {
    auto m = std::make_shared<Lambertian>(Color(1, 1, 1));
    auto box = make_box(Point3(0, 0, 0), Point3(2, 2, 2), m);
    HitRecord h;
    CHECK(box->hit({{1, 1, 10}, {0, 0, -1}}, 1e-3, kInf, h) && approx(h.t, 8));
    Instance moved(box, 0, Vec3(10, 0, 0));
    CHECK(moved.hit({{11, 1, 10}, {0, 0, -1}}, 1e-3, kInf, h) && approx(h.t, 8));
    CHECK(!moved.hit({{1, 1, 10}, {0, 0, -1}}, 1e-3, kInf, h));
    // Rotating a 2x2x2 box by 45 degrees about its corner (0,0,0) widens its footprint to about 2.83.
    // (Flat quads are padded by 1e-4 so their boxes have thickness, hence the tolerance.)
    Instance spun(box, 45, Vec3(0, 0, 0));
    const AABB b = spun.bbox();
    CHECK(approx(b.hi.x - b.lo.x, 2 * std::sqrt(2.0), 1e-3));
    CHECK(b.hi.x - b.lo.x >= 2 * std::sqrt(2.0) - 1e-12);  // never smaller than the true extent
}

static void test_bvh_matches_brute_force() {
    Rng rng(99);
    auto m = std::make_shared<Lambertian>(Color(0.5, 0.5, 0.5));
    std::vector<HittablePtr> objs;
    auto list = std::make_shared<HittableList>();
    for (int i = 0; i < 300; ++i) {
        auto sp = std::make_shared<Sphere>(Point3(rng.uniform(-10, 10), rng.uniform(-3, 3), rng.uniform(-10, 10)), rng.uniform(0.1, 0.8), m);
        objs.push_back(sp);
        list->add(sp);
    }
    BVH bvh(objs);
    int agree = 0, hits = 0;
    for (int i = 0; i < 3000; ++i) {
        const Ray r{{rng.uniform(-12, 12), rng.uniform(-4, 4), 15}, unit(Vec3(rng.uniform(-0.6, 0.6), rng.uniform(-0.3, 0.3), -1))};
        HitRecord a, b;
        const bool ha = bvh.hit(r, 1e-3, kInf, a), hb = list->hit(r, 1e-3, kInf, b);
        if (ha == hb && (!ha || approx(a.t, b.t, 1e-12))) ++agree;
        if (ha) ++hits;
    }
    CHECK(agree == 3000);
    CHECK(hits > 100);  // the test must actually exercise hits
}

static void test_png() {
    const char* iend = "IEND";
    CHECK(png::crc32(reinterpret_cast<const uint8_t*>(iend), 4) == 0xAE426082u);
    const char* digits = "123456789";
    CHECK(png::crc32(reinterpret_cast<const uint8_t*>(digits), 9) == 0xCBF43926u);
    const char* wiki = "Wikipedia";
    CHECK(png::adler32(reinterpret_cast<const uint8_t*>(wiki), 9) == 0x11E60398u);

    Image img(3, 2);
    for (size_t i = 0; i < img.rgb.size(); ++i) img.rgb[i] = static_cast<uint8_t>(i * 10);
    const auto bytes = encode_png(img);
    const uint8_t sig[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    CHECK(bytes.size() > 8 && std::memcmp(bytes.data(), sig, 8) == 0);
    CHECK(std::memcmp(bytes.data() + 12, "IHDR", 4) == 0);
    CHECK(bytes[19] == 3 && bytes[23] == 2);  // width, height
    CHECK(std::memcmp(bytes.data() + bytes.size() - 8, "IEND", 4) == 0);
    // A large image must be split across several stored deflate blocks and still end cleanly.
    Image big(400, 300);
    const auto bb = encode_png(big);
    CHECK(bb.size() > 400u * 300u * 3u);
}

static void test_render() {
    // Lighting sanity: a light pointing at the camera is seen at its emitted brightness.
    auto light = std::make_shared<DiffuseLight>(Color(1, 0.5, 0.25));
    Scene sc{std::make_shared<Quad>(Point3(-5, -5, -3), Vec3(10, 0, 0), Vec3(0, 10, 0), light),
             Camera(Point3(0, 0, 0), Point3(0, 0, -1), Vec3(0, 1, 0), 40, 1.0, 0, 1)};
    sc.use_sky = false;
    RenderSettings rs;
    rs.width = 4; rs.height = 4; rs.spp = 2; rs.max_depth = 4; rs.threads = 1;
    const Image im = render(sc, rs);
    CHECK(im.pixel(1, 1)[0] > im.pixel(1, 1)[1] && im.pixel(1, 1)[1] > im.pixel(1, 1)[2]);

    // Determinism: thread count must not change a single byte of the output.
    const Scene demo = make_scene("cornell", 1.0, 5);
    RenderSettings a;
    a.width = 24; a.height = 24; a.spp = 4; a.max_depth = 8; a.seed = 3;
    a.threads = 1;
    const Image i1 = render(demo, a);
    a.threads = 4;
    const Image i4 = render(demo, a);
    CHECK(i1.rgb == i4.rgb);
    a.seed = 4;
    CHECK(render(demo, a).rgb != i1.rgb);  // a different seed gives a different image

    // Every built-in scene renders something that is not a flat color.
    for (const auto& name : scene_names()) {
        RenderSettings t;
        t.width = 16; t.height = 9; t.spp = 2; t.max_depth = 6; t.threads = 1;
        const Image im2 = render(make_scene(name, 16.0 / 9, 1), t);
        bool varies = false;
        for (size_t i = 3; i < im2.rgb.size(); ++i) if (im2.rgb[i] != im2.rgb[i % 3]) varies = true;
        CHECK(varies);
    }
    bool threw = false;
    try { make_scene("nope", 1, 1); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
}

inline int run() {
    test_vec3();
    test_rng();
    test_sphere_and_quad();
    test_instance();
    test_bvh_matches_brute_force();
    test_png();
    test_render();
    std::cout << (g_checks - g_failed) << "/" << g_checks << " checks passed\n";
    return g_failed ? 1 : 0;
}

}  // namespace selftest

// ---------------------------------------------------------------------------
// command line program
// ---------------------------------------------------------------------------
namespace {

void usage() {
    std::cout <<
        "raytracer " "1.0.0" ": a physically based path tracer\n\n"
        "Usage: raytracer [options]\n\n"
        "  -s, --scene NAME    showcase | cornell | night   (default: showcase)\n"
        "  -w, --width N       image width in pixels        (default: 800)\n"
        "  -h, --height N      image height in pixels       (default: 16:9 of width)\n"
        "  -n, --spp N         samples per pixel            (default: 50)\n"
        "  -d, --depth N       maximum ray bounces          (default: 50)\n"
        "  -j, --threads N     worker threads, 0 = all      (default: 0)\n"
        "      --seed N        random seed                  (default: 1)\n"
        "  -o, --out FILE      output file, .png or .ppm    (default: out.png)\n"
        "      --list          list the built-in scenes\n"
        "      --selftest      run the built-in test suite\n"
        "      --help          show this message\n\n"
        "Example:\n  raytracer --scene cornell --width 600 --height 600 --spp 400 -o cornell.png\n";
}

bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool parse_int(const char* s, long lo, long hi, long& out) {
    char* end = nullptr;
    const long v = std::strtol(s, &end, 10);
    if (end == s || *end != '\0' || v < lo || v > hi) return false;
    out = v;
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    std::string scene = "showcase", out = "out.png";
    rt::RenderSettings s;
    s.height = 0;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&](long lo, long hi, long& v) {
            if (i + 1 >= argc || !parse_int(argv[i + 1], lo, hi, v)) {
                std::cerr << "error: " << a << " needs an integer between " << lo << " and " << hi << "\n";
                std::exit(2);
            }
            ++i;
        };
        long v = 0;
        if (a == "--help") { usage(); return 0; }
        else if (a == "--selftest") { return selftest::run(); }
        else if (a == "--list") { for (const auto& n : rt::scene_names()) std::cout << n << "\n"; return 0; }
        else if (a == "-s" || a == "--scene") { if (i + 1 >= argc) { std::cerr << "error: " << a << " needs a name\n"; return 2; } scene = argv[++i]; }
        else if (a == "-o" || a == "--out") { if (i + 1 >= argc) { std::cerr << "error: " << a << " needs a file name\n"; return 2; } out = argv[++i]; }
        else if (a == "-w" || a == "--width") { value(1, 16384, v); s.width = static_cast<int>(v); }
        else if (a == "-h" || a == "--height") { value(1, 16384, v); s.height = static_cast<int>(v); }
        else if (a == "-n" || a == "--spp") { value(1, 100000, v); s.spp = static_cast<int>(v); }
        else if (a == "-d" || a == "--depth") { value(1, 1000, v); s.max_depth = static_cast<int>(v); }
        else if (a == "-j" || a == "--threads") { value(0, 256, v); s.threads = static_cast<int>(v); }
        else if (a == "--seed") { value(0, 2147483647L, v); s.seed = static_cast<uint64_t>(v); }
        else { std::cerr << "error: unknown option '" << a << "' (try --help)\n"; return 2; }
    }
    if (s.height == 0) s.height = std::max(1, s.width * 9 / 16);
    if (!ends_with(out, ".png") && !ends_with(out, ".ppm")) { std::cerr << "error: output file must end in .png or .ppm\n"; return 2; }

    try {
        const rt::Scene sc = rt::make_scene(scene, static_cast<double>(s.width) / s.height, s.seed);
        std::cerr << "Rendering '" << scene << "' " << s.width << "x" << s.height << " at " << s.spp
                  << " spp, depth " << s.max_depth << "\n";
        const auto t0 = std::chrono::steady_clock::now();
        std::mutex io;
        const rt::Image img = rt::render(sc, s, [&](int done, int total) {
            std::lock_guard<std::mutex> lock(io);
            const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            const double eta = el / done * (total - done);
            std::cerr << "\r  " << done * 100 / total << "%  (" << static_cast<int>(el) << "s elapsed, ~"
                      << static_cast<int>(eta) << "s left)   " << std::flush;
        });
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const bool ok = ends_with(out, ".png") ? rt::write_png(out, img) : rt::write_ppm(out, img);
        if (!ok) { std::cerr << "\nerror: could not write '" << out << "'\n"; return 1; }
        std::cerr << "\rDone in " << secs << " s. Wrote " << out << "                    \n";
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << " (scenes: ";
        for (const auto& n : rt::scene_names()) std::cerr << n << " ";
        std::cerr << ")\n";
        return 2;
    }
    return 0;
}
