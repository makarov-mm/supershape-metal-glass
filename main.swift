// supershape-metal.swift
// Ray-traced glass supershape — Swift + Metal (MetalKit / MTKView), macOS.
//
// A hardware ray-tracing version of the supershape renderer. The procedural
// mesh is built on the CPU exactly as before (positions from the parametric
// equations, normals by central finite differences), but instead of being
// rasterised it is packed into an MTLAccelerationStructure and traced inline.
// A full-screen fragment shader casts one ray per pixel and follows it through
// a glass material: reflection + refraction with proper dielectric Fresnel.
//
// The surface reads as coloured glass: the animated rainbow hue becomes the
// transmission tint, and the UV grid stays as darker veins inside the glass.
// You can see the studio background through the body, with bright Fresnel
// highlights along the rim.
//
// The camera orbits by transforming rays into object space, so rotating and
// zooming never touch the acceleration structure. Only a parameter change or a
// morph frame rebuilds geometry, via an in-place refit.
//
// Hardware RT runs on the GPU's ray-tracing units on Apple silicon M3 / A17 Pro
// and later; on earlier GPUs the same code runs with software BVH traversal.
//
// Build (single file, no Xcode project):
//   swiftc supershape-metal.swift -o supershape-metal \
//          -framework Cocoa -framework Metal -framework MetalKit
//   ./supershape-metal
//
// Controls:
//   Mouse drag — rotate
//   Scroll     — zoom
//   [ / ]      — previous / next preset
//   Space      — toggle looping morph through presets
//   B          — toggle boomerang morph (forward, then back)
//   , / .      — index of refraction down / up
//   M / N      — symmetry m up / down
//   Q / A      — exponent n1 down / up
//   W / S      — exponent n2 down / up
//   E / D      — exponent n3 down / up
//   R          — reset
//   F          — fullscreen
//   ESC        — quit

import Cocoa
import MetalKit
import simd

// ============================================================
// MARK: - Small simd rotation helpers (same conventions as the raster build)
// ============================================================

func rot3x(_ a: Float) -> simd_float3x3 {
    let c = cosf(a), s = sinf(a)
    return simd_float3x3(columns: (SIMD3<Float>(1, 0, 0),
                                   SIMD3<Float>(0, c, s),
                                   SIMD3<Float>(0, -s, c)))
}

func rot3y(_ a: Float) -> simd_float3x3 {
    let c = cosf(a), s = sinf(a)
    return simd_float3x3(columns: (SIMD3<Float>(c, 0, -s),
                                   SIMD3<Float>(0, 1, 0),
                                   SIMD3<Float>(s, 0, c)))
}

// ============================================================
// MARK: - Uniforms (memory layout MUST match the MSL `RTUniforms` struct)
// ============================================================

struct RTUniforms {
    var invRot: simd_float3x3    // 48 bytes: object-space ray rotation (Rᵀ)
    var camPos: SIMD3<Float>     // 16: camera position in object space
    var lightDir: SIMD3<Float>   // 16
    var time: Float
    var tanHalfFov: Float
    var aspect: Float
    var ior: Float
}

// ============================================================
// MARK: - Supershape parameters
// ============================================================

struct SuperParams {
    var m1: Float, a1: Float, b1: Float, c1: Float   // longitude ring: m, n1, n2, n3
    var m2: Float, a2: Float, b2: Float, c2: Float   // latitude ring
}

func lerp(_ x: SuperParams, _ y: SuperParams, _ t: Float) -> SuperParams {
    func l(_ a: Float, _ b: Float) -> Float { a + (b - a) * t }
    return SuperParams(
        m1: l(x.m1, y.m1), a1: l(x.a1, y.a1), b1: l(x.b1, y.b1), c1: l(x.c1, y.c1),
        m2: l(x.m2, y.m2), a2: l(x.a2, y.a2), b2: l(x.b2, y.b2), c2: l(x.c2, y.c2)
    )
}

let presets: [(name: String, p: SuperParams)] = [
    ("sphere",   SuperParams(m1: 0,  a1: 1,   b1: 1,   c1: 1,    m2: 0,  a2: 1,   b2: 1,   c2: 1)),
    ("star",     SuperParams(m1: 5,  a1: 2,   b1: 7,   c1: 7,    m2: 5,  a2: 2,   b2: 7,   c2: 7)),
    ("gem",      SuperParams(m1: 8,  a1: 1,   b1: 1,   c1: 8,    m2: 8,  a2: 1,   b2: 1,   c2: 8)),
    ("flower",   SuperParams(m1: 7,  a1: 0.2, b1: 1.7, c1: 1.7,  m2: 7,  a2: 0.2, b2: 1.7, c2: 1.7)),
    ("round box",SuperParams(m1: 4,  a1: 14,  b1: 15,  c1: 15,   m2: 4,  a2: 14,  b2: 15,  c2: 15)),
    ("spikes",   SuperParams(m1: 12, a1: 0.3, b1: 0.4, c1: 10,   m2: 12, a2: 0.3, b2: 0.4, c2: 10)),
    ("twist",    SuperParams(m1: 6,  a1: 1,   b1: 7,   c1: 8,    m2: 3,  a2: 4,   b2: 10,  c2: 10)),
    ("shell",    SuperParams(m1: 2,  a1: 0.7, b1: 0.3, c1: 0.3,  m2: 6,  a2: 1,   b2: 1,   c2: 1)),
]

// ============================================================
// MARK: - Supershape mesh (interleaved: pos3, normal3, uv2 = 8 floats / 32 bytes)
// ============================================================

@inline(__always)
func superformula(_ angle: Float, _ m: Float, _ n1: Float, _ n2: Float, _ n3: Float) -> Float {
    let t = m * angle * 0.25
    let c = powf(abs(cosf(t)), n2)
    let s = powf(abs(sinf(t)), n3)
    var base = c + s
    if base < 1e-9 { base = 1e-9 }
    let r = powf(base, -1.0 / n1)
    return min(r, 6.0)
}

func buildSupershape(_ p: SuperParams, targetRadius: Float, Nu: Int, Nv: Int)
    -> (verts: [Float], indices: [UInt32])
{
    var verts = [Float](); verts.reserveCapacity((Nu + 1) * (Nv + 1) * 8)
    var indices = [UInt32](); indices.reserveCapacity(Nu * Nv * 6)

    let pi = Float.pi

    func pos(_ theta: Float, _ phi: Float) -> (Float, Float, Float) {
        let r1 = superformula(theta, p.m1, p.a1, p.b1, p.c1)
        let r2 = superformula(phi,   p.m2, p.a2, p.b2, p.c2)
        let x = r1 * cosf(theta) * r2 * cosf(phi)
        let y = r1 * sinf(theta) * r2 * cosf(phi)
        let z = r2 * sinf(phi)
        return (x, y, z)
    }

    for iv in 0...Nv {
        let phi = -0.5 * pi + pi * Float(iv) / Float(Nv)
        for iu in 0...Nu {
            let theta = -pi + 2.0 * pi * Float(iu) / Float(Nu)
            let pp = pos(theta, phi)

            let eps: Float = 1e-4
            let pu1 = pos(theta + eps, phi), pu0 = pos(theta - eps, phi)
            let pv1 = pos(theta, phi + eps), pv0 = pos(theta, phi - eps)
            let dux = (pu1.0 - pu0.0) / (2 * eps), duy = (pu1.1 - pu0.1) / (2 * eps), duz = (pu1.2 - pu0.2) / (2 * eps)
            let dvx = (pv1.0 - pv0.0) / (2 * eps), dvy = (pv1.1 - pv0.1) / (2 * eps), dvz = (pv1.2 - pv0.2) / (2 * eps)
            var nx = duy * dvz - duz * dvy
            var ny = duz * dvx - dux * dvz
            var nz = dux * dvy - duy * dvx
            var nl = sqrtf(nx * nx + ny * ny + nz * nz); if nl < 1e-9 { nl = 1 }
            nx /= nl; ny /= nl; nz /= nl

            verts.append(contentsOf: [pp.0, pp.1, pp.2, nx, ny, nz,
                                      Float(iu) / Float(Nu), Float(iv) / Float(Nv)])
        }
    }

    // Fit to a stable on-screen size so every preset and morph frame frames well.
    var maxR: Float = 1e-6
    var vi = 0
    while vi < verts.count {
        let x = verts[vi], y = verts[vi + 1], z = verts[vi + 2]
        let d = sqrtf(x * x + y * y + z * z)
        if d > maxR { maxR = d }
        vi += 8
    }
    let k = targetRadius / maxR
    vi = 0
    while vi < verts.count {
        verts[vi] *= k; verts[vi + 1] *= k; verts[vi + 2] *= k
        vi += 8
    }

    for iv in 0..<Nv {
        for iu in 0..<Nu {
            let i0 = UInt32(iv * (Nu + 1) + iu)
            let i1 = i0 + 1
            let i2 = i0 + UInt32(Nu + 1)
            let i3 = i2 + 1
            indices.append(contentsOf: [i0, i1, i2,  i1, i3, i2])
        }
    }
    return (verts, indices)
}

// ============================================================
// MARK: - Metal Shading Language source (full-screen triangle + ray-traced glass)
// ============================================================

let shaderSource = """
#include <metal_stdlib>
using namespace metal;
using namespace metal::raytracing;

constant int   MAX_BOUNCES = 8;
constant float PI = 3.14159265;

struct RTUniforms {
    float3x3 invRot;
    float3   camPos;
    float3   lightDir;
    float    time;
    float    tanHalfFov;
    float    aspect;
    float    ior;
};

struct VOut {
    float4 position [[position]];
    float2 ndc;
};

// Full-screen triangle from vertex_id, no vertex buffer needed.
vertex VOut vmain(uint vid [[vertex_id]]) {
    float2 p = float2(float((vid << 1) & 2), float(vid & 2));   // (0,0)(2,0)(0,2)
    VOut o;
    o.position = float4(p * 2.0 - 1.0, 0.0, 1.0);
    o.ndc      = p * 2.0 - 1.0;
    return o;
}

static float3 hsv2rgb(float3 c) {
    float4 K = float4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    float3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

// A simple studio environment so the glass has something to reflect and to be
// seen against. Object-space (it rotates with the body as the camera orbits).
static float3 envColor(float3 d) {
    float up = clamp(d.y * 0.5 + 0.5, 0.0, 1.0);
    float3 c;
    if (d.y > 0.0) c = mix(float3(0.16, 0.18, 0.23), float3(0.03, 0.04, 0.07), up * up);
    else           c = mix(float3(0.16, 0.18, 0.23), float3(0.015, 0.015, 0.025), -d.y);

    float3 L1 = normalize(float3(0.5, 0.9, 0.6));
    c += float3(1.0, 0.98, 0.95) * pow(max(dot(d, L1), 0.0), 140.0) * 2.2;   // key
    float3 L2 = normalize(float3(-0.6, 0.2, -0.7));
    c += float3(0.6, 0.7, 1.0) * pow(max(dot(d, L2), 0.0), 60.0) * 0.7;      // cool rim
    return c;
}

// Exact dielectric Fresnel (unpolarised average).
static float fresnelDielectric(float cosi, float etai, float etat) {
    float sint = etai / etat * sqrt(max(0.0, 1.0 - cosi * cosi));
    if (sint >= 1.0) return 1.0;                       // total internal reflection
    float cost = sqrt(max(0.0, 1.0 - sint * sint));
    float Rs = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost));
    float Rp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost));
    return clamp(0.5 * (Rs * Rs + Rp * Rp), 0.0, 1.0);
}

fragment float4 fmain(VOut in [[stage_in]],
                      constant RTUniforms& u                 [[buffer(0)]],
                      const device float*  vbuf              [[buffer(1)]],
                      const device uint*   ibuf              [[buffer(2)]],
                      primitive_acceleration_structure accel [[buffer(3)]])
{
    // Primary ray: build in view space, rotate into object space.
    float3 viewDir = normalize(float3(in.ndc.x * u.aspect * u.tanHalfFov,
                                      in.ndc.y * u.tanHalfFov,
                                      -1.0));
    float3 dir = normalize(u.invRot * viewDir);
    float3 org = u.camPos;

    intersector<triangle_data> isect;

    float3 color = float3(0.0);
    float3 thru  = float3(1.0);

    for (int b = 0; b < MAX_BOUNCES; ++b) {
        ray r(org, dir, 1e-3, 1e4);

        intersection_result<triangle_data> hit = isect.intersect(r, accel);

        if (hit.type == intersection_type::none) {
            color += thru * envColor(dir);
            break;
        }

        // Fetch + interpolate the triangle's vertex attributes (normal, uv).
        uint prim = hit.primitive_id;
        uint i0 = ibuf[3 * prim + 0], i1 = ibuf[3 * prim + 1], i2 = ibuf[3 * prim + 2];
        float3 n0 = float3(vbuf[i0*8+3], vbuf[i0*8+4], vbuf[i0*8+5]);
        float3 n1 = float3(vbuf[i1*8+3], vbuf[i1*8+4], vbuf[i1*8+5]);
        float3 n2 = float3(vbuf[i2*8+3], vbuf[i2*8+4], vbuf[i2*8+5]);
        float2 t0 = float2(vbuf[i0*8+6], vbuf[i0*8+7]);
        float2 t1 = float2(vbuf[i1*8+6], vbuf[i1*8+7]);
        float2 t2 = float2(vbuf[i2*8+6], vbuf[i2*8+7]);

        float2 bc = hit.triangle_barycentric_coord;
        float  w  = 1.0 - bc.x - bc.y;
        float3 N  = normalize(w * n0 + bc.x * n1 + bc.y * n2);
        float2 uv = w * t0 + bc.x * t1 + bc.y * t2;
        float3 hitPos = org + hit.distance * dir;

        // Coloured-glass tint + grid veins, same look as the raster shader.
        float hue = fract(uv.y + hitPos.y * 0.12 + u.time * 0.06);
        float3 tint = hsv2rgb(float3(hue, 0.85, 1.0));
        float grid = smoothstep(0.96, 1.0, max(abs(sin(uv.x * PI * 48.0)),
                                               abs(sin(uv.y * PI * 48.0))));

        // Orient the interface: decide entering vs exiting the glass.
        float3 I = normalize(dir);
        float ndi = dot(I, N);
        float etai = 1.0, etat = u.ior;
        float3 Nf = N;
        if (ndi > 0.0) { Nf = -N; etai = u.ior; etat = 1.0; }   // exiting
        float cosi = abs(ndi);

        float F = fresnelDielectric(cosi, etai, etat);
        float3 Rdir = reflect(I, Nf);

        // Reflection terminates into the environment (cheap single bounce).
        color += thru * F * envColor(Rdir);

        // A faint internal glow keeps the rainbow readable on a dark background.
        color += thru * (1.0 - F) * tint * 0.12;

        float3 Tdir = refract(I, Nf, etai / etat);
        if (dot(Tdir, Tdir) < 1e-6) {
            // Total internal reflection: continue along the reflected ray.
            org = hitPos + Nf * 1e-3;
            dir = Rdir;
        } else {
            // Transmit: tint by the rainbow, darken along grid veins.
            float3 glassTint = mix(float3(1.0), tint, 0.6);
            glassTint *= mix(1.0, 0.22, grid * 0.85);
            thru *= (1.0 - F) * glassTint;
            org = hitPos - Nf * 1e-3;
            dir = normalize(Tdir);
        }

        if (max(max(thru.x, thru.y), thru.z) < 0.01) break;
    }

    // Mild filmic-ish tone map + gamma so highlights don't clip harshly.
    color = color / (color + 0.8);
    color = pow(color, float3(1.0 / 2.2));
    return float4(color, 1.0);
}
"""

// ============================================================
// MARK: - Renderer
// ============================================================

final class Renderer: NSObject, MTKViewDelegate {
    let device: MTLDevice
    let queue: MTLCommandQueue
    var pipeline: MTLRenderPipelineState!

    // Geometry + acceleration structure
    var vbuf: MTLBuffer!
    var ibuf: MTLBuffer!
    var indexCount = 0
    var accel: MTLAccelerationStructure!
    var primDesc: MTLPrimitiveAccelerationStructureDescriptor!
    var buildScratch: MTLBuffer!
    var refitScratch: MTLBuffer!
    var meshDirty = false

    // Camera
    var rotX: Float = 0.4, rotY: Float = 0.6, zoom: Float = 4.6
    var ior: Float = 1.5

    // Shape
    var params = presets[2].p          // start on "gem" — reads nicely as glass
    var presetIndex = 2
    let targetRadius: Float = 1.8
    let Nu = 200, Nv = 140

    // Morph
    var morphing = false
    var pingPong = false
    var morphU: Float = 0
    var morphDir: Float = 1
    let morphDuration: Float = 3.0

    var time: Float = 0
    var lastTime = ProcessInfo.processInfo.systemUptime

    init?(mtkView: MTKView) {
        guard let dev = mtkView.device, let q = dev.makeCommandQueue() else { return nil }
        guard dev.supportsRaytracing else {
            FileHandle.standardError.write("This GPU does not support Metal ray tracing.\n".data(using: .utf8)!)
            return nil
        }
        device = dev
        queue = q
        super.init()

        mtkView.colorPixelFormat = .bgra8Unorm
        mtkView.depthStencilPixelFormat = .invalid          // no depth: we trace everything
        mtkView.clearColor = MTLClearColor(red: 0.0, green: 0.0, blue: 0.0, alpha: 1.0)
        mtkView.isPaused = false
        mtkView.enableSetNeedsDisplay = false

        do {
            try buildPipeline(view: mtkView)
        } catch {
            FileHandle.standardError.write("Pipeline error: \(error)\n".data(using: .utf8)!)
            return nil
        }
        regenerateMesh()                                    // first build also builds the AS
    }

    func buildPipeline(view: MTKView) throws {
        let library = try device.makeLibrary(source: shaderSource, options: nil)
        guard let vfn = library.makeFunction(name: "vmain"),
              let ffn = library.makeFunction(name: "fmain") else {
            throw NSError(domain: "Renderer", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "shader functions not found"])
        }
        let desc = MTLRenderPipelineDescriptor()
        desc.vertexFunction = vfn
        desc.fragmentFunction = ffn
        desc.colorAttachments[0].pixelFormat = view.colorPixelFormat
        pipeline = try device.makeRenderPipelineState(descriptor: desc)
    }

    // (Re)generate the mesh. First call allocates buffers and builds the AS;
    // later calls overwrite vertex positions in place and flag a refit.
    func regenerateMesh() {
        let (verts, indices) = buildSupershape(params, targetRadius: targetRadius, Nu: Nu, Nv: Nv)
        if vbuf == nil {
            indexCount = indices.count
            vbuf = device.makeBuffer(bytes: verts,
                                     length: verts.count * MemoryLayout<Float>.stride,
                                     options: .storageModeShared)
            ibuf = device.makeBuffer(bytes: indices,
                                     length: indices.count * MemoryLayout<UInt32>.stride,
                                     options: .storageModeShared)
            buildAccelerationStructure()
        } else {
            memcpy(vbuf.contents(), verts, verts.count * MemoryLayout<Float>.stride)
            meshDirty = true
        }
    }

    func buildAccelerationStructure() {
        let geom = MTLAccelerationStructureTriangleGeometryDescriptor()
        geom.vertexBuffer = vbuf
        geom.vertexBufferOffset = 0
        geom.vertexStride = 32
        geom.vertexFormat = .float3
        geom.indexBuffer = ibuf
        geom.indexBufferOffset = 0
        geom.indexType = .uint32
        geom.triangleCount = indexCount / 3
        geom.opaque = true

        let pd = MTLPrimitiveAccelerationStructureDescriptor()
        pd.geometryDescriptors = [geom]
        primDesc = pd

        let sizes = device.accelerationStructureSizes(descriptor: pd)
        accel = device.makeAccelerationStructure(size: sizes.accelerationStructureSize)
        buildScratch = device.makeBuffer(length: max(sizes.buildScratchBufferSize, 16),
                                         options: .storageModePrivate)
        refitScratch = device.makeBuffer(length: max(sizes.refitScratchBufferSize, 16),
                                         options: .storageModePrivate)

        guard let cmd = queue.makeCommandBuffer(),
              let enc = cmd.makeAccelerationStructureCommandEncoder() else { return }
        enc.build(accelerationStructure: accel, descriptor: pd,
                  scratchBuffer: buildScratch, scratchBufferOffset: 0)
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        let now = ProcessInfo.processInfo.systemUptime
        var dt = Float(now - lastTime); lastTime = now
        if dt > 0.05 { dt = 0.05 }
        time += dt

        // Live morph (loop or boomerang), rebuilding the mesh each frame.
        if morphing {
            let count = presets.count
            morphU += morphDir * dt / morphDuration
            if pingPong {
                let last = Float(count - 1)
                if morphU >= last { morphU = last; morphDir = -1 }
                if morphU <= 0    { morphU = 0;    morphDir =  1 }
                let i = min(Int(morphU), count - 2)
                let f = morphU - Float(i)
                params = lerp(presets[i].p, presets[i + 1].p, f * f * (3 - 2 * f))
            } else {
                let c = Float(count)
                morphU = morphU.truncatingRemainder(dividingBy: c)
                if morphU < 0 { morphU += c }
                let i = Int(morphU) % count
                let f = morphU - floorf(morphU)
                params = lerp(presets[i].p, presets[(i + 1) % count].p, f * f * (3 - 2 * f))
            }
            regenerateMesh()
        }

        let size = view.drawableSize
        let aspect = Float(size.width) / Float(max(size.height, 1))

        // Object-space camera: rotate rays by Rᵀ, place the eye at Rᵀ·(0,0,zoom).
        let R = rot3x(rotX) * rot3y(rotY)
        let invRot = R.transpose
        let camPos = invRot * SIMD3<Float>(0, 0, zoom)

        var uni = RTUniforms(invRot: invRot,
                             camPos: camPos,
                             lightDir: normalize(SIMD3<Float>(0.5, 0.9, 0.6)),
                             time: time,
                             tanHalfFov: tanf(0.8 * 0.5),
                             aspect: aspect,
                             ior: ior)

        guard let cmd = queue.makeCommandBuffer() else { return }

        // Refit the acceleration structure if the geometry moved this frame.
        if meshDirty {
            if let aenc = cmd.makeAccelerationStructureCommandEncoder() {
                aenc.refit(sourceAccelerationStructure: accel,
                           descriptor: primDesc,
                           destinationAccelerationStructure: accel,
                           scratchBuffer: refitScratch,
                           scratchBufferOffset: 0)
                aenc.endEncoding()
            }
            meshDirty = false
        }

        guard let rpd = view.currentRenderPassDescriptor,
              let drawable = view.currentDrawable,
              let enc = cmd.makeRenderCommandEncoder(descriptor: rpd) else {
            cmd.commit(); return
        }

        enc.setRenderPipelineState(pipeline)
        enc.setFragmentBytes(&uni, length: MemoryLayout<RTUniforms>.stride, index: 0)
        enc.setFragmentBuffer(vbuf, offset: 0, index: 1)
        enc.setFragmentBuffer(ibuf, offset: 0, index: 2)
        enc.setFragmentAccelerationStructure(accel, bufferIndex: 3)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        enc.endEncoding()

        cmd.present(drawable)
        cmd.commit()
    }
}

// ============================================================
// MARK: - View (input handling)
// ============================================================

final class ShapeMTKView: MTKView {
    weak var renderer: Renderer?
    var dragging = false
    var lastMouse = NSPoint.zero

    override var acceptsFirstResponder: Bool { true }

    override func mouseDown(with e: NSEvent) {
        dragging = true
        lastMouse = e.locationInWindow
    }

    override func mouseDragged(with e: NSEvent) {
        guard dragging, let r = renderer else { return }
        let p = e.locationInWindow
        r.rotY += Float(p.x - lastMouse.x) * 0.008
        r.rotX -= Float(p.y - lastMouse.y) * 0.008
        lastMouse = p
    }

    override func mouseUp(with e: NSEvent) { dragging = false }

    override func scrollWheel(with e: NSEvent) {
        guard let r = renderer else { return }
        r.zoom -= Float(e.scrollingDeltaY) * 0.01
        r.zoom = min(max(r.zoom, 2.0), 14.0)
    }

    // Manual exponent tweaks apply to both rings and stop any morph.
    func bumpBoth(_ r: Renderer, _ f: (inout SuperParams) -> Void) {
        r.morphing = false
        f(&r.params)
        r.regenerateMesh()
        updateTitle(r)
    }

    override func keyDown(with e: NSEvent) {
        guard let r = renderer, let ch = e.charactersIgnoringModifiers?.lowercased().first else { return }
        switch ch {
        case "\u{1b}": NSApp.terminate(nil)
        case "[":
            r.morphing = false
            r.presetIndex = (r.presetIndex - 1 + presets.count) % presets.count
            r.params = presets[r.presetIndex].p; r.regenerateMesh(); updateTitle(r)
        case "]":
            r.morphing = false
            r.presetIndex = (r.presetIndex + 1) % presets.count
            r.params = presets[r.presetIndex].p; r.regenerateMesh(); updateTitle(r)
        case " ":
            if r.morphing && !r.pingPong {
                r.morphing = false
                r.presetIndex = max(0, min(Int(r.morphU.rounded()), presets.count - 1))
            } else {
                r.morphing = true; r.pingPong = false
                r.morphU = Float(r.presetIndex); r.morphDir = 1
            }
            updateTitle(r)
        case "b":
            if r.morphing && r.pingPong {
                r.morphing = false
                r.presetIndex = max(0, min(Int(r.morphU.rounded()), presets.count - 1))
            } else {
                r.morphing = true; r.pingPong = true
                r.morphU = Float(r.presetIndex); r.morphDir = 1
            }
            updateTitle(r)
        case ",": r.ior = max(1.0, r.ior - 0.05); updateTitle(r)
        case ".": r.ior = min(2.4, r.ior + 0.05); updateTitle(r)
        case "m": bumpBoth(r) { $0.m1 = min(20, $0.m1 + 1); $0.m2 = min(20, $0.m2 + 1) }
        case "n": bumpBoth(r) { $0.m1 = max(0,  $0.m1 - 1); $0.m2 = max(0,  $0.m2 - 1) }
        case "q": bumpBoth(r) { $0.a1 = max(0.1, $0.a1 - 0.3); $0.a2 = max(0.1, $0.a2 - 0.3) }
        case "a": bumpBoth(r) { $0.a1 = min(40,  $0.a1 + 0.3); $0.a2 = min(40,  $0.a2 + 0.3) }
        case "w": bumpBoth(r) { $0.b1 = max(0.1, $0.b1 - 0.5); $0.b2 = max(0.1, $0.b2 - 0.5) }
        case "s": bumpBoth(r) { $0.b1 = min(40,  $0.b1 + 0.5); $0.b2 = min(40,  $0.b2 + 0.5) }
        case "e": bumpBoth(r) { $0.c1 = max(0.1, $0.c1 - 0.5); $0.c2 = max(0.1, $0.c2 - 0.5) }
        case "d": bumpBoth(r) { $0.c1 = min(40,  $0.c1 + 0.5); $0.c2 = min(40,  $0.c2 + 0.5) }
        case "r":
            r.morphing = false; r.presetIndex = 2; r.params = presets[2].p
            r.rotX = 0.4; r.rotY = 0.6; r.zoom = 4.6; r.ior = 1.5
            r.regenerateMesh(); updateTitle(r)
        case "f": window?.toggleFullScreen(nil)
        default: break
        }
    }

    func updateTitle(_ r: Renderer) {
        let tag = r.morphing ? (r.pingPong ? "boomerang" : "loop") : presets[r.presetIndex].name
        window?.title = String(format:
            "Supershape RT glass (Metal)  |  %@  ior=%.2f m=%.0f n1=%.1f n2=%.1f n3=%.1f  |  [ ] preset  Space loop  B boomerang  ,/. ior  R reset  F full  ESC quit",
            tag, r.ior, r.params.m1, r.params.a1, r.params.b1, r.params.c1)
    }
}

// ============================================================
// MARK: - Application entry point
// ============================================================

final class AppDelegate: NSObject, NSApplicationDelegate, NSWindowDelegate {
    var window: NSWindow!
    var view: ShapeMTKView!
    var renderer: Renderer!

    func applicationDidFinishLaunching(_ notification: Notification) {
        guard let device = MTLCreateSystemDefaultDevice() else {
            FileHandle.standardError.write("Metal is not supported on this machine.\n".data(using: .utf8)!)
            NSApp.terminate(nil); return
        }

        let style: NSWindow.StyleMask = [.titled, .closable, .resizable, .miniaturizable]
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1280, height: 720),
                          styleMask: style, backing: .buffered, defer: false)
        window.center()
        window.title = "Supershape RT glass (Metal)"
        window.delegate = self

        let v = ShapeMTKView(frame: window.contentView!.bounds, device: device)
        guard let r = Renderer(mtkView: v) else {
            FileHandle.standardError.write("Failed to create the Metal ray-tracing renderer.\n".data(using: .utf8)!)
            NSApp.terminate(nil); return
        }
        v.delegate = r
        v.renderer = r
        renderer = r
        view = v
        view.autoresizingMask = [.width, .height]

        window.contentView = view
        window.makeFirstResponder(view)
        window.makeKeyAndOrderFront(nil)
        v.updateTitle(r)
        NSApp.activate(ignoringOtherApps: true)
    }

    func windowWillClose(_ notification: Notification) {
        NSApp.terminate(nil)
    }
}

let app = NSApplication.shared
app.setActivationPolicy(.regular)
let delegate = AppDelegate()
app.delegate = delegate
app.run()
