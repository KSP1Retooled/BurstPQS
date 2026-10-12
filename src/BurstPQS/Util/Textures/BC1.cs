using Unity.Mathematics;

namespace BurstPQS.Util.Textures;

internal static unsafe class BC1
{
    /// <summary>Encodes the RGB of 16 pixels in [0, 255] as one BC1 block in 4-colour mode.</summary>
    public static ulong Encode(float4* px)
    {
        float3 mean = 0f;
        for (int i = 0; i < 16; i++)
            mean += px[i].xyz;
        mean /= 16f;

        float3x3 cov = default;
        for (int i = 0; i < 16; i++)
        {
            float3 d = px[i].xyz - mean;
            cov.c0 += d * d.x;
            cov.c1 += d * d.y;
            cov.c2 += d * d.z;
        }

        float3 axis = BlockMath.PrincipalAxis(cov);

        float minT = float.PositiveInfinity;
        float maxT = float.NegativeInfinity;
        for (int i = 0; i < 16; i++)
        {
            float t = math.dot(px[i].xyz - mean, axis);
            minT = math.min(minT, t);
            maxT = math.max(maxT, t);
        }

        uint q0 = Quantize565(mean + axis * maxT);
        uint q1 = Quantize565(mean + axis * minT);

        float best = Fit(px, q0, q1, out uint bestIndices);
        uint best0 = q0;
        uint best1 = q1;

        for (int iter = 0; iter < BlockMath.RefineIterations; iter++)
        {
            float a = 0f,
                b = 0f,
                c = 0f;
            float3 d0 = 0f,
                d1 = 0f;
            for (int i = 0; i < 16; i++)
            {
                uint index = (bestIndices >> (i * 2)) & 3;
                float w = index switch
                {
                    0 => 0f,
                    1 => 1f,
                    2 => 1f / 3f,
                    _ => 2f / 3f,
                };
                float u = 1f - w;

                a += u * u;
                b += u * w;
                c += w * w;
                d0 += u * px[i].xyz;
                d1 += w * px[i].xyz;
            }

            float det = a * c - b * b;
            if (math.abs(det) < 1e-6f)
                break;

            float inv = 1f / det;
            q0 = Quantize565((c * d0 - b * d1) * inv);
            q1 = Quantize565((a * d1 - b * d0) * inv);

            float error = Fit(px, q0, q1, out uint indices);
            if (error >= best)
                break;

            best = error;
            best0 = q0;
            best1 = q1;
            bestIndices = indices;
        }

        return Pack(best0, best1, bestIndices);
    }

    /// <summary>
    /// Encodes 16 values in [0, 255] as a grey BC1 block. Only green is fitted: it is the
    /// channel with 6 bits, and the one a DXT5nm decoder reads.
    /// </summary>
    public static ulong EncodeGrey(float* v)
    {
        float lo = v[0];
        float hi = v[0];
        for (int i = 1; i < 16; i++)
        {
            lo = math.min(lo, v[i]);
            hi = math.max(hi, v[i]);
        }

        int g0 = Quantize6(hi);
        int g1 = Quantize6(lo);

        float best = FitGrey(v, g0, g1, out uint bestIndices);
        int best0 = g0;
        int best1 = g1;

        for (int iter = 0; iter < BlockMath.RefineIterations; iter++)
        {
            float a = 0f,
                b = 0f,
                c = 0f,
                d0 = 0f,
                d1 = 0f;
            for (int i = 0; i < 16; i++)
            {
                uint index = (bestIndices >> (i * 2)) & 3;
                float w = index switch
                {
                    0 => 0f,
                    1 => 1f,
                    2 => 1f / 3f,
                    _ => 2f / 3f,
                };
                float u = 1f - w;

                a += u * u;
                b += u * w;
                c += w * w;
                d0 += u * v[i];
                d1 += w * v[i];
            }

            float det = a * c - b * b;
            if (math.abs(det) < 1e-6f)
                break;

            float inv = 1f / det;
            g0 = Quantize6((c * d0 - b * d1) * inv);
            g1 = Quantize6((a * d1 - b * d0) * inv);

            float error = FitGrey(v, g0, g1, out uint indices);
            if (error >= best)
                break;

            best = error;
            best0 = g0;
            best1 = g1;
            bestIndices = indices;
        }

        return Pack(GreyTo565(best0), GreyTo565(best1), bestIndices);
    }

    static float Fit(float4* px, uint q0, uint q1, out uint indices)
    {
        float3 c0 = Expand565(q0);
        float3 c1 = Expand565(q1);
        float3 c2 = (2f * c0 + c1) / 3f;
        float3 c3 = (c0 + 2f * c1) / 3f;

        float error = 0f;
        indices = 0;

        for (int i = 0; i < 16; i++)
        {
            float3 p = px[i].xyz;
            float4 dist = new(
                math.distancesq(p, c0),
                math.distancesq(p, c1),
                math.distancesq(p, c2),
                math.distancesq(p, c3)
            );

            uint index = NearestOf4(dist, out float d);
            error += d;
            indices |= index << (i * 2);
        }

        return error;
    }

    static float FitGrey(float* v, int g0, int g1, out uint indices)
    {
        float c0 = Expand6(g0);
        float c1 = Expand6(g1);
        float4 palette = new(c0, c1, (2f * c0 + c1) / 3f, (c0 + 2f * c1) / 3f);

        float error = 0f;
        indices = 0;

        for (int i = 0; i < 16; i++)
        {
            float4 diff = palette - v[i];
            uint index = NearestOf4(diff * diff, out float d);
            error += d;
            indices |= index << (i * 2);
        }

        return error;
    }

    static uint NearestOf4(float4 dist, out float min)
    {
        uint index = 0;
        min = dist.x;
        if (dist.y < min)
        {
            min = dist.y;
            index = 1;
        }
        if (dist.z < min)
        {
            min = dist.z;
            index = 2;
        }
        if (dist.w < min)
        {
            min = dist.w;
            index = 3;
        }
        return index;
    }

    // In 4-colour mode the first endpoint must compare greater. Swapping the endpoints swaps
    // index 0 with 1 and 2 with 3, which is flipping each index's low bit.
    static ulong Pack(uint q0, uint q1, uint indices)
    {
        if (q0 == q1)
            indices = 0;
        else if (q0 < q1)
        {
            (q0, q1) = (q1, q0);
            indices ^= 0x55555555u;
        }

        return q0 | (ulong)q1 << 16 | (ulong)indices << 32;
    }

    static uint Quantize565(float3 c)
    {
        uint r = (uint)math.clamp((int)math.round(c.x * (31f / 255f)), 0, 31);
        uint g = (uint)math.clamp((int)math.round(c.y * (63f / 255f)), 0, 63);
        uint b = (uint)math.clamp((int)math.round(c.z * (31f / 255f)), 0, 31);
        return r << 11 | g << 5 | b;
    }

    static float3 Expand565(uint q)
    {
        uint r = (q >> 11) & 31;
        uint g = (q >> 5) & 63;
        uint b = q & 31;
        return new float3(r << 3 | r >> 2, g << 2 | g >> 4, b << 3 | b >> 2);
    }

    static int Quantize6(float v) => math.clamp((int)math.round(v * (63f / 255f)), 0, 63);

    static float Expand6(int g) => g << 2 | g >> 4;

    // Red and blue follow green as closely as their 5 bits allow. Rounding is monotonic, so
    // the packed values order the same way the green values do.
    static uint GreyTo565(int g)
    {
        uint rb = (uint)math.clamp((int)math.round(Expand6(g) * (31f / 255f)), 0, 31);
        return rb << 11 | (uint)g << 5 | rb;
    }
}
