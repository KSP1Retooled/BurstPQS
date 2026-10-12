using Unity.Mathematics;

namespace BurstPQS.Util.Textures;

/// <summary>
/// BC7 encoder covering modes 1, 3, 6 and 7, the same set texconv tries by default apart from
/// the separate-alpha modes 4 and 5.
/// </summary>
/// <remarks>
/// Like texconv, the two-subset modes only get a full fit for the partitions that rank best on
/// a rough estimate, and mode 7 is skipped for opaque blocks since 1 and 3 cover those.
/// </remarks>
internal static unsafe class BC7
{
    const int ShapeCandidates = 8;

    struct Mode
    {
        public int number;
        public int subsets;
        public int colorBits;

        /// <summary>0 for the modes without alpha, which decode it as 255.</summary>
        public int alphaBits;
        public int indexBits;
        public bool sharedPBit;
    }

    static Mode Mode1 =>
        new()
        {
            number = 1,
            subsets = 2,
            colorBits = 6,
            indexBits = 3,
            sharedPBit = true,
        };

    static Mode Mode3 =>
        new()
        {
            number = 3,
            subsets = 2,
            colorBits = 7,
            indexBits = 2,
        };

    static Mode Mode6 =>
        new()
        {
            number = 6,
            subsets = 1,
            colorBits = 7,
            alphaBits = 7,
            indexBits = 4,
        };

    static Mode Mode7 =>
        new()
        {
            number = 7,
            subsets = 2,
            colorBits = 5,
            alphaBits = 5,
            indexBits = 2,
        };

    struct Endpoint
    {
        public int4 bits;
        public int pbit;
        public int4 value;
    }

    struct Candidate
    {
        public Mode mode;
        public int shape;
        public float error;

        public Endpoint start0;
        public Endpoint end0;
        public Endpoint start1;
        public Endpoint end1;

        /// <summary>4 bits per pixel, whatever the mode's index size.</summary>
        public ulong indices;
    }

    /// <summary>Encodes 16 RGBA pixels in [0, 255] as one BC7 block.</summary>
    public static Block128 Encode(float4* px)
    {
        bool opaque = true;
        for (int i = 0; i < 16; i++)
            opaque &= px[i].w >= 255f;

        var best = EncodeShape(px, Mode6, 0);
        if (best.error <= 0f)
            return Pack(best);

        var shapes = stackalloc int[ShapeCandidates];
        RankShapes(px, shapes);

        for (int i = 0; i < ShapeCandidates; i++)
        {
            var candidate = EncodeShape(px, Mode1, shapes[i]);
            if (candidate.error < best.error)
                best = candidate;

            candidate = EncodeShape(px, Mode3, shapes[i]);
            if (candidate.error < best.error)
                best = candidate;

            if (opaque)
                continue;

            candidate = EncodeShape(px, Mode7, shapes[i]);
            if (candidate.error < best.error)
                best = candidate;
        }

        return Pack(best);
    }

    // Picks the partitions whose subsets lie closest to a line each, measured as the variance
    // left over once the principal axis is taken out.
    static void RankShapes(float4* px, int* shapes)
    {
        var scores = stackalloc float[ShapeCandidates];
        for (int i = 0; i < ShapeCandidates; i++)
            scores[i] = float.PositiveInfinity;

        for (int shape = 0; shape < PartitionCount; shape++)
        {
            int mask = PartitionMasks[shape];
            float score = OffAxisVariance(px, ~mask & 0xFFFF) + OffAxisVariance(px, mask);

            // Insertion into the sorted candidate list.
            int slot = ShapeCandidates;
            while (slot > 0 && score < scores[slot - 1])
            {
                if (slot < ShapeCandidates)
                {
                    scores[slot] = scores[slot - 1];
                    shapes[slot] = shapes[slot - 1];
                }
                slot--;
            }

            if (slot < ShapeCandidates)
            {
                scores[slot] = score;
                shapes[slot] = shape;
            }
        }
    }

    static float OffAxisVariance(float4* px, int pixels)
    {
        Moments(px, pixels, out _, out float4x4 cov);

        float4 axis = BlockMath.PrincipalAxis(cov);
        float trace = cov.c0.x + cov.c1.y + cov.c2.z + cov.c3.w;
        return trace - math.dot(axis, math.mul(cov, axis));
    }

    static void Moments(float4* px, int pixels, out float4 mean, out float4x4 cov)
    {
        mean = 0f;
        int count = 0;
        for (int i = 0; i < 16; i++)
        {
            if ((pixels & (1 << i)) == 0)
                continue;

            mean += px[i];
            count++;
        }
        mean /= count;

        cov = default;
        for (int i = 0; i < 16; i++)
        {
            if ((pixels & (1 << i)) == 0)
                continue;

            float4 d = px[i] - mean;
            cov.c0 += d * d.x;
            cov.c1 += d * d.y;
            cov.c2 += d * d.z;
            cov.c3 += d * d.w;
        }
    }

    static Candidate EncodeShape(float4* px, Mode mode, int shape)
    {
        var candidate = new Candidate { mode = mode, shape = shape };

        if (mode.subsets == 1)
        {
            candidate.error = FitSubset(
                px,
                0xFFFF,
                mode,
                out candidate.start0,
                out candidate.end0,
                ref candidate.indices
            );
            return candidate;
        }

        int mask = PartitionMasks[shape];
        candidate.error =
            FitSubset(
                px,
                ~mask & 0xFFFF,
                mode,
                out candidate.start0,
                out candidate.end0,
                ref candidate.indices
            )
            + FitSubset(
                px,
                mask,
                mode,
                out candidate.start1,
                out candidate.end1,
                ref candidate.indices
            );

        return candidate;
    }

    // Fits one subset's endpoints along its principal axis, then refines them with
    // least-squares passes over the chosen indices. Writes the subset's pixels into indices.
    static float FitSubset(
        float4* px,
        int pixels,
        Mode mode,
        out Endpoint start,
        out Endpoint end,
        ref ulong indices
    )
    {
        // Modes without alpha always decode it as 255, so the fit only aims for that.
        var target = stackalloc float4[16];
        for (int i = 0; i < 16; i++)
            target[i] = mode.alphaBits == 0 ? new float4(px[i].xyz, 255f) : px[i];

        Moments(target, pixels, out float4 mean, out float4x4 cov);
        float4 axis = BlockMath.PrincipalAxis(cov);

        float minT = float.PositiveInfinity;
        float maxT = float.NegativeInfinity;
        for (int i = 0; i < 16; i++)
        {
            if ((pixels & (1 << i)) == 0)
                continue;

            float t = math.dot(target[i] - mean, axis);
            minT = math.min(minT, t);
            maxT = math.max(maxT, t);
        }

        QuantizePair(mean + axis * minT, mean + axis * maxT, mode, out start, out end);
        float best = Assign(px, pixels, mode, start, end, out ulong bestIndices);

        int count = 1 << mode.indexBits;
        for (int iter = 0; iter < BlockMath.RefineIterations && best > 0f; iter++)
        {
            float a = 0f,
                b = 0f,
                c = 0f;
            float4 d0 = 0f,
                d1 = 0f;
            for (int i = 0; i < 16; i++)
            {
                if ((pixels & (1 << i)) == 0)
                    continue;

                float w = Weight((int)(bestIndices >> (i * 4)) & 15, count) / 64f;
                float u = 1f - w;

                a += u * u;
                b += u * w;
                c += w * w;
                d0 += u * target[i];
                d1 += w * target[i];
            }

            float det = a * c - b * b;
            if (math.abs(det) < 1e-6f)
                break;

            float inv = 1f / det;
            QuantizePair(
                (c * d0 - b * d1) * inv,
                (a * d1 - b * d0) * inv,
                mode,
                out var newStart,
                out var newEnd
            );

            float error = Assign(px, pixels, mode, newStart, newEnd, out ulong newIndices);
            if (error >= best)
                break;

            best = error;
            start = newStart;
            end = newEnd;
            bestIndices = newIndices;
        }

        indices |= bestIndices;
        return best;
    }

    // Picks the nearest palette entry for each pixel in the subset.
    static float Assign(
        float4* px,
        int pixels,
        Mode mode,
        Endpoint start,
        Endpoint end,
        out ulong indices
    )
    {
        int count = 1 << mode.indexBits;
        var palette = stackalloc float4[16];
        for (int i = 0; i < count; i++)
        {
            int w = Weight(i, count);
            palette[i] = ((64 - w) * start.value + w * end.value + 32) >> 6;
        }

        float error = 0f;
        indices = 0;

        for (int i = 0; i < 16; i++)
        {
            if ((pixels & (1 << i)) == 0)
                continue;

            float4 p = px[i];
            float min = math.distancesq(p, palette[0]);
            ulong index = 0;

            for (int j = 1; j < count; j++)
            {
                float d = math.distancesq(p, palette[j]);
                if (d < min)
                {
                    min = d;
                    index = (ulong)j;
                }
            }

            error += min;
            indices |= index << (i * 4);
        }

        return error;
    }

    // 2 bits: 0, 21, 43, 64
    // 3 bits: 0, 9, 18, 27, 37, 46, 55, 64
    // 4 bits: 0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64
    static int Weight(int index, int count) => (index * 64 + (count - 1) / 2) / (count - 1);

    static void QuantizePair(float4 a, float4 b, Mode mode, out Endpoint start, out Endpoint end)
    {
        if (mode.sharedPBit)
        {
            var start0 = Quantize(a, mode, 0, out float startErr0);
            var end0 = Quantize(b, mode, 0, out float endErr0);
            var start1 = Quantize(a, mode, 1, out float startErr1);
            var end1 = Quantize(b, mode, 1, out float endErr1);

            bool one = startErr1 + endErr1 < startErr0 + endErr0;
            start = one ? start1 : start0;
            end = one ? end1 : end0;
            return;
        }

        start = Quantize(a, mode);
        end = Quantize(b, mode);
    }

    static Endpoint Quantize(float4 e, Mode mode)
    {
        var e0 = Quantize(e, mode, 0, out float err0);
        var e1 = Quantize(e, mode, 1, out float err1);
        return err1 < err0 ? e1 : e0;
    }

    // Each channel is stored as its top bits plus the p-bit as the lowest, and the decoder
    // replicates the high bits of that into whatever is left of 8.
    static Endpoint Quantize(float4 e, Mode mode, int pbit, out float error)
    {
        e = math.clamp(e, 0f, 255f);

        var endpoint = new Endpoint { pbit = pbit };
        error = 0f;

        for (int ch = 0; ch < 4; ch++)
        {
            int bits = ch < 3 ? mode.colorBits : mode.alphaBits;
            if (bits == 0)
            {
                endpoint.value[ch] = 255;
                continue;
            }

            int max = (1 << bits) - 1;
            int guess = (int)math.round((e[ch] * ((2 << bits) - 1) / 255f - pbit) * 0.5f);

            int bestBits = 0;
            int bestValue = 0;
            float bestErr = float.PositiveInfinity;

            for (int c = math.max(guess - 1, 0); c <= math.min(guess + 1, max); c++)
            {
                int value = Expand(c << 1 | pbit, bits + 1);
                float d = value - e[ch];
                if (d * d < bestErr)
                {
                    bestErr = d * d;
                    bestBits = c;
                    bestValue = value;
                }
            }

            endpoint.bits[ch] = bestBits;
            endpoint.value[ch] = bestValue;
            error += bestErr;
        }

        return endpoint;
    }

    static int Expand(int value, int bits) => value << (8 - bits) | value >> (2 * bits - 8);

    static Block128 Pack(Candidate c)
    {
        var mode = c.mode;
        int count = 1 << mode.indexBits;
        int mask = mode.subsets == 2 ? PartitionMasks[c.shape] : 0;
        int anchor1 = mode.subsets == 2 ? PartitionAnchors[c.shape] : -1;

        // Each subset's anchor is stored without its top bit, so it has to be in the lower
        // half. Swapping a subset's endpoints turns its index i into count - 1 - i.
        if (Index(c.indices, 0) >= count / 2)
        {
            (c.start0, c.end0) = (c.end0, c.start0);
            c.indices = Invert(c.indices, ~mask & 0xFFFF, count);
        }

        if (anchor1 >= 0 && Index(c.indices, anchor1) >= count / 2)
        {
            (c.start1, c.end1) = (c.end1, c.start1);
            c.indices = Invert(c.indices, mask, count);
        }

        var writer = new BitWriter();
        writer.Write(1ul << mode.number, mode.number + 1);
        if (mode.subsets == 2)
            writer.Write((ulong)c.shape, 6);

        int channels = mode.alphaBits > 0 ? 4 : 3;
        for (int ch = 0; ch < channels; ch++)
        {
            int bits = ch < 3 ? mode.colorBits : mode.alphaBits;

            writer.Write((ulong)c.start0.bits[ch], bits);
            writer.Write((ulong)c.end0.bits[ch], bits);
            if (mode.subsets == 2)
            {
                writer.Write((ulong)c.start1.bits[ch], bits);
                writer.Write((ulong)c.end1.bits[ch], bits);
            }
        }

        if (mode.sharedPBit)
        {
            writer.Write((ulong)c.start0.pbit, 1);
            writer.Write((ulong)c.start1.pbit, 1);
        }
        else
        {
            writer.Write((ulong)c.start0.pbit, 1);
            writer.Write((ulong)c.end0.pbit, 1);
            if (mode.subsets == 2)
            {
                writer.Write((ulong)c.start1.pbit, 1);
                writer.Write((ulong)c.end1.pbit, 1);
            }
        }

        for (int i = 0; i < 16; i++)
        {
            int bits = i == 0 || i == anchor1 ? mode.indexBits - 1 : mode.indexBits;
            writer.Write((ulong)Index(c.indices, i), bits);
        }

        return new Block128 { lo = writer.lo, hi = writer.hi };
    }

    static int Index(ulong indices, int pixel) => (int)(indices >> (pixel * 4)) & 15;

    static ulong Invert(ulong indices, int pixels, int count)
    {
        for (int i = 0; i < 16; i++)
        {
            if ((pixels & (1 << i)) == 0)
                continue;

            ulong inverted = (ulong)(count - 1 - Index(indices, i));
            indices = (indices & ~(15ul << (i * 4))) | inverted << (i * 4);
        }

        return indices;
    }

    struct BitWriter
    {
        public ulong lo;
        public ulong hi;
        int pos;

        public void Write(ulong value, int bits)
        {
            if (pos < 64)
            {
                lo |= value << pos;
                if (pos + bits > 64)
                    hi |= value >> (64 - pos);
            }
            else
            {
                hi |= value << (pos - 64);
            }

            pos += bits;
        }
    }

    #region Partitions
    // The partitions for the two-subset modes, as given by the D3D11 BC7 specification.

    const int PartitionCount = 64;

    /// <summary>Bit <c>i</c> is set when pixel <c>i</c> belongs to the second subset.</summary>
    // csharpier-ignore
    static readonly ushort[] PartitionMasks =
    [
        0xCCCC, 0x8888, 0xEEEE, 0xECC8, 0xC880, 0xFEEC, 0xFEC8, 0xEC80,
        0xC800, 0xFFEC, 0xFE80, 0xE800, 0xFFE8, 0xFF00, 0xFFF0, 0xF000,
        0xF710, 0x008E, 0x7100, 0x08CE, 0x008C, 0x7310, 0x3100, 0x8CCE,
        0x088C, 0x3110, 0x6666, 0x366C, 0x17E8, 0x0FF0, 0x718E, 0x399C,
        0xAAAA, 0xF0F0, 0x5A5A, 0x33CC, 0x3C3C, 0x55AA, 0x9696, 0xA55A,
        0x73CE, 0x13C8, 0x324C, 0x3BDC, 0x6996, 0xC33C, 0x9966, 0x0660,
        0x0272, 0x04E4, 0x4E40, 0x2720, 0xC936, 0x936C, 0x39C6, 0x639C,
        0x9336, 0x9CC6, 0x817E, 0xE718, 0xCCF0, 0x0FCC, 0x7744, 0xEE22,
    ];

    /// <summary>
    /// The second subset's anchor pixel, whose index is stored without its top bit. The first
    /// subset's anchor is always pixel 0.
    /// </summary>
    // csharpier-ignore
    static readonly byte[] PartitionAnchors =
    [
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        15, 2, 8, 2, 2, 8, 8, 15, 2, 8, 2, 2, 8, 8, 2, 2,
        15, 15, 6, 8, 2, 8, 15, 15, 2, 8, 2, 2, 2, 15, 15, 6,
        6, 2, 6, 8, 15, 15, 2, 2, 15, 15, 15, 15, 15, 2, 2, 15,
    ];
    #endregion
}
