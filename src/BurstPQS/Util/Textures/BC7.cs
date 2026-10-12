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

        /// <summary>
        /// The palette entry a single colour is fitted to, or -1 where quantizing the endpoints
        /// to the colour itself is already exact.
        /// </summary>
        public int solidIndex;
    }

    static Mode Mode1 =>
        new()
        {
            number = 1,
            subsets = 2,
            colorBits = 6,
            indexBits = 3,
            sharedPBit = true,
            solidIndex = 2,
        };

    static Mode Mode3 =>
        new()
        {
            number = 3,
            subsets = 2,
            colorBits = 7,
            indexBits = 2,
            solidIndex = -1,
        };

    static Mode Mode6 =>
        new()
        {
            number = 6,
            subsets = 1,
            colorBits = 7,
            alphaBits = 7,
            indexBits = 4,
            solidIndex = 5,
        };

    static Mode Mode7 =>
        new()
        {
            number = 7,
            subsets = 2,
            colorBits = 5,
            alphaBits = 5,
            indexBits = 2,
            solidIndex = 1,
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

    struct SubsetFit
    {
        public Endpoint start;
        public Endpoint end;
        public ulong indices;
        public float error;
    }

    // Fits one subset's endpoints along its principal axis, then refines them with
    // least-squares passes over the chosen indices. Writes the subset's pixels into indices.
    //
    // The refinements after that follow bc7e: least-squares solves for the indices with their
    // extremes pulled in or spread out, and fitting the subset's average as a single colour.
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

        var fit = new SubsetFit { error = float.PositiveInfinity };

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

        Try(px, pixels, mode, mean + axis * minT, mean + axis * maxT, ref fit);

        if (maxT > minT)
        {
            for (int iter = 0; iter < BlockMath.RefineIterations && fit.error > 0f; iter++)
            {
                if (!TryLeastSquares(px, target, pixels, mode, fit.indices, ref fit))
                    break;
            }

            if (fit.error > 0f)
                TryPerturbedIndices(px, target, pixels, mode, ref fit);
        }

        if (fit.error > 0f)
            TrySolid(px, pixels, mode, mean, ref fit);

        start = fit.start;
        end = fit.end;
        indices |= fit.indices;
        return fit.error;
    }

    const int SpreadSteps = 1;

    static void TryPerturbedIndices(
        float4* px,
        float4* target,
        int pixels,
        Mode mode,
        ref SubsetFit fit
    )
    {
        int max = (1 << mode.indexBits) - 1;
        ulong selected = fit.indices;

        int lo = max;
        int hi = 0;
        for (int i = 0; i < 16; i++)
        {
            if ((pixels & (1 << i)) == 0)
                continue;

            int index = Index(selected, i);
            lo = math.min(lo, index);
            hi = math.max(hi, index);
        }

        // Pull the lowest indices up, then the highest down, then both.
        for (int variant = 0; variant < 3; variant++)
        {
            ulong shifted = 0;
            for (int i = 0; i < 16; i++)
            {
                if ((pixels & (1 << i)) == 0)
                    continue;

                int index = Index(selected, i);
                if (variant != 1 && index == lo && index < max)
                    index++;
                else if (variant != 0 && index == hi && index > 0)
                    index--;

                shifted |= (ulong)index << (i * 4);
            }

            TryLeastSquares(px, target, pixels, mode, shifted, ref fit);
        }

        // Remap the indices as though the endpoints sat a step or so outside or inside where
        // they are now.
        for (int low = -SpreadSteps; low <= 1; low++)
        {
            for (int high = max - 1; high <= max + SpreadSteps; high++)
            {
                if (low == 0 && high == max)
                    continue;

                ulong spread = 0;
                for (int i = 0; i < 16; i++)
                {
                    if ((pixels & (1 << i)) == 0)
                        continue;

                    float remapped = math.floor(
                        max * (float)(Index(selected, i) - low) / (high - low) + 0.5f
                    );
                    spread |= (ulong)math.clamp((int)remapped, 0, max) << (i * 4);
                }

                TryLeastSquares(px, target, pixels, mode, spread, ref fit);
            }
        }
    }

    // Solves for the endpoints that best reproduce the targets given these indices.
    static bool TryLeastSquares(
        float4* px,
        float4* target,
        int pixels,
        Mode mode,
        ulong selected,
        ref SubsetFit fit
    )
    {
        int count = 1 << mode.indexBits;

        float a = 0f,
            b = 0f,
            c = 0f;
        float4 d0 = 0f,
            d1 = 0f;
        for (int i = 0; i < 16; i++)
        {
            if ((pixels & (1 << i)) == 0)
                continue;

            float w = Weight(Index(selected, i), count) / 64f;
            float u = 1f - w;

            a += u * u;
            b += u * w;
            c += w * w;
            d0 += u * target[i];
            d1 += w * target[i];
        }

        float det = a * c - b * b;
        if (math.abs(det) < 1e-6f)
            return false;

        float inv = 1f / det;
        return Try(px, pixels, mode, (c * d0 - b * d1) * inv, (a * d1 - b * d0) * inv, ref fit);
    }

    // Quantizes the endpoints under every p-bit combination the mode allows.
    static bool Try(float4* px, int pixels, Mode mode, float4 a, float4 b, ref SubsetFit fit)
    {
        bool improved = false;

        int combinations = mode.sharedPBit ? 2 : 4;
        for (int p = 0; p < combinations; p++)
        {
            int startPBit = mode.sharedPBit ? p : p & 1;
            int endPBit = mode.sharedPBit ? p : p >> 1;

            var start = Quantize(a, mode, startPBit);
            var end = Quantize(b, mode, endPBit);
            improved |= Evaluate(px, pixels, mode, start, end, ref fit);
        }

        return improved;
    }

    // Fits endpoints that put the mode's solid palette entry on the colour. Quantizing the
    // endpoints to the colour itself only reaches the values an endpoint can hold, while an
    // interpolated entry can land on nearly every value exactly.
    static void TrySolid(float4* px, int pixels, Mode mode, float4 color, ref SubsetFit fit)
    {
        if (mode.solidIndex < 0)
            return;

        int w = Weight(mode.solidIndex, 1 << mode.indexBits);
        int4 target = (int4)math.round(math.clamp(color, 0f, 255f));

        int combinations = mode.sharedPBit ? 2 : 4;
        for (int p = 0; p < combinations; p++)
        {
            var start = new Endpoint { pbit = mode.sharedPBit ? p : p & 1 };
            var end = new Endpoint { pbit = mode.sharedPBit ? p : p >> 1 };

            for (int ch = 0; ch < 4; ch++)
            {
                int bits = ch < 3 ? mode.colorBits : mode.alphaBits;
                if (bits == 0)
                {
                    start.value[ch] = 255;
                    end.value[ch] = 255;
                    continue;
                }

                SolidChannel(target[ch], w, bits, ch, ref start, ref end);
            }

            Evaluate(px, pixels, mode, start, end, ref fit);
        }
    }

    const int SolidSearchSteps = 4;

    static void SolidChannel(int c, int w, int bits, int ch, ref Endpoint start, ref Endpoint end)
    {
        int max = (1 << bits) - 1;
        int center = QuantizeChannel(c, bits, start.pbit);
        int bestErr = int.MaxValue;

        int first = math.max(center - SolidSearchSteps, 0);
        int last = math.min(center + SolidSearchSteps, max);
        for (int lo = first; lo <= last; lo++)
        {
            int loValue = Expand(lo << 1 | start.pbit, bits + 1);

            // The end value that puts the interpolated entry closest to c.
            float ideal = (64f * c - (64 - w) * loValue) / w;
            int guess = QuantizeChannel(ideal, bits, end.pbit);

            for (int hi = math.max(guess - 1, 0); hi <= math.min(guess + 1, max); hi++)
            {
                int hiValue = Expand(hi << 1 | end.pbit, bits + 1);
                int value = ((64 - w) * loValue + w * hiValue + 32) >> 6;
                int err = (value - c) * (value - c);
                if (err >= bestErr)
                    continue;

                bestErr = err;
                start.bits[ch] = lo;
                start.value[ch] = loValue;
                end.bits[ch] = hi;
                end.value[ch] = hiValue;
            }
        }
    }

    static bool Evaluate(
        float4* px,
        int pixels,
        Mode mode,
        Endpoint start,
        Endpoint end,
        ref SubsetFit fit
    )
    {
        float error = Assign(px, pixels, mode, start, end, out ulong indices);
        if (error >= fit.error)
            return false;

        fit = new SubsetFit
        {
            start = start,
            end = end,
            indices = indices,
            error = error,
        };
        return true;
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

    // Each channel is stored as its top bits plus the p-bit as the lowest, and the decoder
    // replicates the high bits of that into whatever is left of 8.
    static Endpoint Quantize(float4 e, Mode mode, int pbit)
    {
        e = math.clamp(e, 0f, 255f);

        var endpoint = new Endpoint { pbit = pbit };

        for (int ch = 0; ch < 4; ch++)
        {
            int bits = ch < 3 ? mode.colorBits : mode.alphaBits;
            if (bits == 0)
            {
                endpoint.value[ch] = 255;
                continue;
            }

            int max = (1 << bits) - 1;
            int guess = QuantizeChannel(e[ch], bits, pbit);

            float bestErr = float.PositiveInfinity;
            for (int c = math.max(guess - 1, 0); c <= math.min(guess + 1, max); c++)
            {
                int value = Expand(c << 1 | pbit, bits + 1);
                float d = value - e[ch];
                if (d * d >= bestErr)
                    continue;

                bestErr = d * d;
                endpoint.bits[ch] = c;
                endpoint.value[ch] = value;
            }
        }

        return endpoint;
    }

    static int QuantizeChannel(float v, int bits, int pbit) =>
        math.clamp(
            (int)math.round((v * ((2 << bits) - 1) / 255f - pbit) * 0.5f),
            0,
            (1 << bits) - 1
        );

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
