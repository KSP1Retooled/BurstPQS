using System.Runtime.CompilerServices;
using Unity.Mathematics;

namespace BurstPQS.Util.Textures;

/// <summary>
/// BC7 encoder covering modes 1, 3, 6 and 7, the same set texconv tries by default apart from
/// the separate-alpha modes 4 and 5.
/// </summary>
/// <remarks>
/// Like texconv, the two-subset modes only get a full fit for the partitions that rank best on
/// a rough estimate, and mode 7 is skipped for opaque blocks since 1 and 3 cover those.
/// The fast setting follows bc7e's fastest preset, which only tries mode 6.
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
    public static Block128 Encode(float4* px, bool fast)
    {
        if (fast)
            return EncodeFast(px);

        bool opaque = true;
        for (int i = 0; i < 16; i++)
            opaque &= px[i].w >= 255f;

        var best = EncodeShape(px, Mode6, 0, false);
        if (best.error <= 0f)
            return Pack(best);

        var shapes = stackalloc int[ShapeCandidates];
        RankShapes(px, shapes);

        for (int i = 0; i < ShapeCandidates; i++)
        {
            var candidate = EncodeShape(px, Mode1, shapes[i], false);
            if (candidate.error < best.error)
                best = candidate;

            candidate = EncodeShape(px, Mode3, shapes[i], false);
            if (candidate.error < best.error)
                best = candidate;

            if (opaque)
                continue;

            candidate = EncodeShape(px, Mode7, shapes[i], false);
            if (candidate.error < best.error)
                best = candidate;
        }

        return Pack(best);
    }

    // bc7e's fastest approach: mode 6 alone, with solid blocks going straight to the fit that
    // puts a palette entry on their colour.
    static Block128 EncodeFast(float4* px)
    {
        bool solid = true;
        for (int i = 1; i < 16; i++)
            solid &= math.all(px[i] == px[0]);

        if (!solid)
            return Pack(EncodeShape(px, Mode6, 0, true));

        var s = new Subset(px, 0xFFFF);
        var fit = new SubsetFit { error = float.PositiveInfinity };
        TrySolid(ref s, Mode6, px[0], ref fit);

        return Pack(
            new Candidate
            {
                mode = Mode6,
                error = fit.error,
                start0 = fit.start,
                end0 = fit.end,
                indices = fit.indices,
            }
        );
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

    static Candidate EncodeShape(float4* px, Mode mode, int shape, bool fast)
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
                ref candidate.indices,
                fast
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
                ref candidate.indices,
                fast
            )
            + FitSubset(
                px,
                mask,
                mode,
                out candidate.start1,
                out candidate.end1,
                ref candidate.indices,
                fast
            );

        return candidate;
    }

    // A subset's pixels packed into consecutive lanes, so Assign can test 8 at a time.
    struct Subset
    {
        public const int Used = 4;

        // 16 lanes for each of R, G, B, A, then Used: 1 for the lanes holding a pixel, 0 past
        // the end.
        fixed float lanes[5 * 16];
        public fixed byte position[16];
        public int count;
        public float4 sum;

        // The endpoint pairs already assigned. The least-squares solves land on the same
        // quantized endpoints often enough that skipping repeats saves a lot of work.
        const int TriedSlots = 128;
        fixed ulong tried[TriedSlots];
        int triedCount;

        public Subset(float4* px, int pixels)
        {
            this = default;
            for (int i = 0; i < 16; i++)
            {
                if ((pixels & (1 << i)) == 0)
                    continue;

                for (int ch = 0; ch < 4; ch++)
                    lanes[ch * 16 + count] = px[i][ch];
                lanes[Used * 16 + count] = 1f;
                position[count] = (byte)i;
                sum += px[i];
                count++;
            }
        }

        public float8 Lanes(int channel, int half)
        {
            fixed (float* p = lanes)
                return Load(p + channel * 16 + half * 8);
        }

        public bool FirstTry(Endpoint start, Endpoint end)
        {
            if (triedCount >= TriedSlots / 2)
                return true;

            // Channels hold at most 7 bits. The top bit keeps a key from reading as empty.
            ulong key = 1ul << 63 | (ulong)start.pbit << 57 | (ulong)end.pbit << 56;
            for (int ch = 0; ch < 4; ch++)
                key |= (ulong)start.bits[ch] << (ch * 7) | (ulong)end.bits[ch] << (28 + ch * 7);

            int slot = (int)(key * 0x9E3779B97F4A7C15ul >> 57);
            while (tried[slot] != 0)
            {
                if (tried[slot] == key)
                    return false;
                slot = (slot + 1) & (TriedSlots - 1);
            }

            tried[slot] = key;
            triedCount++;
            return true;
        }
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
        ref ulong indices,
        bool fast
    )
    {
        // Modes without alpha always decode it as 255, so the fit only aims for that.
        var target = stackalloc float4[16];
        for (int i = 0; i < 16; i++)
            target[i] = mode.alphaBits == 0 ? new float4(px[i].xyz, 255f) : px[i];

        var fit = new SubsetFit { error = float.PositiveInfinity };
        var s = new Subset(px, pixels);

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

        Try(ref s, mode, mean + axis * minT, mean + axis * maxT, ref fit);

        if (maxT > minT)
        {
            for (int iter = 0; iter < BlockMath.RefineIterations && fit.error > 0f; iter++)
            {
                if (!TryLeastSquares(ref s, mode, fit.indices, ref fit))
                    break;
            }

            if (!fast && fit.error > 0f)
                TryPerturbedIndices(ref s, pixels, mode, ref fit);
        }

        if (fit.error > 0f)
            TrySolid(ref s, mode, mean, ref fit);

        start = fit.start;
        end = fit.end;
        indices |= fit.indices;
        return fit.error;
    }

    const int SpreadSteps = 1;

    static void TryPerturbedIndices(ref Subset s, int pixels, Mode mode, ref SubsetFit fit)
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

            TryLeastSquares(ref s, mode, shifted, ref fit);
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

                TryLeastSquares(ref s, mode, spread, ref fit);
            }
        }
    }

    // Solves for the endpoints that best reproduce the targets given these indices.
    static bool TryLeastSquares(ref Subset s, Mode mode, ulong selected, ref SubsetFit fit)
    {
        int count = 1 << mode.indexBits;

        var weights = stackalloc float[16];
        for (int k = 0; k < 16; k++)
            weights[k] = Weight(Index(selected, s.position[k]), count) / 64f;

        float8 w0 = Load(weights) * s.Lanes(Subset.Used, 0),
            w1 = Load(weights + 8) * s.Lanes(Subset.Used, 1);

        float sumW = Sum(w0 + w1);
        float c = Sum(w0 * w0 + w1 * w1);
        float b = sumW - c;
        float a = s.count - sumW - b;

        var d1 = new float4(
            Sum(w0 * s.Lanes(0, 0) + w1 * s.Lanes(0, 1)),
            Sum(w0 * s.Lanes(1, 0) + w1 * s.Lanes(1, 1)),
            Sum(w0 * s.Lanes(2, 0) + w1 * s.Lanes(2, 1)),
            Sum(w0 * s.Lanes(3, 0) + w1 * s.Lanes(3, 1))
        );
        float4 total = s.sum;

        // Modes without alpha always decode it as 255, so the fit only aims for that.
        if (mode.alphaBits == 0)
        {
            d1.w = 255f * sumW;
            total.w = 255f * s.count;
        }

        float4 d0 = total - d1;

        float det = a * c - b * b;
        if (math.abs(det) < 1e-6f)
            return false;

        float inv = 1f / det;
        return Try(ref s, mode, (c * d0 - b * d1) * inv, (a * d1 - b * d0) * inv, ref fit);
    }

    // Quantizes the endpoints under every p-bit combination the mode allows.
    static bool Try(ref Subset s, Mode mode, float4 a, float4 b, ref SubsetFit fit)
    {
        var start0 = Quantize(a, mode, 0);
        var start1 = Quantize(a, mode, 1);
        var end0 = Quantize(b, mode, 0);
        var end1 = Quantize(b, mode, 1);

        bool improved = Evaluate(ref s, mode, start0, end0, ref fit);
        if (!mode.sharedPBit)
        {
            improved |= Evaluate(ref s, mode, start1, end0, ref fit);
            improved |= Evaluate(ref s, mode, start0, end1, ref fit);
        }
        improved |= Evaluate(ref s, mode, start1, end1, ref fit);

        return improved;
    }

    // Fits endpoints that put the mode's solid palette entry on the colour. Quantizing the
    // endpoints to the colour itself only reaches the values an endpoint can hold, while an
    // interpolated entry can land on nearly every value exactly.
    static void TrySolid(ref Subset s, Mode mode, float4 color, ref SubsetFit fit)
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

            // Alpha has either the colour's bits or none, so every channel can share them.
            SolidChannels(target, w, mode.colorBits, ref start, ref end);

            if (mode.alphaBits == 0)
            {
                start.bits.w = 0;
                start.value.w = 255;
                end.bits.w = 0;
                end.value.w = 255;
            }

            Evaluate(ref s, mode, start, end, ref fit);
        }
    }

    const int SolidSearchSteps = 4;

    // Searches start values around the colour, and for each the end values either side of the
    // one that would put the interpolated entry on it. Each channel keeps the first of its
    // closest pairs.
    static void SolidChannels(int4 c, int w, int bits, ref Endpoint pstart, ref Endpoint pend)
    {
        var start = pstart;
        var end = pend;

        int max = (1 << bits) - 1;
        int4 center = QuantizeBits(c, bits, start.pbit);
        int4 bestErr = int.MaxValue;

        for (int k = -SolidSearchSteps; k <= SolidSearchSteps; k++)
        {
            int4 lo = center + k;
            bool4 loValid = lo >= 0 & lo <= max;
            int4 loValue = Expand(lo << 1 | start.pbit, bits + 1);

            float4 ideal = (64f * (float4)c - (float4)((64 - w) * loValue)) / w;
            int4 guess = QuantizeBits(ideal, bits, end.pbit);

            for (int j = -1; j <= 1; j++)
            {
                int4 hi = guess + j;
                int4 hiValue = Expand(hi << 1 | end.pbit, bits + 1);
                int4 value = ((64 - w) * loValue + w * hiValue + 32) >> 6;
                int4 err = (value - c) * (value - c);
                bool4 better = loValid & hi >= 0 & hi <= max & err < bestErr;

                bestErr = math.select(bestErr, err, better);
                start.bits = math.select(start.bits, lo, better);
                start.value = math.select(start.value, loValue, better);
                end.bits = math.select(end.bits, hi, better);
                end.value = math.select(end.value, hiValue, better);
            }
        }

        pstart = start;
        pend = end;
    }

    static bool Evaluate(ref Subset s, Mode mode, Endpoint start, Endpoint end, ref SubsetFit fit)
    {
        // A repeat can't beat the fit, which is already at least as good as its first try.
        if (!s.FirstTry(start, end))
            return false;

        float error = Assign(ref s, mode, start, end, out ulong indices);
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

    // Picks the nearest palette entry for each pixel in the subset, 8 pixels per vector. Every
    // distance is a whole number well under 2^24, so the float sums are exact.
    static float Assign(ref Subset s, Mode mode, Endpoint start, Endpoint end, out ulong indices)
    {
        int count = 1 << mode.indexBits;
        bool upper = s.count > 8;

        float8 r0 = s.Lanes(0, 0),
            g0 = s.Lanes(1, 0),
            b0 = s.Lanes(2, 0),
            a0 = s.Lanes(3, 0);
        float8 r1 = s.Lanes(0, 1),
            g1 = s.Lanes(1, 1),
            b1 = s.Lanes(2, 1),
            a1 = s.Lanes(3, 1);

        float8 min0 = float.PositiveInfinity,
            min1 = float.PositiveInfinity;
        float8 index0 = 0f,
            index1 = 0f;

        for (int j = 0; j < count; j++)
        {
            int w = Weight(j, count);
            float4 p = ((64 - w) * start.value + w * end.value + 32) >> 6;

            float8 d = Distance(r0, g0, b0, a0, p);
            bool8 closer = d < min0;
            min0 = MathUtil.Select(min0, d, closer);
            index0 = MathUtil.Select(index0, j, closer);

            if (!upper)
                continue;

            d = Distance(r1, g1, b1, a1, p);
            closer = d < min1;
            min1 = MathUtil.Select(min1, d, closer);
            index1 = MathUtil.Select(index1, j, closer);
        }

        var lanes = stackalloc float4[8];
        lanes[0] = min0.lo;
        lanes[1] = min0.hi;
        lanes[2] = min1.lo;
        lanes[3] = min1.hi;
        lanes[4] = index0.lo;
        lanes[5] = index0.hi;
        lanes[6] = index1.lo;
        lanes[7] = index1.hi;

        var mins = (float*)lanes;
        var selected = mins + 16;

        float error = 0f;
        ulong packed = 0;
        int pixels = s.count;
        for (int k = 0; k < pixels; k++)
        {
            error += mins[k];
            packed |= (ulong)(int)selected[k] << (s.position[k] * 4);
        }

        indices = packed;
        return error;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static float8 Load(float* v) => new(*(float4*)v, *(float4*)(v + 4));

    static float Sum(float8 v) => math.csum(v.lo + v.hi);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static float8 Distance(float8 r, float8 g, float8 b, float8 a, float4 p)
    {
        float8 dr = r - p.x;
        float8 dg = g - p.y;
        float8 db = b - p.z;
        float8 da = a - p.w;
        return MathUtil.Mad(dr, dr, MathUtil.Mad(dg, dg, MathUtil.Mad(db, db, da * da)));
    }

    static int Weight(int index, int count) => Weights[count + index];

    // The interpolation weights for each palette size, starting at the palette size so that
    // count + index finds them.
    // csharpier-ignore
    static readonly byte[] Weights =
    [
        0, 0, 0, 0,
        0, 21, 43, 64,
        0, 9, 18, 27, 37, 46, 55, 64,
        0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64,
    ];

    // Each channel is stored as its top bits plus the p-bit as the lowest, and the decoder
    // replicates the high bits of that into whatever is left of 8.
    static Endpoint Quantize(float4 e, Mode mode, int pbit)
    {
        e = math.clamp(e, 0f, 255f);

        // Alpha has either the colour's bits or none, so every channel can share them.
        int bits = mode.colorBits;
        int max = (1 << bits) - 1;
        int4 guess = QuantizeBits(e, bits, pbit);

        var endpoint = new Endpoint { pbit = pbit };
        float4 bestErr = float.PositiveInfinity;
        for (int k = -1; k <= 1; k++)
        {
            // Clamping repeats an edge candidate, which can't win a second time.
            int4 c = math.clamp(guess + k, 0, max);
            int4 value = Expand(c << 1 | pbit, bits + 1);
            float4 d = value - e;
            bool4 better = d * d < bestErr;

            bestErr = math.select(bestErr, d * d, better);
            endpoint.bits = math.select(endpoint.bits, c, better);
            endpoint.value = math.select(endpoint.value, value, better);
        }

        if (mode.alphaBits == 0)
        {
            endpoint.bits.w = 0;
            endpoint.value.w = 255;
        }

        return endpoint;
    }

    // The stored bits whose expanded value lies nearest, give or take rounding.
    static int4 QuantizeBits(float4 v, int bits, int pbit) =>
        math.clamp(
            (int4)math.round((v * ((2 << bits) - 1) / 255f - pbit) * 0.5f),
            0,
            (1 << bits) - 1
        );

    static int4 Expand(int4 value, int bits) => value << (8 - bits) | value >> (2 * bits - 8);

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
